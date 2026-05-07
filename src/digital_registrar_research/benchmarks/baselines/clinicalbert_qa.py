"""
ClinicalBERT extractive QA baseline.

For numeric / span-style fields (tumor size, AJCC version, grade as
integer in colorectal, etc.) we frame extraction as SQuAD-style
extractive QA:
    question = "What is the tumor size in mm?"
    context  = <report>
and train a span-prediction head over Bio_ClinicalBERT.

Design notes
------------
* Span supervision is **silver / weak**: at train time we auto-locate
  each gold numeric value inside the report with a best-effort
  substring search (with mm/cm variants). Cases where the value is not
  findable in the text are dropped (counted in per-organ stats).
* At inference we ask the per-organ question subset against the test
  report and keep the highest-confidence span per field.
* Training uses every CMUH case (cross-corpus contract: TCGA is held
  out for the LLM-comparable evaluation). The annotated pool per organ
  is modest, so we accept undertraining as the methodologically honest
  baseline rather than pool TCGA in and lose held-out status.
* This module complements `clinicalbert_cls.py` (which handles
  categorical / boolean / enum-grade fields). Together they populate
  the BERT-eligible fields for the head-to-head comparison.

Scope NOT covered: nested lists (`margins`, `biomarkers`,
`regional_lymph_node`) and free-text fields. These are reported as N/A
in the results table — encoder-only architectures cannot emit
variable-length nested objects without an additional generative decoder.

Usage:
    python -m digital_registrar_research.benchmarks.baselines.clinicalbert_qa \
        --phase train --data-root dummy
    python -m digital_registrar_research.benchmarks.baselines.clinicalbert_qa \
        --phase predict --data-root dummy --dataset both
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from ...paths import BENCHMARKS_RESULTS
from .. import organs as _organs
from ..eval.scope import get_field_value
from ._data import load_cases, load_predict_cases, per_dataset_counts

_LOGGER = logging.getLogger(__name__)

MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DOC_STRIDE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cross-corpus baseline scope: the 5 organs present in BOTH TCGA and CMUH
# (= TCGA's full set, since TCGA's organ list is a subset of CMUH's).
# Configured in configs/organ_code.yaml; do not duplicate the literal here.
DEFAULT_ORGANS = list(_organs.common_organs("cmuh", "tcga"))
DEFAULT_DATASETS = list(_organs.all_datasets())

# Per-organ question bank. Only fields whose gold value is a literal
# number that should appear in the report belong here. Enum-valued
# fields (breast nuclear_grade, esophagus grade, ...) live in
# clinicalbert_cls.py.
ORGAN_QUESTIONS: dict[str, dict[str, str]] = {
    "breast": {
        "tumor_size":   "What is the tumor size in millimeters?",
        "dcis_size":    "What is the size of the ductal carcinoma in situ in millimeters?",
        "ajcc_version": "Which AJCC edition is used for staging?",
    },
    "colorectal": {
        "grade":         "What is the histologic grade?",
        "tumor_budding": "What is the tumor budding score?",
        "ajcc_version":  "Which AJCC edition is used for staging?",
    },
    "esophagus": {
        "ajcc_version": "Which AJCC edition is used for staging?",
    },
    "liver": {
        "tumor_size":   "What is the tumor size in millimeters?",
        "ajcc_version": "Which AJCC edition is used for staging?",
    },
    "stomach": {
        "ajcc_version": "Which AJCC edition is used for staging?",
    },
    "thyroid": {
        "tumor_size":   "What is the tumor size in millimeters?",
        "ajcc_version": "Which AJCC edition is used for staging?",
    },
}


# --- Silver span alignment ----------------------------------------------------

def find_span(report: str, value) -> tuple[int, int] | None:
    """Best-effort locate the gold value as a literal substring."""
    if value is None:
        return None
    s = str(value)
    variants = [s, f"{s} mm", f"{s}mm"]
    if s.isdigit():
        cm = float(s) / 10
        variants += [
            f"{s} cm", f"{s}cm",
            f"{cm:g} cm", f"{cm:g}cm",
            f"{cm:.1f} cm", f"{cm:.1f}cm",
        ]
    lower = report.lower()
    for v in variants:
        idx = lower.find(v.lower())
        if idx != -1:
            return idx, idx + len(v)
    m = re.search(rf"\b{re.escape(s)}\b", report)
    if m:
        return m.start(), m.end()
    return None


class QADataset(Dataset):
    """(question, context, answer_span) triples from gold annotations."""

    def __init__(self, cases: list[dict], tokenizer):
        self.examples: list[dict] = []
        self.tok = tokenizer
        seen: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for c in cases:
            organ = c.get("cancer_category")
            qbank = ORGAN_QUESTIONS.get(organ, {})
            if not qbank:
                continue
            report = Path(c["report_path"]).read_text(encoding="utf-8")
            with open(c["annotation_path"], encoding="utf-8") as f:
                gold = json.load(f)
            for field, question in qbank.items():
                seen[(organ, field)][1] += 1
                val = get_field_value(gold, field)
                span = find_span(report, val) if val is not None else None
                if span is None:
                    continue
                seen[(organ, field)][0] += 1
                self.examples.append({
                    "question": question,
                    "context": report,
                    "answer_start": span[0],
                    "answer_text": report[span[0]:span[1]],
                })
        print(f"Silver-aligned training examples: {len(self.examples)}")
        for (organ, field), (hits, total) in sorted(seen.items()):
            print(f"  {organ}.{field}: {hits}/{total} aligned")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        enc = self.tok(
            ex["question"], ex["context"],
            max_length=MAX_LEN, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=False,
            return_offsets_mapping=True, padding="max_length",
        )
        offsets = enc.pop("offset_mapping")
        start_char = ex["answer_start"]
        end_char = start_char + len(ex["answer_text"])
        start_idx = end_idx = 0
        for i_tok, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_idx = i_tok
            if s < end_char <= e:
                end_idx = i_tok
                break
        enc["start_positions"] = start_idx
        enc["end_positions"] = end_idx
        return {k: torch.tensor(v) for k, v in enc.items()}


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def train(args) -> None:
    seed = int(getattr(args, "seed", 42))
    set_seed(seed)  # random / numpy / torch (CPU + all CUDA devices) before head init
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID).to(DEVICE)

    organs = set(parse_csv(args.organs))
    datasets = parse_csv(args.datasets)
    cases = load_cases(
        datasets=datasets,
        root=Path(args.data_root),
        organs=organs,
        included_only=True,  # QA needs a real cancer_category to pick a question bank
    )
    counts = per_dataset_counts(cases)
    pretty = ", ".join(f"{d}: {n}" for d, n in sorted(counts.items()))
    print(f"Training cases: {len(cases)} ({pretty})  organs={sorted(organs)}")
    if not cases:
        raise SystemExit("no training cases — check --data-root, --datasets, --organs")

    ds = QADataset(cases, tok)
    if len(ds) == 0:
        raise SystemExit("no silver-aligned training examples — every gold value was "
                         "missing from its report. Check --data-root or the question bank.")
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=g, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=len(loader) * args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            scheduler.step()
            total += out.loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(loader):.4f}")

    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.ckpt)
    tok.save_pretrained(args.ckpt)
    # Sidecar with training metadata. HuggingFace save_pretrained doesn't
    # carry custom keys, so we write a separate file. The predict step
    # reads this back to enforce dataset-disjointness against the predict
    # corpus.
    train_meta = {
        "method": "clinicalbert_qa",
        "datasets": datasets,
        "organs": sorted(organs),
        "n_train_cases": len(cases),
        "per_dataset_counts": counts,
        "seed": seed,
    }
    (Path(args.ckpt) / "_train_meta.json").write_text(
        json.dumps(train_meta, indent=2), encoding="utf-8",
    )
    print(f"Saved to {args.ckpt}")


def predict(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt).to(DEVICE)
    model.eval()

    organs = set(parse_csv(args.organs))
    datasets = parse_csv(args.datasets) if args.dataset == "both" else [args.dataset]

    # Leakage guard: refuse to predict on a dataset that was in the
    # checkpoint's training set. Mirrors the CLS predict path.
    train_meta_path = Path(args.ckpt) / "_train_meta.json"
    if train_meta_path.is_file():
        with train_meta_path.open(encoding="utf-8") as f:
            train_datasets = set(json.load(f).get("datasets") or [])
        if train_datasets:
            overlap = train_datasets & set(datasets)
            if overlap:
                raise SystemExit(
                    f"refusing to predict: dataset(s) {sorted(overlap)} "
                    f"were in the QA checkpoint's training set "
                    f"({sorted(train_datasets)})."
                )
    else:
        _LOGGER.warning(
            "%s not found — leakage guard disabled. Retrain with the "
            "current train_bert.py to enable.", train_meta_path)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = load_predict_cases(
        datasets=datasets,
        root=Path(args.data_root),
        organs=organs,
    )
    counts = per_dataset_counts(cases)
    pretty = ", ".join(f"{d}: {n}" for d, n in sorted(counts.items()))
    print(f"Predicting on {len(cases)} cases ({pretty})  "
          f"organs={sorted(organs)}")
    if not cases:
        reports_dir = Path(args.data_root) / "data" / datasets[0] / "reports"
        raise SystemExit(
            f"refusing to predict on 0 cases for {datasets} — check that "
            f"{reports_dir} is populated and --organs={sorted(organs)} "
            f"matches non-empty organ_n dirs there.")

    for case in tqdm(cases, desc="predict"):
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        organ = case.get("cancer_category")
        # Fall back to the union of every organ's questions when the
        # case has no category (e.g. 'others' / non-excision).
        qbank = ORGAN_QUESTIONS.get(organ) or {
            f: q for org_q in ORGAN_QUESTIONS.values() for f, q in org_q.items()
        }
        cancer_data: dict = {}
        for field, question in qbank.items():
            enc = tok(question, report, max_length=MAX_LEN,
                      truncation="only_second", padding="max_length",
                      return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0]
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            s = out.start_logits.argmax(dim=-1).item()
            e = out.end_logits.argmax(dim=-1).item()
            if e < s:
                continue
            start_char = int(offsets[s][0])
            end_char = int(offsets[e][1])
            answer = report[start_char:end_char].strip()
            m = re.search(r"\d+(?:\.\d+)?", answer)
            if m:
                val = float(m.group())
                cancer_data[field] = int(val) if val.is_integer() else val

        # QA only populates cancer_data scalars — categorical / cancer_category
        # routing is handled by clinicalbert_cls.py. The merge step under
        # scripts/baselines/run_bert.py reconciles the two heads into
        # clinicalbert/merged/.
        result = {
            "cancer_excision_report": None,
            "cancer_category": None,
            "cancer_category_others_description": None,
            "cancer_data": cancer_data,
        }
        # Canonical layout: <out>/<organ_n>/<case_id>.json.
        organ_dir = out_root / case["organ_n"]
        organ_dir.mkdir(parents=True, exist_ok=True)
        with (organ_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_qa")
    ap.add_argument("--out", default=str(BENCHMARKS_RESULTS / "clinicalbert_qa"),
                    help="Output dir; per-dataset subdirs are created under it.")
    ap.add_argument("--data-root", default="dummy",
                    help="Root containing data/<dataset>/ subtrees (default: dummy).")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS),
                    help="CSV of cancer_category values to include.")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="CSV of dataset names to pool over.")
    ap.add_argument("--dataset", default="both",
                    choices=["cmuh", "tcga", "both"],
                    help="Predict-time dataset selector. 'both' uses --datasets.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for head init + data-shuffle order.")
    args = ap.parse_args()

    if args.phase == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
