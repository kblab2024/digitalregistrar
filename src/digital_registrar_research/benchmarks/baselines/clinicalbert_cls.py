"""
ClinicalBERT multi-head classifier baseline.

One shared Bio_ClinicalBERT encoder + one linear head per categorical
(`Literal[...]`) field. Trained on the pooled cmuh.train + tcga.train
split, evaluated per-dataset (cmuh.test, tcga.test).

Why this design
---------------
ClinicalBERT (Alsentzer et al., 2019) is encoder-only and cannot generate
structured outputs. We therefore reduce every categorical field to a
classification task over the `[CLS]` pooled embedding. Nullable fields
use an extra "null" class so the model can say "absent / not mentioned".

Training across organs is pooled because the annotated pool is modest;
per-organ fine-tuning would severely undertrain a 110M-parameter encoder.
The cross-corpus contract is **train on full CMUH, predict on full TCGA**:
disjointness is guaranteed by the dataset boundary, enforced at predict
time by a check that the predict datasets do not overlap the checkpoint's
training datasets. Evaluation stays per-dataset so per-corpus accuracy
stays visible.

Scope covered
-------------
- All fields in the DSPy signatures declared with `Literal[...]` type.
- Boolean fields (True / False / null → 3-way classification).
- The top-level `cancer_category` field.

Scope NOT covered (by design — see docs/literature_review.md for the
architectural-scope argument):
- Nested list fields (`margins`, `biomarkers`, `regional_lymph_node`).
- Continuous numeric fields (tumor_size in mm, distance in mm) — see
  `clinicalbert_qa.py` instead.
- Free-text fields (description, station_name, etc.).

Usage:
    # Default cross-corpus train (CMUH only), 5-organ scope, dummy data:
    python -m digital_registrar_research.benchmarks.baselines.clinicalbert_cls \
        --phase train --data-root dummy

    # Predict on TCGA (held out from CMUH training):
    python -m digital_registrar_research.benchmarks.baselines.clinicalbert_cls \
        --phase predict --data-root dummy --dataset tcga
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

from ...paths import BENCHMARKS_RESULTS
from .. import organs as _organs
from ..eval.scope import (
    CANCER_CATEGORIES,
    CATEGORICAL_FIELDS,
    get_field_value,
)
from ._data import load_cases, load_predict_cases, per_dataset_counts

_LOGGER = logging.getLogger(__name__)

MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cross-corpus baseline scope: the 5 organs present in BOTH TCGA and CMUH
# (= TCGA's full set, since TCGA's organ list is a subset of CMUH's).
# Configured in configs/organ_code.yaml; do not duplicate the literal here.
DEFAULT_ORGANS = list(_organs.common_organs("cmuh", "tcga"))
DEFAULT_DATASETS = list(_organs.all_datasets())


# --- Dataset ------------------------------------------------------------------

class PathologyCases(Dataset):
    def __init__(self, cases: list[dict], tokenizer,
                 field_to_idx: dict[str, dict[str, int]]):
        self.cases = cases
        self.tok = tokenizer
        self.field_to_idx = field_to_idx

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> dict:
        c = self.cases[i]
        report = Path(c["report_path"]).read_text(encoding="utf-8")
        with open(c["annotation_path"], encoding="utf-8") as f:
            gold = json.load(f)

        enc = self.tok(
            report, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        labels: dict[str, int] = {}
        for field, idx_map in self.field_to_idx.items():
            val = get_field_value(gold, field)
            key = "null" if val is None else str(val).lower()
            labels[field] = idx_map.get(key, idx_map.get("null", 0))

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate(batch: list[dict]) -> dict:
    keys_labels = batch[0]["labels"].keys()
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": {
            k: torch.tensor([b["labels"][k] for b in batch], dtype=torch.long)
            for k in keys_labels
        },
    }


# --- Model --------------------------------------------------------------------

class MultiHeadClassifier(nn.Module):
    def __init__(self, field_cardinalities: dict[str, int]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            f: nn.Linear(self.encoder.config.hidden_size, k)
            for f, k in field_cardinalities.items()
        })

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return {f: head(h) for f, head in self.heads.items()}


# --- Vocab / loaders ----------------------------------------------------------

def build_field_vocab() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    field_to_idx: dict[str, dict[str, int]] = {}
    for field, options in {**CATEGORICAL_FIELDS,
                            "cancer_category": CANCER_CATEGORIES}.items():
        vocab = {str(v).lower(): i for i, v in enumerate(options)}
        vocab.setdefault("null", len(vocab))
        field_to_idx[field] = vocab
    card = {f: len(v) for f, v in field_to_idx.items()}
    return field_to_idx, card


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# --- Train / predict ----------------------------------------------------------

def train(args) -> None:
    seed = int(getattr(args, "seed", 42))
    set_seed(seed)  # random / numpy / torch (CPU + all CUDA devices) before head init
    field_to_idx, card = build_field_vocab()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MultiHeadClassifier(card).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    organs = set(parse_csv(args.organs))
    datasets = parse_csv(args.datasets)

    cases = load_cases(
        datasets=datasets,
        root=Path(args.data_root),
        organs=organs,
        included_only=args.included_only,
    )
    counts = per_dataset_counts(cases)
    pretty = ", ".join(f"{d}: {n}" for d, n in sorted(counts.items()))
    print(f"Training on {len(cases)} cases ({pretty})"
          f"{' [included-only]' if args.included_only else ''}"
          f"  organs={sorted(organs)}")
    if not cases:
        raise SystemExit("no training cases — check --data-root, --datasets, --organs")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(
        PathologyCases(cases, tok, field_to_idx),
        batch_size=4, shuffle=True, collate_fn=collate,
        generator=g, num_workers=0,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(ids, mask)
            loss = torch.stack([
                loss_fn(logits[f], batch["labels"][f].to(DEVICE))
                for f in logits
            ]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss={total / len(train_loader):.4f}")

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "field_to_idx": field_to_idx,
                "card": card,
                "included_only": args.included_only,
                "organs": sorted(organs),
                "datasets": datasets,
                "n_train_cases": len(cases),
                "per_dataset_counts": counts,
                "seed": seed}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def predict(args) -> None:
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    field_to_idx, card = ckpt["field_to_idx"], ckpt["card"]
    idx_to_field = {f: {i: v for v, i in vocab.items()}
                    for f, vocab in field_to_idx.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = MultiHeadClassifier(card).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()

    organs = set(parse_csv(args.organs))
    if args.dataset == "both":
        datasets = parse_csv(args.datasets)
    else:
        datasets = [args.dataset]

    # Leakage guard: refuse to predict on a dataset that was in the
    # checkpoint's training set. The cross-corpus baseline relies on
    # train and predict corpora being disjoint (CMUH-train, TCGA-test).
    train_datasets = set(ckpt.get("datasets") or [])
    if not train_datasets:
        _LOGGER.warning(
            "checkpoint lacks 'datasets' metadata — leakage guard disabled. "
            "Retrain with the current train_bert.py.")
    else:
        overlap = train_datasets & set(datasets)
        if overlap:
            raise SystemExit(
                f"refusing to predict: dataset(s) {sorted(overlap)} were "
                f"in the checkpoint's training set "
                f"({sorted(train_datasets)}). Cross-corpus baseline "
                f"predicts on datasets disjoint from training."
            )

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
        enc = tok(report, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        preds = {f: idx_to_field[f][logit.argmax(dim=-1).item()]
                 for f, logit in logits.items()}

        cancer_category = preds.pop("cancer_category", None)
        if cancer_category == "null":
            cancer_category = None
        cancer_data = {
            f: (None if v == "null" else v) for f, v in preds.items()
        }
        result = {
            "cancer_excision_report": cancer_category is not None,
            "cancer_category": cancer_category,
            "cancer_category_others_description": None,
            "cancer_data": cancer_data,
        }

        # Canonical layout: <out>/<organ_n>/<case_id>.json. The caller
        # (scripts/baselines/run_bert.py) sets <out> to
        # {folder}/results/predictions/{dataset}/clinicalbert/cls/.
        organ_dir = out_root / case["organ_n"]
        organ_dir.mkdir(parents=True, exist_ok=True)
        with (organ_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "predict"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--ckpt", default="ckpts/clinicalbert_cls.pt")
    ap.add_argument("--out", default=str(BENCHMARKS_RESULTS / "clinicalbert_cls"),
                    help="Output dir; per-organ subdirs are created under it.")
    ap.add_argument("--data-root", default="dummy",
                    help="Root containing data/<dataset>/ subtrees (default: dummy).")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS),
                    help="CSV of cancer_category values to include.")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="CSV of dataset names. Train pools all of them; predict "
                         "uses these unless --dataset overrides.")
    ap.add_argument("--dataset", default="both",
                    choices=["cmuh", "tcga", "both"],
                    help="Predict-time dataset selector (default: both). 'both' uses --datasets.")
    ap.add_argument(
        "--included-only", action="store_true",
        help="Drop cases where cancer_excision_report is False (no organ-specific "
             "fields to learn from).",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for head init + data-shuffle order.")
    args = ap.parse_args()

    if args.phase == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
