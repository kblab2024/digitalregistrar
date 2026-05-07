#!/usr/bin/env python3
"""Orchestrate the gpt-oss multi-run experiment (Part B).

Reads the frozen protocol YAML (e.g. configs/multirun_gpt_oss.yaml), then
for each (run_k, case_id) tuple:

  1. Loads the raw report from the CMUH dataset folder.
  2. Selects the organ schema (from the gold annotation's cancer_category).
  3. Formats the prompt with the frozen template + few-shot examples.
  4. Submits to an OpenAI-compatible endpoint (vLLM by default) with
     seed=seed_k, temperature=config.decoding.temperature.
  5. Parses the JSON response. On parse failure retries up to N times
     with bumped seeds; after exhaustion emits a sentinel parse-error
     prediction so downstream eval counts it honestly (not silently).
  6. Writes `<run_dir>/<case_id>.json` and a row to `_log.jsonl`.

The script is idempotent/resumable: valid existing outputs are skipped.

Usage:
    python scripts/run_gpt_oss_multirun.py \
        --config configs/multirun_gpt_oss.yaml \
        --output workspace/results/benchmarks/gpt_oss/ \
        [--runs run1 run2] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("gpt_oss_multirun")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for _run_id

from _run_id import format_run_id, machine_slug  # noqa: E402


# --- Config loading / hashing ------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Protocol:
    experiment_id: str
    config_path: Path
    config_hash: str
    model_name: str
    serving: str
    api_base: str
    api_key: str
    timeout_s: int
    temperature: float
    top_p: float
    max_tokens: int
    stop: list
    seeds: list[int]
    prompt_template_path: Path
    prompt_template_hash: str
    few_shot_path: Path
    few_shot_hash: str
    scope_name: str
    cases_glob: str
    gold_glob: str
    max_parse_retries: int
    max_transient_retries: int
    transient_backoff_s: list[int]
    parse_error_rate_max: float
    missing_case_rate_max: float
    raw: dict = field(default_factory=dict)


def load_protocol(path: Path) -> Protocol:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_hash = _sha256_file(path)

    def _check(val, where):
        if isinstance(val, str) and "FILL_IN" in val:
            raise ValueError(f"Protocol has unfilled placeholder at {where}: {val}")

    _check(cfg["model"].get("checkpoint_hash"), "model.checkpoint_hash")
    _check(cfg["model"].get("serving_version"), "model.serving_version")
    _check(cfg["prompt"].get("template_hash"), "prompt.template_hash")
    _check(cfg["prompt"]["few_shot"].get("hash"), "prompt.few_shot.hash")
    _check(cfg["schema"].get("scope_hash"), "schema.scope_hash")
    _check(cfg["data"]["cases_glob"], "data.cases_glob")
    _check(cfg["data"]["gold_glob"], "data.gold_glob")

    tpl_path = REPO_ROOT / cfg["prompt"]["template_path"]
    fs_path = REPO_ROOT / cfg["prompt"]["few_shot"]["path"]
    return Protocol(
        experiment_id=cfg["experiment_id"],
        config_path=path,
        config_hash=cfg_hash,
        model_name=cfg["model"]["name"],
        serving=cfg["model"]["serving"],
        api_base=cfg["endpoint"]["api_base"],
        api_key=cfg["endpoint"].get("api_key") or "EMPTY",
        timeout_s=int(cfg["endpoint"].get("timeout_s", 120)),
        temperature=float(cfg["decoding"]["temperature"]),
        top_p=float(cfg["decoding"].get("top_p", 1.0)),
        max_tokens=int(cfg["decoding"].get("max_tokens", 2048)),
        stop=list(cfg["decoding"].get("stop") or []),
        seeds=list(cfg["runs"]["seeds"]),
        prompt_template_path=tpl_path,
        prompt_template_hash=cfg["prompt"]["template_hash"],
        few_shot_path=fs_path,
        few_shot_hash=cfg["prompt"]["few_shot"]["hash"],
        scope_name=cfg["schema"]["scope"],
        cases_glob=cfg["data"]["cases_glob"],
        gold_glob=cfg["data"]["gold_glob"],
        max_parse_retries=int(cfg["retry"]["max_parse_retries"]),
        max_transient_retries=int(cfg["retry"]["max_transient_retries"]),
        transient_backoff_s=list(cfg["retry"]["transient_backoff_s"]),
        parse_error_rate_max=float(cfg["acceptance"]["parse_error_rate_max"]),
        missing_case_rate_max=float(cfg["acceptance"]["missing_case_rate_max"]),
        raw=cfg,
    )


# --- Prompt construction -----------------------------------------------------

def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_few_shot(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _organ_of(gold: dict) -> str | None:
    v = gold.get("cancer_category")
    if v is None:
        return None
    return str(v).strip().lower()


def _load_schema_for_organ(organ: str | None) -> dict | None:
    """Load the organ-specific JSON schema used to ground the prompt."""
    if organ is None:
        return None
    from digital_registrar_research import paths as dr_paths
    p = dr_paths.SCHEMAS_DATA / f"{organ}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def format_prompt(template: str, few_shot: list[dict], report: str,
                  organ: str | None, schema: dict | None) -> list[dict]:
    """Render the chat-format message list consumed by the OpenAI client.

    The template is treated as the system message; few-shot examples
    become alternating user/assistant pairs; the final user message
    carries the current report plus its schema reference.

    Using a simple .format() dispatch keeps this dependency-light — swap
    in Jinja2 at the call site if the template grows.
    """
    system = template.format(organ=organ or "unknown",
                             schema_json=json.dumps(schema or {}, indent=2))
    msgs = [{"role": "system", "content": system}]
    for ex in few_shot:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["output"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": report})
    return msgs


# --- Case discovery ----------------------------------------------------------

@dataclass
class CaseRef:
    case_id: str
    report_path: Path
    gold_path: Path
    organ: str | None


def discover_cases(cases_glob: str, gold_glob: str) -> list[CaseRef]:
    """Enumerate cases for which we have both a report file and a gold
    annotation. Matches by stem (`<case_id>.txt` ↔ `<case_id>_gold.json`).
    """
    report_paths = list(REPO_ROOT.glob(cases_glob))
    gold_paths = list(REPO_ROOT.glob(gold_glob))
    reports_by_id = {p.stem: p for p in report_paths}
    gold_by_id = {}
    for p in gold_paths:
        stem = p.stem
        if stem.endswith("_gold"):
            gold_by_id[stem[:-5]] = p
        else:
            gold_by_id[stem] = p

    cases: list[CaseRef] = []
    for cid, gpath in sorted(gold_by_id.items()):
        rpath = reports_by_id.get(cid)
        if rpath is None:
            logger.warning("[discover] case %s: missing report", cid)
            continue
        try:
            with gpath.open(encoding="utf-8") as f:
                gold = json.load(f)
            organ = _organ_of(gold)
        except Exception as exc:
            logger.warning("[discover] case %s: bad gold (%s)", cid, exc)
            organ = None
        cases.append(CaseRef(cid, rpath, gpath, organ))
    return cases


# --- Inference client (OpenAI-compatible) ------------------------------------

def _make_client(proto: Protocol):
    """Return an initialised OpenAI client pointing at the configured
    endpoint. Import is lazy so the module imports cleanly without the
    dependency at unit-test time."""
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=proto.api_key, base_url=proto.api_base,
                  timeout=proto.timeout_s)


def _call_once(client, proto: Protocol, messages: list[dict], seed: int
               ) -> dict:
    """Single inference call. Returns {'content', 'tokens_in', 'tokens_out',
    'raw_response'}. Raises on transient errors so the caller can retry.
    """
    resp = client.chat.completions.create(
        model=proto.model_name,
        messages=messages,
        temperature=proto.temperature,
        top_p=proto.top_p,
        max_tokens=proto.max_tokens,
        stop=proto.stop or None,
        seed=seed,
    )
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return {
        "content": choice.message.content or "",
        "tokens_in": getattr(usage, "prompt_tokens", None),
        "tokens_out": getattr(usage, "completion_tokens", None),
        "finish_reason": getattr(choice, "finish_reason", None),
    }


def _parse_prediction(content: str) -> dict | None:
    """Extract a JSON object from the model's response. Accepts both pure
    JSON and ````json ... ```` fenced blocks.
    Returns None on parse failure."""
    text = content.strip()
    if text.startswith("```"):
        # Strip the first fence and everything after the closing fence.
        parts = text.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("{"):
                text = chunk
                break
    # Best-effort: find the first '{' and parse from there.
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.loads(text[i:])
    except json.JSONDecodeError:
        pass
    # Fallback: try to find a balanced trailing brace.
    for j in range(len(text), i, -1):
        try:
            return json.loads(text[i:j])
        except Exception:
            continue
    return None


# --- Main per-case loop ------------------------------------------------------

def _run_valid_pred(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("_parse_error")


def process_case(client, proto: Protocol, case: CaseRef, seed: int,
                 template: str, few_shot: list[dict],
                 run_dir: Path, log_file) -> dict:
    """Predict for one (run, case). Writes the prediction JSON and a log
    row. Returns the log row for aggregation."""
    out_path = run_dir / f"{case.case_id}.json"
    if _run_valid_pred(out_path):
        return {"case_id": case.case_id, "status": "cached"}

    report = case.report_path.read_text(encoding="utf-8")
    schema = _load_schema_for_organ(case.organ)
    messages = format_prompt(template, few_shot, report, case.organ, schema)

    transient_retries = 0
    parse_retries = 0
    t0 = time.time()
    content = ""
    tokens_in = tokens_out = None
    finish_reason = None
    effective_seed = seed

    while True:
        try:
            resp = _call_once(client, proto, messages, effective_seed)
            content = resp["content"]
            tokens_in = resp["tokens_in"]
            tokens_out = resp["tokens_out"]
            finish_reason = resp["finish_reason"]
        except Exception as exc:
            if transient_retries >= proto.max_transient_retries:
                logger.error("[%s/%s] transient exhausted: %s",
                             run_dir.name, case.case_id, exc)
                content = ""
                break
            delay = proto.transient_backoff_s[min(transient_retries,
                                                 len(proto.transient_backoff_s) - 1)]
            logger.warning("[%s/%s] transient %s (retry in %ds)",
                           run_dir.name, case.case_id, exc, delay)
            time.sleep(delay)
            transient_retries += 1
            continue

        parsed = _parse_prediction(content)
        if parsed is not None:
            # Success — write out the prediction.
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            wall_ms = int((time.time() - t0) * 1000)
            row = {
                "case_id": case.case_id, "run_id": run_dir.name,
                "seed": effective_seed, "temperature": proto.temperature,
                "tokens_in": tokens_in, "tokens_out": tokens_out,
                "wall_ms": wall_ms, "parse_success": True,
                "parse_retries": parse_retries,
                "transient_retries": transient_retries,
                "finish_reason": finish_reason,
                "config_hash": proto.config_hash,
                "status": "ok",
            }
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            return row

        # Parse failure
        if parse_retries >= proto.max_parse_retries:
            break
        parse_retries += 1
        effective_seed = seed + 10_000 * parse_retries
        logger.warning("[%s/%s] parse failed, retry %d (seed %d)",
                       run_dir.name, case.case_id, parse_retries, effective_seed)

    # Exhausted retries — write sentinel so eval counts it wrong.
    sentinel = {
        "_parse_error": True,
        "reason": "exceeded_parse_retries",
        "last_raw": content[:2000],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sentinel, f, ensure_ascii=False, indent=2)
    wall_ms = int((time.time() - t0) * 1000)
    row = {
        "case_id": case.case_id, "run_id": run_dir.name,
        "seed": effective_seed, "temperature": proto.temperature,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "wall_ms": wall_ms, "parse_success": False,
        "parse_retries": parse_retries,
        "transient_retries": transient_retries,
        "finish_reason": finish_reason,
        "config_hash": proto.config_hash,
        "status": "parse_failed",
    }
    log_file.write(json.dumps(row) + "\n")
    log_file.flush()
    return row


# --- Run driver --------------------------------------------------------------

def run_experiment(proto: Protocol, output_root: Path,
                   *, run_filter: list[str] | None = None,
                   dry_run: bool = False) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Persist the frozen protocol next to the outputs so the experiment
    # is fully self-documenting.
    (output_root / "configs").mkdir(exist_ok=True)
    import shutil
    shutil.copy2(proto.config_path, output_root / "configs" / proto.config_path.name)

    cases = discover_cases(proto.cases_glob, proto.gold_glob)
    logger.info("discovered %d cases", len(cases))
    if dry_run:
        logger.info("dry run: would execute %d (run × case) = %d predictions",
                    len(cases) * len(proto.seeds),
                    len(cases) * len(proto.seeds))
        return

    template = _load_template(proto.prompt_template_path)
    few_shot = _load_few_shot(proto.few_shot_path)
    client = _make_client(proto)

    import socket
    manifest: dict = {
        "experiment_id": proto.experiment_id,
        "config_hash": proto.config_hash,
        "model": {"name": proto.model_name, "serving": proto.serving},
        "started_at": dt.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "machine_id": machine_slug(),
        "runs": {},
    }

    for k, seed in enumerate(proto.seeds, start=1):
        run_id = format_run_id(k, padded=False)
        if run_filter and run_id not in run_filter:
            continue
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "_log.jsonl"

        summary = {"seed": seed, "n_cases": 0, "n_parse_error": 0,
                   "n_cached": 0, "tokens_in": 0, "tokens_out": 0,
                   "wall_s": 0}
        run_t0 = time.time()
        with log_path.open("a", encoding="utf-8") as log_file:
            for case in cases:
                row = process_case(client, proto, case, seed, template,
                                   few_shot, run_dir, log_file)
                summary["n_cases"] += 1
                if row.get("status") == "parse_failed":
                    summary["n_parse_error"] += 1
                elif row.get("status") == "cached":
                    summary["n_cached"] += 1
                else:
                    summary["tokens_in"] += int(row.get("tokens_in") or 0)
                    summary["tokens_out"] += int(row.get("tokens_out") or 0)
        summary["wall_s"] = int(time.time() - run_t0)
        total = max(summary["n_cases"], 1)
        parse_rate = summary["n_parse_error"] / total
        valid = parse_rate <= proto.parse_error_rate_max
        summary["parse_error_rate"] = parse_rate
        summary["valid"] = valid
        manifest["runs"][run_id] = summary
        with (run_dir / "_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    manifest["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    with (output_root / "_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("done. manifest at %s", output_root / "_manifest.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _default_config = (
        REPO_ROOT / "configs" / "local" / "multirun_gpt_oss.yaml"
        if (REPO_ROOT / "configs" / "local" / "multirun_gpt_oss.yaml").is_file()
        else REPO_ROOT / "configs" / "multirun_gpt_oss.yaml"
    )
    ap.add_argument("--config", type=Path, default=_default_config,
                    help="Frozen protocol YAML (default: configs/local/ if present, "
                         "else configs/multirun_gpt_oss.yaml).")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "workspace" / "results" / "benchmarks" / "gpt_oss",
                    help="Output root (default: %(default)s).")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Optional subset of run-ids to (re)execute, e.g. run1 run2.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover cases and print the workload without calling the model.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    proto = load_protocol(args.config)
    run_experiment(proto, args.output, run_filter=args.runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
