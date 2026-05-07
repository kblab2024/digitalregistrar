# Obfuscated workspace — schema-conformant synthetic data for PHI-free debugging

## Why this exists

Real patient pathology reports under `workspace/` carry PHI (case IDs, dates, surgeon names, hospital codes, narrative text). Running a Claude session against them is unacceptable.

The obfuscator produces `workspace_obfustrated/`: a sibling tree that is **byte-for-byte different** from `workspace/` but **schema-conformant**, so the same eval / ablation / inference scripts run end-to-end and produce numbers a debugging session can reason about.

Three workspace roots coexist as first-class peers:

| Root | Purpose | Gitignored? |
| --- | --- | --- |
| `workspace/` | Real PHI — production data | yes |
| `workspace_obfustrated/` | Synthetic-from-real, real-shape, real-scale | yes |
| `dummy/` | Small synthetic-from-nothing fixture for unit tests | partly |

`workspace_obfustrated/` is **gitignored** but **not Claude-ignored** — Claude needs read access for the debugging workflow this is built for.

## Threat model

Treat `workspace_obfustrated/` as **public**. Assume it leaks. The obfuscator therefore:

- **Never** copies a source `.txt` or unrecognized binary verbatim.
- Synthesizes report text from scratch using a layered noise model — original text is never `open()`ed for content.
- Strips `.pt` / `.bin` / `.safetensors` model weights to `.placeholder` markers.
- Discards log content (replaces with a banner) — pipeline tracebacks sometimes embed report excerpts.
- Records every skipped file in `_obfuscator_skipped.json` with sha256 for auditability.

## Generate the obfustrated workspace

From the inference repo root:

```bash
python scripts/obfuscate_workspace.py                    # workspace/ -> workspace_obfustrated/
python scripts/obfuscate_workspace.py --src dummy --out /tmp/dummy_obf --seed 1234
```

Or install the standalone CLI:

```bash
pip install -e obfuscator/
obfuscate-workspace --seed 42
```

Flags:
- `--src` source workspace (default: `workspace/`)
- `--out` destination (default: `workspace_obfustrated/`)
- `--seed` master seed; sub-seeds derived deterministically per file/case (default 42)
- `--shapes-only` degraded mode — eval F1 will be 0; failure is *obvious*
- `--force` overwrite an existing non-empty `--out`
- `--no-validate` skip jsonschema validation (faster, riskier)

## Generation model (the layered noise pipeline)

Per case, five layers add deterministic noise from each previous layer:

```
Layer 0: canonical_state    (schema-driven random fill from schemas/data/{organ}.json)
Layer 1: report_realization (omission + ambiguity noise on canonical)
Layer 2a: report .txt        (template render of report_realization)
Layer 2b: 5 annotations      (gold + nhc/kpc x with/without_preann; per-slot noise profile)
Layer 3: N predictions       (per-(model, run) noise profile from configs/models/*.yaml)
```

This produces realistic eval signal: `gold` ≠ `text` in some fields (omissions), IAA disagreement among the 5 annotators, with-vs-without-preann effects, and run-to-run prediction variance.

Default rates (configurable in `obfuscator/src/obfuscator/profiles.yaml`):

- `gold` annotator noise: 5% per field
- `nhc/kpc_without_preann`: 10% per field
- `nhc/kpc_with_preann`: 8% per field, with 30% probability of following preann when it disagrees
- LLM predictions: ~15-20% per field (per-model overrides)
- ClinicalBERT/rule-based: 18-35%

## Point eval / ablation / inference scripts at it

All scripts that take `--folder` or `--root` accept an `--obfustrated` shortcut alongside the existing `dummy` and `workspace` options. **Existing flags are unchanged** — this is purely additive.

### Eval subcommands

> Cascade redesign (2026-05): the `non_nested` and `nested` subcommands have been replaced by the unified `cascade` subcommand. Examples below have been updated.

```bash
python -m scripts.eval.cli cascade --root dummy     --dataset cmuh --model gpt_oss_20b --annotator gold
python -m scripts.eval.cli cascade --root workspace --dataset tcga --model gpt_oss_20b --annotator gold

# With --obfustrated shortcut:
python -m scripts.eval.cli cascade --obfustrated --dataset tcga --model gpt_oss_20b --annotator gold
python -m scripts.eval.cli iaa --obfustrated --dataset cmuh
python -m scripts.eval.cli completeness --obfustrated --dataset tcga --model gpt_oss_20b
```

When `--obfustrated` is set without an explicit `--root`, defaults to `--root workspace_obfustrated` and writes outputs under `workspace_obfustrated/results/eval/<subcommand>/`. Explicit `--root` always wins.

### Ablation runners (all 15 cell wrappers)

```bash
# Existing — unchanged:
python scripts/ablations/run_cell_a.py --folder dummy --dataset tcga --model gptoss
python scripts/ablations/run_cell_a.py --folder workspace --dataset tcga --model gptoss

# New:
python scripts/ablations/run_cell_a.py --obfustrated --dataset tcga --model gptoss
# Or equivalently:
python scripts/ablations/run_cell_a.py --folder obfustrated --dataset tcga --model gptoss
```

### Pipeline inference

```bash
# Existing — unchanged:
python scripts/pipeline/run_dspy_ollama_single.py --folder dummy --dataset tcga --model gptoss --run smoke

# New:
python scripts/pipeline/run_dspy_ollama_single.py --obfustrated --dataset tcga --model gptoss --run smoke
```

### Library code

If you import from `digital_registrar_research.paths`, set the env var to redirect at import time:

```bash
DIGITAL_REGISTRAR_WORKSPACE=workspace_obfustrated python my_script.py
```

Or use the new helpers:

```python
from digital_registrar_research.paths import workspace_root, results_root
data = workspace_root("workspace_obfustrated") / "data" / "tcga" / "reports"
```

## Resolution priority

Highest wins; all options coexist:

1. Explicit `--folder <path>` / `--root <path>` (any of `dummy`, `workspace`, `workspace_obfustrated`, abs path)
2. `--obfustrated` flag → `workspace_obfustrated/`
3. `DIGITAL_REGISTRAR_WORKSPACE` env var → `<repo_root>/<value>`
4. Existing default (`workspace/` for production, `dummy/` where current code already defaults to dummy — unchanged)

## Audit trail

Every obfuscation run writes two files to the destination root:

- `_obfuscator_outputs.json` — every output with src→dst mapping + handler used
- `_obfuscator_skipped.json` — every skipped src path with reason + sha256 (including stripped model weights)

Inspect these to verify the obfuscator handled every input and to see what was discarded.

## What does NOT work against `workspace_obfustrated/`

- BERT / torch code paths — `.pt` checkpoints were stripped to `.placeholder` markers.
- Anything that re-reads the original report text expecting clinical realism — synthetic reports follow a template, not real prose.
- Anything that depends on real TCGA case IDs being resolvable — they're remapped.

LLM extraction, rule-based extraction, schema validation, IAA scoring, completeness analysis, and the eval / ablation aggregators all work end-to-end.

## Do NOT

- Pipe outputs of a run against `workspace_obfustrated/` back into `workspace/`. The obfustrated tree carries no clinical truth.
- Add `workspace_obfustrated/` to any Claude-ignore mechanism (`.claude/settings.json` deny rules, `.claudeignore` files, etc.). Claude needs read access; that's the whole point. Gitignore is fine.
