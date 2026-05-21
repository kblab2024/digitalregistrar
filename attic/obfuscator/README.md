# obfuscator

Produces `workspace_obfustrated/` — a schema-conformant, content-randomized copy of `workspace/` — so Claude (or any external collaborator) can debug eval/ablation pipelines without exposure to PHI.

## Threat model

Treat the output as **public**. Assume it leaks. The obfuscator therefore:

- Never copies a source `.txt` or unrecognized binary verbatim.
- Synthesizes report text from scratch using a layered noise model — original text is never `open()`ed for content.
- Strips `.pt` / `.bin` / `.safetensors` model weights to `.placeholder` markers.
- Discards log content (replaces with a banner) — pipeline tracebacks sometimes embed report excerpts.
- Records every skipped file in `_obfuscator_skipped.json` with sha256 for auditability.

## Usage

```
pip install -e obfuscator/
obfuscate-workspace                       # workspace/ -> workspace_obfustrated/
obfuscate-workspace --src dummy/ --out /tmp/dummy_obf/ --seed 1234
```

Flags:
- `--src`         source workspace root (default `workspace/` resolved against repo root)
- `--out`         destination root (default `workspace_obfustrated/`)
- `--seed`        master seed; sub-seeds derived deterministically per file/case (default 42)
- `--shapes-only` degraded mode: empty annotations + lorem-ipsum reports. Failure is *obvious* (eval F1 = 0)
- `--force`       allow overwriting an existing non-empty `--out`

Then point any eval/ablation/inference script at the obfustrated workspace via the `--obfustrated` flag (or `--folder obfustrated` shorthand) — both added by the script-integration patch alongside the existing `dummy`/`workspace` options.

## Generation model

Per case, five layers add deterministic noise from each previous layer:

```
Layer 0: canonical_state    (jsf-style fill from schemas/data/{organ}.json)
Layer 1: report_realization (omission + ambiguity noise on canonical)
Layer 2a: report .txt        (template render of report_realization)
Layer 2b: 5 annotations      (gold + nhc/kpc x with/without_preann; per-slot noise profile)
Layer 3: N predictions       (per-(model, run) noise profile from configs/models/*.yaml)
```

This produces realistic eval signal: gold ≠ text in some fields, IAA disagreement among annotators, with-vs-without-preann effects, and run-to-run prediction variance.

## Do NOT

- Pipe outputs back into the source `workspace/`. The obfustrated tree carries no clinical truth.
- Add `workspace_obfustrated/` to any Claude-ignore mechanism — Claude needs to read it freely. (gitignore is fine; that's the point.)
