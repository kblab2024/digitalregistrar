# Changelog

All notable changes to the Digital Registrar are documented here.

The project follows [Semantic Versioning](https://semver.org/) within the PEP 440 grammar.
Pre-release suffixes (`b1`, `b2`, `rc1`) iterate within a target version until it stabilises.

## [Unreleased]

## [0.2.0b4] — 2026-09-27

**Feature beta.** `registrar-eval` works again. All three subcommands now score a folder of `registrar-pipeline` outputs against a gold folder. The LLM endpoint is configurable (a remote Ollama host, vLLM, llama.cpp or any OpenAI-compatible server), the default Ollama context is 16384, and `gpt5_4_mini` works on dspy ≥ 3.4.

### Added
- Configurable LLM endpoint ([src/digital_registrar/models/common.py](src/digital_registrar/models/common.py)): the Ollama base URL is resolved per call from `overrides["api_base"]` → `$DIGITAL_REGISTRAR_OLLAMA_HOST` → `$OLLAMA_HOST` → `http://localhost:11434`. It accepts `host` / `host:port` without a scheme and maps `0.0.0.0` to `localhost`. New helpers: `resolve_ollama_api_base`, `resolve_model_id`.
- Raw LiteLLM model ids: `ollama_chat/<tag>` for any Ollama model, and `hosted_vllm/<name>` / `openai/<name>` for vLLM, llama.cpp or other OpenAI-compatible servers. The latter require `api_base`; their key comes from `$DIGITAL_REGISTRAR_API_KEY` (default `EMPTY`), and `OPENAI_API_KEY` is never forwarded to them.
- `registrar-pipeline` flags: `--api-base`, `--num-ctx`, `--think` / `--no-think`.
- `ollama_chat/qwen3:30b` decoding profile (the sampler values it previously inherited from the default profile).
- Optional `think` profile/override key, forwarded as Ollama's top-level `think` flag and stripped for non-Ollama backends. No shipped profile sets it.
- [docs/llm_backends.md](docs/llm_backends.md): backend, endpoint, context-window and thinking-mode guide.

### Changed
- Default Ollama `num_ctx` raised from 8192 to 16384 (12288 for `gemma4:*`), per [docs/architecture/dspy_ollama_model_compatibility.md](docs/architecture/dspy_ollama_model_compatibility.md) §5.2. Pass `--num-ctx 8192` (`PAPER_NUM_CTX`) to reproduce the paper.
- `registrar-eval --scope` now takes `cascade` (default) or `fair` instead of the unused `all|attempted|applicable`. The per-case table is written as CSV (`atomic.csv`), so evaluation no longer needs `pyarrow`.
- `digital_registrar.eval.pairwise_compare` now provides `compare_runs()`. The retired `main()` stub is gone.
- Rewrote `docs/eval/` to document `registrar-eval`: [index](docs/eval/index.md), [recipes](docs/eval/recipes.md), [comparing runs](docs/eval/comparing_runs.md) and [reading outputs](docs/eval/reading_outputs.md). It no longer refers to the retired `python -m scripts.eval.cli`.
- Bumped version pins to `0.2.0b4` in all four `pyproject.toml` files, in [packaging/hosted-demo/requirements.txt](packaging/hosted-demo/requirements.txt) and in `uv.lock`.

### Fixed
- `completeness.method_pair_deltas` no longer raises `ImportError`: it imported `ci_gpu`, which exists only in the optional drr-attic package. Each cell now uses `ci.mcnemar_test`; output columns are unchanged.
- `load_model(..., overrides={"api_base": ...})` no longer raises `TypeError: got multiple values for keyword argument 'api_base'`.
- `load_model("gpt5_4_mini")` works on dspy ≥ 3.4. dspy 3.4 treats dotted `gpt-5.x` ids as reasoning models: it rejected the profile's `temperature: 0.3` with `LMConfigurationError`, and the pre-renamed `max_completion_tokens` with a `TypeError`. For gpt-5 / o-series ids, both values are now set on `lm.kwargs` after `dspy.LM` is built. The request still carries temperature 0.3 and a 4096-token cap, the values the rebuttal runs used on dspy 3.2.1.
- `registrar-eval` works again. All three subcommands crashed at v0.2.0b3, and each now runs end-to-end on a folder of `registrar-pipeline` outputs against a gold folder:
  - `metrics` writes `atomic.csv` and `summary.csv`, with per-field accuracy or F1, coverage and 95% CIs.
  - `compare` writes `compare.csv`: per-field paired Δ, a paired-bootstrap CI and McNemar's test.
  - `completeness` writes `completeness.csv`, `refusal_calibration.csv` and `out_of_vocab.csv`.

  Files are paired by case id after stripping `_output` / `_annotation` from the stem. Both folders are searched recursively. The same code is exposed in [src/digital_registrar/eval/](src/digital_registrar/eval/) as `load_pairs`, `score_pairs`, `summarize_scores`, `compare_runs` and `completeness_atomic`.
- Per-case F1 no longer drops to 0 when `margins` / `biomarkers` / lymph-node lists are empty on both sides. The same goes for whitelisted biomarkers that neither side lists: those cases are now skipped, since there is nothing to score.
- `completeness.out_of_vocab_rate` no longer flags integer and boolean enum values (e.g. `grade: 2`, `perineural_invasion: true`) as out-of-vocabulary.

## [0.2.0b3] — 2026-05-28

**Hotfix beta** for end users routing through `openai/gpt-5.4-mini` (and any future OpenAI gpt-5.x / o-series reasoning model). The default decoding profile passed a `max_tokens` cap that the new OpenAI Chat Completions surface rejects with `BadRequestError: Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.`, breaking the GUI inference path the moment a registrar pointed it at OpenAI instead of Ollama.

### Fixed
- [src/digital_registrar/models/common.py](src/digital_registrar/models/common.py): when `load_model()` dispatches through the `openai/` provider for a gpt-5.x or o-series model id (`gpt-5*`, `gpt5*`, `o1*`, `o3*`, `o4*`), rename the resolved `max_tokens` kwarg to `max_completion_tokens` at the dspy.LM build site. The translation mirrors the existing `_OLLAMA_ONLY_KEYS` strip — applied only when the LM is actually built so `compute_lm_kwargs()` (and the manifests that consume it) still record the intended sampler config under the canonical `max_tokens` name.

### Changed
- Bumped version pins to `0.2.0b3` in all four `pyproject.toml` files and in [packaging/hosted-demo/requirements.txt](packaging/hosted-demo/requirements.txt).

## [0.2.0b2] — 2026-05-27

**README-only beta refresh** prompted by the Diagnostics launch announcement. No code or schema changes; PyPI metadata republished so cold-install instructions reach end users.

### Changed
- Root README: added a TL;DR hook above the intro, restructured `## Quickstart (end users)` into a "Prerequisites → Run the GUI → Other packages" flow. The Prerequisites block makes the BYO-LLM gate explicit (one of `gpt-oss:20b` / `qwen3:30b` / `gemma3:27b` via Ollama, **or** an `OPENAI_API_KEY`), so first-time pip installers from the launch tweet don't hit a silent `ConnectionError` against Ollama.
- `digital-registrar-gui` README: added a `## Prerequisites` section above `## What it does` mirroring the root README gate, and a one-line "Requires Ollama or OpenAI key" disclaimer in the top blurb.
- Bumped version pins to `0.2.0b2` in all four `pyproject.toml` files and in [packaging/hosted-demo/requirements.txt](packaging/hosted-demo/requirements.txt).

## [0.2.0b1] — 2026-05-23

**First tagged release.** The companion medRxiv preprint v8 ([10.1101/2025.10.21.25338475](https://www.medrxiv.org/content/10.1101/2025.10.21.25338475v8)) has been accepted, so the project shifts from a research codebase to a beta toolkit for end-users (cancer registrars, pathology informatics teams, registry IT). This release ships four installable Python packages plus PyInstaller native bundles.

`b1` is a PEP 440 beta marker: `pip install` will skip it by default and requires `--pre` to opt in. The API surface and per-organ schemas are not yet promised stable; iterate to `b2`/`b3` from feedback, then to plain `0.2.0` final.

### Added
- First public packaging on PyPI: `digital-registrar`, `digital-registrar-gui`, `digital-registrar-annotator`, `digital-registrar-schema-editor`.
- PyInstaller scaffolding for Windows, macOS, and Linux native bundles ([packaging/pyinstaller/](packaging/pyinstaller/), per-app spec files).
- `digital_registrar.eval.*` public API for prediction-vs-annotation scoring: `score_case`, `aggregate_cases_to_df`, `summary_table`, `field_correct`, `score_lymph_nodes`, `score_margins`, completeness metrics, Wilson confidence intervals.
- `digital_registrar.schemas.*` factory: `CASE_MODELS`, `build_case_model`, `list_organs`, `load_json_schema`, `load_pydantic_model`.
- DSPy signature builders: `ExtractionStep`, `build_extraction_signatures`, `build_router_signature`, `build_jsonize_signature`.
- CAP-aligned schemas for 10 cancer types: breast, cervix, colorectal, esophagus, liver, lung, pancreas, prostate, stomach, thyroid.
- Vendored AJCC TNM staging engine (`tnmhelper`) for stage-group derivation.
- Repository-wide CI (`ci.yml`) — lint, install, test, schema concordance on Python 3.11 and 3.12.

### Changed
- **Repository reorganization** ([d9e8588](https://github.com/kblab2024/digitalregistrar/commit/d9e8588)): the previous monorepo with everything under `src/` was split into `src/digital_registrar/` (core) + `apps/{infer-gui,annotator,schema-editor}/` + `attic/` (research scaffolding). The four shippable packages each get their own `pyproject.toml`.
- The three Streamlit apps moved from monorepo top-level into `apps/`. Console scripts (`registrar-infer-gui`, `registrar-annotate*`, `registrar-schema-gui`) are unchanged.
- Public Python API consolidated under `digital_registrar.*`. Imports from research-era paths (`drr_attic.benchmarks.eval.*`, `scripts.eval._common.*`) are no longer part of the supported surface — those modules now live in `attic/` and are not installed by `pip install digital-registrar`.

### Fixed
- `scipy` added to `digital-registrar` runtime dependencies — `eval/ci.py:wilson_ci` imports it unconditionally; pre-release installs from a clean env would have failed at first `import digital_registrar.eval`.
- Lint, test-collection, and schema-concordance regressions introduced by the reorg ([PR #1](https://github.com/kblab2024/digitalregistrar/pull/1)).

### Known limitations
- `apps.annotator.io.discover_folders` still expects the legacy `{prefix}_{dataset|result|annotation}_{date}/` directory layout. The dummy fixture under `examples/dummy/data/` was restructured to a flatter `{tcga,cmuh}/{reports,preannotation,annotations}/` shape during the reorg, so the discovery test for that fixture is skipped pending a product decision (update `discover_folders` or restore a legacy fixture).
- The eval helpers in `tests/eval/test_paths_and_pairing.py` reference `scripts.eval._common.*` which moved into `attic/eval_scripts/` (not importable). The test is module-level skipped pending a rewrite against the canonical `digital_registrar.eval.*` API.
- `digital-registrar-schema-editor` declares `ruff` as a runtime dependency — needs confirmation whether intentional (app shells out to ruff) or a leftover from dev tools.

### Distribution channels in scope for this beta
- **PyPI** — four wheels published in dependency order: core first, then the three apps.
- **PyInstaller native bundles** — Windows / macOS / Linux, built via the GitHub Actions release workflow's OS matrix and attached to the GitHub Release.

### Out of scope for this beta (deferred)
- Hosted Streamlit demo (Path B in [docs/release.md](docs/release.md)) — separate operational concern, not bound to this tag.
- Docker images (Dockerfiles remain in [packaging/docker/](packaging/docker/) but are not built / published from this tag).

[Unreleased]: https://github.com/kblab2024/digitalregistrar/compare/v0.2.0b3...HEAD
[0.2.0b3]: https://github.com/kblab2024/digitalregistrar/releases/tag/v0.2.0b3
[0.2.0b2]: https://github.com/kblab2024/digitalregistrar/releases/tag/v0.2.0b2
[0.2.0b1]: https://github.com/kblab2024/digitalregistrar/releases/tag/v0.2.0b1
