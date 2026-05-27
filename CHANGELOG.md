# Changelog

All notable changes to the Digital Registrar are documented here.

The project follows [Semantic Versioning](https://semver.org/) within the PEP 440 grammar.
Pre-release suffixes (`b1`, `b2`, `rc1`) iterate within a target version until it stabilises.

## [Unreleased]

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

[Unreleased]: https://github.com/kblab2024/digitalregistrar/compare/v0.2.0b2...HEAD
[0.2.0b2]: https://github.com/kblab2024/digitalregistrar/releases/tag/v0.2.0b2
[0.2.0b1]: https://github.com/kblab2024/digitalregistrar/releases/tag/v0.2.0b1
