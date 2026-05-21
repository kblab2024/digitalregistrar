# `packaging/` — release pipeline scaffolding

Build artifacts and configs for the three distribution paths. See [../docs/release.md](../docs/release.md) for the full release procedure (and the mandatory safe-practice checklist for the hosted demo).

## Subdirs

| Path | Purpose |
|---|---|
| `pyinstaller/` | PyInstaller `.spec` files for the three apps. |
| `docker/` | Multi-stage `Dockerfile` per app. Build with `make docker-build`. |
| `hosted-demo/` | Streamlit Community Cloud / Hugging Face Spaces entry point. **Safe-practice checklist required before going public** — see [../docs/release.md](../docs/release.md). |
| `windows/` *(legacy)* | Pre-existing Windows annotator bundle (`build.py`, `.bat`, `.ps1`). Retained from the original `digital-registrar-research` distribution; update module paths from `digital_registrar_research.annotation` → `digital_registrar_annotator` before reuse. |
| `precompute_section_groups.py` | Build-time codegen materialising per-organ section group metadata; consumed by the annotator parser. |

## Build commands

```bash
# Native bundles (must run on each target OS)
make bundle

# Docker images
make docker-build
```

## Three distribution paths

| Path | Audience | Status |
|---|---|---|
| **A. PyPI** — `pip install digital-registrar-gui` | Python users | Wheels build cleanly; `release.yml` workflow scaffold below. |
| **B. Hosted Streamlit demo** | Paper reviewers / casual visitors | `hosted-demo/streamlit_app.py` is a scaffold. Implement against the 5-layer safety checklist before going public. |
| **C. PyInstaller + Docker** | Non-technical end users | `pyinstaller/*.spec` and `docker/*.Dockerfile` are scaffolds. Test on each target OS. |

## Adding a new app to the bundle pipeline

1. Create `pyinstaller/<app>.spec` modelled on the existing specs.
2. Create `docker/<app>.Dockerfile` modelled on existing Dockerfiles.
3. Add a target to the `Makefile`.
4. Wire into `.github/workflows/release.yml` so tagged releases build it automatically.

## Hosted demo (mandatory before sharing the URL)

`hosted-demo/streamlit_app.py` is the Streamlit Cloud entry point. **Do not deploy until the five-layer safety checklist in [../docs/release.md](../docs/release.md) is satisfied**:

1. Secret management (dedicated key, hard cap, `.gitignore`).
2. Cost guardrails (per-session caps, max_tokens, caching).
3. Abuse protection (private app + email allowlist OR passcode).
4. PHI / sensitive-input safeguards (banner, dummy examples, length cap, no input logging).
5. Repo hygiene (no real-data imports, pinned deps, footer source link).
