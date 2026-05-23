# Release pipeline

The Digital Registrar supports three distribution paths for end users. Each is documented in this file and scaffolded under [packaging/](../packaging/).

## Path A — PyPI (Python users)

Four packages publish independently:

| Package | Source | Console scripts |
|---|---|---|
| `digital-registrar` | [pyproject.toml](../pyproject.toml) | `registrar-pipeline`, `registrar-schemas`, `registrar-eval` |
| `digital-registrar-gui` | [apps/infer-gui/](../apps/infer-gui/) | `registrar-infer-gui` |
| `digital-registrar-schema-editor` | [apps/schema-editor/](../apps/schema-editor/) | `registrar-schema-gui` |
| `digital-registrar-annotator` | [apps/annotator/](../apps/annotator/) | `registrar-annotate{,-workspace,-dummy}` |

Release procedure:

1. Bump `version` in the relevant `pyproject.toml` (semver).
2. From the package root: `python -m build` → `dist/*.whl` + `dist/*.tar.gz`.
3. `twine upload dist/*` (or push a `v*.*.*` tag and let `.github/workflows/release.yml` do it).
4. Verify on PyPI; smoke install in a fresh venv: `pip install <pkg> && registrar-... --help`.

End-user install:

```bash
pip install digital-registrar-gui    # transitively pulls digital-registrar
registrar-infer-gui
```

## Path B — Hosted Streamlit demo (casual visitors / paper reviewers)

`packaging/hosted-demo/streamlit_app.py` is the Streamlit Community Cloud entry point. Two deployment options:

- **Streamlit Community Cloud** (free): create an app pointing at this file in the public repo. Auto-redeploys on push.
- **Hugging Face Spaces** (free): `packaging/hosted-demo/space.yaml` + `requirements.txt`.

### Safe practice for the hosted demo (mandatory checklist)

Every public demo URL must pass these five gates before being shared.

#### 1. Secret management
- [ ] `.gitignore` includes `*.env` and `.streamlit/secrets.toml`.
- [ ] Only `packaging/hosted-demo/secrets.toml.example` is committed (a template with placeholder values).
- [ ] Real keys set via the Streamlit Cloud dashboard → "Secrets" tab. Accessed in code as `st.secrets["OPENAI_API_KEY"]`.
- [ ] **Dedicated API key** for the demo, scoped to its own OpenAI project. Never reused from production.
- [ ] **Hard monthly spend cap** at provider dashboard (e.g. `$25/month hard limit`, `$10 soft alert`).

#### 2. Cost guardrails inside the app
- [ ] Per-session caps: `MAX_CALLS_PER_SESSION = 20` in `st.session_state`; refuse with `st.stop()` when exceeded.
- [ ] Per-call cap: `max_tokens=512` on completions.
- [ ] Input truncated to N tokens before sending.
- [ ] Cheapest viable model (e.g. `gpt-4o-mini`), not the production model. Document the chosen model name in the demo sidebar.
- [ ] `@st.cache_data(ttl=3600)` on the LLM call so identical inputs don't re-bill.

#### 3. Abuse protection
- [ ] **Preferred**: Streamlit Cloud **private app** with Google SSO + email allowlist of named reviewers.
- [ ] **Fallback**: passcode gate via `st.secrets["DEMO_PASSCODE"]`; distribute passcode in supplementary material.
- [ ] Soft rate limit per IP using `st.context.headers.get("x-forwarded-for")` (Streamlit 1.35+) + in-memory dict.

#### 4. PHI / sensitive-input protection (highest priority)
- [ ] Banner above input: "DO NOT PASTE PROTECTED HEALTH INFORMATION. Use only synthetic or de-identified text."
- [ ] Default textarea pre-populated with 2–3 synthetic examples (from `examples/dummy/`).
- [ ] File uploads disabled in the demo. Only the textarea.
- [ ] Input length capped to ~5000 chars; refuse longer.
- [ ] **No raw-input logging.** Log only: `len(text)`, model name, latency, token counts.
- [ ] Demo writes to neither disk nor any remote DB.
- [ ] Provider Zero Data Retention enabled (OpenAI ZDR requires application).

#### 5. Repo / deployment hygiene
- [ ] `packaging/hosted-demo/streamlit_app.py` imports only dummy schemas + dummy examples. Never `from digital_registrar.paths import RAW_REPORTS`.
- [ ] `packaging/hosted-demo/requirements.txt` pins exact versions; minimal dep set.
- [ ] Sidebar shows: model name, "demo only" disclaimer, link to the public repo.
- [ ] Footer link to this repo for transparency.

The `release.yml` workflow refuses to deploy if `packaging/hosted-demo/secrets.toml.example` contains values that look like real keys (simple grep guardrail).

## Path C — Native bundles + Docker (non-technical end users)

### PyInstaller bundles

`packaging/pyinstaller/*.spec` — one `.spec` per app. Each bundles the Streamlit launcher + `digital-registrar` core into a single-folder distribution.

```bash
make bundle
# Produces dist/registrar-infer-gui/, dist/registrar-schema-gui/, dist/registrar-annotate/
```

Wrap into installers:
- **macOS**: `create-dmg` → `.dmg`.
- **Windows**: `nsis` or `Inno Setup` → `.exe` installer.
- **Linux**: AppImage or `.deb` / `.rpm`.

PyInstaller is platform-specific — Windows `.exe` must be built on Windows, etc. The `.github/workflows/release.yml` matrix should run this across `runs-on: [macos-latest, windows-latest, ubuntu-latest]` and attach the artifacts to the GitHub Release on tag.

### Docker images

`packaging/docker/*.Dockerfile` — one Dockerfile per app, multi-stage builds.

```bash
make docker-build
docker run -p 8502:8502 digitalregistrar/gui    # then open http://localhost:8502
```

Images published to Docker Hub or GHCR. Expected size: ~1.5 GB (`python:3.11-slim` + dspy + pydantic + streamlit).

## CI / release automation

- `.github/workflows/ci.yml` — runs on every push / PR. Installs core + 3 apps via plain pip, lints with ruff, runs targeted tests, checks schema concordance. Attic excluded.
- `.github/workflows/release.yml` — on `v[0-9]*` tag (or `workflow_dispatch` for TestPyPI dry runs): re-runs CI on the tagged commit, builds all 4 wheels + sdists, publishes to PyPI via Trusted Publishing (OIDC), builds PyInstaller bundles via `[windows-latest, macos-latest, ubuntu-latest]` matrix, attaches everything to a **draft** GitHub Release (you manually publish after eyeballing).

### One-time PyPI Trusted Publishing setup

Before the first tag fires `release.yml`, configure a *Pending Publisher* on PyPI for each of the four packages:

1. Create the four projects on PyPI (you can do this by uploading a zero-byte wheel manually, or by creating them via the PyPI web UI as part of registering a Pending Publisher — see [docs.pypi.org/trusted-publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)).
2. For each project, under **Manage → Publishing → Add a new publisher → GitHub**:
   - Owner: `kblab2024`
   - Repository: `digitalregistrar`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. Repeat for TestPyPI ([test.pypi.org](https://test.pypi.org/)) with environment name `testpypi` if you want `workflow_dispatch` dry runs to work.
4. In the repo settings (Settings → Environments), create environments named `pypi` and `testpypi`. Add deployment protection rules (e.g., manual approval) on `pypi` to require a human click before each PyPI publish.

After that, no API tokens or secrets are needed — OIDC handles auth.

### Docker (not published from this tag)

Docker is documented above (Path B) and the Dockerfiles ship in [packaging/docker/](../packaging/docker/), but the release workflow does not currently build or push them. Re-add a `docker-build` job to `release.yml` when Docker becomes a supported channel.

## Updating the citation on journal publication

When the medRxiv preprint is replaced by a published-journal version, three edits ship together:

1. `CITATION.cff` — `preferred-citation.{doi,journal,year,url,notes}`.
2. `README.md` — Citation block (top of file + bottom).
3. `docs/index.md` — any preprint references.

A `TODO: replace with published journal citation` comment in `CITATION.cff:preferred-citation.notes` is the discovery point.
