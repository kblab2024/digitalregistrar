# `packaging/` — annotator bundle build matrix

> Last updated: 2026-05-11 · Reflects: `d11d072`

This directory builds the self-contained Digital Registrar Annotator
distributable (Python + Streamlit + the annotation UI, frozen as a zip
for Windows or a tar.gz for Unix/macOS). Recipients don't need Python
installed.

## Layout

```
packaging/
├── build.py                   ← canonical entry point (Python dispatcher)
├── build_<os>_<mode>[_kpc].*  ← thin wrappers calling build.py
├── _build_common.ps1          ← shared bundle logic (Windows host)
├── _build_common.sh           ← shared bundle logic (Unix host)
├── precompute_section_groups.py
├── run_<target>_<lock>_<mode>[_kpc][_macos].*  ← end-user launchers shipped INSIDE the bundle
└── dist/                      ← outputs land here (gitignored)
```

## Build the bundle

Preferred (any host with Python):

```bash
python packaging/build.py --platform windows --annotators single
python packaging/build.py --platform windows --annotators single_kpc
python packaging/build.py --platform unix    --annotators multi
python packaging/build.py --platform macos   --annotators single_kpc
```

Targets:

| `--platform` | Host that can build | Bundle extension |
|---|---|---|
| `windows`    | Windows or Unix     | `.zip` |
| `unix`       | Windows or Unix     | `.tar.gz` |
| `macos`      | macOS (or cross-build from Windows for Apple Silicon) | `.tar.gz` |

`--annotators` controls the annotator list baked into the bundle's
`defaults.json`:

| `--annotators` | Names baked in |
|---|---|
| `single`       | NHC only |
| `single_kpc`   | KPC only |
| `multi`        | NHC + KPC |

Output lands at
`packaging/dist/digital-registrar-annotator-<platform>-<annotators>.<ext>`.

The legacy `.bat` / `.sh` wrappers (`build_windows_single.bat` etc.)
are kept as thin shims that forward to `build.py` — convenient for
double-clicking and for existing CI invocations.

## Run wrappers (shipped inside the bundle)

The `run_*` files are end-user launchers packed into the bundle's
zip/tarball. They live here so they're versioned alongside the build
recipe. End users won't have Python on `PATH`, so these stay as native
`.bat` / `.sh` scripts that point at the bundled Python interpreter
inside the unpacked tree.

Variant matrix:

| Filename pattern | Data root | Locked? | Annotator set |
|---|---|---|---|
| `run_workspace_locked_<mode>[_kpc]`        | `workspace/` | yes | per-mode |
| `run_dummy_unlocked_<mode>[_kpc][_macos]`  | `dummy/`     | no  | per-mode |

`<mode>` is `single` or `multi`.

## End-user instructions

A copy of these instructions also ships in the bundle as `README.txt`:

1. Unpack anywhere (e.g. Desktop). The folder is self-contained.
2. Pick a launcher:
   - `run` (production, locked) — annotates against `workspace/`.
   - `run_demo` (demo, unlocked) — annotates against `dummy/`.
3. First Windows launch may prompt about the firewall — choose "Allow access".
4. Browser opens at `http://localhost:8501`.
5. Pick the mode (`with_preann` / `without_preann`) and dataset in the sidebar, then annotate.
6. Close the launcher window (or Ctrl+C) to stop the server.

### Where to drop data

```
workspace/
  with_preann/
    data/<dataset>/
      reports/<organ>/<case_id>.txt
      preannotation/<model>/<organ>/<case_id>.json
      annotations/<annotator>/<organ>/<case_id>.json
  without_preann/
    data/<dataset>/
      reports/<organ>/<case_id>.txt
      annotations/<annotator>/<organ>/<case_id>.json
```

The sibling `dummy/` folder shows the example file format.

### Troubleshooting

- **Browser didn't open** — go to `http://localhost:8501` manually.
- **Port 8501 in use** — edit the launcher file, change
  `--server.port=8501` to e.g. `8502`, run again.
- **`run.bat` window flashes and closes** — open Command Prompt and
  drag `run.bat` in to keep the window open and see the error.

## System requirements

- Windows 10 / 11 (64-bit) or Linux x86_64 or macOS Apple Silicon
- ~200 MB disk space (Python + dependencies are bundled)
- No Python installation needed by the recipient
