# Annotation UI

A Streamlit app for doctors to review GPT-OSS pre-annotations and save corrected annotations.

## Workflow

The flow is **GPT-OSS pre-annotates → doctor corrects in app → save as final annotation**. Doctors need to see at a glance what the model pre-filled vs. what they changed; the UI highlights diffs with a `✎` marker plus a "pre-annotated" caption.

## Three coexisting entry points

| Command | App | Default base dir | Purpose |
|---|---|---|---|
| `registrar-annotate` | `app.py` (legacy) | — (user picks) | Original flat-sibling layout; kept for regression comparison |
| `registrar-annotate-workspace` | `app_canonical.py` | `<repo>/workspace/` | Live patient data (gitignored) |
| `registrar-annotate-dummy` | `app_canonical.py` | `<repo>/dummy/` | Public skeleton / demo / smoke test |

The canonical launchers export `REGISTRAR_ANNOTATE_BASE_DIR` before spawning `streamlit run`; setting that env var yourself before calling either command overrides the default. Set it to `<repo>/workspace_obfustrated/` to point the annotator UI at the obfuscator-produced synthetic workspace for PHI-free debugging — see [obfuscation.md](obfuscation.md).

## Canonical dataset layout (`registrar-annotate-workspace` / `-dummy`)

```
<base_dir>/
├── with_preann/
│   └── data/<dataset>/                             # e.g. cmuh, tcga
│       ├── reports/<n>/<case_id>.txt
│       ├── preannotation/gpt_oss_20b/<n>/<case_id>.json
│       └── annotations/<annotator>/<n>/<case_id>.json
└── without_preann/
    └── data/<dataset>/                             # independent subset (no preannotation/)
        ├── reports/<n>/<case_id>.txt
        └── annotations/<annotator>/<n>/<case_id>.json
```

`with_preann` and `without_preann` are fully independent datasets — the mode picker in the sidebar switches between the two subtrees.

Chrome is in 繁體中文. Sidebar selectors:

- **標註者** — who is labelling (e.g. NHC, KPC).
- **標註模式** — `含預標註 (with_preann)` browses `with_preann/data/<dataset>/`, loads `preannotation/gpt_oss_20b/`, and pre-fills the form; `不含預標註 (without_preann)` browses `without_preann/data/<dataset>/`, hides the pre-annotation panel, and starts from blank. Switching modes re-discovers datasets and samples from scratch.
- **資料集** — dropdown of subfolders found under `<base_dir>/<mode>/data/` that contain a `reports/` directory (so `cmuh` and `tcga` show up automatically in the bundled `dummy/`).

Pre-annotation source is fixed to `gpt_oss_20b` today (only model present in `dummy/`).

## Legacy layout (`registrar-annotate`)

The original app expects a base folder with three timestamped sibling subfolders:

```
<base_dir>/
├── tcga_dataset_20251117/      # raw .txt reports
├── tcga_result_20251117/       # GPT-OSS pre-annotations (one *_output.json per case)
└── tcga_annotation_20251117/   # doctor-saved final annotations (auto-created on first save)
```

Folder discovery is by regex on `{prefix}_{kind}_{date}` — see `annotation.io.discover_folders`.

## JSON contract

Saved annotations match the canonical Pydantic schema for the organ — top-level routing fields plus a flat `cancer_data` object (no section-based intermediate layer). For example:

```json
{
  "cancer_excision_report": true,
  "cancer_category": "lung",
  "cancer_category_others_description": null,
  "cancer_data": {
    "procedure": "lobectomy",
    "histology": "adenocarcinoma",
    "grade": 2,
    "margins": [{"margin_category": "bronchial", "margin_involved": false, "distance": 12}],
    "regional_lymph_node": [...]
  }
}
```

This shape is identical to what `CancerPipeline` emits at runtime — the annotation tool just hand-corrects it.

## Launching

```bash
registrar-annotate-workspace                    # canonical layout, defaults to workspace/
registrar-annotate-dummy                        # canonical layout, defaults to dummy/
registrar-annotate                              # legacy layout (flat sibling folders)
registrar-annotate-workspace --server.port 8502 # forward args to streamlit
```

Or directly:

```bash
streamlit run src/digital_registrar_research/annotation/app_canonical.py
streamlit run src/digital_registrar_research/annotation/app.py             # legacy
```

Both apps can run concurrently on different ports for side-by-side comparison.

## UI principles

The user is a busy clinician. Defaults that should not change without a strong reason:

- Use **native Streamlit widgets** — no over-customisation.
- Folder selection uses a **native GUI dialog** (Windows + macOS); both must work.
- Pre-annotation diffs are marked with `✎` and a caption — never hide the model's input.
