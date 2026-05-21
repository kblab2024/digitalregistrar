# registrar-infer-gui

Streamlit GUI for running the cancer-extraction pipeline interactively.
Mirrors the contract of the `registrar-pipeline` CLI so single-report
and folder outputs are interchangeable with the batch runner.

## Quick start

```bash
pip install -e .[schema-gui]
registrar-infer-gui
```

The app binds to `localhost:8502` by default (the schema editor uses
8501; both can coexist). Add `--no-browser` if running headless.

## What it does

A thin Streamlit wrapper around `pipeline_factory.run_cancer_pipeline_v2`
(or `pipeline.run_cancer_pipeline` when the legacy engine is selected),
with two run modes:

- **Single report** — paste text or upload a `.txt`; one extraction; JSON
  output rendered live.
- **Folder** — point at a directory of `*.txt`; the GUI runs the pipeline
  over each file with a progress bar; pick a result from the dropdown to
  view the report and its JSON side by side.

Output JSON is written to `runs/run_<timestamp>/<stem>_output.json`
under the repo root, byte-identical to `runner.run_folder` (the write
path is `save_output` in [runner_bridge.py](runner_bridge.py)).

## Sidebar controls

| Control | Values | Notes |
|---|---|---|
| Mode | `single` / `folder` | |
| Model | keys from `models.common.model_list` | Ollama (local) or `openai/*`; provider shown below the picker |
| Engine | `factory` / `legacy` | `factory` matches `--engine factory` on the CLI |
| Decomposition | `auto` / `per_group` / `monolithic` | Factory engine only |
| Jsonize preprocessing | bool | Factory engine only; OFF by default — modern local models extract directly |
| Validate output | bool | Factory engine only; degraded-mode Pydantic validation, logs warnings |

Switching engine or model triggers a global `dspy.configure(...)`
reconfigure on the next **Run**. **Do not run other DSPy notebooks
against this process simultaneously** — they will fight over
`dspy.settings.lm`. The sidebar carries the same caveat inline (see
[app.py](app.py)).

## Layout

Two-column, sticky right pane:

- **Left** — input report (single mode: textarea; folder mode: file
  picker + report preview).
- **Right** — JSON output with download button and elapsed-time caption;
  on failure, an expandable traceback. Reuses the annotation app's
  sticky-column CSS so the JSON stays visible while scrolling long reports.

## Pipeline contract

The GUI uses `inference_gui.runner_bridge` as its only pipeline surface,
which:

1. Wraps `setup_pipeline_v2(model)` / `setup_pipeline(model)` behind a
   typed `PipelineSetupError` so missing Ollama / `OPENAI_API_KEY`
   failures surface as a clean inline error rather than a stack trace.
2. Calls `run_cancer_pipeline_v2(...)` / `run_cancer_pipeline(...)`
   with identical kwargs to the CLI runner.
3. Writes outputs via `save_output` with the same JSON formatting the
   batch runner uses, so GUI-produced runs feed straight into eval
   pipelines.

## Limitations

- **Single-process DSPy state**: the global `dspy.settings.lm` is shared
  across the Streamlit session; this is fine for one user but breaks
  if you embed the page in a multi-tenant deployment.
- **No streaming output**: the LM call blocks until the full structured
  output is ready. Per-group decompositions show no intermediate
  progress beyond a spinner.
- **AppTest harness recursion**: `streamlit.testing.v1.AppTest` hits a
  recursion limit when reflecting on this page (same `st.data_editor`
  quirk noted in the schema-gui README). The app itself runs cleanly
  under `streamlit run`; only the headless test framework is affected.
  Smoke-test by opening the URL in a browser.
