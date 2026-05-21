# digital-registrar-gui

Streamlit inference GUI for the Digital Registrar pipeline. Lets you paste a pathology report (or pick a folder of `.txt` files) and see the structured extraction in real time.

## Install

```bash
pip install digital-registrar-gui
registrar-infer-gui
```

The default port is 8502. Override with `registrar-infer-gui --port 9000`.

## What it does

- Two-column layout: input on the left, sticky JSON output on the right.
- Lets you switch engine (`factory` v2 or `legacy` v1), model, decomposition strategy, and optional jsonize / output validation.
- Folder mode: pick a directory, see one row per `.txt`, click to preview.
- Expander shows the DSPy LM trace for each run (router + group extractors).

Depends on `digital-registrar` for the actual pipeline.
