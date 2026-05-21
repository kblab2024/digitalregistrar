# digital-registrar-annotator

Streamlit annotation tool for Digital Registrar pathology reports. Lets clinicians review and correct pipeline pre-annotations against the gold schema.

## Install

```bash
pip install digital-registrar-annotator

# Workspace mode (with/without pre-annotations subdirs):
registrar-annotate-workspace

# Dummy mode (for trying without real data):
registrar-annotate-dummy

# Legacy flat layout:
registrar-annotate
```

Default port: 8500.

## Folder layouts

- **Canonical (workspace / dummy)**: `{base}/with_preann/{dataset}/{dataset,result}/` and `{base}/without_preann/{dataset}/dataset/`.
- **Legacy flat**: `{base}/{prefix}_dataset_{date}/`, `{prefix}_result_{date}/`, `{prefix}_annotation_{date}/`.

The annotator reads pre-generated JSON schemas from the installed `digital_registrar.schemas.data` package. No DSPy / LLM dependency.
