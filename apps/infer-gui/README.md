# digital-registrar-gui

Streamlit inference GUI for the Digital Registrar pipeline. Lets you paste a pathology report (or pick a folder of `.txt` files) and see the structured extraction in real time. **Requires** a local Ollama server (with one of the paper-benchmarked models) **or** an `OPENAI_API_KEY` — see [Prerequisites](#prerequisites--pick-one-llm-backend) below.

## Install

```bash
pip install digital-registrar-gui
registrar-infer-gui
```

The default port is 8502. Override with `registrar-infer-gui --port 9000`.

## Prerequisites — pick one LLM backend

The GUI is BYO-LLM. You need **either**:

- **Local Ollama** with one of the three paper-benchmarked models:
  ```bash
  ollama pull gpt-oss:20b      # default
  ollama pull qwen3:30b        # alt (Qwen3-30B-A3B)
  ollama pull gemma3:27b       # alt
  ```

- **OpenAI**: `export OPENAI_API_KEY=sk-...`, then pick `gpt5_4_mini` in the GUI's model dropdown.

Ollama on another machine or outside the GUI's container: set `DIGITAL_REGISTRAR_OLLAMA_HOST` (or `OLLAMA_HOST`), e.g. `DIGITAL_REGISTRAR_OLLAMA_HOST=192.168.1.20:11434 registrar-infer-gui`. See [LLM backends](https://github.com/kblab2024/digitalregistrar/blob/main/docs/llm_backends.md).

See the [main repo](https://github.com/kblab2024/digitalregistrar) for the full pipeline docs and four-package overview.

## What it does

- Two-column layout: input on the left, sticky JSON output on the right.
- Lets you switch engine (`factory` v2 or `legacy` v1), model, decomposition strategy, and optional jsonize / output validation.
- Folder mode: pick a directory, see one row per `.txt`, click to preview.
- Expander shows the DSPy LM trace for each run (router + group extractors).

Depends on `digital-registrar` for the actual pipeline.
