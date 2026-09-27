# Choosing and configuring the LLM backend

Digital Registrar is BYO-LLM. Every entry point (the `registrar-pipeline` CLI, the inference GUI, and the Python API) builds its model through `digital_registrar.models.common.load_model`. This page covers:

- where that model is served from;
- how big its context window is;
- whether "thinking" is on.

## Backends at a glance

| Backend | `--model` | Endpoint | API key |
|---|---|---|---|
| Local Ollama (default) | an alias such as `gpt`, `qwen30b`, `gemma3` | `http://localhost:11434` | none |
| Ollama on another machine, or in Docker | same aliases, or `ollama_chat/<tag>` for any pulled tag | `--api-base`, `$DIGITAL_REGISTRAR_OLLAMA_HOST` or `$OLLAMA_HOST` | none |
| vLLM | `hosted_vllm/<served-model-name>` | `--api-base http://host:8000/v1` (required) | `$DIGITAL_REGISTRAR_API_KEY`, else `EMPTY` |
| llama.cpp `llama-server`, LM Studio, other OpenAI-compatible servers | `openai/<model-name>` | `--api-base http://host:8080/v1` (required) | `$DIGITAL_REGISTRAR_API_KEY`, else `EMPTY` |
| Hosted OpenAI | `gpt5_4_mini` | OpenAI | `OPENAI_API_KEY` (env or `~/.config/digital-registrar/.env`) |

The aliases are the keys of `model_list` in [`src/digital_registrar/models/common.py`](../src/digital_registrar/models/common.py). `--model` also accepts raw LiteLLM ids that start with `ollama_chat/`, `hosted_vllm/` or `openai/`.

## Ollama on another host or in Docker

The endpoint is resolved each time a model is loaded. The first non-empty value wins:

1. `--api-base` on `registrar-pipeline` (or `overrides={"api_base": ...}` in Python)
2. `DIGITAL_REGISTRAR_OLLAMA_HOST`
3. `OLLAMA_HOST`, the variable the `ollama` CLI itself reads, so a shell already set up for a remote `ollama run` works unchanged
4. `http://localhost:11434`

Accepted forms:

| You write | Used as |
|---|---|
| `gpu-box` | `http://gpu-box:11434` |
| `gpu-box:11500` | `http://gpu-box:11500` |
| `0.0.0.0` / `0.0.0.0:11434` / `:11434` | `http://localhost:11434` (a server's bind-all address is mapped to `localhost`) |
| `http://gpu-box:11434/` | `http://gpu-box:11434` |
| `https://ollama.example.org` | unchanged (an explicit scheme keeps its own port and path) |

Examples:

```bash
# Ollama on a GPU box on the LAN
registrar-pipeline --input reports/ --model gpt --api-base 192.168.1.20:11434

# The same thing via env var (also picked up by the GUI, which has no --api-base)
export DIGITAL_REGISTRAR_OLLAMA_HOST=192.168.1.20:11434
registrar-infer-gui

# Pipeline or GUI running in Docker, Ollama running on the Docker host
docker run -e DIGITAL_REGISTRAR_OLLAMA_HOST=http://host.docker.internal:11434 ...
# On Linux, add --add-host=host.docker.internal:host-gateway, or use --network host with localhost:11434.

# Any Ollama tag that has no alias
registrar-pipeline --input reports/ --model ollama_chat/llama3.3:70b
```

The Ollama server has to listen beyond loopback for any of this to work, e.g. `OLLAMA_HOST=0.0.0.0 ollama serve` on the GPU box.

From Python:

```python
from digital_registrar import setup_pipeline
setup_pipeline("qwen30b", overrides={"api_base": "gpu-box:11434"})
```

## vLLM, llama.cpp and other OpenAI-compatible servers

These servers don't speak Ollama's `/api/chat` protocol, so instead of an alias you pass a raw id with an OpenAI-style provider prefix, plus the server's `/v1` base URL:

```bash
# vLLM:  vllm serve Qwen/Qwen3-30B-A3B --max-model-len 16384
registrar-pipeline --input reports/ --model hosted_vllm/Qwen/Qwen3-30B-A3B --api-base http://gpu-box:8000/v1

# llama.cpp:  llama-server -m qwen3-30b-a3b-q4_k_m.gguf -c 16384 --port 8080
registrar-pipeline --input reports/ --model openai/qwen3-30b-a3b --api-base http://gpu-box:8080/v1
```

- **`--api-base` is required.** The Ollama env vars are not used for these ids.
- **API key.** It is read from `DIGITAL_REGISTRAR_API_KEY`, e.g. for `llama-server --api-key` or `vllm serve --api-key`. When that is unset, the placeholder `EMPTY` is sent. `OPENAI_API_KEY` is **never** forwarded to a custom server.
- **Ollama-only knobs are dropped.** These are `num_ctx`, `top_k`, `repeat_penalty`, `keep_alive` and `think`. Set the context window on the server instead: vLLM `--max-model-len`, llama.cpp `-c`.
- **Unprofiled.** These ids use the default sampler profile (temperature 0.2, top_p 0.95, max_tokens 4096). None of them has been benchmarked against the paper's results.

## Context window (`num_ctx`)

Ollama silently drops the **start** of any prompt longer than `num_ctx`, without raising an error. For the long nested extraction signatures, that drops the report itself; see [architecture/dspy_ollama_model_compatibility.md](architecture/dspy_ollama_model_compatibility.md) §2 Layer C/D. The defaults follow that audit's §5.2:

| Model (Ollama) | Default `num_ctx` |
|---|---|
| `gpt-oss:20b`, `qwen3:30b`, `qwen3.5:27b`, `gemma3:27b`, `medgemma:*`, any model without a profile | 16384 |
| `gemma4:26b`, `gemma4:e2b` | 12288 (Gemma 4 is reported to blow up VRAM above ~16k on consumer GPUs) |

- **Override:** use `--num-ctx N` on the CLI or `overrides={"num_ctx": N}` in Python.
- **VRAM:** a larger context costs KV-cache VRAM. If a model no longer fits in VRAM, lower `--num-ctx`.

### Reproducing the paper

The published results (Diagnostics 2026;16(11):1644) were produced with `num_ctx=8192`. To reproduce them, pass that value explicitly:

```bash
registrar-pipeline --input reports/ --model gpt --num-ctx 8192
```

```python
from digital_registrar.models.common import PAPER_NUM_CTX
setup_pipeline("gpt", overrides={"num_ctx": PAPER_NUM_CTX})
```

## Thinking mode (`think`)

`--think` / `--no-think` (Python: `overrides={"think": True | False}`) sets Ollama's top-level `think` request flag. When neither is given, which is the default, each model uses its own default. For `qwen3:30b` that means thinking is on, and its thinking tokens count against `max_tokens`.

Known caveats (audit §5.3):
- Ollama issue #15260: `think=false` breaks structured output (`format`) for `gemma4`.
- Qwen3 structured output is fragile with thinking disabled.

Validate on a few reports before pinning either setting. The flag is Ollama-only, and it is dropped for every other backend.

## Reference

Environment variables:

| Variable | Used for |
|---|---|
| `DIGITAL_REGISTRAR_OLLAMA_HOST` | Ollama endpoint. Beats `OLLAMA_HOST`; `--api-base` beats both. |
| `OLLAMA_HOST` | Ollama endpoint fallback, the same variable the `ollama` CLI reads. |
| `DIGITAL_REGISTRAR_API_KEY` | API key for `hosted_vllm/` / `openai/<name>` servers (default `EMPTY`). |
| `OPENAI_API_KEY` | Hosted OpenAI only (`gpt5_4_mini`). |

`registrar-pipeline` flags:

| Flag | Meaning |
|---|---|
| `--model` | An alias, or a raw `ollama_chat/…`, `hosted_vllm/…` or `openai/…` id. |
| `--api-base URL` | The server URL: Ollama host, or an OpenAI-compatible `/v1` base. |
| `--num-ctx N` | Ollama context window. Default: per model, as above. |
| `--think` / `--no-think` | Ollama thinking mode. Default: the model's own. |

Python (`digital_registrar.models.common`):

| Name | What |
|---|---|
| `load_model(name, overrides=None)` / `setup_pipeline(name, overrides=None)` / `setup_pipeline_v2(name, overrides=None)` | `overrides` may hold any sampler key (`temperature`, `top_p`, `top_k`, `num_ctx`, `max_tokens`, `seed`, …) plus `api_base` and `think`. `None` values are ignored. |
| `compute_lm_kwargs(name, overrides=None)` | The kwargs that would be sent, for logging and manifests. |
| `resolve_ollama_api_base(override=None)` | The Ollama URL that would be used. |
| `resolve_model_id(name)` | Alias or raw id → LiteLLM model id. |
| `PAPER_NUM_CTX` | `8192`, the paper's context window. |
