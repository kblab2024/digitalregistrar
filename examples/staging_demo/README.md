# staging_demo

Runnable examples for `digital_registrar_research.staging` — the
`tnmhelper`-backed AJCC TNM staging service.

These scripts are **examples, not part of the shipped pipeline**. They
live outside `src/` deliberately. Nothing in `pipeline_factory.py`,
the runner, or the eval harness imports them.

| Script | What it shows |
|---|---|
| [`post_process.py`](post_process.py) | Plain-callable surface: run the factory pipeline on a synthetic report, inspect `observable_schema(...)`, build the observations dict, call `stage_from_observations(...)`. Highlights the adapter friction between drr-next extraction output and `tnmhelper`'s observable names. |
| [`react_agent.py`](react_agent.py) | DSPy-tool surface: wrap `STAGING_TOOLS` in a `dspy.ReAct` module against a synthetic report; print the ReAct trajectory so the tool calls are visible. |

## Prerequisites

- `pip install -e .[all]` from the repo root.
- A configured LM. Either:
  - Ollama running locally with `gpt-oss:20b` pulled (default model in
    these scripts), **or**
  - `OPENAI_API_KEY` set, and edit `MODEL = "gpt"` to `MODEL = "gpt5_4_mini"`.
  - Model aliases live in
    [`src/digital_registrar_research/models/common.py`](../../src/digital_registrar_research/models/common.py).

## Running

```bash
python examples/staging_demo/post_process.py
python examples/staging_demo/react_agent.py
```

Each script is self-contained. The synthetic report is embedded in the
script; no external data needed.

## Background reading

- Module API → [`src/digital_registrar_research/staging/README.md`](../../src/digital_registrar_research/staging/README.md)
- Architecture → [`docs/architecture/staging.md`](../../docs/architecture/staging.md)
- Vendored wheel & data bundle → [`vendor/README.md`](../../vendor/README.md)
