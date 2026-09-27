"""
models/common.py
This script sets up a series of data extraction models using the dspy library for pathology reports. Common model includes basic dspy functionality, cancer examination, and json handling. It includes model loading, signature definitions for various cancer types, and functions to convert model predictions into structured JSON formats.

author: Hong-Kai (Walther) Chen, Po-Yen Tzeng and Kai-Po Chang @ Med NLP Lab, China Medical University
date: 2025-10-13
"""
__version__ = "1.0.0"
__date__ = "2025-10-13"
__author__ = ["Hong-Kai (Walther) Chen", "Po-Yen Tzeng", "Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import os
from typing import Literal
from urllib.parse import urlsplit

import dspy

model_list = {
    # --- Legacy keys (still used by pipeline.py __main__, ablations, existing
    # runner.py invocations). Do not remove without sweeping callers. ---
    "gemma4b": "ollama_chat/gemma3:4b",
    "gemma1b": "ollama_chat/gemma3:1b",
    "gemma4e2b": "ollama_chat/gemma4:e2b",
    "med8b": "ollama_chat/thewindmom/llama3-med42-8b",
    "gemma12b": "ollama_chat/gemma3:12b",
    "gemma27b": "ollama_chat/gemma3:27b",
    "med70b": "ollama_chat/thewindmom/llama3-med42-70b",
    "gpt": "ollama_chat/gpt-oss:20b",
    "phi4": "ollama_chat/phi4",
    "qwen30b": "ollama_chat/qwen3:30b",
    # --- Unified aliases consumed by the consolidated runners
    # scripts/run_dspy_ollama_{single,smoke}.py via --model <alias>. ---
    "gptoss":         "ollama_chat/gpt-oss:20b",
    "gemma3":         "ollama_chat/gemma3:27b",
    "gemma4":         "ollama_chat/gemma4:26b",
    "gemma4large":         "ollama_chat/gemma4:31b",
    "qwen3_5":        "ollama_chat/qwen3.5:27b",
    "medgemmalarge":  "ollama_chat/medgemma:27b",
    "medgemmasmall":  "ollama_chat/medgemma:4b",
    "qwen3_6":        "ollama_chat/qwen3.6:27b",
    # OpenAI-hosted top-class model. Routed through dspy.LM via LiteLLM's
    # `openai/` provider; api_key is loaded from util.secrets.load_openai_key
    # (see load_model below). Used as a "yet another top-class model"
    # comparator on the public TCGA set for the rebuttal vs reviewer (a).
    "gpt5_4_mini":    "openai/gpt-5.4-mini",
}

# Fallback Ollama endpoint. Kept as a module constant for backward
# compatibility (attic runners import it); the endpoint load_model actually
# uses is resolved per call by resolve_ollama_api_base(), which honours
# --api-base / overrides["api_base"] and the env vars below.
localaddr = "http://localhost:11434"
_OLLAMA_DEFAULT_PORT = 11434

# Env vars consulted, in order, for the Ollama endpoint. OLLAMA_HOST is the
# variable the Ollama CLI itself reads, so a shell already set up for
# `ollama run` against a remote server works unchanged.
OLLAMA_HOST_ENV_VARS = ("DIGITAL_REGISTRAR_OLLAMA_HOST", "OLLAMA_HOST")

# API key sent to a custom OpenAI-compatible server (vLLM, llama.cpp, ...).
# OPENAI_API_KEY is deliberately never forwarded to a user-supplied endpoint.
API_KEY_ENV_VAR = "DIGITAL_REGISTRAR_API_KEY"

# LiteLLM provider prefixes accepted as raw model ids (anything that is not a
# model_list alias): any pulled Ollama tag, or a model served by vLLM /
# llama.cpp / another OpenAI-compatible server reached via api_base.
_RAW_MODEL_PREFIXES = ("ollama_chat/", "hosted_vllm/", "openai/")

# Bind-all addresses: valid for a server to listen on (OLLAMA_HOST=0.0.0.0 is
# a common server-side setting) but not for a client to connect to.
_UNSPECIFIED_HOSTS = ("", "0.0.0.0", "::")

# num_ctx used for the published results (Diagnostics 2026;16(11):1644). The
# profile defaults below are larger, per
# docs/architecture/dspy_ollama_model_compatibility.md §5.2; pass
# ``--num-ctx 8192`` (``overrides={"num_ctx": PAPER_NUM_CTX}``) to reproduce
# the paper's runs.
PAPER_NUM_CTX = 8192

# Per-model decoding profiles. Tuned for deterministic structured-JSON
# extraction on Ollama. Seed / cache are in _BASE_KWARGS so profiles stay
# focused on sampler choices.
#
# num_ctx: Ollama silently drops the *start* of any prompt longer than
# num_ctx, so default to 16384; gemma4 gets 12288 (VRAM blow-up above ~16k on
# consumer GPUs, audit §5.2).
#
# An optional ``"think": bool`` key is forwarded as Ollama's top-level think
# flag. No profile sets it, so each model keeps its own default; see audit
# §5.3 for the gemma4 / qwen3 structured-output caveats before pinning it.
MODEL_PROFILES: dict[str, dict] = {
    "ollama_chat/gpt-oss:20b":   {"temperature": 0.3,  "top_p": 1.0,  "top_k": 40, "num_ctx": 16384, "max_tokens": 4096},
    "ollama_chat/gemma3:27b":    {"temperature": 0.15, "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "max_tokens": 4096},
    "ollama_chat/gemma4:26b":    {"temperature": 0.1,  "top_p": 0.95, "top_k": 64, "num_ctx": 12288, "max_tokens": 4096},
    "ollama_chat/gemma4:e2b":    {"temperature": 0.1,  "top_p": 0.95, "top_k": 64, "num_ctx": 12288, "max_tokens": 4096},
    "ollama_chat/qwen3.5:27b":   {"temperature": 0.15, "top_p": 0.9,  "top_k": 40, "num_ctx": 16384, "max_tokens": 4096},
    # Sampler values are the _DEFAULT_PROFILE ones qwen3:30b fell back to
    # before it had a profile, so only num_ctx differs from the paper runs.
    "ollama_chat/qwen3:30b":     {"temperature": 0.2,  "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "max_tokens": 4096},
    "ollama_chat/medgemma:27b":  {"temperature": 0.15, "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "max_tokens": 4096},
    "ollama_chat/medgemma:4b":   {"temperature": 0.2,  "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "max_tokens": 4096},
    # OpenAI: stochastic profile mirrors the gpt-oss shape so K-run
    # reliability metrics (ICC, flip-rate, paired bootstrap) are meaningful.
    # No top_k / num_ctx (not supported by the chat-completions API).
    # temperature 0.3 is what the rebuttal runs sent. OpenAI accepts it only
    # while reasoning_effort is "none" (this model's default), so do not add
    # reasoning_effort here. load_model sets temperature and the token cap
    # after building the LM (see _OPENAI_POST_INIT_KEYS).
    "openai/gpt-5.4-mini":       {"temperature": 0.3,  "top_p": 1.0,  "max_tokens": 4096},
}
_DEFAULT_PROFILE = {"temperature": 0.2, "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "max_tokens": 4096}
_BASE_KWARGS = {"repeat_penalty": 1.05, "keep_alive": "30m", "cache": False, "seed": 10}

# Sampler / runtime kwargs that only make sense for the local Ollama backend.
# When a model_id is dispatched through a non-Ollama provider (e.g. ``openai/``)
# these are stripped from the kwargs passed to ``dspy.LM`` so LiteLLM does not
# forward them to the upstream API (which would 400).
_OLLAMA_ONLY_KEYS = ("top_k", "num_ctx", "repeat_penalty", "keep_alive", "think")

# OpenAI model families that reject ``max_tokens`` and require
# ``max_completion_tokens`` instead (gpt-5.x + reasoning o-series).
_OPENAI_COMPLETION_TOKENS_PREFIXES = ("gpt-5", "gpt5", "o1", "o3", "o4")

# For the families above, these kwargs are set on ``lm.kwargs`` after
# ``dspy.LM`` is built instead of being passed to it. dspy >= 3.4 treats every
# ``openai/gpt-5*`` id (dotted ones like gpt-5.4-mini included) as a reasoning
# model and refuses temperature != 1.0 or a token cap below 16000 at
# construction; a pre-renamed ``max_completion_tokens`` raises TypeError
# there. gpt-5.1+ accept both when reasoning_effort is "none", their default
# (LiteLLM's model map agrees), and dspy merges ``lm.kwargs`` into every
# request, so the profile values still reach the API.
_OPENAI_POST_INIT_KEYS = ("temperature", "max_completion_tokens")


def _needs_max_completion_tokens(model_id: str) -> bool:
    if not model_id.startswith("openai/"):
        return False
    bare = model_id.split("/", 1)[1]
    return bare.startswith(_OPENAI_COMPLETION_TOKENS_PREFIXES)


def _normalize_base_url(raw: str, *, default_port: int | None = None, source: str = "api_base") -> str:
    """Turn a user-supplied host or URL into a base URL LiteLLM can call.

    ``host`` / ``host:port`` without a scheme get ``http://``, plus
    *default_port* when no port is given (mirrors the Ollama CLI's own
    ``OLLAMA_HOST`` parsing). Bind-all hosts (``0.0.0.0``, ``::``, empty as
    in ``:11434``) become ``localhost``. A URL with an explicit scheme keeps
    its port and path as given; trailing slashes are dropped.
    """
    value = raw.strip().rstrip("/")
    has_scheme = "://" in value
    if not has_scheme:
        value = f"http://{value}"
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError as e:
        raise ValueError(f"Invalid {source} {raw!r}: {e}") from None
    host = parts.hostname or ""
    if host in _UNSPECIFIED_HOSTS:
        host = "localhost"
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    if port is None and not has_scheme and default_port is not None:
        port = default_port
    userinfo = parts.netloc.rpartition("@")[0]
    netloc = (f"{userinfo}@" if userinfo else "") + (host if port is None else f"{host}:{port}")
    return f"{parts.scheme}://{netloc}{parts.path}".rstrip("/")


def resolve_ollama_api_base(override: str | None = None) -> str:
    """Return the Ollama base URL to call, resolved at call time.

    Precedence (first non-empty wins): *override* (``--api-base`` /
    ``overrides["api_base"]``), ``$DIGITAL_REGISTRAR_OLLAMA_HOST``,
    ``$OLLAMA_HOST``, then :data:`localaddr`. Values may be a bare
    ``host`` / ``host:port`` (see :func:`_normalize_base_url`).
    """
    candidates = [("api_base", override)] + [(var, os.environ.get(var)) for var in OLLAMA_HOST_ENV_VARS]
    for source, value in candidates:
        if value is not None and str(value).strip():
            return _normalize_base_url(str(value), default_port=_OLLAMA_DEFAULT_PORT, source=source)
    return localaddr


def _redact_url(url: str) -> str:
    """Hide ``user:password@`` in a URL before it is printed."""
    scheme, sep, rest = url.partition("://")
    netloc, slash, path = rest.partition("/")
    if "@" in netloc:
        netloc = "***@" + netloc.rpartition("@")[2]
    return f"{scheme}{sep}{netloc}{slash}{path}"


def resolve_model_id(model_name: str) -> str:
    """Map a ``model_list`` alias, or a raw LiteLLM id, to the LiteLLM model id.

    Raw ids are accepted for the providers in ``_RAW_MODEL_PREFIXES``:
    ``ollama_chat/<tag>`` for any pulled Ollama model, and
    ``hosted_vllm/<name>`` / ``openai/<name>`` for an OpenAI-compatible
    server (vLLM, llama.cpp, ...) whose URL is passed as ``api_base``.
    """
    if model_name in model_list:
        return model_list[model_name]
    if model_name.startswith(_RAW_MODEL_PREFIXES) and model_name.split("/", 1)[1]:
        return model_name
    raise ValueError(
        f"Model {model_name} not found. Available models: {list(model_list.keys())}, "
        f"or a raw LiteLLM id starting with one of {list(_RAW_MODEL_PREFIXES)}.")


def compute_lm_kwargs(model_name: str, overrides: dict | None = None) -> dict:
    """Resolve the final dspy.LM kwargs for *model_name*.

    Layered: ``_BASE_KWARGS`` ⊕ ``MODEL_PROFILES[model_id]`` (or
    ``_DEFAULT_PROFILE``) ⊕ caller overrides (non-None only). Returned
    verbatim so callers (runners, manifests) can record what was actually
    sent to the LM. ``load_model`` calls this internally; runners that
    need to log the kwargs without constructing an LM should call it
    directly. Provider-incompatible keys are not stripped here — that
    happens in :func:`load_model` only when the LM is actually built,
    so the manifest still records the intended sampler config.

    Besides sampler keys, overrides may carry ``api_base`` (server URL;
    consumed by :func:`load_model`) and ``think`` (Ollama thinking mode).
    *model_name* is a ``model_list`` alias or a raw id (see
    :func:`resolve_model_id`).
    """
    model_id = resolve_model_id(model_name)
    kwargs = {**_BASE_KWARGS, **MODEL_PROFILES.get(model_id, _DEFAULT_PROFILE)}
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def load_model(model_name: str, overrides: dict | None = None):
    """Build the ``dspy.LM`` for *model_name* (alias or raw LiteLLM id).

    Three routes:

    * hosted OpenAI alias (``gpt5_4_mini``): key from
      :func:`~digital_registrar.util.secrets.load_openai_key`;
    * ``ollama_chat/...``: endpoint from :func:`resolve_ollama_api_base`
      (``overrides["api_base"]`` > env vars > localhost);
    * raw ``hosted_vllm/...`` / ``openai/...``: an OpenAI-compatible server
      at ``overrides["api_base"]`` (required), key from
      ``$DIGITAL_REGISTRAR_API_KEY`` or ``"EMPTY"``.

    Ollama-only kwargs are stripped on both OpenAI-style routes.
    """
    model_id = resolve_model_id(model_name)
    kwargs = compute_lm_kwargs(model_name, overrides=overrides)

    if model_name in model_list and model_id.startswith("openai/"):
        from digital_registrar.util.secrets import load_openai_key
        api_key = load_openai_key()
        api_kwargs = {k: v for k, v in kwargs.items() if k not in _OLLAMA_ONLY_KEYS}
        post_init = {}
        if _needs_max_completion_tokens(model_id):
            if "max_tokens" in api_kwargs:
                api_kwargs["max_completion_tokens"] = api_kwargs.pop("max_tokens")
            post_init = {k: api_kwargs.pop(k) for k in _OPENAI_POST_INIT_KEYS if k in api_kwargs}
        lm = dspy.LM(
            model=model_id,
            api_key=api_key,
            model_type="chat",
            **api_kwargs,
        )
        lm.kwargs.update(post_init)
        # Print the redacted kwargs (api_key never logged).
        print(f"Loaded model: {model_name} (openai) with {api_kwargs | post_init}")
        return lm

    if model_id.startswith("ollama_chat/"):
        api_base = resolve_ollama_api_base(kwargs.pop("api_base", None))
        lm = dspy.LM(
            model=model_id,
            api_base=api_base,
            api_key="",
            model_type="chat",
            **kwargs,
        )
        print(f"Loaded model: {model_name} at {_redact_url(api_base)} with {kwargs}")
        return lm

    # Raw hosted_vllm/ or openai/ id: an OpenAI-compatible server. The Ollama
    # env vars are not consulted here; the URL must be given explicitly.
    raw_base = kwargs.pop("api_base", None)
    if not raw_base or not str(raw_base).strip():
        raise ValueError(
            f"Model {model_name} needs the URL of an OpenAI-compatible server: pass "
            "--api-base (e.g. http://gpu-box:8000/v1) or overrides={'api_base': ...}. "
            "For hosted OpenAI use the 'gpt5_4_mini' alias.")
    api_base = _normalize_base_url(str(raw_base))
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip() or "EMPTY"
    api_kwargs = {k: v for k, v in kwargs.items() if k not in _OLLAMA_ONLY_KEYS}
    lm = dspy.LM(
        model=model_id,
        api_base=api_base,
        api_key=api_key,
        model_type="chat",
        **api_kwargs,
    )
    # Print the redacted kwargs (api_key never logged).
    print(f"Loaded model: {model_name} at {_redact_url(api_base)} with {api_kwargs}")
    return lm

# 2 . define classes and set up Signatures

def autoconf_dspy (model_name: str, overrides: dict | None = None):
    lm = load_model(model_name, overrides=overrides)
    dspy.configure(lm=lm)

class is_cancer(dspy.Signature):
    """You are a cancer registrar, you need to identify whether or not this report belongs to PRIMARY cancer excision eligible for cancer registry, and if so, which organ the cancer belongs to. If no viable tumor is present after excision, you should not register this case. If only carcinoma in situ or high-grade dysplasia, you should not register this case."""

    report: list = dspy.InputField(desc = 'this is a pathologic report, separated into paragraphs. you should determine whether or not this report belongs to cancer excision eligible for cancer registry')

    cancer_excision_report: bool = dspy.OutputField(desc= 'identify whether or not this report belongs to PRIMARY cancer excision eligible for registry for cancer excision. If no viable tumor is present after excision, you should not register this case. If only carcinoma in situ or high-grade dysplasia, you should not register this case.')#a point
    #exp:
    cancer_category: Literal['stomach','colorectal','breast','esophagus', 'lung', 'prostate', "thyroid", "pancreas", "cervix", "liver", "others"]|None = dspy.OutputField(desc = 'identify which organ the primary cancer arises from. Currently only ten are implemented, if it IS a cancer excision report, but primary site not included in these standard organs, should be classified as others.')
    cancer_category_others_description: str|None = dspy.OutputField(desc = 'if is cancer_excision report AND cancer_category is others, please specify the organ here. if not, return null.')

class ReportJsonize(dspy.Signature):
    """You are cancer registrar, and you are assigned a task to manually convert the raw pathology report into a roughly structured json format. Keep original wording as much as possible. Try to follow the order of cancer checklists."""
    report: list = dspy.InputField(desc = 'this is a raw pathological report, separated into paragraphs. You need to convert it into a roughly structured json format, keeping original wording as much as possible.')
    cancer_category: Literal['stomach','colorectal','breast','esophagus', 'lung', 'prostate', "pancreas", "thyroid", "cervix", "liver"]|None = dspy.InputField(desc = 'which part the cancer belongs to. You need to convert it into a roughly structured json format, keeping original wording as much as possible.')
    output: dict = dspy.OutputField(desc = 'You are cancer registrar, and you are assigned a task to manually convert the raw pathology report into a roughly structured json format. Keep original wording as much as possible. Try to follow the order of cancer checklists.')
