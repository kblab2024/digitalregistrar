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

from typing import Literal

import dspy

model_list = {
    # --- Legacy keys (still used by pipeline.py __main__, ablations, existing
    # experiment.py invocations). Do not remove without sweeping callers. ---
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

localaddr = "http://localhost:11434"

# Per-model decoding profiles. Tuned for deterministic structured-JSON
# extraction on Ollama. Seed / cache are in _BASE_KWARGS so profiles stay
# focused on sampler choices.
MODEL_PROFILES: dict[str, dict] = {
    "ollama_chat/gpt-oss:20b":   {"temperature": 0.3,  "top_p": 1.0,  "top_k": 40, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/gemma3:27b":    {"temperature": 0.15, "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/gemma4:26b":    {"temperature": 0.1,  "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/gemma4:e2b":    {"temperature": 0.1,  "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/qwen3.5:27b":   {"temperature": 0.15, "top_p": 0.9,  "top_k": 40, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/medgemma:27b":  {"temperature": 0.15, "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/medgemma:4b":   {"temperature": 0.2,  "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    # OpenAI: stochastic profile mirrors the gpt-oss shape so K-run
    # reliability metrics (ICC, flip-rate, paired bootstrap) are meaningful.
    # No top_k / num_ctx (not supported by the chat-completions API).
    "openai/gpt-5.4-mini":       {"temperature": 0.3,  "top_p": 1.0,  "max_tokens": 4096},
}
_DEFAULT_PROFILE = {"temperature": 0.2, "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096}
_BASE_KWARGS = {"repeat_penalty": 1.05, "keep_alive": "30m", "cache": False, "seed": 10}

# Sampler / runtime kwargs that only make sense for the local Ollama backend.
# When a model_id is dispatched through a non-Ollama provider (e.g. ``openai/``)
# these are stripped from the kwargs passed to ``dspy.LM`` so LiteLLM does not
# forward them to the upstream API (which would 400).
_OLLAMA_ONLY_KEYS = ("top_k", "num_ctx", "repeat_penalty", "keep_alive")


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
    """
    if model_name not in model_list:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(model_list.keys())}")
    model_id = model_list[model_name]
    kwargs = {**_BASE_KWARGS, **MODEL_PROFILES.get(model_id, _DEFAULT_PROFILE)}
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def load_model(model_name: str, overrides: dict | None = None):
    model_id = model_list[model_name] if model_name in model_list else None
    if model_id is None:
        raise ValueError(f"Model {model_name} not found. Available models: {list(model_list.keys())}")

    kwargs = compute_lm_kwargs(model_name, overrides=overrides)

    if model_id.startswith("openai/"):
        from digital_registrar_research.util.secrets import load_openai_key
        api_key = load_openai_key()
        api_kwargs = {k: v for k, v in kwargs.items() if k not in _OLLAMA_ONLY_KEYS}
        lm = dspy.LM(
            model=model_id,
            api_key=api_key,
            model_type="chat",
            **api_kwargs,
        )
        # Print the redacted kwargs (api_key never logged).
        print(f"Loaded model: {model_name} (openai) with {api_kwargs}")
        return lm

    lm = dspy.LM(
        model=model_id,
        api_base=localaddr,
        api_key="",
        model_type="chat",
        **kwargs,
    )
    print(f"Loaded model: {model_name} with {kwargs}")
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
