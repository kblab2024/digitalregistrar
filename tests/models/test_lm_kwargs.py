"""LM kwargs / endpoint resolution in ``models.common`` and the runner flags.

No network: ``dspy.LM`` is constructed but never called, and the tests
inspect ``lm.model`` / ``lm.kwargs`` to see what would be sent.
"""
from __future__ import annotations

import pytest

from digital_registrar import runner
from digital_registrar.models.common import (
    MODEL_PROFILES,
    PAPER_NUM_CTX,
    compute_lm_kwargs,
    load_model,
    model_list,
    resolve_model_id,
    resolve_ollama_api_base,
)

_ENV_VARS = ("DIGITAL_REGISTRAR_OLLAMA_HOST", "OLLAMA_HOST", "DIGITAL_REGISTRAR_API_KEY")
_OLLAMA_ONLY = ("top_k", "num_ctx", "repeat_penalty", "keep_alive", "think")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Dev machines often export OLLAMA_HOST; keep the tests hermetic.
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --- compute_lm_kwargs -------------------------------------------------------

def test_default_profile_num_ctx_is_16384():
    assert model_list["phi4"] not in MODEL_PROFILES
    assert compute_lm_kwargs("phi4")["num_ctx"] == 16384


@pytest.mark.parametrize(("alias", "num_ctx"), [("gpt", 16384), ("gemma3", 16384), ("gemma4", 12288)])
def test_profile_num_ctx(alias, num_ctx):
    assert compute_lm_kwargs(alias)["num_ctx"] == num_ctx


def test_paper_num_ctx_is_below_defaults():
    assert PAPER_NUM_CTX == 8192
    assert all(p.get("num_ctx", 16384) > PAPER_NUM_CTX for p in MODEL_PROFILES.values())


def test_qwen3_30b_has_explicit_profile():
    assert "ollama_chat/qwen3:30b" in MODEL_PROFILES
    assert compute_lm_kwargs("qwen30b")["num_ctx"] == 16384


def test_no_profile_sets_think():
    assert not any("think" in p for p in MODEL_PROFILES.values())


def test_overrides_win_and_none_is_ignored():
    kwargs = compute_lm_kwargs("gpt", overrides={"num_ctx": PAPER_NUM_CTX, "temperature": None})
    assert kwargs["num_ctx"] == PAPER_NUM_CTX
    assert kwargs["temperature"] == MODEL_PROFILES["ollama_chat/gpt-oss:20b"]["temperature"]


def test_raw_ollama_id_uses_matching_profile():
    assert resolve_model_id("ollama_chat/qwen3:30b") == "ollama_chat/qwen3:30b"
    assert compute_lm_kwargs("ollama_chat/qwen3:30b") == compute_lm_kwargs("qwen30b")


@pytest.mark.parametrize("name", ["no-such-alias", "anthropic/claude-x", "hosted_vllm/"])
def test_unknown_model_raises(name):
    with pytest.raises(ValueError, match="not found"):
        compute_lm_kwargs(name)


# --- resolve_ollama_api_base -------------------------------------------------

def test_api_base_default_is_localhost():
    assert resolve_ollama_api_base() == "http://localhost:11434"


@pytest.mark.parametrize(("raw", "expected"), [
    ("gpu-box:11434", "http://gpu-box:11434"),
    ("gpu-box", "http://gpu-box:11434"),
    ("192.168.1.20:11500", "http://192.168.1.20:11500"),
    ("0.0.0.0", "http://localhost:11434"),
    ("0.0.0.0:11500", "http://localhost:11500"),
    (":11434", "http://localhost:11434"),
    ("http://host.docker.internal:11434/", "http://host.docker.internal:11434"),
    ("https://ollama.example.org/", "https://ollama.example.org"),
    ("  gpu-box:11434  ", "http://gpu-box:11434"),
])
def test_api_base_normalization(monkeypatch, raw, expected):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert resolve_ollama_api_base() == expected


def test_api_base_env_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "ollama-host:11434")
    assert resolve_ollama_api_base() == "http://ollama-host:11434"
    monkeypatch.setenv("DIGITAL_REGISTRAR_OLLAMA_HOST", "dr-host:11434")
    assert resolve_ollama_api_base() == "http://dr-host:11434"
    assert resolve_ollama_api_base("cli-host") == "http://cli-host:11434"


def test_api_base_empty_env_falls_through(monkeypatch):
    monkeypatch.setenv("DIGITAL_REGISTRAR_OLLAMA_HOST", "  ")
    monkeypatch.setenv("OLLAMA_HOST", "ollama-host")
    assert resolve_ollama_api_base() == "http://ollama-host:11434"


def test_api_base_keeps_credentials_but_does_not_print_them(capsys):
    lm = load_model("gpt", overrides={"api_base": "https://user:s3cret@ollama.example.org"})
    assert lm.kwargs["api_base"] == "https://user:s3cret@ollama.example.org"
    out = capsys.readouterr().out
    assert "s3cret" not in out
    assert "https://***@ollama.example.org" in out


def test_api_base_bad_port_raises(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:notaport")
    with pytest.raises(ValueError, match="OLLAMA_HOST"):
        resolve_ollama_api_base()


# --- load_model: Ollama ------------------------------------------------------

def test_load_model_ollama_defaults():
    lm = load_model("gpt")
    assert lm.model == "ollama_chat/gpt-oss:20b"
    assert lm.kwargs["api_base"] == "http://localhost:11434"
    assert lm.kwargs["num_ctx"] == 16384
    assert "think" not in lm.kwargs


def test_load_model_ollama_env_api_base(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    assert load_model("gpt").kwargs["api_base"] == "http://gpu-box:11434"


def test_load_model_api_base_override_no_typeerror(monkeypatch):
    # Used to raise "got multiple values for keyword argument 'api_base'".
    monkeypatch.setenv("DIGITAL_REGISTRAR_OLLAMA_HOST", "env-host")
    lm = load_model("gpt", overrides={"api_base": "10.0.0.5:11434"})
    assert lm.kwargs["api_base"] == "http://10.0.0.5:11434"


def test_load_model_ollama_num_ctx_and_think_overrides():
    lm = load_model("qwen30b", overrides={"num_ctx": PAPER_NUM_CTX, "think": False})
    assert lm.kwargs["num_ctx"] == PAPER_NUM_CTX
    assert lm.kwargs["think"] is False


def test_load_model_raw_ollama_tag(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    lm = load_model("ollama_chat/llama3.3:70b")
    assert lm.model == "ollama_chat/llama3.3:70b"
    assert lm.kwargs["api_base"] == "http://gpu-box:11434"
    assert lm.kwargs["num_ctx"] == 16384


# --- load_model: OpenAI-compatible server (vLLM / llama.cpp) -----------------

def test_load_model_hosted_vllm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    lm = load_model(
        "hosted_vllm/Qwen/Qwen3-30B-A3B",
        overrides={"api_base": "http://vllm-box:8000/v1/", "num_ctx": 8192, "think": False},
    )
    assert lm.model == "hosted_vllm/Qwen/Qwen3-30B-A3B"
    assert lm.kwargs["api_base"] == "http://vllm-box:8000/v1"
    # The real OpenAI key must never go to a user-supplied server.
    assert lm.kwargs["api_key"] == "EMPTY"
    assert not set(_OLLAMA_ONLY) & set(lm.kwargs)
    assert lm.kwargs["max_tokens"] == 4096


def test_load_model_compat_server_api_key_env(monkeypatch):
    monkeypatch.setenv("DIGITAL_REGISTRAR_API_KEY", "local-secret")
    lm = load_model("openai/qwen3-30b-a3b", overrides={"api_base": "llama-box:8080/v1"})
    assert lm.kwargs["api_base"] == "http://llama-box:8080/v1"
    assert lm.kwargs["api_key"] == "local-secret"


def test_load_model_compat_server_requires_api_base(monkeypatch):
    # The Ollama env vars must not be reused for an OpenAI-compatible server.
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    with pytest.raises(ValueError, match="--api-base"):
        load_model("openai/local-model")


# --- load_model: hosted OpenAI alias ----------------------------------------

@pytest.fixture
def _fake_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("digital_registrar.util.secrets.load_openai_key", lambda: "sk-test")


@pytest.mark.usefixtures("_fake_openai_key")
def test_load_model_openai_alias_builds_real_lm(monkeypatch):
    # dspy >= 3.4 refuses temperature != 1.0 / max_tokens < 16000 for gpt-5
    # ids at construction; the profile values must still land in lm.kwargs.
    monkeypatch.setenv("DIGITAL_REGISTRAR_OLLAMA_HOST", "gpu-box")
    lm = load_model("gpt5_4_mini", overrides={"num_ctx": 8192, "think": False})
    profile = MODEL_PROFILES["openai/gpt-5.4-mini"]
    assert lm.model == "openai/gpt-5.4-mini"
    assert lm.kwargs["api_key"] == "sk-test"
    assert "api_base" not in lm.kwargs
    assert not set(_OLLAMA_ONLY) & set(lm.kwargs)
    assert lm.kwargs["temperature"] == profile["temperature"]
    assert lm.kwargs["top_p"] == profile["top_p"]
    assert lm.kwargs["max_completion_tokens"] == profile["max_tokens"]
    assert lm.kwargs.get("max_tokens") is None


@pytest.mark.usefixtures("_fake_openai_key")
def test_load_model_openai_alias_sampler_overrides():
    lm = load_model("gpt5_4_mini", overrides={"temperature": 0.7, "max_tokens": 2048})
    assert lm.kwargs["temperature"] == 0.7
    assert lm.kwargs["max_completion_tokens"] == 2048
    assert lm.kwargs.get("max_tokens") is None


# --- runner flags --------------------------------------------------------------

def test_runner_no_flags_means_no_overrides():
    args = runner._build_parser().parse_args([])
    assert runner._lm_overrides(args) is None


def test_runner_flags_become_overrides():
    args = runner._build_parser().parse_args(
        ["--api-base", "gpu-box:11434", "--num-ctx", "8192", "--no-think"])
    assert runner._lm_overrides(args) == {"api_base": "gpu-box:11434", "num_ctx": 8192, "think": False}
    args = runner._build_parser().parse_args(["--think"])
    assert runner._lm_overrides(args) == {"think": True}
