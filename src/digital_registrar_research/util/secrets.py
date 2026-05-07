"""Out-of-repo secret loading for hosted-LLM runners.

The OpenAI API key is intentionally stored *outside* the repository tree
(at ``~/.config/digital-registrar/.env``) so it cannot be committed,
shipped in a tarball, or read by repo-rooted tooling that scans for
secrets. Existing in-repo paths (``.env`` at repo root, ``secrets/``)
are git-ignored but still live alongside source code; the out-of-repo
location is one level safer.

Resolution order for ``OPENAI_API_KEY`` (first non-empty wins):
  1. ``OPENAI_API_KEY`` already in process env (CI / one-shot override).
  2. ``OPENAI_API_KEY=...`` line in ``$XDG_CONFIG_HOME/digital-registrar/.env``
     (falls back to ``~/.config/digital-registrar/.env``).

The loader never prints the key. Callers should also avoid logging it.

This is a deliberately tiny parser (no python-dotenv dependency): we
only support ``KEY=VALUE`` lines, optional surrounding whitespace, and
optional ``"`` / ``'`` quoting around the value. Comments (``#``) and
blank lines are skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILENAME = ".env"
_CONFIG_SUBDIR = "digital-registrar"


def _config_dir() -> Path:
    """Return the directory that holds the secrets ``.env`` file.

    Honours ``XDG_CONFIG_HOME`` per the XDG Base Directory Spec; otherwise
    falls back to ``~/.config`` (the spec's default). Works on Windows too:
    ``Path.home()`` resolves to ``C:\\Users\\<user>``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / _CONFIG_SUBDIR


def _parse_env_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2 and value[0] == value[-1]
                and value[0] in ("'", '"')):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_openai_key() -> str:
    """Return the OpenAI API key, reading from env or the out-of-repo file.

    Caches the value in ``os.environ["OPENAI_API_KEY"]`` after first read so
    repeated calls within one process do not re-parse the file. Raises
    ``RuntimeError`` with an actionable message (no key contents) if the key
    cannot be found.
    """
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    path = _config_dir() / _ENV_FILENAME
    if not path.is_file():
        raise RuntimeError(
            "OPENAI_API_KEY is not set and no out-of-repo secrets file was "
            f"found at {path}. Create it with one line:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "or export OPENAI_API_KEY in your shell before running.")

    parsed = _parse_env_file(path.read_text(encoding="utf-8"))
    key = parsed.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found in {path}. The file exists but does "
            "not contain a non-empty 'OPENAI_API_KEY=...' line.")

    os.environ["OPENAI_API_KEY"] = key
    return key


__all__ = ["load_openai_key"]
