"""Annotator list persistence — lives next to app.py as annotators.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "annotators.json"

DEFAULTS: list[dict] = [
    {"name": "Nan-Haw Chow", "suffix": "nhc"},
    {"name": "Kai-Po Chang", "suffix": "kpc"},
]

_SUFFIX_RE = re.compile(r"^[a-z0-9]{1,6}$")

RESERVED_SUFFIXES = {"gold"}


def load_annotators() -> list[dict]:
    """Return the annotator list. Writes DEFAULTS on first run."""
    if not CONFIG_PATH.exists():
        save_annotators(DEFAULTS)
        return [dict(a) for a in DEFAULTS]
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        annotators = data.get("annotators", [])
        return [
            {"name": str(a["name"]), "suffix": str(a["suffix"])}
            for a in annotators
            if isinstance(a, dict) and a.get("name") and a.get("suffix")
        ]
    except (OSError, json.JSONDecodeError, KeyError):
        return [dict(a) for a in DEFAULTS]


def save_annotators(annotators: list[dict]) -> None:
    payload = {"annotators": annotators}
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def add_annotator(name: str, suffix: str) -> tuple[bool, str]:
    """Append a new annotator. Returns (ok, message)."""
    name = (name or "").strip()
    suffix = (suffix or "").strip().lower()

    if not name:
        return False, "請輸入全名。"
    if not _SUFFIX_RE.match(suffix):
        return False, "縮寫需為 1–6 個小寫英數字元。"
    if suffix in RESERVED_SUFFIXES:
        return False, f"縮寫「{suffix}」為系統保留，請改用其他縮寫。"

    annotators = load_annotators()
    if any(a["suffix"] == suffix for a in annotators):
        return False, f"縮寫「{suffix}」已存在。"
    if any(a["name"] == name for a in annotators):
        return False, f"全名「{name}」已存在。"

    annotators.append({"name": name, "suffix": suffix})
    save_annotators(annotators)
    return True, f"已新增 {name} ({suffix})。"
