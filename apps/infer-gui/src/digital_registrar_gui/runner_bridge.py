"""Thin adapter between the Streamlit UI and the extraction pipeline.

Keeps ``app.py`` free of pipeline / DSPy imports so the page module
loads quickly and is easy to reason about. Mirrors the contract of
``runner._select_runner`` so GUI output is interchangeable with the
``registrar-pipeline`` CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from digital_registrar.paths import RUNS_ROOT


class PipelineSetupError(RuntimeError):
    """Raised when configuring DSPy for the chosen engine/model fails."""


def setup_for(engine: str, model: str) -> dict:
    """Build the LM and the ``dspy.context`` kwargs for ``engine`` × ``model``.

    Returns a dict suitable for ``dspy.context(**ctx)``. We deliberately do
    NOT call ``dspy.configure`` here: DSPy 3.x pins ``configure`` to the
    thread that first invoked it (otherwise raising "dspy.settings can only
    be changed by the thread that initially configured it"). Streamlit
    reruns its script on different worker threads from a pool, so any
    second model swap from the GUI would hit that guard. ``dspy.context``
    is thread-safe and applied per-call in :func:`run_one`.
    """
    try:
        from digital_registrar.models.common import load_model
        lm = load_model(model)
    except Exception as e:
        raise PipelineSetupError(f"{engine}/{model}: {e}") from e

    ctx: dict = {"lm": lm}
    if engine == "factory":
        # Mirror setup_pipeline_v2's side effect: silence DSPy 3.2's
        # field-mismatch warnings, which fire frequently for dynamically
        # built signatures even when the LM output is valid.
        ctx["disable_typeguard_warnings"] = True
    return ctx


def run_one(
    report: str,
    fname: str,
    *,
    engine: str,
    lm_context: dict,
    decomposition: Literal["per_group", "monolithic", "auto"] = "auto",
    jsonize: bool = False,
    validate_output: bool = True,
) -> tuple[dict, float]:
    """Run the pipeline on a single report under ``lm_context``.

    Wraps the call in ``dspy.context(**lm_context)`` so every internal
    ``dspy.Predict`` sees the chosen LM without us having to call
    ``dspy.configure`` (which would be thread-pinned). Returns
    ``(output, elapsed_s)``.
    """
    import dspy

    with dspy.context(**lm_context):
        if engine == "factory":
            from digital_registrar.pipeline_factory import run_cancer_pipeline_v2
            return run_cancer_pipeline_v2(
                report=report,
                fname=fname,
                decomposition=decomposition,
                jsonize_enabled=jsonize,
                validate_output=validate_output,
            )
        from digital_registrar.pipeline import run_cancer_pipeline
        return run_cancer_pipeline(report=report, fname=fname)


def save_output(run_dir: Path, stem: str, output: dict) -> Path:
    """Write ``output`` to ``run_dir/<stem>_output.json``.

    Byte-identical to the CLI runner's write at ``runner.run_folder``.
    """
    out_path = run_dir / f"{stem}_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return out_path


def make_run_dir() -> Path:
    """Create ``workspace/runs/run_<timestamp>/`` (override via ``DIGITAL_REGISTRAR_WORKSPACE``)."""
    from digital_registrar.runner import create_run_folder
    base = RUNS_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return Path(create_run_folder(str(base)))


# ── LM trace capture ────────────────────────────────────────────────────────
# Each ``dspy.Predict`` call (router, optional jsonize, every group extractor)
# appends one entry to ``lm.history``. To isolate a single pipeline run's
# calls in folder mode (where history accumulates across files) we snapshot
# the length before, then render only the new slice.


def lm_history_len(lm_context: dict | None) -> int:
    """Length of the LM history on ``lm_context['lm']`` (0 if unset)."""
    if not lm_context:
        return 0
    lm = lm_context.get("lm")
    return len(lm.history) if lm is not None and hasattr(lm, "history") else 0


def capture_lm_trace(lm_context: dict | None, since: int) -> tuple[str, int]:
    """Capture ``lm.inspect_history()`` for entries appended since ``since``.

    Returns ``(trace_text, n_calls)``. ANSI colour codes are stripped so the
    text renders cleanly in Streamlit. Writes to a ``StringIO`` via
    ``inspect_history(file=...)`` — no stdout redirection needed.
    """
    import io
    import re
    if not lm_context:
        return "", 0
    lm = lm_context.get("lm")
    if lm is None or not hasattr(lm, "history"):
        return "", 0
    n = max(len(lm.history) - since, 0)
    if n <= 0:
        return "", 0
    buf = io.StringIO()
    lm.inspect_history(n=n, file=buf)
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", buf.getvalue())
    return text, n


__all__ = [
    "PipelineSetupError",
    "setup_for",
    "run_one",
    "save_output",
    "make_run_dir",
    "lm_history_len",
    "capture_lm_trace",
]
