"""Streamlit entry point for the inference GUI.

Run via the console script ``registrar-infer-gui`` (see ``pyproject.toml``),
which delegates to :func:`main_cli`. The CLI shells out to ``streamlit run``
against this file.

UI shape mirrors the annotation tool: sidebar holds controls, the main area
is a two-column layout with the input report on the left (scrollable) and the
pipeline output JSON on the right (sticky).
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import subprocess
import sys
from pathlib import Path


# ── CSS (copied verbatim from annotation/app.py:47-82) ────────────────────────

_STICKY_CSS = """
<style>
[data-testid="stHorizontalBlock"]:has(.report-col-marker) {
    align-items: flex-start !important;
    overflow: visible !important;
}

[data-testid="stHorizontalBlock"]:has(.report-col-marker) > [data-testid="stColumn"]:last-child,
[data-testid="stHorizontalBlock"]:has(.report-col-marker) > [data-testid="column"]:last-child,
[data-testid="stColumn"]:has(> div .report-col-marker),
[data-testid="stColumn"]:has(.report-col-marker) {
    position: sticky !important;
    top: 1rem !important;
    align-self: flex-start !important;
    max-height: calc(100vh - 2rem) !important;
    overflow-y: auto !important;
}

.report-text-pre {
    background: #f0f2f6;
    border: 1px solid #d6d8dc;
    border-radius: 4px;
    padding: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px;
    line-height: 1.5;
    margin: 0;
}
</style>
"""


def main() -> None:  # noqa: C901 — one-file Streamlit page is conventional
    """Streamlit page renderer — invoked by ``streamlit run app.py``."""
    import streamlit as st

    st.set_page_config(page_title="drr inference", layout="wide")
    st.markdown(_STICKY_CSS, unsafe_allow_html=True)

    from digital_registrar_research.models.common import model_list
    from digital_registrar_research.paths import RAW_REPORTS

    from digital_registrar_research.inference_gui.runner_bridge import (
        PipelineSetupError,
        capture_lm_trace,
        lm_history_len,
        make_run_dir,
        run_one,
        save_output,
        setup_for,
    )

    # ── Session state ─────────────────────────────────────────────────────────

    defaults = {
        "mode": "single",
        "engine": "factory",
        "model": "gpt",
        "decomposition": "auto",
        "jsonize": False,
        "validate_output": True,
        "input_text": "",
        "uploaded_name": "",
        "single_result": None,        # dict | None
        "folder_path": str(RAW_REPORTS),
        "run_dir": None,              # Path | None
        "results": [],                # list[{stem, output, elapsed, error}]
        "viewed_idx": 0,
        "last_error": "",
        "lm_context": None,           # dict | None — kwargs for dspy.context(**ctx)
        "lm_context_for": None,       # (engine, model) tuple
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.title("drr inference")
        st.session_state.mode = st.radio(
            "Mode", ["single", "folder"],
            index=["single", "folder"].index(st.session_state.mode),
            horizontal=True,
        )

        model_keys = list(model_list.keys())
        st.session_state.model = st.selectbox(
            "Model",
            model_keys,
            index=model_keys.index(st.session_state.model)
            if st.session_state.model in model_keys else 0,
        )
        provider = "openai" if model_list[st.session_state.model].startswith("openai/") else "ollama (local)"
        st.caption(f"`{model_list[st.session_state.model]}` — {provider}")

        st.session_state.engine = st.radio(
            "Engine", ["factory", "legacy"],
            index=["factory", "legacy"].index(st.session_state.engine),
            horizontal=True,
        )

        if st.session_state.engine == "factory":
            st.session_state.decomposition = st.selectbox(
                "Decomposition", ["auto", "per_group", "monolithic"],
                index=["auto", "per_group", "monolithic"].index(st.session_state.decomposition),
            )
            st.session_state.jsonize = st.checkbox(
                "Jsonize preprocessing", value=st.session_state.jsonize,
            )
            st.session_state.validate_output = st.checkbox(
                "Validate output", value=st.session_state.validate_output,
            )

        st.divider()

        if st.session_state.mode == "single":
            uploaded = st.file_uploader("Upload .txt", type=["txt"])
            if uploaded is not None:
                st.session_state.input_text = uploaded.getvalue().decode("utf-8", errors="replace")
                st.session_state.uploaded_name = Path(uploaded.name).stem
            run_clicked = st.button("Run", use_container_width=True, type="primary")
        else:
            st.session_state.folder_path = st.text_input(
                "Folder", value=st.session_state.folder_path,
            )
            run_clicked = st.button("Run all", use_container_width=True, type="primary")

        st.caption(
            "Switching engine/model triggers a global DSPy reconfigure on the "
            "next Run. Avoid running other DSPy notebooks against this process "
            "simultaneously."
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _ensure_pipeline_ready() -> bool:
        key = (st.session_state.engine, st.session_state.model)
        if st.session_state.lm_context_for == key and st.session_state.lm_context is not None:
            return True
        try:
            st.session_state.lm_context = setup_for(
                st.session_state.engine, st.session_state.model,
            )
        except PipelineSetupError as e:
            st.error(f"Model setup failed — {e}")
            return False
        st.session_state.lm_context_for = key
        return True

    def _run_single() -> None:
        text = st.session_state.input_text.strip()
        if not text:
            st.warning("Paste or upload a report first.")
            return
        if not _ensure_pipeline_ready():
            return
        stem = st.session_state.uploaded_name or "pasted_report"
        ctx = st.session_state.lm_context
        before = lm_history_len(ctx)
        with st.spinner(f"Running {st.session_state.engine}/{st.session_state.model}…"):
            try:
                out, elapsed = run_one(
                    st.session_state.input_text, stem,
                    engine=st.session_state.engine,
                    lm_context=ctx,
                    decomposition=st.session_state.decomposition,
                    jsonize=st.session_state.jsonize,
                    validate_output=st.session_state.validate_output,
                )
            except Exception as e:
                trace, n_calls = capture_lm_trace(ctx, before)
                st.session_state.single_result = {
                    "stem": stem, "output": None, "elapsed": 0.0, "error": repr(e),
                    "lm_trace": trace, "lm_calls": n_calls,
                }
                return
        trace, n_calls = capture_lm_trace(ctx, before)
        run_dir = make_run_dir()
        save_path = save_output(run_dir, stem, out)
        st.session_state.run_dir = run_dir
        st.session_state.single_result = {
            "stem": stem, "output": out, "elapsed": elapsed, "error": None,
            "save_path": str(save_path),
            "lm_trace": trace, "lm_calls": n_calls,
        }

    def _run_folder() -> None:
        folder = Path(st.session_state.folder_path)
        if not folder.exists():
            st.warning(f"Folder not found: {folder}")
            return
        files = sorted(folder.glob("*.txt"))
        if not files:
            st.warning(f"No .txt files in {folder}")
            return
        if not _ensure_pipeline_ready():
            return
        st.session_state.run_dir = make_run_dir()
        st.session_state.results = []
        st.session_state.viewed_idx = 0
        prog = st.progress(0.0, text="Starting…")
        status = st.status("Running…", expanded=False)
        ctx = st.session_state.lm_context
        for i, fp in enumerate(files, 1):
            before = lm_history_len(ctx)
            try:
                report = fp.read_text(encoding="utf-8")
                out, sec = run_one(
                    report, fp.stem,
                    engine=st.session_state.engine,
                    lm_context=ctx,
                    decomposition=st.session_state.decomposition,
                    jsonize=st.session_state.jsonize,
                    validate_output=st.session_state.validate_output,
                )
                save_output(st.session_state.run_dir, fp.stem, out)
                trace, n_calls = capture_lm_trace(ctx, before)
                st.session_state.results.append({
                    "stem": fp.stem, "output": out, "elapsed": sec, "error": None,
                    "source_path": str(fp),
                    "lm_trace": trace, "lm_calls": n_calls,
                })
            except Exception as e:
                trace, n_calls = capture_lm_trace(ctx, before)
                st.session_state.results.append({
                    "stem": fp.stem, "output": None, "elapsed": 0.0, "error": repr(e),
                    "source_path": str(fp),
                    "lm_trace": trace, "lm_calls": n_calls,
                })
                status.write(f"FAIL {fp.name}: {e}")
            prog.progress(i / len(files), text=f"{i}/{len(files)} {fp.name}")
        ok = sum(1 for r in st.session_state.results if not r["error"])
        status.update(label=f"Done — {ok}/{len(files)} ok → {st.session_state.run_dir}", state="complete")

    if run_clicked:
        if st.session_state.mode == "single":
            _run_single()
        else:
            _run_folder()

    # ── Main two-column layout ────────────────────────────────────────────────

    st.title("pipeline inference")

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.markdown('<span class="report-col-marker"></span>', unsafe_allow_html=True)
        _render_output_panel(st)

    with col_left:
        _render_input_panel(st)


def _render_input_panel(st) -> None:
    """Left column: textarea (single) or file-list + preview (folder)."""
    if st.session_state.mode == "single":
        st.subheader("Input")
        st.session_state.input_text = st.text_area(
            "Report text",
            value=st.session_state.input_text,
            height=520,
            label_visibility="collapsed",
            placeholder="Paste a pathology report here, or upload a .txt in the sidebar.",
        )
        return

    # Folder mode
    st.subheader("Files")
    results = st.session_state.results
    if not results:
        st.info("Pick a folder and click **Run all** in the sidebar.")
        return

    labels = [
        f"[{'OK' if not r['error'] else 'FAIL'}] {r['stem']}"
        for r in results
    ]
    idx = st.selectbox(
        "Pick a file",
        options=list(range(len(results))),
        format_func=lambda i: labels[i],
        index=min(st.session_state.viewed_idx, len(results) - 1),
        label_visibility="collapsed",
    )
    st.session_state.viewed_idx = idx
    r = results[idx]

    source_path = Path(r.get("source_path", ""))
    if source_path.exists():
        try:
            report_text = source_path.read_text(encoding="utf-8")
        except Exception as e:
            st.warning(f"Could not read source: {e}")
            return
        report_html = _html.escape(report_text).replace("|", "<br>")
        st.markdown(
            f'<pre class="report-text-pre">{report_html}</pre>',
            unsafe_allow_html=True,
        )


def _render_output_panel(st) -> None:
    """Right column (sticky): JSON output + download + traceback if any."""
    st.subheader("Output")

    if st.session_state.mode == "single":
        r = st.session_state.single_result
        if r is None:
            st.info("Click **Run** to extract structured fields.")
            return
        _render_result(st, r)
        return

    results = st.session_state.results
    if not results:
        st.info("Output appears here as the folder run progresses.")
        return
    _render_result(st, results[st.session_state.viewed_idx])


def _render_result(st, r: dict) -> None:
    """Render one ``{stem, output, elapsed, error}`` record."""
    if r.get("error"):
        st.error(f"{r['stem']} — failed.")
        with st.expander("Traceback", expanded=False):
            st.code(r["error"])
        _render_lm_trace(st, r)
        return

    elapsed = r.get("elapsed", 0.0)
    save_hint = r.get("save_path") or (
        f"{st.session_state.run_dir}/{r['stem']}_output.json"
        if st.session_state.run_dir else "(not saved)"
    )
    st.caption(f"elapsed: {elapsed:.2f}s — saved to `{save_hint}`")
    st.json(r["output"], expanded=True)
    st.download_button(
        "Download JSON",
        data=json.dumps(r["output"], ensure_ascii=False, indent=2),
        file_name=f"{r['stem']}_output.json",
        mime="application/json",
        use_container_width=True,
    )
    _render_lm_trace(st, r)


def _render_lm_trace(st, r: dict) -> None:
    """Expander showing the dspy.inspect_history() output for this run.

    One entry per ``dspy.Predict`` call: router → optional jsonize → one per
    group extractor. Useful for debugging which group signature produced bad
    output (look at its prompt + the LM's raw completion).
    """
    trace = r.get("lm_trace", "")
    n_calls = r.get("lm_calls", 0)
    if not trace:
        return
    with st.expander(f"LM trace — {n_calls} call{'s' if n_calls != 1 else ''}", expanded=False):
        st.download_button(
            "Download trace",
            data=trace,
            file_name=f"{r['stem']}_lm_trace.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.text(trace)


def main_cli() -> int:
    """Console-script wrapper: launches ``streamlit run`` on this module.

    Mirrors the ``registrar-schema-gui`` pattern. Returns the subprocess
    exit code.
    """
    parser = argparse.ArgumentParser(
        prog="registrar-infer-gui",
        description="Streamlit GUI for running the cancer-extraction pipeline.",
    )
    parser.add_argument(
        "--port", type=int, default=8502,
        help="port for streamlit to bind (default: 8502)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="do not auto-open a browser",
    )
    args = parser.parse_args()

    app_path = Path(__file__).resolve()
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
    ]
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    env = dict(os.environ)
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    main()
