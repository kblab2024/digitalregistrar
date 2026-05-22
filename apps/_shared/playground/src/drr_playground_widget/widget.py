"""Single-file Streamlit playground widget.

The widget renders a self-contained chunker + signature-inspector UI
inside whatever container the host app drops it into. State is keyed
under ``st.session_state.playground__*`` so the host's other state is
left alone.

Host integration:

    from drr_playground_widget import render_playground

    render_playground(
        st,
        lm_context=st.session_state.get("lm_context"),   # optional dspy.context kwargs
        input_state_key="input_text",                    # share text with another tab
        organ_state=None,                                # schema-editor passes OrganState
    )

When ``organ_state`` is provided, the organ picker defaults to that
organ and the live ``group_instructions`` from the state override what's
on disk — that's how the schema-editor's edits show up immediately in
the rendered LM prompt.
"""
from __future__ import annotations

import html as _html
import io
import json
import re
from typing import Any

# Color palette for chunk highlighting; cycled.
_PALETTE = [
    "#ffd8a8", "#a8e6cf", "#dcedc1", "#cce5ff", "#fdcae1",
    "#e4c1f9", "#fcf6bd", "#c9e4ca", "#d8b4fe", "#fde2e4",
]


def render_playground(
    st,
    *,
    lm_context: dict | None = None,
    input_state_key: str = "input_text",
    organ_state: Any = None,
) -> None:
    """Top-level renderer.

    ``st`` is the streamlit module (passed in so the widget doesn't
    pin a streamlit version). All other arguments are optional.
    """
    from digital_registrar.chunking import CHUNKER_REGISTRY, ROUTER_REGISTRY
    from digital_registrar.schemas import CASE_MODELS
    from digital_registrar.schemas.extraction import EXTRACTION_META

    _init_state(st, input_state_key)
    organ_default = getattr(organ_state, "organ_key", None) if organ_state else None

    # ─── Controls row ───────────────────────────────────────────────────
    cols = st.columns([1.2, 1.2, 1.0, 1.4])

    chunker_names = sorted(CHUNKER_REGISTRY)
    chunker_idx = _safe_index(chunker_names, st.session_state.playground__chunker, default=0)
    with cols[0]:
        chunker_name = st.selectbox("Chunker", chunker_names, index=chunker_idx, key="playground__chunker_picker")
        st.session_state.playground__chunker = chunker_name

    router_options = ["(none — chunker self-labels)"] + sorted(ROUTER_REGISTRY)
    router_idx = _safe_index(router_options, st.session_state.playground__router, default=0)
    with cols[1]:
        router_name = st.selectbox("Router", router_options, index=router_idx, key="playground__router_picker")
        st.session_state.playground__router = router_name

    organ_keys = sorted(CASE_MODELS)
    if organ_default and organ_default in organ_keys:
        organ_idx = organ_keys.index(organ_default)
    else:
        organ_idx = _safe_index(organ_keys, st.session_state.playground__organ, default=0)
    with cols[2]:
        organ = st.selectbox("Organ", organ_keys, index=organ_idx, key="playground__organ_picker")
        st.session_state.playground__organ = organ

    groups_map = _resolve_groups(organ, organ_state, EXTRACTION_META)
    group_names = list(groups_map)
    group_idx = _safe_index(group_names, st.session_state.playground__group, default=0)
    with cols[3]:
        group = st.selectbox("Group", group_names, index=group_idx, key="playground__group_picker")
        st.session_state.playground__group = group

    if organ_state is not None:
        st.caption(
            f"Using **live** group instructions from the editor (organ "
            f"`{organ_state.organ_key}`). Field types & per-field descriptions "
            "still come from the on-disk schema."
        )

    # ─── Input + visualization ─────────────────────────────────────────
    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("Report")
        text_default = st.session_state.get(input_state_key, "")
        report_text = st.text_area(
            "Report text",
            value=text_default,
            height=320,
            key="playground__report_input",
            label_visibility="collapsed",
            placeholder="Paste a pathology report here…",
        )
        # Mirror into the host's input key so the Pipeline tab sees the same text.
        if report_text != text_default:
            st.session_state[input_state_key] = report_text

        chunk_clicked = st.button("Chunk", use_container_width=True, type="secondary")
        if chunk_clicked or st.session_state.playground__last_text != report_text:
            st.session_state.playground__chunks = _do_chunk(
                report_text, chunker_name, router_name, groups_map,
            )
            st.session_state.playground__last_text = report_text
            # Default-check chunks routed to the current group.
            st.session_state.playground__picked = {
                c.id for c in st.session_state.playground__chunks
                if group in c.labels
            }

    chunks: list = st.session_state.playground__chunks
    with right:
        st.subheader(f"Chunks ({len(chunks)})")
        if not chunks:
            st.info("Click **Chunk** to split the report.")
        else:
            _render_chunk_picker(st, chunks, group)

    if chunks and report_text:
        st.markdown("##### Highlighted report")
        st.markdown(
            _highlighted_html(report_text, chunks, st.session_state.playground__picked),
            unsafe_allow_html=True,
        )

    # ─── Run ───────────────────────────────────────────────────────────
    st.divider()
    run_clicked = st.button(
        "Run signature on selected chunks", use_container_width=True, type="primary",
        disabled=not chunks,
    )

    if run_clicked:
        picked_chunks = [c for c in chunks if c.id in st.session_state.playground__picked]
        if not picked_chunks:
            st.warning("Pick at least one chunk first.")
        else:
            with st.spinner("Running signature…"):
                result = _run_single_signature(
                    organ=organ, group=group, picked=picked_chunks,
                    groups_map=groups_map, lm_context=lm_context,
                )
            st.session_state.playground__last_result = result

    result = st.session_state.playground__last_result
    if result is not None:
        _render_result(st, result)


# ─── State init ────────────────────────────────────────────────────────


def _init_state(st, input_state_key: str) -> None:
    defaults = {
        "playground__chunker": "regex_section",
        "playground__router": "(none — chunker self-labels)",
        "playground__organ": None,
        "playground__group": None,
        "playground__chunks": [],
        "playground__picked": set(),
        "playground__last_text": "",
        "playground__last_result": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault(input_state_key, "")


def _safe_index(options: list, value, *, default: int) -> int:
    if value in options:
        return options.index(value)
    return default if 0 <= default < len(options) else 0


def _resolve_groups(organ: str, organ_state, EXTRACTION_META) -> dict[str, str]:
    """Group name → instruction text. Live state overrides on-disk when given."""
    disk = dict(EXTRACTION_META[organ]["groups"])
    if organ_state is None or organ_state.organ_key != organ:
        return disk
    live = {gi.group_name: gi.instruction for gi in organ_state.group_instructions}
    return live if live else disk


# ─── Chunk computation ─────────────────────────────────────────────────


def _do_chunk(
    text: str,
    chunker_name: str,
    router_name: str,
    groups_map: dict[str, str],
) -> list:
    from digital_registrar.chunking import CHUNKER_REGISTRY, ROUTER_REGISTRY

    if not text.strip():
        return []
    try:
        chunker = CHUNKER_REGISTRY[chunker_name]()
        chunks = list(chunker.chunk(text))
    except Exception as e:
        import streamlit as st
        st.error(f"Chunker {chunker_name!r} failed: {e!r}")
        return []
    if router_name in ROUTER_REGISTRY:
        try:
            router = ROUTER_REGISTRY[router_name]()
            chunks = list(router.route(chunks, groups_map))
        except Exception as e:
            import streamlit as st
            st.error(f"Router {router_name!r} failed: {e!r}")
    return chunks


# ─── Chunk picker UI ───────────────────────────────────────────────────


def _render_chunk_picker(st, chunks: list, group: str) -> None:
    picked: set = st.session_state.playground__picked
    new_picked: set = set()
    for c in chunks:
        labels_s = ", ".join(sorted(c.labels)) if c.labels else "—"
        label = f"`{c.id}` · {labels_s}"
        default = c.id in picked
        checked = st.checkbox(label, value=default, key=f"playground__pick_{c.id}")
        if checked:
            new_picked.add(c.id)
        with st.expander(f"preview {c.id}", expanded=False):
            section = c.meta.get("section") if isinstance(c.meta, dict) else None
            if section:
                st.caption(f"section: `{section}` · span: {c.span[0]}..{c.span[1]}")
            else:
                st.caption(f"span: {c.span[0]}..{c.span[1]}")
            st.text(c.text[:600] + ("…" if len(c.text) > 600 else ""))
    st.session_state.playground__picked = new_picked


def _highlighted_html(text: str, chunks: list, picked: set) -> str:
    """Render the report as HTML with chunk spans color-highlighted.

    Picked chunks get a thicker outline so the user can see exactly what
    will be fed to the LM. Overlapping chunks are not handled (none of
    the built-in chunkers produce them).
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.span[0])
    parts: list[str] = []
    cursor = 0
    color_for: dict[str, str] = {}
    for i, c in enumerate(sorted_chunks):
        s, e = c.span
        if s > cursor:
            parts.append(_html.escape(text[cursor:s]))
        color = _PALETTE[i % len(_PALETTE)]
        color_for[c.id] = color
        outline = "2px solid #333" if c.id in picked else "1px dashed #aaa"
        title = f"{c.id} · {','.join(sorted(c.labels)) or '—'}"
        parts.append(
            f'<span style="background:{color}; outline:{outline}; '
            f'outline-offset:1px; border-radius:3px;" title="{_html.escape(title)}">'
            f'{_html.escape(text[s:e])}</span>'
        )
        cursor = e
    if cursor < len(text):
        parts.append(_html.escape(text[cursor:]))
    body = "".join(parts).replace("\n", "<br>")
    return (
        '<pre style="background:#f8f9fb; border:1px solid #d6d8dc; '
        'border-radius:4px; padding:12px; white-space:pre-wrap; '
        'word-wrap:break-word; font-family:ui-monospace, SFMono-Regular, '
        f'Menlo, Consolas, monospace; font-size:13px; line-height:1.5;">{body}</pre>'
    )


# ─── Run single signature ──────────────────────────────────────────────


def _run_single_signature(
    *, organ: str, group: str, picked: list,
    groups_map: dict[str, str], lm_context: dict | None,
) -> dict:
    """Build the per-group extraction signature, run it on picked chunks."""
    import dspy

    from digital_registrar.schemas import CASE_MODELS
    from digital_registrar.schemas.extraction import EXTRACTION_META
    from digital_registrar.signatures.factory import build_extraction_signatures

    organ_meta = EXTRACTION_META[organ]
    # Use live groups_map only for the chosen group's instruction; other
    # group metadata isn't needed since we run per_group.
    groups_override = dict(organ_meta["groups"])
    if group in groups_map:
        groups_override[group] = groups_map[group]

    steps = build_extraction_signatures(
        CASE_MODELS[organ],
        organ_meta["fields"],
        groups_override,
        decomposition="per_group",
    )
    step = next((s for s in steps if s.group == group), None)
    if step is None:
        return {"error": f"no per-group step found for {group!r} in {organ!r}"}

    joined = [c.text for c in picked]
    lm = (lm_context or {}).get("lm")
    history_before = len(lm.history) if (lm is not None and hasattr(lm, "history")) else 0

    predictor = dspy.Predict(step.signature)
    try:
        if lm_context:
            with dspy.context(**lm_context):
                pred = predictor(report=joined, report_jsonized={})
        else:
            pred = predictor(report=joined, report_jsonized={})
    except Exception as e:
        return {"error": repr(e), "step": step.name}

    trace = _capture_trace(lm, history_before)
    parsed = {k: getattr(pred, k, None) for k in step.output_field_names}
    return {
        "step": step.name,
        "joined_input": joined,
        "parsed": parsed,
        "trace": trace,
        "chunk_ids": [c.id for c in picked],
    }


def _capture_trace(lm, since: int) -> str:
    if lm is None or not hasattr(lm, "history"):
        return ""
    n = max(len(lm.history) - since, 0)
    if n <= 0:
        return ""
    buf = io.StringIO()
    lm.inspect_history(n=n, file=buf)
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", buf.getvalue())


# ─── Result rendering ──────────────────────────────────────────────────


def _render_result(st, result: dict) -> None:
    st.divider()
    st.subheader(f"Result — `{result.get('step', '?')}`")
    if "error" in result and result["error"]:
        st.error(result["error"])

    if "joined_input" in result:
        with st.expander("Input sent to LM", expanded=False):
            for i, t in enumerate(result["joined_input"]):
                st.markdown(f"**Chunk {i+1}** (`{result['chunk_ids'][i]}`)")
                st.text(t)

    if "parsed" in result:
        st.markdown("**Parsed output**")
        st.json(_json_safe(result["parsed"]), expanded=True)

    if result.get("trace"):
        with st.expander("LM trace (prompt + completion)", expanded=False):
            st.text(result["trace"])


def _json_safe(obj):
    """Best-effort jsonify for pydantic BaseModel instances."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(x) for x in obj]
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                return repr(obj)
        return repr(obj)
