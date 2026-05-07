"""Digital Registrar Compare / Consensus Tool — Streamlit app.

Two modes share a folder picker and sample list:
  • Consensus   — A vs B vs Gold (editable). Saves _gold.json.
  • Evaluation  — Reference annotator vs machine output. Read-only.

Reuses annotation.io, annotation.parser, annotation.annotator_config, annotation.ui.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from digital_registrar_research.annotation.annotator_config import (
    RESERVED_SUFFIXES,
    load_annotators,
)
from digital_registrar_research.annotation.diff_utils import (
    ARRAY_KEY_FIELDS,
    FieldDiff,
    aggregate_stats,
    align_arrays_by_key,
    diff_flat_fields,
    values_differ,
)
from digital_registrar_research.annotation.io import (
    NA_SENTINEL,
    FolderSet,
    build_save_payload,
    discover_folders,
    list_samples,
    load_json,
    load_report_text,
    rehydrate_sentinels,
    save_annotation,
    strip_meta,
)
from digital_registrar_research.annotation.parser import (
    CANCER_CATEGORIES,
    CANCER_TO_FILE,
    FieldSpec,
    SectionSpec,
    parse_cancer_schema,
)
from digital_registrar_research.annotation.ui import pick_folder

NOT_SET_LABEL = "— not set —"
NA_LABEL = "— N/A —"

st.set_page_config(page_title="Compare / Consensus", layout="wide")


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.diff-cell-a    { background:#fff7d6; padding:3px 8px; border-radius:4px;
                  border:1px solid #f1d67c; display:inline-block; min-width:80%; }
.diff-cell-b    { background:#d6ecff; padding:3px 8px; border-radius:4px;
                  border:1px solid #86b8e6; display:inline-block; min-width:80%; }
.diff-cell-plain{ padding:3px 8px; display:inline-block; }
.null-val       { color:#9a9a9a; font-style:italic; }
.agree-tag      { color:#2e7d32; font-size:0.8em; margin-left:6px; }
.disagree-tag   { color:#b36200; font-size:0.8em; margin-left:6px; font-weight:600; }
.sec-summary    { color:#555; font-size:0.85em; margin-bottom:6px; }
.field-label    { font-weight:600; font-size:0.92em; margin-bottom:2px; }
.cell-head-a    { color:#8a6a00; font-size:0.78em; font-weight:600; }
.cell-head-b    { color:#1560a8; font-size:0.78em; font-weight:600; }
.cell-head-g    { color:#333; font-size:0.78em; font-weight:600; }
</style>
"""


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "mode": "consensus",          # "consensus" | "evaluation"
        "base_dir": "",
        "folders": None,              # FolderSet | None
        "annotators": load_annotators(),
        "suffix_a": None,
        "suffix_b": None,
        "eval_ref_suffix": None,      # used in evaluation mode
        "samples": [],                # list[CompareSampleRef]
        "sample_idx": 0,
        "stem_filter": "all",
        "report_text": "",
        "annotation_a": {},
        "annotation_b": {},
        "gold_annotation": {},        # editable gold (consensus only)
        "last_sample_id": "",
        "save_message": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
st.markdown(_CSS, unsafe_allow_html=True)


# ── Sample ref for compare mode ────────────────────────────────────────────────

@dataclass
class CompareSampleRef:
    sample_id: str
    stem: str
    n: str
    dataset_path: str
    result_path: str
    path_a: str
    path_b: str
    path_gold: str


def _build_compare_samples(
    folders: FolderSet, suffix_a: str, suffix_b: str, gold_suffix: str = "gold",
) -> list[CompareSampleRef]:
    base = list_samples(folders, suffix_a)  # discovery + sort order
    out: list[CompareSampleRef] = []
    token_a = f"_annotation_{suffix_a}.json"
    token_b = f"_annotation_{suffix_b}.json"
    token_g = f"_annotation_{gold_suffix}.json"
    for s in base:
        out.append(CompareSampleRef(
            sample_id=s.sample_id,
            stem=s.stem,
            n=s.n,
            dataset_path=s.dataset_path,
            result_path=s.result_path,
            path_a=s.annotation_path,
            path_b=s.annotation_path.replace(token_a, token_b),
            path_gold=s.annotation_path.replace(token_a, token_g),
        ))
    return out


def _visible_samples() -> list[tuple[int, CompareSampleRef]]:
    samples = st.session_state.samples
    stem = st.session_state.stem_filter
    if stem == "all":
        return list(enumerate(samples))
    return [(i, s) for i, s in enumerate(samples) if s.stem == stem]


# ── Sample loading ─────────────────────────────────────────────────────────────

def _reload_samples() -> tuple[bool, str]:
    folders: FolderSet | None = st.session_state.folders
    if not folders:
        return False, "尚未載入資料夾。"
    if st.session_state.mode == "consensus":
        a = st.session_state.suffix_a
        b = st.session_state.suffix_b
        if not a or not b:
            st.session_state.samples = []
            return False, "請選擇 Annotator A 與 B。"
        if a == b:
            st.session_state.samples = []
            return False, "A 和 B 不能是同一位 annotator。"
        st.session_state.samples = _build_compare_samples(folders, a, b)
    else:
        ref = st.session_state.eval_ref_suffix
        if not ref:
            st.session_state.samples = []
            return False, "請選擇 Reference annotator。"
        # In evaluation mode, B-slot is the machine output (result_path).
        st.session_state.samples = _build_compare_samples(folders, ref, ref)
    st.session_state.sample_idx = 0
    st.session_state.stem_filter = "all"
    _on_file_change()
    return True, f"載入 {len(st.session_state.samples)} 個樣本。"


def _reload_from_base(base_dir: str) -> tuple[bool, str]:
    folders = discover_folders(base_dir)
    if not folders:
        return False, "找不到符合 `{prefix}_{dataset|result|annotation}_{date}` 格式的子資料夾。"
    st.session_state.base_dir = base_dir
    st.session_state.folders = folders
    return _reload_samples()


def _on_file_change():
    """Load A, B (or machine), and reset gold to a pre-filled copy of A."""
    samples: list[CompareSampleRef] = st.session_state.samples
    idx = st.session_state.sample_idx
    if not samples or idx >= len(samples):
        st.session_state.report_text = ""
        st.session_state.annotation_a = {}
        st.session_state.annotation_b = {}
        st.session_state.gold_annotation = {}
        st.session_state.last_sample_id = ""
        return

    sample = samples[idx]
    st.session_state.report_text = load_report_text(sample.dataset_path)

    raw_a = load_json(sample.path_a)
    st.session_state.annotation_a = strip_meta(raw_a)
    rehydrate_sentinels(st.session_state.annotation_a, raw_a.get("_meta"))

    if st.session_state.mode == "consensus":
        raw_b = load_json(sample.path_b)
        st.session_state.annotation_b = strip_meta(raw_b)
        rehydrate_sentinels(st.session_state.annotation_b, raw_b.get("_meta"))
        # Gold: resume from existing _gold file if saved; else copy of A.
        if os.path.exists(sample.path_gold):
            raw_gold = load_json(sample.path_gold)
            st.session_state.gold_annotation = strip_meta(raw_gold)
            rehydrate_sentinels(st.session_state.gold_annotation, raw_gold.get("_meta"))
        else:
            st.session_state.gold_annotation = copy.deepcopy(st.session_state.annotation_a)
    else:
        # Evaluation: B-slot is the machine output JSON (no _meta stripping needed;
        # real result files have no _meta, but strip defensively anyway).
        raw_b = load_json(sample.result_path)
        st.session_state.annotation_b = strip_meta(raw_b)
        rehydrate_sentinels(st.session_state.annotation_b, raw_b.get("_meta"))
        st.session_state.gold_annotation = {}  # unused in evaluation mode

    st.session_state.last_sample_id = sample.sample_id
    st.session_state.save_message = ""


# ── Value rendering (read-only A/B cells) ──────────────────────────────────────

def _fmt_scalar(v) -> str:
    if v == NA_SENTINEL:
        return NA_LABEL
    if v is None or v == "":
        return NOT_SET_LABEL
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v).replace("_", " ")


def _cell_html(v, kind: str) -> str:
    """kind: 'a' | 'b' | 'plain'. Returns an inline-styled markdown span."""
    cls = {"a": "diff-cell-a", "b": "diff-cell-b", "plain": "diff-cell-plain"}[kind]
    display = _fmt_scalar(v)
    if v is None or v == "" or v == NA_SENTINEL:
        return f"<span class='{cls}'><span class='null-val'>{display}</span></span>"
    return f"<span class='{cls}'>{display}</span>"


# ── Container / section helpers ────────────────────────────────────────────────

def _container_for(annotation: dict, section_name: str) -> dict:
    if section_name == "IsCancer":
        return annotation or {}
    return (annotation or {}).get("cancer_data") or {}


def _ensure_container(annotation: dict, section_name: str) -> dict:
    if section_name == "IsCancer":
        return annotation
    return annotation.setdefault("cancer_data", {})


# ── Gold editable widget (mirrors render_field in app.py) ──────────────────────

def _gold_input(field: FieldSpec, current_val, key: str, disabled: bool = False):
    help_text = field.description or None
    if field.field_type in ("enum", "int_enum"):
        options = [None, NA_SENTINEL, *list(field.enum_values)]
        idx = options.index(current_val) if current_val in options else 0
        return st.selectbox(
            "Gold", options=options, index=idx,
            format_func=_fmt_scalar, key=key, help=help_text,
            label_visibility="collapsed", disabled=disabled,
        )
    if field.field_type == "bool":
        options = [NA_SENTINEL, True, False]
        if current_val is None or current_val == NA_SENTINEL:
            idx = 0
        elif current_val in (True, False):
            idx = options.index(current_val)
        else:
            idx = 0
        return st.selectbox(
            "Gold", options=options, index=idx,
            format_func=_fmt_scalar, key=key, help=help_text,
            label_visibility="collapsed", disabled=disabled,
        )
    if field.field_type == "int":
        col_num, col_null = st.columns([3, 1])
        is_null = current_val is None or current_val == NA_SENTINEL
        with col_null:
            null_checked = st.checkbox("N/A", value=is_null,
                                       key=key + "__null", disabled=disabled)
        with col_num:
            num_init = (
                int(current_val)
                if isinstance(current_val, int) and not isinstance(current_val, bool)
                else 0
            )
            num_val = st.number_input(
                "Gold", value=num_init,
                min_value=0, step=1, disabled=null_checked or disabled,
                key=key + "__num", help=help_text,
                label_visibility="collapsed",
            )
        return NA_SENTINEL if null_checked else int(num_val)
    if field.field_type == "string":
        col_text, col_na = st.columns([3, 1])
        is_na = current_val == NA_SENTINEL
        with col_na:
            na_checked = st.checkbox("N/A", value=is_na,
                                     key=key + "__na", disabled=disabled)
        with col_text:
            text_init = "" if is_na else (current_val or "")
            raw = st.text_input(
                "Gold", value=text_init,
                key=key + "__text", help=help_text,
                label_visibility="collapsed", disabled=na_checked or disabled,
            )
        if na_checked:
            return NA_SENTINEL
        return raw if raw else None
    return current_val


# ── Consensus flat-field row renderer ──────────────────────────────────────────

def _apply_override(key: str, val):
    """Write an override directly into widget state, clearing stale sub-keys.

    Needed because `int` uses `__num`/`__null` and `string` uses `__text`/`__na`.
    """
    for suffix in ("", "__num", "__null", "__text", "__na"):
        st.session_state.pop(key + suffix, None)
    # Leave gold_annotation mutation to the caller.


def _render_field_row_consensus(
    field: FieldSpec,
    section_name: str,
    a_val, b_val,
    sample_id: str,
    key_prefix: str,
) -> None:
    differs = values_differ(a_val, b_val)
    key_base = f"{key_prefix}__{field.name}__{sample_id}"

    # Header: title + diff tag
    tag = ("<span class='disagree-tag'>✎ Disagree</span>" if differs
           else "<span class='agree-tag'>✓ Agree</span>")
    st.markdown(f"<div class='field-label'>{field.title} {tag}</div>",
                unsafe_allow_html=True)

    col_a, col_b, col_g = st.columns([3, 3, 6])
    with col_a:
        st.markdown(
            f"<div class='cell-head-a'>A</div>{_cell_html(a_val, 'a' if differs else 'plain')}",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='cell-head-b'>B</div>{_cell_html(b_val, 'b' if differs else 'plain')}",
            unsafe_allow_html=True,
        )
    with col_g:
        st.markdown("<div class='cell-head-g'>Gold</div>", unsafe_allow_html=True)
        gold_container = _ensure_container(st.session_state.gold_annotation, section_name)
        new_val = _gold_input(field, gold_container.get(field.name), key=key_base)
        gold_container[field.name] = new_val

        if differs:
            btn_a, btn_b, _ = st.columns([1, 1, 3])
            with btn_a:
                if st.button("⇐ Use A", key=key_base + "__usea"):
                    _apply_override(key_base, a_val)
                    gold_container[field.name] = a_val
                    st.rerun()
            with btn_b:
                if st.button("⇐ Use B", key=key_base + "__useb"):
                    _apply_override(key_base, b_val)
                    gold_container[field.name] = b_val
                    st.rerun()

    st.markdown("<hr style='margin:6px 0; border:none; border-top:1px solid #eee;'>",
                unsafe_allow_html=True)


# ── Evaluation (read-only) flat-field row ──────────────────────────────────────

def _render_field_row_eval(field: FieldSpec, a_val, b_val) -> None:
    differs = values_differ(a_val, b_val)
    tag = ("<span class='disagree-tag'>✎ Differs</span>" if differs
           else "<span class='agree-tag'>✓ Match</span>")
    st.markdown(f"<div class='field-label'>{field.title} {tag}</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            f"<div class='cell-head-a'>Reference</div>{_cell_html(a_val, 'a' if differs else 'plain')}",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='cell-head-b'>Machine</div>{_cell_html(b_val, 'b' if differs else 'plain')}",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:4px 0; border:none; border-top:1px solid #eee;'>",
                unsafe_allow_html=True)


# ── Array section rendering ────────────────────────────────────────────────────

def _default_gold_item(slot_a, slot_b) -> dict | None:
    """Initial gold item for a slot: prefer A; fall back to B."""
    if slot_a is not None:
        return copy.deepcopy(slot_a)
    if slot_b is not None:
        return copy.deepcopy(slot_b)
    return None


def _render_array_consensus(
    section: SectionSpec, sample_id: str, key_prefix: str,
) -> None:
    afield = section.array_field_name
    atype = section.array_field_type
    a_container = _container_for(st.session_state.annotation_a, section.name)
    b_container = _container_for(st.session_state.annotation_b, section.name)
    gold_container = _ensure_container(st.session_state.gold_annotation, section.name)

    st.markdown(f"#### Array: {afield.replace('_', ' ').title()}")

    if atype == "array_of_strings_enum":
        a_list = list(a_container.get(afield) or [])
        b_list = list(b_container.get(afield) or [])
        differ = set(a_list) != set(b_list)
        col_a, col_b, col_g = st.columns([3, 3, 6])
        with col_a:
            st.markdown("<div class='cell-head-a'>A</div>", unsafe_allow_html=True)
            st.markdown(", ".join(_fmt_scalar(v) for v in a_list) or "_— empty —_")
        with col_b:
            st.markdown("<div class='cell-head-b'>B</div>", unsafe_allow_html=True)
            st.markdown(", ".join(_fmt_scalar(v) for v in b_list) or "_— empty —_")
        with col_g:
            st.markdown("<div class='cell-head-g'>Gold</div>", unsafe_allow_html=True)
            current_gold = gold_container.get(afield) or []
            # Default: union of A + B for initial gold load is done via _on_file_change,
            # but gold was copied from A. If user wants union, quick buttons help.
            picked = st.multiselect(
                "Gold", options=section.array_item_enum_values,
                default=[v for v in current_gold if v in section.array_item_enum_values],
                key=f"{key_prefix}__{afield}__ms__{sample_id}",
                label_visibility="collapsed",
            )
            gold_container[afield] = picked or None
            if differ:
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("⇐ Use A", key=f"{key_prefix}_{afield}_usea_{sample_id}"):
                        st.session_state.pop(f"{key_prefix}__{afield}__ms__{sample_id}", None)
                        gold_container[afield] = list(a_list) or None
                        st.rerun()
                with b2:
                    if st.button("⇐ Use B", key=f"{key_prefix}_{afield}_useb_{sample_id}"):
                        st.session_state.pop(f"{key_prefix}__{afield}__ms__{sample_id}", None)
                        gold_container[afield] = list(b_list) or None
                        st.rerun()
                with b3:
                    if st.button("∪ Union", key=f"{key_prefix}_{afield}_union_{sample_id}"):
                        st.session_state.pop(f"{key_prefix}__{afield}__ms__{sample_id}", None)
                        gold_container[afield] = sorted(set(a_list) | set(b_list)) or None
                        st.rerun()
        return

    # array_of_objects
    a_list = list(a_container.get(afield) or [])
    b_list = list(b_container.get(afield) or [])
    key_field = ARRAY_KEY_FIELDS.get(afield)

    if key_field:
        slots = align_arrays_by_key(a_list, b_list, key_field)
    else:
        # Fallback: align by position.
        slots = []
        for i in range(max(len(a_list), len(b_list))):
            ai = a_list[i] if i < len(a_list) else None
            bi = b_list[i] if i < len(b_list) else None
            slots.append((ai, bi, f"#{i + 1}"))

    # Ensure gold array exists and is aligned to slot count.
    gold_list = gold_container.get(afield) or []
    if len(gold_list) != len(slots):
        gold_list = [_default_gold_item(ai, bi) for ai, bi, _ in slots]
        gold_container[afield] = gold_list

    include_states: list[bool] = []
    for slot_idx, (item_a, item_b, key_val) in enumerate(slots):
        key_suffix = str(key_val).replace(" ", "_") if key_val is not None else f"slot{slot_idx}"
        slot_key_prefix = f"{key_prefix}_{afield}_{key_suffix}"

        # Determine slot status
        both = item_a is not None and item_b is not None
        if both:
            inner_differ = any(
                values_differ(item_a.get(f.name), item_b.get(f.name))
                for f in section.array_item_fields
            )
            slot_label = f"{'✎' if inner_differ else '✓'} {key_val}  (A vs B)"
        elif item_a is not None:
            slot_label = f"◐ {key_val}  (A-only)"
        else:
            slot_label = f"◑ {key_val}  (B-only)"

        with st.expander(slot_label, expanded=both or item_a is None or item_b is None):
            col_inc, _ = st.columns([1, 4])
            with col_inc:
                included = st.checkbox(
                    "Include in Gold", value=gold_list[slot_idx] is not None,
                    key=slot_key_prefix + "__include__" + sample_id,
                )
            include_states.append(included)

            if not included:
                gold_list[slot_idx] = None
                continue

            if gold_list[slot_idx] is None:
                gold_list[slot_idx] = _default_gold_item(item_a, item_b) or {
                    f.name: None for f in section.array_item_fields
                }

            for field in section.array_item_fields:
                va = (item_a or {}).get(field.name)
                vb = (item_b or {}).get(field.name)
                differs = values_differ(va, vb)
                tag = ("<span class='disagree-tag'>✎</span>" if differs
                       else "<span class='agree-tag'>✓</span>")
                st.markdown(f"<div class='field-label'>{field.title} {tag}</div>",
                            unsafe_allow_html=True)
                col_a, col_b, col_g = st.columns([3, 3, 6])
                with col_a:
                    if item_a is None:
                        st.markdown("<span class='null-val'>(not in A)</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(_cell_html(va, 'a' if differs else 'plain'),
                                    unsafe_allow_html=True)
                with col_b:
                    if item_b is None:
                        st.markdown("<span class='null-val'>(not in B)</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(_cell_html(vb, 'b' if differs else 'plain'),
                                    unsafe_allow_html=True)
                with col_g:
                    widget_key = f"{slot_key_prefix}_{field.name}_{sample_id}"
                    new_val = _gold_input(
                        field, gold_list[slot_idx].get(field.name), key=widget_key,
                    )
                    gold_list[slot_idx][field.name] = new_val
                    if differs and item_a is not None and item_b is not None:
                        bA, bB, _ = st.columns([1, 1, 3])
                        with bA:
                            if st.button("⇐ A", key=widget_key + "__usea"):
                                _apply_override(widget_key, va)
                                gold_list[slot_idx][field.name] = va
                                st.rerun()
                        with bB:
                            if st.button("⇐ B", key=widget_key + "__useb"):
                                _apply_override(widget_key, vb)
                                gold_list[slot_idx][field.name] = vb
                                st.rerun()

    # Write back (cleaning None entries doesn't happen here — save payload cleans).


def _render_array_eval(section: SectionSpec) -> None:
    afield = section.array_field_name
    atype = section.array_field_type
    a_container = _container_for(st.session_state.annotation_a, section.name)
    b_container = _container_for(st.session_state.annotation_b, section.name)
    a_list = a_container.get(afield) or []
    b_list = b_container.get(afield) or []

    st.markdown(f"#### Array: {afield.replace('_', ' ').title()}")

    if atype == "array_of_strings_enum":
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("<div class='cell-head-a'>Reference</div>", unsafe_allow_html=True)
            st.markdown(", ".join(_fmt_scalar(v) for v in a_list) or "_— empty —_")
        with col_b:
            st.markdown("<div class='cell-head-b'>Machine</div>", unsafe_allow_html=True)
            st.markdown(", ".join(_fmt_scalar(v) for v in b_list) or "_— empty —_")
        return

    key_field = ARRAY_KEY_FIELDS.get(afield)
    if key_field:
        slots = align_arrays_by_key(list(a_list), list(b_list), key_field)
    else:
        slots = []
        for i in range(max(len(a_list), len(b_list))):
            ai = a_list[i] if i < len(a_list) else None
            bi = b_list[i] if i < len(b_list) else None
            slots.append((ai, bi, f"#{i + 1}"))

    for _slot_idx, (item_a, item_b, key_val) in enumerate(slots):
        both = item_a is not None and item_b is not None
        if both:
            inner_differ = any(
                values_differ(item_a.get(f.name), item_b.get(f.name))
                for f in section.array_item_fields
            )
            label = f"{'✎' if inner_differ else '✓'} {key_val}"
        elif item_a is not None:
            label = f"◐ {key_val} (Ref only)"
        else:
            label = f"◑ {key_val} (Machine only)"
        with st.expander(label, expanded=not both):
            for field in section.array_item_fields:
                va = (item_a or {}).get(field.name)
                vb = (item_b or {}).get(field.name)
                _render_field_row_eval(field, va, vb)


# ── Section dispatcher ─────────────────────────────────────────────────────────

def _section_field_diffs(section: SectionSpec) -> list[FieldDiff]:
    a_container = _container_for(st.session_state.annotation_a, section.name)
    b_container = _container_for(st.session_state.annotation_b, section.name)
    return diff_flat_fields(a_container, b_container, section.flat_fields)


def _render_section(section: SectionSpec, tab_index: int, mode: str) -> None:
    sample_id = st.session_state.last_sample_id
    key_prefix = f"sec_{mode}_{section.name}_{tab_index}"
    field_diffs = _section_field_diffs(section)
    stats = aggregate_stats(field_diffs)
    total_label = f"{stats['agree']}/{stats['total']} fields match"
    st.markdown(f"<div class='sec-summary'>{total_label}</div>", unsafe_allow_html=True)

    a_container = _container_for(st.session_state.annotation_a, section.name)
    b_container = _container_for(st.session_state.annotation_b, section.name)

    for field in section.flat_fields:
        va = a_container.get(field.name)
        vb = b_container.get(field.name)
        if mode == "consensus":
            _render_field_row_consensus(field, section.name, va, vb, sample_id, key_prefix)
        else:
            _render_field_row_eval(field, va, vb)

    if section.array_field_name:
        st.divider()
        if mode == "consensus":
            _render_array_consensus(section, sample_id, key_prefix)
        else:
            _render_array_eval(section)


# ── Classification (IsCancer) panel ────────────────────────────────────────────

def _render_classification(mode: str) -> None:
    sample_id = st.session_state.last_sample_id
    a = st.session_state.annotation_a
    b = st.session_state.annotation_b

    st.subheader("Classification")

    for field_name, title, field_type, enum_vals in [
        ("cancer_excision_report", "Cancer Excision Report", "bool", []),
        ("cancer_category", "Cancer Category", "enum",
         [c for c in CANCER_CATEGORIES if c is not None]),
        ("cancer_category_others_description", "Others Description", "string", []),
    ]:
        va = a.get(field_name)
        vb = b.get(field_name)
        field = FieldSpec(
            name=field_name, title=title, description="",
            field_type=field_type, enum_values=enum_vals,
        )
        if mode == "consensus":
            # Use the same row renderer but target the top-level gold container.
            differs = values_differ(va, vb)
            tag = ("<span class='disagree-tag'>✎ Disagree</span>" if differs
                   else "<span class='agree-tag'>✓ Agree</span>")
            st.markdown(f"<div class='field-label'>{title} {tag}</div>",
                        unsafe_allow_html=True)
            col_a, col_b, col_g = st.columns([3, 3, 6])
            with col_a:
                st.markdown(
                    f"<div class='cell-head-a'>A</div>{_cell_html(va, 'a' if differs else 'plain')}",
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"<div class='cell-head-b'>B</div>{_cell_html(vb, 'b' if differs else 'plain')}",
                    unsafe_allow_html=True,
                )
            with col_g:
                st.markdown("<div class='cell-head-g'>Gold</div>",
                            unsafe_allow_html=True)
                gold = st.session_state.gold_annotation
                key = f"cls__{field_name}__{sample_id}"
                new_val = _gold_input(field, gold.get(field_name), key=key)
                # If the category changes, drop cancer_data so it gets re-seeded
                if field_name == "cancer_category" and new_val != gold.get(field_name):
                    gold.pop("cancer_data", None)
                gold[field_name] = new_val

                if differs:
                    bA, bB, _ = st.columns([1, 1, 3])
                    with bA:
                        if st.button("⇐ Use A", key=key + "__usea"):
                            _apply_override(key, va)
                            gold[field_name] = va
                            if field_name == "cancer_category":
                                gold.pop("cancer_data", None)
                            st.rerun()
                    with bB:
                        if st.button("⇐ Use B", key=key + "__useb"):
                            _apply_override(key, vb)
                            gold[field_name] = vb
                            if field_name == "cancer_category":
                                gold.pop("cancer_data", None)
                            st.rerun()
        else:
            _render_field_row_eval(field, va, vb)
        st.markdown("<hr style='margin:6px 0; border:none; border-top:1px solid #eee;'>",
                    unsafe_allow_html=True)


# ── Cancer-specific tabs ───────────────────────────────────────────────────────

@st.cache_data
def _get_sections(cancer_type: str) -> list[SectionSpec]:
    return parse_cancer_schema(cancer_type)


def _render_cancer_sections(mode: str) -> None:
    a = st.session_state.annotation_a
    b = st.session_state.annotation_b
    cat_a = a.get("cancer_category")
    cat_b = b.get("cancer_category")

    # If top-level disagrees on whether this is cancer at all, skip cancer-specific.
    if a.get("cancer_excision_report") is False and b.get("cancer_excision_report") is False:
        st.info("兩位 annotator 皆標記為非 cancer excision report — 無 cancer-specific 欄位。")
        return

    # Choose category to drive section layout: prefer agreement, otherwise A.
    cat = cat_a if cat_a else cat_b
    if not cat or cat == "others" or not CANCER_TO_FILE.get(cat):
        if cat_a != cat_b:
            st.warning(
                f"A 與 B 的 cancer_category 不同（A={cat_a}, B={cat_b}）。"
                "先在 Classification 區達成共識後，cancer-specific 欄位才會顯示。"
            )
        return

    if cat_a and cat_b and cat_a != cat_b:
        st.warning(
            f"Categories differ: A={cat_a}, B={cat_b}. Tabs below follow A's schema."
        )

    sections = _get_sections(cat)
    if not sections:
        return
    tab_names = [s.display_name for s in sections]
    tabs = st.tabs(tab_names)
    for i, (tab, section) in enumerate(zip(tabs, sections, strict=True)):
        with tab:
            _render_section(section, tab_index=i, mode=mode)


# ── Save Gold ──────────────────────────────────────────────────────────────────

def _save_gold() -> None:
    samples: list[CompareSampleRef] = st.session_state.samples
    idx = st.session_state.sample_idx
    if not samples or idx >= len(samples):
        return
    sample = samples[idx]
    # Clean array Nones (include=False slots) before save.
    gold = copy.deepcopy(st.session_state.gold_annotation)
    cd = gold.get("cancer_data")
    if isinstance(cd, dict):
        for k, v in list(cd.items()):
            if isinstance(v, list):
                cleaned = [x for x in v if x is not None]
                cd[k] = cleaned if cleaned else None
    payload = build_save_payload(gold, f"{sample.sample_id}.txt")
    payload["_meta"]["consolidated_from"] = [
        st.session_state.suffix_a, st.session_state.suffix_b,
    ]
    save_annotation(payload, sample.path_gold)
    st.session_state.save_message = f"Saved → {sample.path_gold}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _mode_selector() -> None:
    options = ["consensus", "evaluation"]
    labels = {"consensus": "Consensus (A + B → Gold)",
              "evaluation": "Evaluation (Ref vs Machine)"}
    selected = st.sidebar.radio(
        "Mode",
        options=options,
        format_func=lambda x: labels[x],
        index=options.index(st.session_state.mode),
        key="mode_radio",
    )
    if selected != st.session_state.mode:
        st.session_state.mode = selected
        _reload_samples()
        st.rerun()


def _selectable_annotators() -> list[dict]:
    return [a for a in st.session_state.annotators
            if a["suffix"] not in RESERVED_SUFFIXES]


def _consensus_annotator_pickers() -> None:
    annotators = _selectable_annotators()
    if len(annotators) < 2:
        st.sidebar.error("需要至少兩位 annotator。請先在主 app 新增 annotator。")
        return
    suffixes = [a["suffix"] for a in annotators]
    labels = [f"{a['name']} ({a['suffix']})" for a in annotators]

    cur_a = st.session_state.suffix_a
    cur_b = st.session_state.suffix_b

    idx_a = suffixes.index(cur_a) if cur_a in suffixes else 0
    chosen_a = st.sidebar.selectbox(
        "Annotator A", options=list(range(len(annotators))),
        index=idx_a, format_func=lambda i: labels[i], key="sel_a",
    )
    idx_b = (suffixes.index(cur_b) if cur_b in suffixes
             else (1 if len(suffixes) > 1 else 0))
    chosen_b = st.sidebar.selectbox(
        "Annotator B", options=list(range(len(annotators))),
        index=idx_b, format_func=lambda i: labels[i], key="sel_b",
    )
    new_a = annotators[chosen_a]["suffix"]
    new_b = annotators[chosen_b]["suffix"]
    if new_a != cur_a or new_b != cur_b:
        st.session_state.suffix_a = new_a
        st.session_state.suffix_b = new_b
        if st.session_state.folders:
            _reload_samples()
        st.rerun()


def _eval_ref_picker() -> None:
    # Allow nhc/kpc/any non-reserved annotator PLUS gold as a choice.
    annotators = _selectable_annotators()
    options = [a["suffix"] for a in annotators] + ["gold"]
    labels = {a["suffix"]: f"{a['name']} ({a['suffix']})" for a in annotators}
    labels["gold"] = "Gold (consensus)"
    cur = st.session_state.eval_ref_suffix or (options[0] if options else None)
    idx = options.index(cur) if cur in options else 0
    chosen = st.sidebar.selectbox(
        "Reference", options=options, index=idx,
        format_func=lambda s: labels.get(s, s), key="sel_eval_ref",
    )
    if chosen != st.session_state.eval_ref_suffix:
        st.session_state.eval_ref_suffix = chosen
        if st.session_state.folders:
            _reload_samples()
        st.rerun()
    st.sidebar.caption("Machine source: result_dir `*_output.json`")


def render_sidebar() -> None:
    st.sidebar.title("🔍 Compare / Consensus")
    _mode_selector()
    st.sidebar.divider()

    if st.session_state.mode == "consensus":
        _consensus_annotator_pickers()
    else:
        _eval_ref_picker()

    st.sidebar.divider()

    if st.sidebar.button("📂 選擇資料夾", use_container_width=True):
        picked = pick_folder(initial=st.session_state.base_dir or "")
        if picked:
            ok, msg = _reload_from_base(picked)
            (st.sidebar.success if ok else st.sidebar.error)(msg)
            if ok:
                st.rerun()

    if st.session_state.base_dir:
        st.sidebar.caption(f"📁 {st.session_state.base_dir}")

    with st.sidebar.expander("手動輸入路徑", expanded=not bool(st.session_state.samples)):
        path_val = st.text_input(
            "Base folder path", value=st.session_state.base_dir, key="manual_base_cmp"
        )
        if st.button("Load", key="manual_load_cmp"):
            if not os.path.isdir(path_val):
                st.error("Directory not found.")
            else:
                ok, msg = _reload_from_base(path_val)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()

    samples = st.session_state.samples
    if not samples:
        st.sidebar.info("選擇 base 資料夾後載入樣本。")
        return

    # Completion stats (consensus: how many golds saved)
    if st.session_state.mode == "consensus":
        gold_done = sum(1 for s in samples if os.path.exists(s.path_gold))
        st.sidebar.markdown(f"**Gold saved: {gold_done} / {len(samples)}**")
        st.sidebar.progress(gold_done / len(samples))

    # Stem filter
    stems = sorted({s.stem for s in samples}, key=lambda x: (len(x), x))
    stem_options = ["all"] + stems
    cur_filter = st.session_state.stem_filter
    if cur_filter not in stem_options:
        cur_filter = "all"
    new_filter = st.sidebar.selectbox(
        "Filter by stem", stem_options,
        index=stem_options.index(cur_filter), key="stem_filter_cmp",
    )
    if new_filter != st.session_state.stem_filter:
        st.session_state.stem_filter = new_filter
        st.rerun()

    st.sidebar.divider()
    visible = _visible_samples()
    current_sample_id = st.session_state.last_sample_id

    with st.sidebar.container(height=520):
        for orig_idx, sample in visible:
            mark = _sample_badge(sample)
            is_current = sample.sample_id == current_sample_id
            label = f"{'▶ ' if is_current else ''}{mark} {sample.sample_id}"
            if st.button(
                label, key=f"cmp_sample_btn_{sample.sample_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.sample_idx = orig_idx
                _on_file_change()
                st.rerun()


def _sample_badge(sample: CompareSampleRef) -> str:
    if st.session_state.mode == "consensus":
        has_a = os.path.exists(sample.path_a)
        has_b = os.path.exists(sample.path_b)
        has_g = os.path.exists(sample.path_gold)
        if has_a and has_b and has_g:
            return "✓G"
        if has_a and has_b:
            return "✓✓"
        if has_a or has_b:
            return "½"
        return "·"
    # evaluation
    return "✓" if os.path.exists(sample.path_a) else "·"


# ── Report panel ───────────────────────────────────────────────────────────────

def render_report_panel() -> None:
    samples: list[CompareSampleRef] = st.session_state.samples
    idx = st.session_state.sample_idx
    if not samples or idx >= len(samples):
        st.info("尚未載入樣本。")
        return
    sample = samples[idx]
    total = len(samples)

    nav_l, nav_m, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("← Prev", disabled=(idx == 0), key="cmp_prev"):
            st.session_state.sample_idx = max(0, idx - 1)
            _on_file_change()
            st.rerun()
    with nav_m:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>"
            f"<b>{sample.sample_id}</b><br>"
            f"<small>{idx + 1} / {total} · stem={sample.stem}</small></div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("Next →", disabled=(idx >= total - 1), key="cmp_next"):
            st.session_state.sample_idx = min(total - 1, idx + 1)
            _on_file_change()
            st.rerun()

    st.text_area(
        "Report Text", value=st.session_state.report_text,
        height=640, disabled=True,
        key=f"cmp_report_area_{sample.sample_id}",
        label_visibility="collapsed",
    )


# ── Main panel renderers ───────────────────────────────────────────────────────

def render_consensus_main() -> None:
    samples = st.session_state.samples
    if not samples:
        st.info("👈 Select annotators + base folder to begin.")
        return
    sample = samples[st.session_state.sample_idx]

    # Overall match summary
    a = st.session_state.annotation_a
    b = st.session_state.annotation_b
    overall_total, overall_agree = _overall_counts(a, b)
    pct = (overall_agree / overall_total * 100) if overall_total else 100
    st.markdown(
        f"### Consensus · {sample.sample_id}  "
        f"<span class='sec-summary'>({overall_agree}/{overall_total} fields match, {pct:.0f}%)</span>",
        unsafe_allow_html=True,
    )

    # Save button
    col_save, col_status = st.columns([2, 5])
    with col_save:
        if st.button("🏁 Save Gold", type="primary", use_container_width=True,
                     key=f"cmp_save_{sample.sample_id}"):
            _save_gold()
            st.rerun()
    with col_status:
        if os.path.exists(sample.path_gold):
            st.success(f"Gold exists: {os.path.basename(sample.path_gold)}")
        if st.session_state.save_message:
            st.info(st.session_state.save_message)

    st.divider()
    _render_classification(mode="consensus")
    _render_cancer_sections(mode="consensus")


def render_evaluation_main() -> None:
    samples = st.session_state.samples
    if not samples:
        st.info("👈 Select reference + base folder to begin.")
        return
    sample = samples[st.session_state.sample_idx]
    a = st.session_state.annotation_a  # reference
    b = st.session_state.annotation_b  # machine

    overall_total, overall_agree = _overall_counts(a, b)
    pct = (overall_agree / overall_total * 100) if overall_total else 100
    ref_label = st.session_state.eval_ref_suffix
    st.markdown(
        f"### Evaluation · {sample.sample_id}  "
        f"<span class='sec-summary'>"
        f"Reference=<b>{ref_label}</b> vs Machine · "
        f"{overall_agree}/{overall_total} match ({pct:.0f}%)</span>",
        unsafe_allow_html=True,
    )
    if not os.path.exists(sample.path_a):
        st.warning(f"Reference annotation missing for this sample: {sample.path_a}")
        return

    st.divider()
    _render_classification(mode="evaluation")
    _render_cancer_sections(mode="evaluation")


def _overall_counts(a: dict, b: dict) -> tuple[int, int]:
    """Cheap aggregate across classification + cancer sections."""
    total = 0
    agree = 0
    top_fields = ["cancer_excision_report", "cancer_category",
                  "cancer_category_others_description"]
    for f in top_fields:
        total += 1
        if not values_differ(a.get(f), b.get(f)):
            agree += 1

    cat = a.get("cancer_category") or b.get("cancer_category")
    if cat and cat != "others" and CANCER_TO_FILE.get(cat):
        a_cd = a.get("cancer_data") or {}
        b_cd = b.get("cancer_data") or {}
        for section in _get_sections(cat):
            for f in section.flat_fields:
                total += 1
                if not values_differ(a_cd.get(f.name), b_cd.get(f.name)):
                    agree += 1
    return total, agree


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    if not st.session_state.samples:
        st.title("🔍 Compare / Consensus")
        st.info("Select mode, annotators, and a base folder in the sidebar.")
        return

    col_left, col_right = st.columns([3, 2])
    with col_right:
        render_report_panel()
    with col_left:
        if st.session_state.mode == "consensus":
            render_consensus_main()
        else:
            render_evaluation_main()


def main_cli():
    """Console-script entry: spawn `streamlit run` pointing at this file.

    Mirrors `app.main_cli` so the compare tool can also be invoked as a
    console script (e.g. wired into pyproject as `registrar-compare`).
    """
    import subprocess
    import sys
    script = str(Path(__file__).resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", script, *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


# Only execute under `streamlit run` (which imports the module with the Streamlit runtime active).
if os.environ.get("STREAMLIT_SERVER_PORT") or st.runtime.exists():
    main()
