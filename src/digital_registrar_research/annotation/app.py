"""Digital Registrar Annotation Tool — Streamlit app."""

from __future__ import annotations

import copy
import html as _html
import os
from pathlib import Path

import streamlit as st

from digital_registrar_research.annotation.annotator_config import (
    add_annotator,
    load_annotators,
)
from digital_registrar_research.annotation.io import (
    NA_SENTINEL,
    FolderSet,
    SampleRef,
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
EMPTY_LABEL = "— empty —"

st.set_page_config(page_title="Digital Registrar", layout="wide")


# ── CSS ────────────────────────────────────────────────────────────────────────

_STICKY_CSS = """
<style>
/* Force the row holding the report column to align children at the top,
   otherwise the sticky child gets stretched to row height and sticky breaks. */
[data-testid="stHorizontalBlock"]:has(.report-col-marker) {
    align-items: flex-start !important;
    overflow: visible !important;
}

/* Make the report column sticky. Use both :has() (modern, scoped) and
   :last-child as a fallback selector. */
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
st.markdown(_STICKY_CSS, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "base_dir": "",
        "folders": None,            # FolderSet | None
        "samples": [],              # list[SampleRef]
        "sample_idx": 0,
        "report_text": "",
        "annotation": {},           # flat: same layout as on-disk JSON
        "pre_annotation": {},       # pre-annotation snapshot (flat) for diff
        "annotation_status": "",    # "completed" | "new"
        "stem_filter": "all",
        "last_sample_id": "",
        "save_message": "",
        "annotators": load_annotators(),
        "current_annotator": None,  # dict {"name": ..., "suffix": ...}
        "add_annotator_message": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Folder / sample loading ────────────────────────────────────────────────────

def _reload_from_base(base_dir: str) -> tuple[bool, str]:
    annotator = st.session_state.current_annotator
    if not annotator:
        return False, "請先在上方選擇 annotator。"
    folders = discover_folders(base_dir)
    if not folders:
        return False, "找不到符合 `{prefix}_{dataset|result|annotation}_{date}` 格式的子資料夾。"
    samples = list_samples(folders, annotator["suffix"])
    if not samples:
        return False, f"在 {folders.result_dir} 下找不到任何 *_output.json。"
    st.session_state.base_dir = base_dir
    st.session_state.folders = folders
    st.session_state.samples = samples
    st.session_state.sample_idx = 0
    st.session_state.stem_filter = "all"
    _on_file_change()
    return True, f"載入 {len(samples)} 個樣本（{folders.prefix}_{folders.date}）。"


def _on_annotator_change():
    """Rebuild sample list with new suffix, keeping base_dir/folders."""
    folders: FolderSet | None = st.session_state.folders
    annotator = st.session_state.current_annotator
    if not folders or not annotator:
        st.session_state.samples = []
        st.session_state.sample_idx = 0
        _on_file_change()
        return
    st.session_state.samples = list_samples(folders, annotator["suffix"])
    st.session_state.sample_idx = 0
    st.session_state.stem_filter = "all"
    _on_file_change()


def _visible_samples() -> list[tuple[int, SampleRef]]:
    """Return (original_index, sample) pairs filtered by current stem_filter."""
    samples = st.session_state.samples
    stem = st.session_state.stem_filter
    if stem == "all":
        return list(enumerate(samples))
    return [(i, s) for i, s in enumerate(samples) if s.stem == stem]


def _on_file_change():
    samples: list[SampleRef] = st.session_state.samples
    idx = st.session_state.sample_idx
    if not samples or idx >= len(samples):
        st.session_state.report_text = ""
        st.session_state.annotation = {}
        st.session_state.pre_annotation = {}
        st.session_state.annotation_status = ""
        st.session_state.last_sample_id = ""
        return
    sample = samples[idx]
    st.session_state.report_text = load_report_text(sample.dataset_path)
    pre = strip_meta(load_json(sample.result_path))
    st.session_state.pre_annotation = pre
    if os.path.exists(sample.annotation_path):
        raw = load_json(sample.annotation_path)
        st.session_state.annotation = strip_meta(raw)
        rehydrate_sentinels(st.session_state.annotation, raw.get("_meta"))
        st.session_state.annotation_status = "completed"
    else:
        st.session_state.annotation = copy.deepcopy(pre)
        st.session_state.annotation_status = "new"
    st.session_state.last_sample_id = sample.sample_id
    st.session_state.save_message = ""


# ── Flat session-state adapters ────────────────────────────────────────────────

def _get(section: str, field: str):
    ann = st.session_state.annotation
    if section == "IsCancer":
        return ann.get(field)
    return (ann.get("cancer_data") or {}).get(field)


def _get_pre(section: str, field: str):
    pre = st.session_state.pre_annotation
    if section == "IsCancer":
        return pre.get(field)
    return (pre.get("cancer_data") or {}).get(field)


def _set(section: str, field: str, val):
    ann = st.session_state.annotation
    if section == "IsCancer":
        ann[field] = val
    else:
        ann.setdefault("cancer_data", {})[field] = val


def _get_items(section: str, array_field: str) -> list:
    ann = st.session_state.annotation
    container = ann if section == "IsCancer" else (ann.get("cancer_data") or {})
    return container.get(array_field) or []


def _get_items_pre(section: str, array_field: str) -> list:
    pre = st.session_state.pre_annotation
    container = pre if section == "IsCancer" else (pre.get("cancer_data") or {})
    return container.get(array_field) or []


def _set_items(section: str, array_field: str, items: list):
    ann = st.session_state.annotation
    container = ann if section == "IsCancer" else ann.setdefault("cancer_data", {})
    container[array_field] = items if items else None


# ── Formatting / diff helpers ──────────────────────────────────────────────────

def _fmt_option(val, field_type: str = "enum") -> str:
    if val == NA_SENTINEL:
        return NA_LABEL
    if val is None:
        return NOT_SET_LABEL
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val).replace("_", " ")


_NULLISH = (None, "", NA_SENTINEL)


def _values_differ(a, b) -> bool:
    # Null-equivalent states (pre: None/""; annotation: None/""/NA_SENTINEL) all
    # serialise to the same null, so they shouldn't light up the diff marker.
    if a in _NULLISH and b in _NULLISH:
        return False
    return a != b


def _diff_marker(current, pre) -> str:
    return "✎ " if _values_differ(current, pre) else ""


def _pre_caption(pre_val) -> str | None:
    if pre_val is None or pre_val == "":
        return f"Pre-annotated: {NOT_SET_LABEL}"
    return f"Pre-annotated: {_fmt_option(pre_val)}"


# ── Widget rendering ───────────────────────────────────────────────────────────

def render_field(field: FieldSpec, current_val, pre_val, key: str) -> object:
    """Render one field; returns the (possibly new) value."""
    changed = _values_differ(current_val, pre_val)
    label = f"{_diff_marker(current_val, pre_val)}{field.title}"
    help_text = field.description or None

    if field.field_type in ("enum", "int_enum"):
        options = [None, NA_SENTINEL, *field.enum_values]
        idx = options.index(current_val) if current_val in options else 0
        new_val = st.selectbox(
            label, options=options, index=idx,
            format_func=_fmt_option, key=key, help=help_text,
        )
    elif field.field_type == "bool":
        # Per spec: bool fields expose exactly three options, no 尚未設定.
        # A None coming from pre-annotation or a fresh session is treated as
        # intentional N/A (first option) — the selectbox never emits None.
        options = [NA_SENTINEL, True, False]
        if current_val is None or current_val == NA_SENTINEL:
            idx = 0
        elif current_val in (True, False):
            idx = options.index(current_val)
        else:
            idx = 0
        new_val = st.selectbox(
            label, options=options, index=idx,
            format_func=_fmt_option, key=key, help=help_text,
        )
    elif field.field_type == "int":
        col_num, col_null = st.columns([3, 1])
        is_null = current_val is None or current_val == NA_SENTINEL
        with col_null:
            null_checked = st.checkbox("N/A", value=is_null, key=key + "__null")
        with col_num:
            num_init = (
                int(current_val)
                if isinstance(current_val, int) and not isinstance(current_val, bool)
                else 0
            )
            num_val = st.number_input(
                label, value=num_init,
                min_value=0, step=1, disabled=null_checked,
                key=key + "__num", help=help_text,
            )
        # Store the sentinel so the N/A intent is carried into save_payload.
        new_val = NA_SENTINEL if null_checked else int(num_val)
    elif field.field_type == "string":
        col_text, col_na = st.columns([3, 1])
        is_na = current_val == NA_SENTINEL
        with col_na:
            na_checked = st.checkbox("N/A", value=is_na, key=key + "__na")
        with col_text:
            text_init = "" if is_na else (current_val or "")
            raw = st.text_input(
                label, value=text_init,
                disabled=na_checked,
                key=key + "__text", help=help_text,
            )
        if na_checked:
            new_val = NA_SENTINEL
        else:
            new_val = raw if raw else None
    else:
        new_val = current_val

    if changed:
        st.caption(_pre_caption(pre_val))

    return new_val


def render_flat_fields(section: SectionSpec, section_key_prefix: str):
    for fspec in section.flat_fields:
        current = _get(section.name, fspec.name)
        pre = _get_pre(section.name, fspec.name)
        new_val = render_field(
            fspec, current, pre,
            key=f"{section_key_prefix}__{fspec.name}",
        )
        if new_val != current:
            _set(section.name, fspec.name, new_val)


def render_array_section(section: SectionSpec, section_key_prefix: str):
    afield = section.array_field_name
    atype = section.array_field_type

    # Array-level N/A toggle: the entire section can be marked as intentionally
    # null, distinct from an empty list that simply means "not yet set".
    raw_current = _get(section.name, afield)
    is_na_now = raw_current == NA_SENTINEL
    na_key = f"{section_key_prefix}__{afield}__na"
    backup_key = f"{section_key_prefix}__{afield}__na_backup"

    col_hdr, col_na = st.columns([5, 2])
    with col_hdr:
        st.markdown(f"**{afield.replace('_', ' ').title()}**")
    with col_na:
        na_checked = st.checkbox("N/A (intentional)", value=is_na_now, key=na_key)

    if na_checked and not is_na_now:
        # Entering N/A: backup current list so we can restore on uncheck.
        if isinstance(raw_current, list):
            st.session_state[backup_key] = copy.deepcopy(raw_current)
        _set(section.name, afield, NA_SENTINEL)
        st.rerun()
    if not na_checked and is_na_now:
        # Leaving N/A: restore previous list if we have one.
        restored = st.session_state.pop(backup_key, None)
        _set(section.name, afield, restored)
        st.rerun()

    if na_checked:
        st.caption(NA_LABEL)
        return

    if atype == "array_of_strings_enum":
        current_list = raw_current if isinstance(raw_current, list) else []
        pre_list = _get_items_pre(section.name, afield) or []
        changed = set(current_list) != set(pre_list)
        label = f"{'✎ ' if changed else ''}{afield.replace('_', ' ').title()}"
        new_list = st.multiselect(
            label,
            options=section.array_item_enum_values,
            default=[v for v in current_list if v in section.array_item_enum_values],
            key=f"{section_key_prefix}__{afield}__multiselect",
        )
        if changed:
            st.caption(f"Pre-annotated: {', '.join(_fmt_option(v) for v in pre_list) or EMPTY_LABEL}")
        _set_items(section.name, afield, new_list if new_list else None)
        return

    # array_of_objects
    items = list(raw_current) if isinstance(raw_current, list) else []
    pre_items = _get_items_pre(section.name, afield)

    to_remove = None
    for idx, item in enumerate(items):
        col_card, col_rm = st.columns([11, 1])
        with col_rm:
            if st.button("✕", key=f"{section_key_prefix}_{afield}_rm_{idx}"):
                to_remove = idx

        pre_item = pre_items[idx] if idx < len(pre_items) else {}
        is_new = idx >= len(pre_items)
        item_changed = is_new or any(
            _values_differ(item.get(f.name), pre_item.get(f.name))
            for f in section.array_item_fields
        )

        with col_card:
            base_label = f"Entry {idx + 1}"
            for hint in ("margin_category", "lymph_node_category",
                         "biomarker_category", "pattern_name"):
                if item.get(hint):
                    base_label = f"{idx + 1}. {str(item[hint]).replace('_', ' ')}"
                    break
            if is_new:
                base_label = f"✎ (new) {base_label}"
            elif item_changed:
                base_label = f"✎ {base_label}"
            with st.expander(base_label, expanded=True):
                for fspec in section.array_item_fields:
                    current = item.get(fspec.name)
                    pre = pre_item.get(fspec.name) if not is_new else None
                    key = f"{section_key_prefix}_{afield}_{idx}__{fspec.name}"
                    new_val = render_field(fspec, current, pre, key=key)
                    items[idx][fspec.name] = new_val

    if to_remove is not None:
        items.pop(to_remove)
        _set_items(section.name, afield, items)
        st.rerun()

    btn_label = f"+ Add {afield.replace('_', ' ').title().rstrip('s')}"
    if st.button(btn_label, key=f"{section_key_prefix}_{afield}_add"):
        empty = {f.name: None for f in section.array_item_fields}
        items.append(empty)
        _set_items(section.name, afield, items)
        st.rerun()

    _set_items(section.name, afield, items if items else None)


def render_section(section: SectionSpec, tab_index: int, sample_id: str):
    key_prefix = f"sec_{section.name}_{tab_index}_{sample_id}"
    render_flat_fields(section, key_prefix)
    if section.array_field_name:
        st.divider()
        render_array_section(section, key_prefix)


# ── Sidebar: folder picker + sample list ───────────────────────────────────────

def _render_annotator_picker():
    annotators: list[dict] = st.session_state.annotators
    current = st.session_state.current_annotator

    labels = [f"{a['name']} ({a['suffix']})" for a in annotators]
    suffixes = [a["suffix"] for a in annotators]
    current_idx = (
        suffixes.index(current["suffix"])
        if current and current["suffix"] in suffixes
        else None
    )

    selected = st.sidebar.selectbox(
        "Annotator",
        options=list(range(len(annotators))),
        index=current_idx,
        format_func=lambda i: labels[i],
        placeholder="— 選擇 annotator —",
        key="annotator_select",
    )

    new_annotator = annotators[selected] if selected is not None else None
    current_suffix = current["suffix"] if current else None
    new_suffix = new_annotator["suffix"] if new_annotator else None
    if new_suffix != current_suffix:
        st.session_state.current_annotator = new_annotator
        _on_annotator_change()
        st.rerun()

    with st.sidebar.expander("➕ 新增 annotator", expanded=False):
        name = st.text_input("全名", key="new_annotator_name")
        suffix = st.text_input("縮寫 (1–6 小寫英數)", key="new_annotator_suffix")
        if st.button("新增", key="btn_add_annotator", use_container_width=True):
            ok, msg = add_annotator(name, suffix)
            if ok:
                st.session_state.annotators = load_annotators()
                added = next(
                    (a for a in st.session_state.annotators
                     if a["suffix"] == suffix.strip().lower()),
                    None,
                )
                if added:
                    st.session_state.current_annotator = added
                    _on_annotator_change()
                st.session_state.add_annotator_message = ("success", msg)
                st.rerun()
            else:
                st.session_state.add_annotator_message = ("error", msg)
        msg_pair = st.session_state.add_annotator_message
        if msg_pair:
            level, text = msg_pair
            (st.success if level == "success" else st.error)(text)


def render_sidebar():
    st.sidebar.title("📋 標註清單")

    _render_annotator_picker()

    folders: FolderSet | None = st.session_state.folders
    samples: list[SampleRef] = st.session_state.samples

    if not st.session_state.current_annotator:
        st.sidebar.info("請先選擇 annotator 再載入資料夾。")
        return

    st.sidebar.divider()

    if st.sidebar.button("📂 選擇資料夾", use_container_width=True):
        picked = pick_folder(initial=st.session_state.base_dir or "")
        if picked:
            ok, msg = _reload_from_base(picked)
            if ok:
                st.sidebar.success(msg)
                st.rerun()
            else:
                st.sidebar.error(msg)

    if st.session_state.base_dir:
        st.sidebar.caption(f"📁 {st.session_state.base_dir}")

    with st.sidebar.expander("手動輸入路徑", expanded=not bool(samples)):
        path_val = st.text_input(
            "Base folder path", value=st.session_state.base_dir, key="manual_base_input"
        )
        if st.button("Load", key="manual_base_load"):
            if not os.path.isdir(path_val):
                st.error("Directory not found.")
            else:
                ok, msg = _reload_from_base(path_val)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    if not samples or not folders:
        st.sidebar.info("選擇 base 資料夾以載入樣本。")
        return

    # Completion stats
    completed_total = sum(1 for s in samples if os.path.exists(s.annotation_path))
    st.sidebar.markdown(
        f"**Completed: {completed_total} / {len(samples)}**"
    )
    st.sidebar.progress(completed_total / len(samples))

    # Stem filter
    stems = sorted({s.stem for s in samples}, key=lambda x: (len(x), x))
    stem_options = ["all"] + stems
    current_filter = st.session_state.stem_filter
    if current_filter not in stem_options:
        current_filter = "all"
    new_filter = st.sidebar.selectbox(
        "Filter by stem", stem_options,
        index=stem_options.index(current_filter), key="stem_filter_select",
    )
    if new_filter != st.session_state.stem_filter:
        st.session_state.stem_filter = new_filter
        st.rerun()

    # Sample list
    st.sidebar.divider()
    visible = _visible_samples()
    current_sample_id = st.session_state.last_sample_id

    # Scroll container
    with st.sidebar.container(height=520):
        for orig_idx, sample in visible:
            done = os.path.exists(sample.annotation_path)
            mark = "✓" if done else "○"
            is_current = sample.sample_id == current_sample_id
            label = f"{'▶ ' if is_current else ''}{mark} {sample.sample_id}"
            if st.button(
                label,
                key=f"sample_btn_{sample.sample_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.sample_idx = orig_idx
                _on_file_change()
                st.rerun()


# ── Left column: annotation form ───────────────────────────────────────────────

def render_annotation_panel():
    ann = st.session_state.annotation
    pre = st.session_state.pre_annotation

    # Classification (IsCancer)
    st.subheader("Classification")

    excision = ann.get("cancer_excision_report")
    pre_excision = pre.get("cancer_excision_report")
    excision_opts = [NA_SENTINEL, True, False]
    if excision is None or excision == NA_SENTINEL:
        eidx = 0
    elif excision in (True, False):
        eidx = excision_opts.index(excision)
    else:
        eidx = 0
    label = f"{_diff_marker(excision, pre_excision)}Cancer Excision Report"
    new_excision = st.selectbox(
        label, options=excision_opts, index=eidx,
        format_func=_fmt_option, key=f"is_cancer_excision_{st.session_state.last_sample_id}",
        help="Is this a primary cancer excision report eligible for registry?",
    )
    if _values_differ(excision, pre_excision):
        st.caption(_pre_caption(pre_excision))
    if new_excision != excision:
        ann["cancer_excision_report"] = new_excision

    category = ann.get("cancer_category")
    pre_category = pre.get("cancer_category")
    cat_opts = CANCER_CATEGORIES
    cidx = cat_opts.index(category) if category in cat_opts else 0
    cat_label = f"{_diff_marker(category, pre_category)}Cancer Category"
    new_cat = st.selectbox(
        cat_label, options=cat_opts, index=cidx,
        format_func=lambda x: "— select —" if x is None else x.replace("_", " ").title(),
        key=f"cancer_category_select_{st.session_state.last_sample_id}",
    )
    if _values_differ(category, pre_category):
        st.caption(_pre_caption(pre_category))
    if new_cat != category:
        # Clear cancer_data when switching category
        if "cancer_data" in ann:
            del ann["cancer_data"]
        ann["cancer_category"] = new_cat
        st.rerun()

    if new_cat == "others":
        others_desc = ann.get("cancer_category_others_description") or ""
        pre_desc = pre.get("cancer_category_others_description") or ""
        label = f"{_diff_marker(others_desc, pre_desc)}Specify organ"
        new_desc = st.text_input(
            label, value=others_desc,
            key=f"others_desc_{st.session_state.last_sample_id}",
        )
        if _values_differ(others_desc, pre_desc):
            st.caption(_pre_caption(pre_desc or None))
        ann["cancer_category_others_description"] = new_desc or None

    # Cancer-specific sections
    if new_excision is False:
        st.info("Non-cancer report — only classification will be saved.")
        _render_save_button()
        return

    if new_cat and new_cat != "others" and CANCER_TO_FILE.get(new_cat):
        st.divider()
        st.subheader(f"{new_cat.title()} Cancer")
        sections = _get_sections(new_cat)
        if sections:
            tab_names = [s.display_name for s in sections]
            tabs = st.tabs(tab_names)
            sample_id = st.session_state.last_sample_id
            for i, (tab, section) in enumerate(zip(tabs, sections, strict=True)):
                with tab:
                    render_section(section, tab_index=i, sample_id=sample_id)

    st.divider()
    _render_save_button()


@st.cache_data
def _get_sections(cancer_type: str) -> list[SectionSpec]:
    return parse_cancer_schema(cancer_type)


def _render_save_button():
    samples = st.session_state.samples
    idx = st.session_state.sample_idx
    if not samples or idx >= len(samples):
        st.button("💾 Save Annotation", type="primary", use_container_width=True, disabled=True)
        return

    sample: SampleRef = samples[idx]
    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        payload = build_save_payload(st.session_state.annotation, f"{sample.sample_id}.txt")
        save_annotation(payload, sample.annotation_path)
        st.session_state.annotation_status = "completed"
        st.session_state.save_message = f"Saved → {sample.annotation_path}"
        st.rerun()

    if st.session_state.save_message:
        st.success(st.session_state.save_message)


# ── Right column: report viewer ────────────────────────────────────────────────

def render_report_panel():
    samples = st.session_state.samples
    idx = st.session_state.sample_idx

    if not samples:
        st.info("尚未載入樣本。左側選擇 base 資料夾開始。")
        return

    if idx >= len(samples):
        st.warning("No sample selected.")
        return

    sample: SampleRef = samples[idx]
    total = len(samples)

    nav_l, nav_mid, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("← Prev", disabled=(idx == 0), key="btn_prev"):
            st.session_state.sample_idx = max(0, idx - 1)
            _on_file_change()
            st.rerun()
    with nav_mid:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px'>"
            f"<b>{sample.sample_id}</b><br>"
            f"<small>{idx + 1} / {total} · stem={sample.stem}</small></div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button("Next →", disabled=(idx >= total - 1), key="btn_next"):
            st.session_state.sample_idx = min(total - 1, idx + 1)
            _on_file_change()
            st.rerun()

    status = st.session_state.annotation_status
    if status == "completed":
        st.success("✓ Existing annotation loaded")
    elif status == "new":
        st.info("New annotation — pre-filled from GPT-OSS result")

    # Original report uses "|" as line separator. Convert to <br> (rather than
    # "\n") because st.markdown collapses raw newlines inside HTML blocks to
    # spaces before rendering, even inside <pre>.
    report_html = _html.escape(st.session_state.report_text).replace("|", "<br>")
    st.markdown(
        f'<pre class="report-text-pre">{report_html}</pre>',
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    if not st.session_state.samples:
        st.title("Cancer Pathology Report Annotator")
        if not st.session_state.current_annotator:
            st.info("👈 請先在左側選擇 annotator。")
        else:
            st.info("👈 從左側選擇 base 資料夾開始標註。")
        return

    st.title("Cancer Pathology Report Annotator")

    col_left, col_right = st.columns([1, 1])
    with col_right:
        st.markdown('<span class="report-col-marker"></span>', unsafe_allow_html=True)
        render_report_panel()
    with col_left:
        render_annotation_panel()


def main_cli():
    """Console-script entry: spawn `streamlit run` pointing at this file.

    Wired into pyproject as `registrar-annotate`. Any CLI args after the
    command are forwarded to streamlit (e.g. `registrar-annotate --server.port 8080`).
    """
    import subprocess
    import sys
    script = str(Path(__file__).resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", script, *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


# Only execute under `streamlit run` (which imports the module with the Streamlit runtime active).
# Suppress when imported by Python directly (e.g. by `main_cli` before subprocess dispatch).
if os.environ.get("STREAMLIT_SERVER_PORT") or st.runtime.exists():
    main()
