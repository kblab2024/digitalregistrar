"""Digital Registrar Annotation Tool — canonical-layout Streamlit app.

Reads datasets that follow the layout in :mod:`.io_canonical` (``dummy/``,
``workspace/`` and anything produced by ``scripts/gen_dummy_skeleton.py``).
``with_preann`` and ``without_preann`` are fully independent subtrees
under the base dir; the sidebar mode radio picks which one to browse.

Chrome is in 繁體中文; schema/field names stay English because they come
from the Pydantic schema.

Default ``base_dir`` is read from the ``REGISTRAR_ANNOTATE_BASE_DIR`` env
var, which the dedicated launcher scripts (``registrar-annotate-workspace``
and ``registrar-annotate-dummy``) set before spawning streamlit.
"""

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
from digital_registrar_research.annotation.io_canonical import (
    MODES,
    NA_SENTINEL,
    SampleRef,
    WorkspaceSet,
    build_save_payload,
    list_datasets,
    list_samples,
    load_json,
    load_report_text,
    load_workspace,
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

st.set_page_config(page_title="Digital Registrar (Canonical)", layout="wide")


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


# ── Label translations ────────────────────────────────────────────────────────

MODE_LABELS = {
    "with_preann": "含預標註 (with_preann)",
    "without_preann": "不含預標註 (without_preann)",
}
NOT_SET = "— 尚未設定 —"
NA_LABEL = "— N/A —"
EMPTY = "— 空 —"
SELECT = "— 請選擇 —"


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "base_dir": os.environ.get("REGISTRAR_ANNOTATE_BASE_DIR", ""),
        "available_datasets": [],
        "dataset": "",
        "mode": "with_preann",
        "workspace": None,           # WorkspaceSet | None
        "samples": [],
        "sample_idx": 0,
        "report_text": "",
        "annotation": {},
        "pre_annotation": {},
        "annotation_status": "",
        "stem_filter": "all",
        "last_sample_id": "",
        "save_message": "",
        "annotators": load_annotators(),
        "current_annotator": None,
        "add_annotator_message": "",
        "auto_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Base-dir / dataset / samples loading ──────────────────────────────────────

def _reload_from_base(base_dir: str) -> tuple[bool, str]:
    """Index the base dir under the current mode, pick a dataset, load samples."""
    annotator = st.session_state.current_annotator
    if not annotator:
        return False, "請先在上方選擇標註者。"
    mode = st.session_state.mode
    datasets = list_datasets(base_dir, mode)
    if not datasets:
        sub = "reports_without_preann" if mode == "without_preann" else "reports"
        return False, f"此資料夾下找不到 data/<dataset>/{sub}/ 結構。"
    st.session_state.base_dir = base_dir
    st.session_state.available_datasets = datasets
    dataset = st.session_state.dataset if st.session_state.dataset in datasets else datasets[0]
    st.session_state.dataset = dataset
    ws = load_workspace(base_dir, dataset, mode)
    if ws is None:
        return False, f"無法載入資料集「{dataset}」({mode})。"
    st.session_state.workspace = ws
    _rebuild_samples()
    return True, f"已載入「{dataset}」({mode}, {len(st.session_state.samples)} 筆樣本)。"


def _rebuild_samples():
    ws: WorkspaceSet | None = st.session_state.workspace
    annotator = st.session_state.current_annotator
    if not ws or not annotator:
        st.session_state.samples = []
        st.session_state.sample_idx = 0
        _on_file_change()
        return
    st.session_state.samples = list_samples(ws, annotator["suffix"])
    st.session_state.sample_idx = 0
    st.session_state.stem_filter = "all"
    _on_file_change()


def _on_annotator_change():
    _rebuild_samples()


def _on_mode_change():
    # Mode changes the entire base subtree — re-discover datasets for the new mode.
    base = st.session_state.base_dir
    if base:
        _reload_from_base(base)
    else:
        _rebuild_samples()


def _on_dataset_change():
    base = st.session_state.base_dir
    dataset = st.session_state.dataset
    ws = load_workspace(base, dataset, st.session_state.mode)
    if ws is None:
        st.session_state.workspace = None
        st.session_state.samples = []
        _on_file_change()
        return
    st.session_state.workspace = ws
    _rebuild_samples()


def _visible_samples() -> list[tuple[int, SampleRef]]:
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
    st.session_state.report_text = load_report_text(sample.report_path)

    is_with_preann = st.session_state.mode == "with_preann"
    if is_with_preann and os.path.exists(sample.preannotation_path):
        pre = strip_meta(load_json(sample.preannotation_path))
    else:
        pre = {}
    st.session_state.pre_annotation = pre

    if os.path.exists(sample.annotation_path):
        raw = load_json(sample.annotation_path)
        st.session_state.annotation = strip_meta(raw)
        rehydrate_sentinels(st.session_state.annotation, raw.get("_meta"))
        st.session_state.annotation_status = "completed"
    elif is_with_preann:
        st.session_state.annotation = copy.deepcopy(pre)
        st.session_state.annotation_status = "new_preann"
    else:
        st.session_state.annotation = {}
        st.session_state.annotation_status = "new_blank"

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

def _show_diff() -> bool:
    return st.session_state.mode == "with_preann"


def _fmt_option(val, field_type: str = "enum") -> str:
    if val == NA_SENTINEL:
        return NA_LABEL
    if val is None:
        return NOT_SET
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val).replace("_", " ")


_NULLISH = (None, "", NA_SENTINEL)


def _values_differ(a, b) -> bool:
    if a in _NULLISH and b in _NULLISH:
        return False
    return a != b


def _diff_marker(current, pre) -> str:
    if not _show_diff():
        return ""
    return "✎ " if _values_differ(current, pre) else ""


def _pre_caption(pre_val) -> str | None:
    if not _show_diff():
        return None
    if pre_val is None or pre_val == "":
        return f"預標註值：{NOT_SET}"
    return f"預標註值：{_fmt_option(pre_val)}"


# ── Widget rendering ───────────────────────────────────────────────────────────

def render_field(field: FieldSpec, current_val, pre_val, key: str) -> object:
    show_diff = _show_diff()
    changed = show_diff and _values_differ(current_val, pre_val)
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
        # Bool fields expose exactly three options; None coming from pre-annotation
        # or a fresh session collapses to the intentional-N/A choice.
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
        cap = _pre_caption(pre_val)
        if cap:
            st.caption(cap)

    return new_val


def render_flat_fields(section: SectionSpec, section_key_prefix: str):
    for fspec in section.flat_fields:
        current = _get(section.name, fspec.name)
        pre = _get_pre(section.name, fspec.name) if _show_diff() else None
        new_val = render_field(
            fspec, current, pre,
            key=f"{section_key_prefix}__{fspec.name}",
        )
        if new_val != current:
            _set(section.name, fspec.name, new_val)


def render_array_section(section: SectionSpec, section_key_prefix: str):
    afield = section.array_field_name
    atype = section.array_field_type
    show_diff = _show_diff()

    # Array-level N/A toggle: whole section marked intentionally null.
    raw_current = _get(section.name, afield)
    is_na_now = raw_current == NA_SENTINEL
    na_key = f"{section_key_prefix}__{afield}__na"
    backup_key = f"{section_key_prefix}__{afield}__na_backup"

    col_hdr, col_na = st.columns([5, 2])
    with col_hdr:
        st.markdown(f"**{afield.replace('_', ' ').title()}**")
    with col_na:
        na_checked = st.checkbox("N/A (刻意留空)", value=is_na_now, key=na_key)

    if na_checked and not is_na_now:
        if isinstance(raw_current, list):
            st.session_state[backup_key] = copy.deepcopy(raw_current)
        _set(section.name, afield, NA_SENTINEL)
        st.rerun()
    if not na_checked and is_na_now:
        restored = st.session_state.pop(backup_key, None)
        _set(section.name, afield, restored)
        st.rerun()

    if na_checked:
        st.caption(NA_LABEL)
        return

    if atype == "array_of_strings_enum":
        current_list = raw_current if isinstance(raw_current, list) else []
        pre_list = _get_items_pre(section.name, afield) or [] if show_diff else []
        changed = show_diff and set(current_list) != set(pre_list)
        label = f"{'✎ ' if changed else ''}{afield.replace('_', ' ').title()}"
        new_list = st.multiselect(
            label,
            options=section.array_item_enum_values,
            default=[v for v in current_list if v in section.array_item_enum_values],
            key=f"{section_key_prefix}__{afield}__multiselect",
        )
        if changed:
            st.caption(f"預標註值：{', '.join(_fmt_option(v) for v in pre_list) or EMPTY}")
        _set_items(section.name, afield, new_list if new_list else None)
        return

    items = list(raw_current) if isinstance(raw_current, list) else []
    pre_items = _get_items_pre(section.name, afield) if show_diff else []

    to_remove = None
    for idx, item in enumerate(items):
        col_card, col_rm = st.columns([11, 1])
        with col_rm:
            if st.button("✕", key=f"{section_key_prefix}_{afield}_rm_{idx}"):
                to_remove = idx

        pre_item = pre_items[idx] if idx < len(pre_items) else {}
        is_new = show_diff and idx >= len(pre_items)
        item_changed = show_diff and (is_new or any(
            _values_differ(item.get(f.name), pre_item.get(f.name))
            for f in section.array_item_fields
        ))

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
                    pre = pre_item.get(fspec.name) if (show_diff and not is_new) else None
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


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_annotator_picker():
    annotators: list[dict] = st.session_state.annotators
    current = st.session_state.current_annotator
    locked = bool(os.environ.get("REGISTRAR_ANNOTATE_LOCK_ANNOTATORS"))

    # In locked mode with a single annotator, auto-select to skip the placeholder.
    if locked and current is None and len(annotators) == 1:
        st.session_state.current_annotator = annotators[0]
        _on_annotator_change()
        st.rerun()

    labels = [f"{a['name']} ({a['suffix']})" for a in annotators]
    suffixes = [a["suffix"] for a in annotators]
    current_idx = (
        suffixes.index(current["suffix"])
        if current and current["suffix"] in suffixes
        else None
    )

    selected = st.sidebar.selectbox(
        "標註者",
        options=list(range(len(annotators))),
        index=current_idx,
        format_func=lambda i: labels[i],
        placeholder="— 請選擇標註者 —",
        key="annotator_select",
    )

    new_annotator = annotators[selected] if selected is not None else None
    current_suffix = current["suffix"] if current else None
    new_suffix = new_annotator["suffix"] if new_annotator else None
    if new_suffix != current_suffix:
        st.session_state.current_annotator = new_annotator
        _on_annotator_change()
        st.rerun()

    if locked:
        return

    with st.sidebar.expander("➕ 新增標註者", expanded=False):
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


def _render_mode_picker():
    current = st.session_state.mode
    new_mode = st.sidebar.radio(
        "標註模式",
        options=list(MODES),
        index=list(MODES).index(current),
        format_func=lambda m: MODE_LABELS[m],
        key="mode_radio",
    )
    if new_mode != current:
        st.session_state.mode = new_mode
        _on_mode_change()
        st.rerun()


def _render_dataset_picker():
    datasets: list[str] = st.session_state.available_datasets
    if not datasets:
        return
    current = st.session_state.dataset
    if current not in datasets:
        current = datasets[0]
    new_ds = st.sidebar.selectbox(
        "資料集",
        options=datasets,
        index=datasets.index(current),
        key="dataset_select",
    )
    if new_ds != st.session_state.dataset:
        st.session_state.dataset = new_ds
        _on_dataset_change()
        st.rerun()


def _maybe_auto_load():
    """Auto-load the env-var base dir once, after annotator is chosen."""
    if st.session_state.auto_loaded:
        return
    if not st.session_state.current_annotator:
        return
    base = st.session_state.base_dir
    if not base or not os.path.isdir(base):
        st.session_state.auto_loaded = True
        return
    ok, msg = _reload_from_base(base)
    st.session_state.auto_loaded = True
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.warning(msg)


def render_sidebar():
    st.sidebar.title("📋 標註清單")

    _render_annotator_picker()

    if not st.session_state.current_annotator:
        st.sidebar.info("請先選擇標註者再載入資料夾。")
        return

    _render_mode_picker()

    _maybe_auto_load()

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

    with st.sidebar.expander("手動輸入路徑", expanded=not bool(st.session_state.samples)):
        path_val = st.text_input(
            "Base folder path", value=st.session_state.base_dir, key="manual_base_input"
        )
        if st.button("載入", key="manual_base_load"):
            if not os.path.isdir(path_val):
                st.error("找不到資料夾。")
            else:
                ok, msg = _reload_from_base(path_val)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    _render_dataset_picker()

    samples: list[SampleRef] = st.session_state.samples
    ws: WorkspaceSet | None = st.session_state.workspace
    if not samples or not ws:
        st.sidebar.info("選擇資料夾以載入樣本。")
        return

    completed_total = sum(1 for s in samples if os.path.exists(s.annotation_path))
    st.sidebar.markdown(f"**已完成: {completed_total} / {len(samples)}**")
    st.sidebar.progress(completed_total / len(samples))

    stems = sorted({s.stem for s in samples}, key=lambda x: (len(x), x))
    stem_options = ["all"] + stems
    current_filter = st.session_state.stem_filter
    if current_filter not in stem_options:
        current_filter = "all"
    new_filter = st.sidebar.selectbox(
        "依前綴篩選", stem_options,
        index=stem_options.index(current_filter), key="stem_filter_select",
        format_func=lambda s: "全部" if s == "all" else s,
    )
    if new_filter != st.session_state.stem_filter:
        st.session_state.stem_filter = new_filter
        st.rerun()

    st.sidebar.divider()
    visible = _visible_samples()
    current_sample_id = st.session_state.last_sample_id

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


# ── Annotation panel ──────────────────────────────────────────────────────────

def render_annotation_panel():
    ann = st.session_state.annotation
    pre = st.session_state.pre_annotation

    st.subheader("Classification")

    excision = ann.get("cancer_excision_report")
    pre_excision = pre.get("cancer_excision_report") if _show_diff() else None
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
        cap = _pre_caption(pre_excision)
        if cap:
            st.caption(cap)
    if new_excision != excision:
        ann["cancer_excision_report"] = new_excision

    category = ann.get("cancer_category")
    pre_category = pre.get("cancer_category") if _show_diff() else None
    cat_opts = CANCER_CATEGORIES
    cidx = cat_opts.index(category) if category in cat_opts else 0
    cat_label = f"{_diff_marker(category, pre_category)}Cancer Category"
    new_cat = st.selectbox(
        cat_label, options=cat_opts, index=cidx,
        format_func=lambda x: SELECT if x is None else x.replace("_", " ").title(),
        key=f"cancer_category_select_{st.session_state.last_sample_id}",
    )
    if _values_differ(category, pre_category):
        cap = _pre_caption(pre_category)
        if cap:
            st.caption(cap)
    if new_cat != category:
        if "cancer_data" in ann:
            del ann["cancer_data"]
        ann["cancer_category"] = new_cat
        st.rerun()

    if new_cat == "others":
        others_desc = ann.get("cancer_category_others_description") or ""
        pre_desc = (pre.get("cancer_category_others_description") or "") if _show_diff() else ""
        label = f"{_diff_marker(others_desc, pre_desc)}Specify organ"
        new_desc = st.text_input(
            label, value=others_desc,
            key=f"others_desc_{st.session_state.last_sample_id}",
        )
        if _values_differ(others_desc, pre_desc):
            cap = _pre_caption(pre_desc or None)
            if cap:
                st.caption(cap)
        ann["cancer_category_others_description"] = new_desc or None

    if new_excision is False:
        st.info("非癌症報告 — 僅儲存分類結果。")
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
        st.button("💾 儲存標註", type="primary", use_container_width=True, disabled=True)
        return

    sample: SampleRef = samples[idx]
    if st.button("💾 儲存標註", type="primary", use_container_width=True):
        payload = build_save_payload(st.session_state.annotation, f"{sample.sample_id}.txt")
        save_annotation(payload, sample.annotation_path)
        st.session_state.annotation_status = "completed"
        st.session_state.save_message = f"已儲存 → {sample.annotation_path}"
        st.rerun()

    if st.session_state.save_message:
        st.success(st.session_state.save_message)


# ── Right column: report viewer ────────────────────────────────────────────────

def render_report_panel():
    samples = st.session_state.samples
    idx = st.session_state.sample_idx

    if not samples:
        st.info("尚未載入樣本。左側選擇資料夾開始。")
        return

    if idx >= len(samples):
        st.warning("未選取樣本。")
        return

    sample: SampleRef = samples[idx]
    total = len(samples)

    nav_l, nav_mid, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("← 上一筆", disabled=(idx == 0), key="btn_prev"):
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
        if st.button("下一筆 →", disabled=(idx >= total - 1), key="btn_next"):
            st.session_state.sample_idx = min(total - 1, idx + 1)
            _on_file_change()
            st.rerun()

    status = st.session_state.annotation_status
    if status == "completed":
        st.success("✓ 已載入既有標註")
    elif status == "new_preann":
        st.info("新標註 — 已由 GPT-OSS 預填")
    elif status == "new_blank":
        st.info("新標註 — 從空白開始")

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
        st.title("Cancer Pathology Report Annotator (Canonical)")
        if not st.session_state.current_annotator:
            st.info("👈 請先在左側選擇標註者。")
        else:
            st.info("👈 從左側選擇資料夾開始標註。")
        return

    st.title("Cancer Pathology Report Annotator (Canonical)")

    col_left, col_right = st.columns([1, 1])
    with col_right:
        st.markdown('<span class="report-col-marker"></span>', unsafe_allow_html=True)
        render_report_panel()
    with col_left:
        render_annotation_panel()


def _repo_root() -> Path:
    # src/digital_registrar_research/annotation/app_canonical.py → parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def _spawn_streamlit(default_base_dir: Path):
    import subprocess
    import sys
    env = os.environ.copy()
    env.setdefault("REGISTRAR_ANNOTATE_BASE_DIR", str(default_base_dir))
    script = str(Path(__file__).resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", script, *sys.argv[1:]]
    sys.exit(subprocess.call(cmd, env=env))


def main_cli_workspace():
    """Console-script entry for `registrar-annotate-workspace`."""
    _spawn_streamlit(_repo_root() / "workspace")


def main_cli_dummy():
    """Console-script entry for `registrar-annotate-dummy`."""
    _spawn_streamlit(_repo_root() / "dummy")


if os.environ.get("STREAMLIT_SERVER_PORT") or st.runtime.exists():
    main()
