"""Smoke tests for the annotation IO layer (three-folder discovery + flat-JSON contract)."""
from digital_registrar_research.annotation import io as ann_io
from digital_registrar_research.paths import DATA_ROOT


def test_module_exposes_public_helpers():
    for name in ("FolderSet", "SampleRef", "build_save_payload",
                 "discover_folders", "list_samples", "load_json",
                 "load_report_text", "save_annotation", "strip_meta",
                 "NA_SENTINEL", "rehydrate_sentinels"):
        assert hasattr(ann_io, name), f"annotation.io missing {name!r}"


def test_discover_folders_on_packaged_data():
    """The shipped TCGA example folders should be discoverable."""
    from pathlib import Path
    fs = ann_io.discover_folders(str(DATA_ROOT))
    assert fs is not None
    # Three-folder convention: {prefix}_dataset_{date}/ + _result_/ + _annotation_/.
    assert Path(fs.dataset_dir).exists()
    assert Path(fs.result_dir).exists()
    assert Path(fs.annotation_dir).exists()
    assert fs.prefix == "tcga"


# ── Intentional-null sentinel round trip ───────────────────────────────────────

def test_sentinel_round_trip_preserves_intent():
    """build_save_payload strips sentinels to null and records their paths;
    rehydrate_sentinels puts them back when reloading."""
    NA = ann_io.NA_SENTINEL
    session = {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_category_others_description": None,
        "cancer_data": {
            "treatment_effect": NA,
            "tumor_size": None,
            "margins": [
                {"margin_category": "anterior", "distance_mm": NA},
                {"margin_category": "posterior", "distance_mm": 5},
            ],
            "biomarker_categories": NA,
        },
    }

    payload = ann_io.build_save_payload(session, "fake.txt")

    # Values collapse to null.
    assert payload["cancer_data"]["treatment_effect"] is None
    assert payload["cancer_data"]["margins"][0]["distance_mm"] is None
    assert payload["cancer_data"]["biomarker_categories"] is None
    # Paths recorded.
    intent = payload["_meta"]["intentional_nulls"]
    assert "cancer_data.treatment_effect" in intent
    assert "cancer_data.margins[0].distance_mm" in intent
    assert "cancer_data.biomarker_categories" in intent
    # Un-flagged nulls stay out.
    assert "cancer_data.tumor_size" not in intent

    # Reload: strip meta into a fresh session, rehydrate.
    reloaded = ann_io.strip_meta(payload)
    ann_io.rehydrate_sentinels(reloaded, payload["_meta"])
    assert reloaded["cancer_data"]["treatment_effect"] == NA
    assert reloaded["cancer_data"]["tumor_size"] is None
    assert reloaded["cancer_data"]["margins"][0]["distance_mm"] == NA
    assert reloaded["cancer_data"]["margins"][1]["distance_mm"] == 5
    assert reloaded["cancer_data"]["biomarker_categories"] == NA


def test_rehydrate_skips_stale_paths():
    """Paths that no longer resolve (removed array entries etc.) are ignored."""
    NA = ann_io.NA_SENTINEL
    reloaded = {"cancer_data": {"margins": [{"distance_mm": None}]}}
    meta = {"intentional_nulls": [
        "cancer_data.margins[0].distance_mm",
        "cancer_data.margins[5].distance_mm",   # out of range
        "cancer_data.missing_field",            # missing key
        "cancer_data.margins[0].missing_field", # missing per-item key
    ]}
    ann_io.rehydrate_sentinels(reloaded, meta)
    assert reloaded["cancer_data"]["margins"][0]["distance_mm"] == NA
