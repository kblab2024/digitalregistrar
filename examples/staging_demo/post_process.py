"""Demo: post-extraction staging via the plain-callable surface.

Pipeline:

    raw report  →  run_cancer_pipeline_v2  →  cancer_data dict
                                                   │
                                                   ▼
                          observable_schema(organ, edition)
                                                   │
                  (hand-adapter: cancer_data → observations dict)
                                                   │
                                                   ▼
                              stage_from_observations(...)
                                                   │
                                                   ▼
                                  DerivedStage(T, N, M, stage, ...)

The hand-adapter step is the friction this example exposes: field
names in `cancer_data` (drr-next extraction output) do not match the
observable names that `tnmhelper` expects. Wiring this into the
pipeline-proper would mean codifying the mapping per organ.

This script is NOT part of the shipped pipeline; it lives in
``examples/`` deliberately.
"""
from __future__ import annotations

import json
from pprint import pprint

from digital_registrar.pipeline_factory import (
    run_cancer_pipeline_v2,
    setup_pipeline_v2,
)
from digital_registrar.staging import (
    active_data_source,
    editions_for,
    observable_schema,
    organs,
    stage_from_observations,
)

MODEL = "gpt"  # ollama_chat/gpt-oss:20b — see models/common.py for the catalogue
ORGAN = "breast"

REPORT = """\
Pathology report — left breast lumpectomy, 2026-03-14.

Specimen: 4.2 x 3.1 x 2.0 cm fibrofatty tissue with an ill-defined firm
tan-white nodule measuring 1.8 cm in greatest dimension.

Microscopy: Invasive ductal carcinoma, histologic grade 2 (Nottingham
score 6/9: tubules 3, nuclei 2, mitoses 1). No lymphovascular invasion
identified. Surgical margins free of tumor (closest margin 4 mm,
anterior).

Sentinel lymph nodes (3): 0 of 3 nodes involved by carcinoma.

Biomarkers: ER positive (95%), PR positive (80%), HER2 negative
(IHC 1+).

Staging: pT1c pN0 (sn) pMX — Stage IA. AJCC 8th edition.
"""


def main() -> None:
    print("=" * 72)
    print("Step 1 — configure DSPy")
    print("=" * 72)
    setup_pipeline_v2(MODEL)

    print()
    print("=" * 72)
    print("Step 2 — run the factory pipeline on the synthetic report")
    print("=" * 72)
    output, elapsed = run_cancer_pipeline_v2(report=REPORT, fname="demo", decomposition="auto")
    print(f"elapsed: {elapsed:.2f}s")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    cancer_data = output.get("cancer_data") or {}
    if not cancer_data:
        print("\n(no cancer_data extracted — staging step skipped)")
        return

    print()
    print("=" * 72)
    print("Step 3 — inspect tnmhelper's expectations")
    print("=" * 72)
    print(f"organs available: {organs()}")
    print(f"editions for {ORGAN!r}: {editions_for(ORGAN)}")
    print(f"data source: {active_data_source()}")

    edition = editions_for(ORGAN)[0]  # pick the first available; usually the most recent
    print(f"\nobservable_schema({ORGAN!r}, {edition!r}):")
    schema = observable_schema(ORGAN, edition)
    pprint({name: (spec.type, spec.unit, spec.values) for name, spec in schema.items()})

    print()
    print("=" * 72)
    print("Step 4 — hand-adapter: cancer_data → tnmhelper observations")
    print("=" * 72)
    print(
        "NOTE: this mapping is ILLUSTRATIVE. drr-next field names will not\n"
        "match tnmhelper observable names verbatim — codifying the per-organ\n"
        "mapping is the real work hidden inside any future wiring of this\n"
        "module into the pipeline.\n"
    )

    # Illustrative-only mapping. Real fields depend on what `schema` contains
    # at runtime AND what the factory pipeline emits for `cancer_data`. Adjust
    # the keys after inspecting both printouts above.
    observations: dict = {
        # "tumor_size_cm": cancer_data.get("tumor_size_cm"),
        # "ln_positive_count": cancer_data.get("ln_positive_count"),
        # "er_status": cancer_data.get("er_status"),
        # "pr_status": cancer_data.get("pr_status"),
        # "her2_status": cancer_data.get("her2_status"),
    }
    print("observations (illustrative — uncomment + fix names per schema above):")
    pprint(observations)

    if not observations:
        print("\n(nothing to stage; bail out before tnmhelper raises)")
        return

    print()
    print("=" * 72)
    print("Step 5 — call stage_from_observations")
    print("=" * 72)
    derived = stage_from_observations(ORGAN, edition, observations, Classification="p")
    print(derived)


if __name__ == "__main__":
    main()
