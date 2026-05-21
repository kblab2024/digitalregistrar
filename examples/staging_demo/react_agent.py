"""Demo: DSPy ReAct agent driving STAGING_TOOLS.

Builds a single-call ``dspy.ReAct`` module wired with the two tools
exposed by ``digital_registrar_research.staging``:

  - ``cancer_observable_schema`` — look up what observations the engine
    expects for an (organ, edition) pair.
  - ``cancer_stage_from_observations`` — compute the AJCC TNM stage
    given those observations.

The model is expected to call the observable-schema tool first, extract
the relevant values from the report, then call
``cancer_stage_from_observations`` and return the resulting stage.

This is single-call demo code, not a benchmarked module. STAGING_TOOLS
is not consumed anywhere else in this repo; this script is its only
caller.
"""
from __future__ import annotations

import dspy

from digital_registrar_research.models.common import autoconf_dspy
from digital_registrar_research.staging import STAGING_TOOLS

MODEL = "gpt"  # ollama_chat/gpt-oss:20b — see models/common.py for the catalogue

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

Staging: pT1c pN0 (sn) pMX. AJCC 8th edition.
"""


class StageFromReport(dspy.Signature):
    """Derive the AJCC TNM stage for the cancer described in the report.

    First call ``cancer_observable_schema`` with the given organ and
    edition to learn which observations the staging engine expects and
    what their valid values are. Then read those observations off the
    report (T, N, M categories, tumor size, lymph-node counts,
    biomarkers — only the ones the schema lists). Finally call
    ``cancer_stage_from_observations`` with the values you found and
    return its ``stage`` field as your output.
    """

    report: str = dspy.InputField(desc="A pathology report describing one cancer case.")
    organ: str = dspy.InputField(desc="The cancer organ, e.g. 'breast'.")
    edition: str = dspy.InputField(desc="The AJCC edition string, e.g. 'AJCC 8'.")
    stage: str = dspy.OutputField(desc="The derived stage group (e.g. 'IIA').")
    reasoning: str = dspy.OutputField(
        desc="One short sentence on which observations drove the stage assignment."
    )


def main() -> None:
    print("=" * 72)
    print("Step 1 — configure DSPy")
    print("=" * 72)
    autoconf_dspy(MODEL)

    print()
    print("=" * 72)
    print("Step 2 — build the ReAct agent")
    print("=" * 72)
    agent = dspy.ReAct(StageFromReport, tools=STAGING_TOOLS, max_iters=6)
    print(f"agent: {agent}")
    print(f"tools: {[t.name for t in STAGING_TOOLS]}")

    print()
    print("=" * 72)
    print("Step 3 — run the agent against the synthetic report")
    print("=" * 72)
    result = agent(report=REPORT, organ="breast", edition="AJCC 8")
    print(f"stage:     {result.stage}")
    print(f"reasoning: {result.reasoning}")

    print()
    print("=" * 72)
    print("Step 4 — inspect the ReAct trajectory")
    print("=" * 72)
    dspy.inspect_history(n=1)


if __name__ == "__main__":
    main()
