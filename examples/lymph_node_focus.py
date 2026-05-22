"""Companion to docs/learning/03_chunking.md.

Runs three flavors of "feed only lymph-node chunks to the lymph-node
extraction signature" against a single synthetic breast pathology report:

  Way 1: RegexSectionChunker with default patterns (self-labeling).
  Way 2: ParagraphChunker + DSPyEmbedderRouter (sentence-transformers).
  Way 3: Custom chunker that labels paragraphs mentioning LN terms.

Prints `out["step_chunks"]` for each — the dict mapping
``{step_name: [chunk_ids_fed_to_that_step]}``. The lymph_nodes step
should only see the LN-tagged chunks; other steps fall back to the
full report.

Requires:
  - `pip install digital-registrar[chunking]` (for sentence-transformers)
  - A local Ollama with the configured model, or OpenAI creds for an
    ``openai/*`` model in ``models/common.py:model_list``.

Run:
  python examples/lymph_node_focus.py

Adjust MODEL below if you're not running ollama_chat/gpt-oss:20b.
"""
from __future__ import annotations

import json
import re

from digital_registrar.chunking import (
    Chunk,
    DSPyEmbedderRouter,
    ParagraphChunker,
    RegexSectionChunker,
    register_chunker,
)
from digital_registrar.pipeline_factory import (
    run_cancer_pipeline_v2,
    setup_pipeline_v2,
)

MODEL = "gpt"  # ollama_chat/gpt-oss:20b — adjust to whatever your environment supports

# Synthetic breast pathology report with conventional headers, so all
# three approaches can light up. Note the explicit MARGINS / LYMPH NODES
# headers — these are what RegexSectionChunker keys off.
REPORT = """\
Pathology report — left breast lumpectomy, 2026-04-12.

SPECIMEN:
4.0 x 3.2 x 2.1 cm fibrofatty breast tissue with an ill-defined firm
tan-white nodule measuring 1.9 cm in greatest dimension. Specimen
oriented with sutures: short = superior, long = lateral.

MICROSCOPIC DESCRIPTION:
Sections show invasive ductal carcinoma, no special type, Nottingham
histologic grade 2 of 3 (tubules 3, nuclear pleomorphism 2,
mitotic count 1). No lymphovascular invasion identified. Associated
ductal carcinoma in situ, intermediate nuclear grade, comedo pattern,
involves approximately 15% of the tumor area.

MARGINS:
All surgical margins are free of invasive carcinoma. Closest margin
is the anterior margin at 4 mm. DCIS extends to within 2 mm of the
anterior margin. Other margins (superior, inferior, medial, lateral,
posterior) are all >5 mm from tumor.

LYMPH NODES:
Sentinel lymph nodes, three (3) submitted. All three negative for
metastatic carcinoma on H&E and cytokeratin immunostains. No
extranodal extension. No isolated tumor cells or micrometastases.

BIOMARKERS:
ER positive (Allred score 8/8, ~95% staining).
PR positive (Allred score 7/8, ~80% staining).
HER2 IHC 1+ (negative).
Ki-67 proliferation index ~12%.

STAGING:
pT1c pN0(sn) pMX. AJCC 8th edition. Stage group: IA.

COMMENTS:
Recommend medical oncology consult for endocrine therapy. HER2 IHC 1+
does not require reflex ISH testing per current guidelines.
"""


# ----------------------------------------------------------------------
# Way 3 — custom chunker that labels only LN paragraphs.
# Defined at module level so the @register_chunker decorator fires once
# (it raises on double-registration).
# ----------------------------------------------------------------------
_LN_TERMS = re.compile(
    r"\b(?:lymph\s*nodes?|LN|sentinel|axillary|station\s*\d+|N\d|micrometastas[ei]s|extranodal)\b",
    re.IGNORECASE,
)


@register_chunker("lymph_node_focus_only")
class LymphNodeFocusOnlyChunker:
    """Tag paragraphs mentioning LN terms with {'lymph_nodes'}; leave others unlabeled."""

    def chunk(self, report: str) -> list[Chunk]:
        base = ParagraphChunker().chunk(report)
        out: list[Chunk] = []
        for c in base:
            if _LN_TERMS.search(c.text):
                out.append(c.with_labels(frozenset({"lymph_nodes"})))
            else:
                out.append(c)
        return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _print_step_chunks(label: str, out: dict, elapsed: float) -> None:
    print(f"\n--- {label}  ({elapsed:.2f}s) ---")
    if "step_chunks" not in out:
        print("  (no step_chunks — routing was disabled or the report was not classified)")
        return
    print(f"  cancer_category: {out.get('cancer_category')!r}")
    print("  step_chunks:")
    print(json.dumps(out["step_chunks"], indent=4, ensure_ascii=False))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    setup_pipeline_v2(MODEL)

    # ----- Way 1: regex section chunker, self-labeling --------------------
    out1, t1 = run_cancer_pipeline_v2(
        report=REPORT,
        chunker=RegexSectionChunker(),
        routing_mode="filter",
    )
    _print_step_chunks("Way 1: RegexSectionChunker (default patterns)", out1, t1)

    # ----- Way 2: paragraph + embedding router ----------------------------
    # Uses sentence-transformers/all-MiniLM-L6-v2 by default; swap the
    # `model=` kwarg for another LiteLLM-resolvable embedder if you prefer.
    try:
        router = DSPyEmbedderRouter(
            model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=2,
            min_similarity=0.20,
        )
        out2, t2 = run_cancer_pipeline_v2(
            report=REPORT,
            chunker=ParagraphChunker(),
            chunk_router=router,
            routing_mode="filter",
        )
        _print_step_chunks("Way 2: ParagraphChunker + DSPyEmbedderRouter", out2, t2)
    except ImportError as e:
        print(f"\n--- Way 2: SKIPPED ({e}) ---")
        print("  Install with: pip install sentence-transformers")

    # ----- Way 3: custom chunker, labels only LN paragraphs ---------------
    out3, t3 = run_cancer_pipeline_v2(
        report=REPORT,
        chunker=LymphNodeFocusOnlyChunker(),
        routing_mode="filter",
    )
    _print_step_chunks("Way 3: LymphNodeFocusOnlyChunker (custom)", out3, t3)

    print("\nDone. The `lymph_nodes` extraction step should have")
    print("received only LN-relevant chunk(s) in each case.")


if __name__ == "__main__":
    main()
