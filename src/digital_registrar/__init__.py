"""Digital Registrar — schema-first, privacy-preserving pathology abstraction.

This package provides the core extraction pipeline, the CAP-aligned clinical
ontology (schemas), DSPy signatures, and a prediction-vs-annotation eval
toolkit. Higher-level apps (inference GUI, schema editor, annotator) are
shipped as separate packages: `digital-registrar-gui`,
`digital-registrar-schema-editor`, `digital-registrar-annotator`.

See `docs/api.md` for the public API reference.
"""
__version__ = "0.2.0"
__author__ = [
    "Nan-Haw Chow", "Han Chang", "Hung-Kai Chen", "Chen-Yuan Lin",
    "Ying-Lung Liu", "Po-Yen Tseng", "Li-Ju Shiu", "Yen-Wei Chu",
    "Pau-Choo Chung", "Kai-Po Chang",
]
__license__ = "MIT"

from digital_registrar.eval import (
    completeness,
    field_metrics,  # alias for digital_registrar.eval.metrics submodule
    nested_field_metrics,  # alias for digital_registrar.eval.nested_metrics submodule
    pairwise_compare,
    score_case,
    score_lymph_nodes,
    score_margins,
)
from digital_registrar.paths import (
    SCHEMAS_DATA,
    WORKSPACE_ROOT,
    results_root,
    workspace_root,
)
from digital_registrar.pipeline import (
    run_cancer_pipeline as run_pipeline_legacy,
)
from digital_registrar.pipeline_factory import (
    run_cancer_pipeline_v2 as run_pipeline,
    setup_pipeline_v2 as setup_pipeline,
)
from digital_registrar.schemas import (
    CASE_MODELS,
    build_case_model,
    list_organs,
    load_json_schema,
    load_pydantic_model,
)
from digital_registrar.signatures import (
    ExtractionStep,
    build_extraction_signatures,
    build_jsonize_signature,
    build_router_signature,
)

__all__ = [
    "__version__",
    # Pipeline
    "run_pipeline", "setup_pipeline", "run_pipeline_legacy",
    # Schemas
    "load_pydantic_model", "load_json_schema", "list_organs", "CASE_MODELS", "build_case_model",
    # Signatures
    "ExtractionStep", "build_extraction_signatures", "build_router_signature", "build_jsonize_signature",
    # Eval
    "field_metrics", "nested_field_metrics", "pairwise_compare", "completeness",
    "score_case", "score_lymph_nodes", "score_margins",
    # Paths
    "WORKSPACE_ROOT", "workspace_root", "results_root", "SCHEMAS_DATA",
]
