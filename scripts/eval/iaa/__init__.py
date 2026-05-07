"""Inter-annotator agreement subcommand.

Pairwise IAA across human annotators (with and without preann) and
gold, plus the headline preann-effect analyses (Δκ paired by case,
anchoring index, convergence to preann, disagreement reduction).

Reuses ``src/.../eval/iaa.py`` for the agreement primitives (Cohen's
κ, Krippendorff's α, Lin's CCC, ICC, etc.) and ``src/.../eval/preann.py``
for the new paired analyses.
"""
