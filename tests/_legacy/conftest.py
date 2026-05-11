"""Exclude all tests in this directory from default pytest collection.

Files here exercise modules that have been archived to
``src/digital_registrar_research/_legacy/``. They are kept for historical
reproducibility, not for CI signal — and they include known pre-existing
failures unrelated to current work.

To run them explicitly:

    pytest tests/_legacy --override-ini "collect_ignore_glob=" -q
"""

collect_ignore_glob = ["*"]
