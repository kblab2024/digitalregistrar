#!/usr/bin/env bash
# Build the Unix (Linux x86_64) bundle with a single annotator (NHC only).
# Output: packaging/dist/digital-registrar-annotator-unix-single.tar.gz
set -euo pipefail
export PLATFORM=unix ANNOTATOR_SET=single
exec bash "$(dirname "$0")/_build_common.sh"
