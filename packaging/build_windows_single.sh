#!/usr/bin/env bash
# Build the Windows bundle with a single annotator (NHC only).
# Output: packaging/dist/digital-registrar-annotator-windows-single.zip
set -euo pipefail
export PLATFORM=windows ANNOTATOR_SET=single
exec bash "$(dirname "$0")/_build_common.sh"
