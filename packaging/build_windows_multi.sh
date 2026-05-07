#!/usr/bin/env bash
# Build the Windows bundle with the multi-annotator list (NHC + KPC).
# Output: packaging/dist/digital-registrar-annotator-windows-multi.zip
set -euo pipefail
export PLATFORM=windows ANNOTATOR_SET=multi
exec bash "$(dirname "$0")/_build_common.sh"
