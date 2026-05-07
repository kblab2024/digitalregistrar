#!/usr/bin/env bash
# Build the Unix (Linux x86_64) bundle with the multi-annotator list (NHC + KPC).
# Output: packaging/dist/digital-registrar-annotator-unix-multi.tar.gz
set -euo pipefail
export PLATFORM=unix ANNOTATOR_SET=multi
exec bash "$(dirname "$0")/_build_common.sh"
