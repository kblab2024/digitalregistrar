#!/usr/bin/env bash
# Thin wrapper — see packaging/build.py for the canonical entry point.
# Cross-builds a Windows bundle from a Unix host (e.g. CI).
set -euo pipefail
exec python3 "$(dirname "$0")/build.py" --platform windows --annotators single "$@"
