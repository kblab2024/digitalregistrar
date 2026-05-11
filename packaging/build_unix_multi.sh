#!/usr/bin/env bash
# Thin wrapper — see packaging/build.py for the canonical entry point.
set -euo pipefail
exec python3 "$(dirname "$0")/build.py" --platform unix --annotators multi "$@"
