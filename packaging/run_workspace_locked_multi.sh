#!/usr/bin/env bash
# Launch the annotator: workspace data root, locked annotator list (NHC / KPC).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

CRED="$HOME/.streamlit/credentials.toml"
if [ ! -f "$CRED" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$CRED"
fi

export REGISTRAR_ANNOTATE_BASE_DIR="$HERE/workspace"
export REGISTRAR_ANNOTATE_LOCK_ANNOTATORS=1
export PYTHONPATH="$HERE/app"
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

cat <<'BANNER'
================================================================
  Digital Registrar - Annotator (workspace, locked annotators)
================================================================
  Data root : workspace/
  Annotators: NHC / KPC (locked list, pick one in sidebar)
  URL       : http://localhost:8501

  Keep this terminal open while annotating.
  Press Ctrl+C to stop the server.
================================================================

BANNER

exec "$HERE/python/bin/python3" -m streamlit run \
    "$HERE/app/digital_registrar_research/annotation/app_canonical.py" \
    --server.address=localhost \
    --server.port=8501
