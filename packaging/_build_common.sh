#!/usr/bin/env bash
# Shared bundle builder. Not meant to be run directly — use one of the
# build_<platform>_<annotator_set>.sh wrappers, which export:
#
#   PLATFORM         windows | unix
#   ANNOTATOR_SET    single  | multi
#
# Output: packaging/dist/digital-registrar-annotator-<platform>-<set>.<ext>
#
# Build host: macOS or Linux. Windows bundles cross-install win_amd64 wheels;
# Unix bundles use python-build-standalone for a portable CPython.
#
# Each bundle contains:
#   python/     Portable Python (embedded on Windows, python-build-standalone on Unix)
#   app/        Slim digital_registrar_research package (annotation subtree only)
#   dummy/      Bundled dummy dataset (smoke-test fixture + format reference)
#   workspace/  Empty-but-structured workspace (user drops real data here)
#   run.<ext>      Default launcher — workspace/, annotators locked
#   run_demo.<ext> Alternate launcher — dummy/, unlocked (add-annotator UI visible)
#   README.txt  Annotator-facing instructions

set -euo pipefail

# ── Wrapper contract ──────────────────────────────────────────────────────────
: "${PLATFORM:?PLATFORM must be exported (windows|unix) by the wrapper}"
: "${ANNOTATOR_SET:?ANNOTATOR_SET must be exported (single|multi) by the wrapper}"

case "$PLATFORM" in
    windows|unix) ;;
    *) echo "ERR: unknown PLATFORM='$PLATFORM' (expected windows|unix)" >&2; exit 1 ;;
esac
case "$ANNOTATOR_SET" in
    single|multi) ;;
    *) echo "ERR: unknown ANNOTATOR_SET='$ANNOTATOR_SET' (expected single|multi)" >&2; exit 1 ;;
esac

# ── Pins ──────────────────────────────────────────────────────────────────────
PYTHON_VERSION="3.12.8"
PYTHON_TAG="cp312"
RUNTIME_REQS=(
    "streamlit>=1.35,<2"
    "pydantic>=2,<3"
)

# python-build-standalone pinned release (Linux x86_64 install_only tarball).
# Update RELEASE together with PYTHON_VERSION when bumping.
PBS_RELEASE="20241016"
PBS_LINUX_X86_64="cpython-${PYTHON_VERSION}+${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$REPO_ROOT/packaging"
DL_DIR="$PKG_DIR/_downloads"
BUILD_DIR="$PKG_DIR/build"
DIST_DIR="$PKG_DIR/dist"
BUNDLE_NAME="digital-registrar-annotator-${PLATFORM}-${ANNOTATOR_SET}"
BUNDLE_DIR="$BUILD_DIR/$BUNDLE_NAME"

PY_HOST="${PYTHON:-python3}"

# ── Helpers ───────────────────────────────────────────────────────────────────
say()  { printf "\033[1;36m==>\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31mERR:\033[0m %s\n" "$*" >&2; exit 1; }

# ── Preconditions ─────────────────────────────────────────────────────────────
command -v "$PY_HOST" >/dev/null || die "host python not found ($PY_HOST)"
command -v curl      >/dev/null || die "curl is required"
command -v unzip     >/dev/null || die "unzip is required"
command -v zip       >/dev/null || die "zip is required"
command -v tar       >/dev/null || die "tar is required"

[ -d "$REPO_ROOT/src/digital_registrar_research/annotation" ] \
    || die "expected to run from the repo root — annotation package not found"
[ -d "$REPO_ROOT/dummy" ] \
    || die "dummy/ not found; nothing to bundle"
[ -d "$REPO_ROOT/workspace" ] \
    || die "workspace/ not found; create it first (cp -r dummy workspace)"

"$PY_HOST" -c "
import digital_registrar_research.models.modellist
import digital_registrar_research.schemas.pydantic._builder
" >/dev/null 2>&1 \
    || die "host python cannot import dspy-backed modules — activate dev venv first"

# ── Clean build dir (keep downloads cached) ───────────────────────────────────
say "Preparing build tree for $BUNDLE_NAME"
rm -rf "$BUNDLE_DIR"
mkdir -p "$DL_DIR" "$DIST_DIR" "$BUNDLE_DIR"

# ── Stage portable Python + cross-install wheels (platform-specific) ──────────
PYTHON_DIR="$BUNDLE_DIR/python"
mkdir -p "$PYTHON_DIR"

if [ "$PLATFORM" = "windows" ]; then
    EMBED_ZIP_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip"
    EMBED_ZIP_PATH="$DL_DIR/python-${PYTHON_VERSION}-embed-amd64.zip"

    if [ ! -f "$EMBED_ZIP_PATH" ]; then
        say "Downloading Windows embeddable Python $PYTHON_VERSION"
        curl -fL --retry 3 -o "$EMBED_ZIP_PATH.part" "$EMBED_ZIP_URL"
        mv "$EMBED_ZIP_PATH.part" "$EMBED_ZIP_PATH"
    else
        say "Using cached $(basename "$EMBED_ZIP_PATH")"
    fi

    say "Extracting embedded Python"
    unzip -q -o "$EMBED_ZIP_PATH" -d "$PYTHON_DIR"

    # Enable `import site` so pip-installed packages are picked up.
    PTH_FILE="$(ls "$PYTHON_DIR"/python*._pth | head -1)"
    [ -n "$PTH_FILE" ] || die "python*._pth not found in embedded distribution"
    say "Patching $(basename "$PTH_FILE")"
    # Paths are relative to python.exe (python/python.exe), so ..\app points
    # to <bundle>/app where our slim digital_registrar_research package lives.
    # ._pth disables PYTHONPATH env var, so this must be listed here or the
    # package won't be importable at runtime.
    cat > "$PTH_FILE" <<EOF
python312.zip
.
..\\app
Lib\\site-packages
import site
EOF

    SITE_PACKAGES="$PYTHON_DIR/Lib/site-packages"
    mkdir -p "$SITE_PACKAGES"

    say "Installing Windows wheels into embedded site-packages"
    "$PY_HOST" -m pip install \
        --disable-pip-version-check \
        --target "$SITE_PACKAGES" \
        --platform "win_amd64" \
        --python-version "${PYTHON_VERSION%.*}" \
        --implementation cp \
        --abi "$PYTHON_TAG" \
        --only-binary=:all: \
        --upgrade \
        "${RUNTIME_REQS[@]}"

else
    # Unix: python-build-standalone (Linux x86_64 only in this plan; macOS is a follow-up).
    PBS_URL="https://github.com/indygreg/python-build-standalone/releases/download/${PBS_RELEASE}/${PBS_LINUX_X86_64}"
    PBS_PATH="$DL_DIR/$PBS_LINUX_X86_64"

    if [ ! -f "$PBS_PATH" ]; then
        say "Downloading python-build-standalone $PYTHON_VERSION ($PBS_RELEASE)"
        curl -fL --retry 3 -o "$PBS_PATH.part" "$PBS_URL"
        mv "$PBS_PATH.part" "$PBS_PATH"
    else
        say "Using cached $(basename "$PBS_PATH")"
    fi

    say "Extracting python-build-standalone"
    # The install_only tarball contains a top-level `python/` directory.
    tar -xzf "$PBS_PATH" -C "$BUNDLE_DIR"
    [ -x "$PYTHON_DIR/bin/python3" ] || die "extracted python-build-standalone layout unexpected"

    SITE_PACKAGES="$PYTHON_DIR/lib/python${PYTHON_VERSION%.*}/site-packages"
    mkdir -p "$SITE_PACKAGES"

    say "Installing Unix wheels into bundled site-packages"
    "$PYTHON_DIR/bin/python3" -m pip install \
        --disable-pip-version-check \
        --target "$SITE_PACKAGES" \
        --upgrade \
        "${RUNTIME_REQS[@]}"
fi

# ── Trim size: drop tests/ and __pycache__ from site-packages ─────────────────
say "Trimming site-packages"
find "$SITE_PACKAGES" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$SITE_PACKAGES" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type d -name "test"  -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type f -name "*.pyc" -delete
rm -rf "$SITE_PACKAGES/bin"

# ── Assemble slim digital_registrar_research package ─────────────────────────
say "Copying annotation package"
APP_ROOT="$BUNDLE_DIR/app"
DRR_DEST="$APP_ROOT/digital_registrar_research"
mkdir -p "$DRR_DEST"

cp "$REPO_ROOT/src/digital_registrar_research/__init__.py" "$DRR_DEST/__init__.py"
cp "$REPO_ROOT/src/digital_registrar_research/paths.py"    "$DRR_DEST/paths.py"

mkdir -p "$DRR_DEST/annotation"
rsync -a \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "generate_dummy_data.py" \
    --exclude "dummy_data" \
    --exclude "annotators.json" \
    "$REPO_ROOT/src/digital_registrar_research/annotation/" \
    "$DRR_DEST/annotation/"

mkdir -p "$DRR_DEST/schemas/data"
touch "$DRR_DEST/schemas/__init__.py"
rsync -a --exclude "__pycache__" --exclude "*.pyc" \
    "$REPO_ROOT/src/digital_registrar_research/schemas/data/" \
    "$DRR_DEST/schemas/data/"

say "Generating precomputed section-groups map"
"$PY_HOST" "$PKG_DIR/precompute_section_groups.py" \
    "$DRR_DEST/annotation/_section_groups.json"

# ── Write annotators.json baked to the chosen annotator set ──────────────────
say "Baking annotators.json ($ANNOTATOR_SET)"
if [ "$ANNOTATOR_SET" = "single" ]; then
    cat > "$DRR_DEST/annotation/annotators.json" <<'EOF'
{
  "annotators": [
    {"name": "Nan-Haw Chow", "suffix": "nhc"}
  ]
}
EOF
else
    cat > "$DRR_DEST/annotation/annotators.json" <<'EOF'
{
  "annotators": [
    {"name": "Nan-Haw Chow", "suffix": "nhc"},
    {"name": "Kai-Po Chang", "suffix": "kpc"}
  ]
}
EOF
fi

# ── Copy dummy dataset + workspace skeleton ──────────────────────────────────
say "Copying dummy dataset"
rsync -a \
    --exclude ".DS_Store" \
    --exclude "__pycache__" \
    --exclude "results" \
    "$REPO_ROOT/dummy/" \
    "$BUNDLE_DIR/dummy/"

say "Copying workspace skeleton"
rsync -a \
    --exclude ".DS_Store" \
    --exclude "__pycache__" \
    --exclude "results" \
    "$REPO_ROOT/workspace/" \
    "$BUNDLE_DIR/workspace/"

# ── Launchers + README ────────────────────────────────────────────────────────
say "Installing launchers"
if [ "$PLATFORM" = "windows" ]; then
    cp "$PKG_DIR/run_workspace_locked_${ANNOTATOR_SET}.bat" "$BUNDLE_DIR/run.bat"
    cp "$PKG_DIR/run_dummy_unlocked_${ANNOTATOR_SET}.bat"   "$BUNDLE_DIR/run_demo.bat"
else
    cp "$PKG_DIR/run_workspace_locked_${ANNOTATOR_SET}.sh"  "$BUNDLE_DIR/run.sh"
    cp "$PKG_DIR/run_dummy_unlocked_${ANNOTATOR_SET}.sh"    "$BUNDLE_DIR/run_demo.sh"
    chmod +x "$BUNDLE_DIR/run.sh" "$BUNDLE_DIR/run_demo.sh"
fi
cp "$PKG_DIR/README.txt" "$BUNDLE_DIR/README.txt"

# ── Final pycache sweep ───────────────────────────────────────────────────────
find "$BUNDLE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | awk '{print $1}')"
say "Bundle tree assembled: $BUNDLE_DIR ($BUNDLE_SIZE)"

# ── Archive ───────────────────────────────────────────────────────────────────
if [ "$PLATFORM" = "windows" ]; then
    ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.zip"
    rm -f "$ARCHIVE_PATH"
    say "Zipping"
    ( cd "$BUILD_DIR" && zip -qr "$ARCHIVE_PATH" "$BUNDLE_NAME" )
else
    ARCHIVE_PATH="$DIST_DIR/${BUNDLE_NAME}.tar.gz"
    rm -f "$ARCHIVE_PATH"
    say "Tarring"
    ( cd "$BUILD_DIR" && tar -czf "$ARCHIVE_PATH" "$BUNDLE_NAME" )
fi
ARCHIVE_SIZE="$(du -sh "$ARCHIVE_PATH" | awk '{print $1}')"

say "DONE"
printf "   bundle: %s  (%s unpacked)\n" "$ARCHIVE_PATH" "$BUNDLE_SIZE"
printf "   archive: %s\n" "$ARCHIVE_SIZE"
if [ "$PLATFORM" = "windows" ]; then
    printf "\nNext: copy %s to the Windows machine, unzip, double-click run.bat.\n" "$ARCHIVE_PATH"
else
    printf "\nNext: copy %s to the target, tar -xzf it, ./run.sh\n" "$ARCHIVE_PATH"
fi
