#!/usr/bin/env bash
# Install a minimal pre-commit hook that runs ruff before every commit.
#
#     bash scripts/install_git_hooks.sh
#
# After installation, every `git commit` will run `ruff check --fix` on
# the staged files. If ruff fixes anything, the hook re-stages the
# modified files so the fix is part of the same commit. If ruff reports
# an error it can't auto-fix, the commit is aborted.
#
# This is intentionally a plain shell script rather than a pre-commit
# framework config — the framework chokes on Microsoft Store Python's
# deep LocalCache path on Windows.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK_PATH" <<'HOOK'
#!/usr/bin/env bash
# Auto-installed by scripts/install_git_hooks.sh — do not edit by hand.
# Runs ruff on staged Python files; re-stages anything ruff auto-fixed.
set -e

# Collect staged .py files (added/copied/modified, NUL-separated to handle spaces)
mapfile -d '' staged < <(git diff --cached --name-only --diff-filter=ACM -z -- '*.py')
[ ${#staged[@]} -eq 0 ] && exit 0

echo "[pre-commit] ruff check --fix on ${#staged[@]} file(s)..."
python -m ruff check --fix "${staged[@]}"

# Re-stage anything ruff modified so the fix is part of the commit
git add -- "${staged[@]}"
HOOK

chmod +x "$HOOK_PATH"
echo "Installed git pre-commit hook at: $HOOK_PATH"
echo "Test it with: git commit (a no-op stage will skip the hook)."
