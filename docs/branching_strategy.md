# Branching strategy

The repo runs 12 branches. This doc is the map — what each branch is for,
how long it lives, and whether it ever merges back.

## The three kinds of branch

| Kind | Lifecycle | Merge-back? |
|---|---|---|
| **Testing** (`testing_*`) | Throwaway; open → poke → reset / delete | No (cherry-pick good bits into `refactor_layout` if earned) |
| **Long-lived refactor** (`refactor_layout`) | Open for weeks, tracks a single multi-file refactor | Yes, squash-merge to `main` when the layout migration ships |
| **Experiment state** (`experiment_cmuh_pilot`) | Open for the lifetime of the experiment | No — code flows `main → pilot`, never reverse |

## Branch inventory

### Core testing — one per system under test

| Branch | What goes on it |
|---|---|
| `testing_llm` | LLM multi-run inference (`scripts/pipeline/run_gpt_oss_multirun.py`, per-model sweeps); trying new prompts, decoding params, retry logic |
| `testing_ui` | Streamlit annotation app (`src/.../annotation/`, session state, discovery, annotator-mode switching) |
| `testing_bert` | ClinicalBERT training / inference (`scripts/baselines/predict_clinicalbert_cls_mac.py`, head architecture, calibration) |
| `testing_rule` | Rule-based baseline (`src/.../benchmarks/baselines/rules.py`), regex tuning |

### Risk-isolation — high-blast-radius code changes

| Branch | Why isolated |
|---|---|
| `testing_migration` | TCGA old-path → new-path script. Touches real data; a bad run orphans 300+ files |
| `testing_paths` | `paths.py` resolver rewrite. Every script imports from here — break this, break everything |
| `testing_schema` | Pydantic schema churn. Adding organs / renaming fields invalidates every existing annotation and prediction |

### Specialised evaluation

| Branch | Focus |
|---|---|
| `testing_iaa` | Inter-annotator agreement (Cohen/QWK/CCC) + `preann_effect_*` comparisons |
| `testing_ensemble` | Majority-vote across LLM runs, ensemble-vs-single bootstrap |

### Long-lived

| Branch | Role |
|---|---|
| `refactor_layout` | Integration branch for the dataset/results layout redesign. Cherry-picks validated bits from `testing_paths`, `testing_schema`, `testing_migration`. Squash-merges to `main` when all pieces land |
| `experiment_cmuh_pilot` | The "working floor" of the 2026-04 experiment. Accumulates real data, annotator outputs, and per-experiment config. Never merges back to `main`; code flows in from `main` as features stabilise |

## Typical workflows

### Trying a new decoding temperature for gpt-oss

```bash
git checkout testing_llm
# edit configs/experiments/multirun_gpt_oss.yaml
python scripts/pipeline/run_gpt_oss_multirun.py --config ...  --dry-run
# iterate until happy
git checkout main && git branch -D testing_llm && git branch testing_llm   # reset & try again
```
No merge back — the config change either gets cherry-picked onto
`main`/`refactor_layout` or thrown away.

### Refactoring `paths.py`

```bash
git checkout testing_paths
# rewrite paths.py, run tests
git checkout refactor_layout
git cherry-pick <testing_paths commit>
# update all callers, land the whole refactor as one squash-merge to main
```

### Starting CMUH annotation

```bash
git checkout experiment_cmuh_pilot
# drop CMUH reports into data/cmuh/reports/, commit (with LFS / git-annex if large)
# launch annotation UI on this branch so annotators' saves go here, not main
```
Keep `main` code-only. Real annotation JSONs live on `experiment_cmuh_pilot`.

## Merge policy

- **Never force-push `main`**, `refactor_layout`, or `experiment_cmuh_pilot`.
- **Testing branches are local-unstable**: feel free to `git reset --hard`
  or delete-and-recreate them. If they're pushed to remote, note it
  publicly before force-pushing.
- **`refactor_layout` receives cherry-picks** from testing branches, not
  merges — keeps history linear.
- **`experiment_cmuh_pilot` receives merges** from `main` (one-way).

## One-time setup per clone — install the pre-commit hook

CI runs `ruff check src/ tests/` on every push. To catch lint issues
**before** they hit CI, install the local git hook:

```bash
bash scripts/repo/install_git_hooks.sh
```

This drops a `.git/hooks/pre-commit` script that runs
`python -m ruff check --fix` on the staged Python files and re-stages
anything ruff auto-fixed. If ruff hits an error it can't auto-fix,
the commit aborts so you see it locally instead of in CI.

Skip the hook for an emergency commit with `git commit --no-verify`
(but expect CI to be red).

The framework `pre-commit` was deliberately **not** used — its
per-hook virtualenv layout exceeds Windows MAX_PATH on Microsoft
Store Python. The plain shell hook above works on Git Bash, macOS,
and Linux without extra setup.

## Branch creation recipes

```bash
# Core testing — isolated poking
git branch testing_<system>

# Long-lived — rename with care, they're public
git branch refactor_<thing>
git branch experiment_<name>_<phase>
```

When in doubt about where code belongs: **if it will ship to `main`, use
a testing branch → cherry-pick to `refactor_layout`. If it's experiment
state that never ships, use `experiment_cmuh_pilot`.**
