#!/usr/bin/env bash
# Portable runner for the GLM-5.1 content-overlap-structure analysis.
# Works from any clone on any machine -- paths are derived relative to THIS file,
# nothing is hardcoded to a particular home dir or worktree.
#
# USAGE
#   ./data_processing/run_overlap.sh verify         # offline E2E on sample data (NO Modal account, NO spend)
#   ./data_processing/run_overlap.sh run [args...]  # the real analysis, dispatched to Modal
#   ./data_processing/run_overlap.sh help
#
# CONFIG (all optional env vars; sensible defaults)
#   MODAL_PROFILE      named ~/.modal.toml profile to use (the one that can see the data's env)
#   OVERLAP_MODAL_ENV  Modal environment holding the tokenized data (default: alessio-dev, baked into the driver)
#   PYTHON             interpreter to use (default: auto-detect a .venv, else python3)
#
# EXAMPLES
#   # First: prove it works on your box, no account, no cost:
#   ./data_processing/run_overlap.sh verify
#   # Real run in the env that holds the data (this BILLS that account):
#   OVERLAP_MODAL_ENV=alessio-dev MODAL_PROFILE=myprofile ./data_processing/run_overlap.sh run
#   # Tune knobs (e.g. if the full corpus OOMs at the default 64 GiB):
#   ./data_processing/run_overlap.sh run --stride 32
#   ./data_processing/run_overlap.sh run --block-sizes 16        # run the heavy size alone
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; return; fi
  for c in "$REPO_ROOT/.venv/bin/python" "$HOME/.venv/bin/python"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  command -v python3 >/dev/null 2>&1 && { echo python3; return; }
  echo python
}
PY="$(pick_python)"

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

require_modal() {
  if ! "$PY" -c 'import modal' >/dev/null 2>&1; then
    echo "ERROR: the 'modal' package is not importable by: $PY" >&2
    echo "  pip install -r data_processing/requirements-overlap.txt   (then 'modal setup' to authenticate)" >&2
    echo "  or set PYTHON=/path/to/python with modal installed." >&2
    exit 1
  fi
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  verify)
    require_modal
    echo ">> Offline E2E on sample data (no Modal account, no spend) using: $PY"
    "$PY" -m pytest data_processing/tests/ -q "$@"
    echo ">> OK -- the parquet-read -> aggregate -> JSON path works on this machine."
    ;;
  run)
    require_modal
    : "${OVERLAP_MODAL_ENV:=}"
    [[ -n "$OVERLAP_MODAL_ENV" ]] && export MODAL_ENVIRONMENT="$OVERLAP_MODAL_ENV"
    export OVERLAP_MODAL_ENV
    echo ">> python=$PY  profile=${MODAL_PROFILE:-<active>}  modal_env=${MODAL_ENVIRONMENT:-<profile default>}  overlap_env=${OVERLAP_MODAL_ENV:-<driver default: alessio-dev>}"
    echo ">> Dispatching to Modal -- this BILLS the account that owns the target environment."
    "$PY" -m modal run data_processing/analyze_overlap_structure.py::analyze_all "$@"
    echo ">> Done. Verify the artifacts (trust the file, not the logs):"
    echo "     $PY -m modal volume get GORGO-glm5-completions overlap_structure /tmp/overlap_out"
    echo ">> Confirm teardown (nothing should be 'running'):"
    echo "     $PY -m modal app list"
    ;;
  help|-h|--help) usage ;;
  *) echo "unknown command: $cmd" >&2; echo; usage; exit 2 ;;
esac
