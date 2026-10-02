#!/usr/bin/env bash
# Integrated gate for CI. Dedicated workflows own component checks on main.
set -euo pipefail
main_ci=false
plan=''
portfolio=full
while [ "$#" -gt 0 ]; do
  case "$1" in
    --routine) portfolio=routine; shift ;;
    --main-ci) main_ci=true; shift ;;
    --plan)
      if [ "$#" -lt 2 ] || [ -n "$plan" ]; then
        echo "Usage: $0 [--routine] [--main-ci] [--plan FILE]" >&2; exit 2
      fi
      plan=$(realpath "$2"); shift 2 ;;
    *) echo "Usage: $0 [--routine] [--main-ci] [--plan FILE]" >&2; exit 2 ;;
  esac
done

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export LMEX_EVIDENCE_DIR=${LMEX_EVIDENCE_DIR:-$(mktemp -d /tmp/lmex-integration-evidence-XXXXXX)}
mkdir -p "$LMEX_EVIDENCE_DIR"
export LMEX_TEST_PORTFOLIO=$portfolio
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python_bin="$repo_dir/backend/.venv/bin/python"
network_args=(acceptance)
if [ -n "$plan" ]; then
  # Command substitution must propagate errors before mapfile can hide them.
  selected=$(python3 .github/scripts/validation_selector.py --plan "$plan" --network-targets)
  mapfile -t network_args <<< "$selected"
fi

# Full local execution keeps contract checks. CI delegates to exact-target API/UI gates.
if [ "$main_ci" = false ]; then
  "${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py
fi

# acceptance/ is not covered by backend-ci's backend/** path filter.
backend/.venv/bin/ruff check --config backend/pyproject.toml acceptance .github/scripts/validation_selector.py
backend/.venv/bin/ruff format --check --config backend/pyproject.toml acceptance .github/scripts/validation_selector.py
"$python_bin" -m pytest "${network_args[@]}" -ra --durations=25 -o junit_family=legacy --junitxml="$LMEX_EVIDENCE_DIR/network.xml"

(
  cd ui
  if [ "$main_ci" = false ]; then
    npm run api:check
  fi
  # Build the production UI from this checkout; never reuse another SHA's build.
  npm run build
  if [ -n "$plan" ]; then
    selector_args=()
    if [ "$main_ci" = true ]; then selector_args=(--main); fi
    python3 ../.github/scripts/validation_selector.py --plan "$plan" --run integration "${selector_args[@]}"
  else
    browser_args=()
    if [ "$main_ci" = false ]; then browser_args=(-- --project=dpr1); fi
    # Main retains both configured DPR projects and their native density tags.
    xvfb-run -a npm run test:acceptance "${browser_args[@]}"
  fi
)

echo "Integration acceptance evidence: $LMEX_EVIDENCE_DIR"
