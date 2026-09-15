#!/usr/bin/env bash
# Integrated gate for CI. Dedicated workflows own component checks on main.
set -euo pipefail
main_ci=false
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--main-ci]" >&2
  exit 2
fi
case "${1:-}" in
  '') ;;
  --main-ci) main_ci=true ;;
  *) echo "Usage: $0 [--main-ci]" >&2; exit 2 ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export LMEX_EVIDENCE_DIR=${LMEX_EVIDENCE_DIR:-$(mktemp -d /tmp/lmex-integration-evidence-XXXXXX)}
mkdir -p "$LMEX_EVIDENCE_DIR"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python_bin="$repo_dir/backend/.venv/bin/python"

# Preserve the existing PR/epic gate. Main delegates these checks to api-contract.
if [ "$main_ci" = false ]; then
  "${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py
  "${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py --write
  git diff --exit-code -- api/fixtures
fi

# acceptance/ is not covered by backend-ci's backend/** path filter.
backend/.venv/bin/ruff check --config backend/pyproject.toml acceptance
backend/.venv/bin/ruff format --check --config backend/pyproject.toml acceptance
"$python_bin" -m pytest acceptance -ra -o junit_family=legacy --junitxml="$LMEX_EVIDENCE_DIR/network.xml"

(
  cd ui
  if [ "$main_ci" = false ]; then
    npm run api:check
    npm run api:generate
    git diff --exit-code -- src/api/generated
  fi
  # Build the production UI from this checkout; never reuse another SHA's build.
  npm run build
  browser_args=()
  if [ "$main_ci" = false ]; then
    browser_args=(-- --project=dpr1)
  fi
  # Main retains both configured DPR projects, with real backend/browser tests.
  xvfb-run -a npm run test:acceptance "${browser_args[@]}"
)

echo "Integration acceptance evidence: $LMEX_EVIDENCE_DIR"
