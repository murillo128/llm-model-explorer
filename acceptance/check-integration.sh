#!/usr/bin/env bash
# Fast integrated gate for pull requests and epic integration branches.
# Dedicated backend/UI workflows own lint, type, unit and component-browser suites.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export LMEX_EVIDENCE_DIR=${LMEX_EVIDENCE_DIR:-$(mktemp -d /tmp/lmex-integration-evidence-XXXXXX)}
mkdir -p "$LMEX_EVIDENCE_DIR"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python_bin="$repo_dir/backend/.venv/bin/python"

"${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py
"${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py --write
git diff --exit-code -- api/fixtures

# acceptance/ is not covered by backend-ci's backend/** path filter.
backend/.venv/bin/ruff check --config backend/pyproject.toml acceptance
backend/.venv/bin/ruff format --check --config backend/pyproject.toml acceptance
"$python_bin" -m pytest acceptance -ra -o junit_family=legacy --junitxml="$LMEX_EVIDENCE_DIR/network.xml"

(
  cd ui
  npm run api:check
  npm run api:generate
  git diff --exit-code -- src/api/generated
  npm run build
  # DPR 1 exercises the real production browser/backend path on intermediate
  # branches; the exhaustive check on main retains DPR 1 and DPR 2.
  xvfb-run -a npm run test:acceptance -- --project=dpr1
)

echo "Integration acceptance evidence: $LMEX_EVIDENCE_DIR"
