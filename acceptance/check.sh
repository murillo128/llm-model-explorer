#!/usr/bin/env bash
# Run from any directory after installing the locked backend/UI dependencies.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export LMEX_EVIDENCE_DIR=${LMEX_EVIDENCE_DIR:-$(mktemp -d /tmp/lmex-evidence-XXXXXX)}
mkdir -p "$LMEX_EVIDENCE_DIR"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python_bin="$repo_dir/backend/.venv/bin/python"
"${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py
"${LMEX_CONTRACT_PYTHON:-$repo_dir/api/.venv/bin/python}" api/validate_contract.py --write
git diff --exit-code -- api/fixtures
backend/.venv/bin/ruff check --config backend/pyproject.toml acceptance
backend/.venv/bin/ruff format --check --config backend/pyproject.toml acceptance
(
  cd backend
  .venv/bin/ruff check .
  .venv/bin/ruff format --check .
  .venv/bin/mypy
  .venv/bin/pytest -o junit_family=legacy --junitxml="$LMEX_EVIDENCE_DIR/backend.xml"
)
"$python_bin" -m pytest acceptance -ra -o junit_family=legacy --junitxml="$LMEX_EVIDENCE_DIR/network.xml"
(
  cd ui
  npm run api:check
  npm run api:generate
  git diff --exit-code -- src/api/generated
  npm run typecheck
  npm run lint
  npm test -- --reporter=default --reporter=json --outputFile="$LMEX_EVIDENCE_DIR/unit.json"
  npm run build
  PLAYWRIGHT_JSON_OUTPUT_FILE="$LMEX_EVIDENCE_DIR/component-browser.json" \
    xvfb-run -a npm run test:browser -- --reporter=list,json
  xvfb-run -a npm run test:acceptance
)
"$python_bin" -m acceptance.report "$LMEX_EVIDENCE_DIR"
