#!/usr/bin/env bash
set -euo pipefail
support=$(cd "$(dirname "$0")/.." && pwd)
cd "$GITHUB_WORKSPACE/target"
expected=7f289d70215420a8b244f6f5c428ed0f660eb486
test "$(git rev-parse HEAD)" = "$expected"
test -z "$(git status --porcelain)"
cat "$support"/.transfer/patch.{0,1,2,3} > "$RUNNER_TEMP/source.patch.gz"
echo "c04710e324f506b564822b6647b00eeb41e54263585d849434035caaa0beba4c  $RUNNER_TEMP/source.patch.gz" | sha256sum --check
gzip -dc "$RUNNER_TEMP/source.patch.gz" > "$RUNNER_TEMP/source.patch"
git apply --check "$RUNNER_TEMP/source.patch"
git apply --index "$RUNNER_TEMP/source.patch"
export PATH="$PWD/backend/.venv/bin:$PATH"
uv sync --locked --project backend --python 3.12
uv venv "$RUNNER_TEMP/model-json-api" --python 3.12
uv pip install --python "$RUNNER_TEMP/model-json-api/bin/python" -r api/requirements.txt
python - <<'PY'
from pathlib import Path
path = Path('backend/src/llm_model_explorer/architecture_service.py')
source = path.read_text()
old = '"Model-supplied architecture.json is invalid or unsupported; no fallback was used."'
new = '("Model-supplied architecture.json is invalid or unsupported; "\n                         "no fallback was used.")'
assert source.count(old) == 1
path.write_text(source.replace(old, new))
PY
python backend/scripts/generate_architecture_records.py
PYTHONPATH=backend/src python -m llm_model_explorer.architecture_analysis.model_defined_schema > docs/spec/backend/architecture-definition.schema.json
npm ci --prefix ui
npm run api:generate --prefix ui
"$RUNNER_TEMP/model-json-api/bin/python" api/validate_contract.py --write
(
  cd backend
  mapfile -t pyfiles < <(grep -E '^backend/.*\.py$' "$support/.transfer/paths.txt" | sed 's@^backend/@@')
  ruff format "${pyfiles[@]}"
  ruff check --fix "${pyfiles[@]}"
  ruff format "${pyfiles[@]}"
  ruff check .
  ruff format --check .
  mypy
)
PYTHONPATH=backend/src python -m pytest backend/tests/test_model_defined_architecture.py backend/tests/test_model_defined_service.py -ra
npm run api:check --prefix ui
npm run typecheck --prefix ui
npm run lint --prefix ui
(cd ui && npm test -- src/architecture-explorer/ArchitectureExplorer.test.tsx)
git diff --check
mapfile -t paths < "$support/.transfer/paths.txt"
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then git add -- "$path"; fi
done
# Deletion of the temporary helper is already staged by git apply --index.
git diff --cached --name-only | sort > "$RUNNER_TEMP/staged-paths"
sort "$support/.transfer/paths.txt" > "$RUNNER_TEMP/allowed-paths"
test -z "$(comm -23 "$RUNNER_TEMP/staged-paths" "$RUNNER_TEMP/allowed-paths")"
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git commit -m 'feat(architecture): load model-owned JSON definitions'
git push origin HEAD:refs/heads/feat/model-owned-architecture-json
git rev-parse HEAD > "$RUNNER_TEMP/published-head.txt"
