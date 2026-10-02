#!/usr/bin/env bash
# Install into an owned temporary environment; dependency environments are read-only.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
backend_python=${1:?Usage: check-wheel.sh ABSOLUTE_BACKEND_PYTHON [--full]}
mode=${2:-smoke}
if [ "$mode" != smoke ] && [ "$mode" != --full ]; then echo "Unknown wheel mode" >&2; exit 2; fi
scratch=$(mktemp -d /tmp/lmex-wheel-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
uv build --project "$repo_dir/backend" --out-dir "$scratch/dist"
uv venv "$scratch/installed" --python "$backend_python"
wheel_python="$scratch/installed/bin/python"
dependency_site=$("$backend_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
installed_site=$("$wheel_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
# Add locked dependencies without executing another environment's editable .pth hooks.
printf '%s\n' "$dependency_site" > "$installed_site/locked-dependencies.pth"
uv pip install --python "$wheel_python" --no-deps "$scratch"/dist/*.whl
cd "$scratch"
LMEX_WHEEL_ROOT="$scratch/installed" "$wheel_python" - <<'PY'
import importlib
import os
from pathlib import Path
import torch
for module in ('llm_model_explorer', 'llm_model_explorer.app', 'llm_model_explorer.cli'):
    location = Path(importlib.import_module(module).__file__).resolve()
    assert location.is_relative_to(Path(os.environ['LMEX_WHEEL_ROOT'])), location
assert torch.version.cuda is None
PY
runtime=(
  "$repo_dir/backend/tests/test_cli.py::test_installed_help_needs_no_paths_or_torch"
  "$repo_dir/backend/tests/test_cli.py::test_installed_server_starts_and_stops_on_cpu"
  "$repo_dir/backend/tests/test_sessions.py::test_sessions_inventory_refresh_isolation_and_restart"
  "$repo_dir/backend/tests/test_tensor_data.py::test_exact_shapes_and_native_source_unchanged"
)
if [ "$mode" = --full ]; then runtime=("$repo_dir/backend/tests"); fi
PYTHONPATH="$repo_dir/backend/tests" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$wheel_python" -m pytest --import-mode=importlib "${runtime[@]}" --durations=10
