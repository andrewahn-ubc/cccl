#!/bin/bash
set -euo pipefail

ROOT=${1:?project root required}
VENV=${2:?venv path required}
PYTHON_MODULE=${RECAP_PYTHON_MODULE:-3.11}

if type module >/dev/null 2>&1; then
  module purge
  module load "python/${PYTHON_MODULE}"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

source "$VENV/bin/activate"
if [[ -n "${RECAP_ALLOW_INDEX:-}" ]]; then
  python -m pip install --editable "$ROOT[test]"
else
  python -m pip install --no-index --editable "$ROOT[test]"
fi

