#!/bin/bash
set -euo pipefail

ROOT=${1:?project root required}
VENV=${2:?venv path required}
PYTHON_MODULE=${RECAP_PYTHON_MODULE:-3.11}
SCRATCH_ROOT=${SCRATCH:?SCRATCH must be set on Narval}
CACHE_ROOT=${RECAP_CACHE_ROOT:-"$SCRATCH_ROOT/recap_pilots_cache"}

# virtualenv normally builds reusable pip/setuptools seed images below $HOME.
# On Alliance systems that can exhaust the much smaller home file/byte quota.
# Keep every high-churn installer/cache/temp path on scratch instead.
export VIRTUALENV_OVERRIDE_APP_DATA="$CACHE_ROOT/virtualenv"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg/cache"
export XDG_DATA_HOME="$CACHE_ROOT/xdg/data"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p \
  "$VIRTUALENV_OVERRIDE_APP_DATA" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_DATA_HOME" \
  "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR" \
  "$(dirname "$VENV")"

if type module >/dev/null 2>&1; then
  # Narval's CCconfig/StdEnv stack is sticky; purging it only prints a scary but
  # harmless warning. Loading Python directly retains the supported site stack.
  module load "python/${PYTHON_MODULE}"
fi

# A quota failure can leave an executable-looking but unusable environment.
# Preserve it for inspection and recreate cleanly instead of blindly sourcing it.
if [[ -e "$VENV" ]] && { [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; }; then
  incomplete="${VENV}.incomplete.$(date +%Y%m%dT%H%M%S)"
  echo "Moving incomplete virtual environment to $incomplete" >&2
  mv "$VENV" "$incomplete"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download --app-data "$VIRTUALENV_OVERRIDE_APP_DATA" "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

source "$VENV/bin/activate"
if [[ -n "${RECAP_ALLOW_INDEX:-}" ]]; then
  python -m pip install "$ROOT[test]"
else
  python -m pip install --no-index "$ROOT[test]"
fi

echo "ReCAP environment: $VENV"
echo "Installer/cache root: $CACHE_ROOT"
