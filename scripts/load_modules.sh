#!/bin/bash

# Source this file before activating the virtual environment. Alliance supplies
# PyArrow through the Arrow module rather than its Python wheelhouse.
PYTHON_MODULE=${RECAP_PYTHON_MODULE:-3.11}
ARROW_MODULE=${RECAP_ARROW_MODULE:-arrow}

if type module >/dev/null 2>&1; then
  module load "python/${PYTHON_MODULE}"
  module load "$ARROW_MODULE"
fi

# Installer caches live on scratch during login-node bootstrap, but compute jobs
# should use their node-local temporary disk for high-churn runtime files.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  RECAP_JOB_CACHE_ROOT="$SLURM_TMPDIR/recap_runtime"
  export TMPDIR="$RECAP_JOB_CACHE_ROOT/tmp"
  export XDG_CACHE_HOME="$RECAP_JOB_CACHE_ROOT/xdg/cache"
  export XDG_DATA_HOME="$RECAP_JOB_CACHE_ROOT/xdg/data"
  export PYTHONPYCACHEPREFIX="$RECAP_JOB_CACHE_ROOT/pycache"
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$PYTHONPYCACHEPREFIX"
fi
