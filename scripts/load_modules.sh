#!/bin/bash

# Source this file before activating the virtual environment. Alliance supplies
# PyArrow through the Arrow module rather than its Python wheelhouse.
PYTHON_MODULE=${RECAP_PYTHON_MODULE:-3.11}
ARROW_MODULE=${RECAP_ARROW_MODULE:-arrow}

if type module >/dev/null 2>&1; then
  module load "python/${PYTHON_MODULE}"
  module load "$ARROW_MODULE"
fi

