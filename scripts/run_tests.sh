#!/usr/bin/env bash
set -euo pipefail

# Ensure run/logs exists
mkdir -p run/logs

# Point device logging to the test log file
export ATLANTICO_DEVICE_LOG="$(pwd)/run/logs/tests.log"
echo "Writing device logs to: $ATLANTICO_DEVICE_LOG"

# Run pytest with any provided args via the project's venv if present
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python -m pytest "$@"
else
  python -m pytest "$@"
fi
