#!/bin/bash
# Double-click this once (Mac). It prepares Python if needed and connects the
# ICPS Focus Alerts extension to the local detector.
cd "$(dirname "$0")"
echo "ICPS Focus Alerts - one-time setup"
if [ ! -x .venv/bin/python ]; then
  echo "Creating the Python environment (first time only, a few minutes)..."
  python3 -m venv .venv && .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt \
    || { echo "Setup failed - see the messages above."; read -r -p "Press Enter to close"; exit 1; }
fi
.venv/bin/python install_native_host.py "$@"
read -r -p "Press Enter to close"
