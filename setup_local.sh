#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv .venv-pathflow
.venv-pathflow/bin/python bootstrap.py
.venv-pathflow/bin/python doctor.py
printf '\nReady. Launch with: PORT=5001 .venv-pathflow/bin/python run.py\n'
