#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON:-python3}"
TTS_ALLOW_LOGIN_NODE=1 "$PYTHON_BIN" -m unittest discover -s collect_de_num_data/tests -v
"$PYTHON_BIN" -m unittest discover -s preprocess_num_of_datasets/tests -v
