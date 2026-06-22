#!/bin/bash
set -e
source .venv/bin/activate
python main_a2c.py train-neurogym "$@"
