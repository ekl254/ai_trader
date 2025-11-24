#!/bin/bash
# Run script for AI Trading System

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the trading system (no venv activation - use system python)
python src/main.py "$@"
