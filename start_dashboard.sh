#!/bin/bash
# Start the AI Trading Dashboard

cd "$(dirname "$0")"

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Start the dashboard
python web/dashboard.py
