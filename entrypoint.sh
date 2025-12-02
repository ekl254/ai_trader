#!/bin/bash

# Start dashboard in background
python web/dashboard.py &
DASHBOARD_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down..."
    kill $DASHBOARD_PID 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start continuous trading (will exit when market closed, but we keep container alive)
while true; do
    echo "Starting trading engine..."
    python -m src.main continuous
    
    # If trading exits, wait before restarting (market might be closed)
    echo "Trading engine stopped. Waiting 60 seconds before restart..."
    sleep 60
done
