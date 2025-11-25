#!/bin/bash

# Stop AI Trading Bot

PROJECT_DIR="/Users/enocklangat/Documents/AI/ai_trader"
PID_FILE="$PROJECT_DIR/trading_bot.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Trading bot is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Trading bot is not running (PID $PID not found)"
    rm "$PID_FILE"
    exit 0
fi

echo "Stopping trading bot (PID: $PID)..."
kill "$PID"

# Wait for process to stop
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Trading bot stopped successfully"
        rm "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Force killing trading bot..."
    kill -9 "$PID"
    rm "$PID_FILE"
fi

echo "Trading bot stopped"
