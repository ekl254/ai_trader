#!/bin/bash

# AI Trading Bot Auto-Starter
# This script ensures continuous trading runs automatically

PROJECT_DIR="/Users/enocklangat/Documents/AI/ai_trader 2"
LOG_FILE="$PROJECT_DIR/logs/trading.log"
PID_FILE="$PROJECT_DIR/trading_bot.pid"

cd "$PROJECT_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Trading bot already running (PID: $OLD_PID)"
        exit 0
    fi
fi

# Start trading bot
echo "Starting AI Trading Bot..."
nohup python -m src.main continuous >> "$LOG_FILE" 2>&1 &
NEW_PID=$!

# Save PID
echo $NEW_PID > "$PID_FILE"

echo "Trading bot started (PID: $NEW_PID)"
echo "Logs: tail -f $LOG_FILE"
