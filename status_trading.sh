#!/bin/bash

# Check AI Trading Bot Status

PROJECT_DIR="/Users/enocklangat/Documents/AI/ai_trader"
PID_FILE="$PROJECT_DIR/trading_bot.pid"
LOG_FILE="$PROJECT_DIR/logs/trading.log"

echo "🤖 AI Trading Bot Status"
echo "========================"
echo

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Status: ❌ NOT RUNNING (no PID file)"
    exit 1
fi

PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Status: ❌ NOT RUNNING (PID $PID not found)"
    echo "Note: Stale PID file exists - run stop_trading.sh to clean up"
    exit 1
fi

echo "Status: ✅ RUNNING"
echo "PID: $PID"
echo

# Show process info
echo "Process Info:"
ps -p "$PID" -o pid,ppid,etime,rss,command | tail -n +2

echo
echo "Recent Activity (last 10 lines):"
tail -10 "$LOG_FILE" | grep -E "scanning_symbol|buy_executed|symbol_scored" | tail -5 || echo "No recent activity"

echo
echo "Commands:"
echo "  Stop:   ./stop_trading.sh"
echo "  Logs:   tail -f logs/trading.log"
