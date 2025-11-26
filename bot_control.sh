#!/bin/bash
# Trading Bot Control Script

PROJECT_DIR="/Users/enocklangat/Documents/AI/ai_trader"
PID_FILE="$PROJECT_DIR/logs/bot.pid"
LOG_FILE="$PROJECT_DIR/logs/bot_stdout.log"

cd "$PROJECT_DIR" || exit 1

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Bot is already running (PID: $PID)"
                exit 1
            else
                rm -f "$PID_FILE"
            fi
        fi
        
        echo "Starting trading bot with auto-restart..."
        # Use PYTHONPATH to ensure modules can be found
        # --auto-restart flag keeps bot running 24/7 and auto-starts when market opens
        PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" nohup python3 -u src/main.py continuous --auto-restart > "$LOG_FILE" 2>&1 &
        BOT_PID=$!
        echo "$BOT_PID" > "$PID_FILE"
        echo "Trading bot started with PID: $BOT_PID"
        echo "Bot will automatically start trading when market opens"
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "No PID file found. Bot may not be running."
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping trading bot (PID: $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            echo "Trading bot stopped"
        else
            echo "Bot process not found. Cleaning up PID file."
            rm -f "$PID_FILE"
        fi
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "running:$PID"
                exit 0
            else
                echo "stopped"
                rm -f "$PID_FILE"
                exit 1
            fi
        else
            echo "stopped"
            exit 1
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
