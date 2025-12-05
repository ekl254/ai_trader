#!/bin/bash
# Trading Bot Control Script
# Prevents duplicate bot instances and integrates with systemd

set -e

PROJECT_DIR="/root/ai_trader"
SERVICE_NAME="ai-trader"
LOG_FILE="$PROJECT_DIR/logs/trading.log"

cd "$PROJECT_DIR" || exit 1

# Function to check if ANY trading bot is running (regardless of how it was started)
check_any_bot_running() {
    local count=$(pgrep -f "src.main.*continuous" | wc -l)
    echo "$count"
}

# Function to get all bot PIDs
get_bot_pids() {
    pgrep -f "src.main.*continuous" 2>/dev/null || true
}

# Function to check systemd service status
check_systemd_service() {
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "running"
    else
        echo "stopped"
    fi
}

case "$1" in
    start)
        # Check if any bot is already running
        BOT_COUNT=$(check_any_bot_running)
        if [ "$BOT_COUNT" -gt 0 ]; then
            PIDS=$(get_bot_pids)
            echo "ERROR: Trading bot is already running!"
            echo "Running processes:"
            ps -fp $PIDS 2>/dev/null || true
            echo ""
            echo "Use '$0 status' for details or '$0 stop' to stop all instances."
            exit 1
        fi
        
        # Start via systemd (preferred method)
        echo "Starting trading bot via systemd..."
        sudo systemctl start "$SERVICE_NAME"
        sleep 2
        
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            PID=$(systemctl show --property MainPID --value "$SERVICE_NAME")
            echo "Trading bot started successfully (PID: $PID)"
            echo "View logs: tail -f $LOG_FILE"
            echo "Check status: $0 status"
        else
            echo "ERROR: Failed to start trading bot"
            systemctl status "$SERVICE_NAME" --no-pager
            exit 1
        fi
        ;;
        
    stop)
        echo "Stopping all trading bot instances..."
        
        # Stop systemd service first
        if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
            echo "Stopping systemd service..."
            sudo systemctl stop "$SERVICE_NAME"
        fi
        
        # Kill any rogue processes not managed by systemd
        PIDS=$(get_bot_pids)
        if [ -n "$PIDS" ]; then
            echo "Killing remaining bot processes: $PIDS"
            kill $PIDS 2>/dev/null || true
            sleep 2
            # Force kill if still running
            PIDS=$(get_bot_pids)
            if [ -n "$PIDS" ]; then
                echo "Force killing: $PIDS"
                kill -9 $PIDS 2>/dev/null || true
            fi
        fi
        
        echo "Trading bot stopped"
        ;;
        
    status)
        echo "=== Trading Bot Status ==="
        echo ""
        
        # Check systemd service
        echo "Systemd Service ($SERVICE_NAME):"
        if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
            PID=$(systemctl show --property MainPID --value "$SERVICE_NAME")
            echo "  Status: RUNNING (PID: $PID)"
            echo "  Started: $(systemctl show --property ExecMainStartTimestamp --value "$SERVICE_NAME")"
        else
            echo "  Status: STOPPED"
        fi
        echo ""
        
        # Check for any running bot processes
        BOT_COUNT=$(check_any_bot_running)
        echo "Running Bot Processes: $BOT_COUNT"
        if [ "$BOT_COUNT" -gt 0 ]; then
            echo ""
            PIDS=$(get_bot_pids)
            ps -fp $PIDS 2>/dev/null || true
        fi
        
        # Warn if duplicates detected
        if [ "$BOT_COUNT" -gt 1 ]; then
            echo ""
            echo "WARNING: Multiple bot instances detected!"
            echo "Run '$0 stop' then '$0 start' to fix."
        fi
        
        # Show if service is enabled
        echo ""
        if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
            echo "Auto-start on boot: ENABLED"
        else
            echo "Auto-start on boot: DISABLED"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 3
        $0 start
        ;;
        
    logs)
        echo "Showing live logs (Ctrl+C to exit)..."
        tail -f "$LOG_FILE"
        ;;
        
    check-duplicates)
        # Silent check for duplicates - useful for cron
        BOT_COUNT=$(check_any_bot_running)
        if [ "$BOT_COUNT" -gt 1 ]; then
            echo "DUPLICATE_DETECTED:$BOT_COUNT"
            PIDS=$(get_bot_pids)
            echo "PIDS:$PIDS"
            exit 1
        elif [ "$BOT_COUNT" -eq 1 ]; then
            echo "OK:1"
            exit 0
        else
            echo "STOPPED:0"
            exit 2
        fi
        ;;
        
    kill-duplicates)
        # Kill all but keep the systemd-managed process
        SERVICE_PID=$(systemctl show --property MainPID --value "$SERVICE_NAME" 2>/dev/null || echo "0")
        PIDS=$(get_bot_pids)
        
        if [ -z "$PIDS" ]; then
            echo "No bot processes running"
            exit 0
        fi
        
        KILLED=0
        for PID in $PIDS; do
            if [ "$PID" != "$SERVICE_PID" ] && [ "$PID" != "0" ]; then
                echo "Killing duplicate process: $PID"
                kill "$PID" 2>/dev/null || true
                KILLED=$((KILLED + 1))
            fi
        done
        
        if [ "$KILLED" -eq 0 ]; then
            echo "No duplicate processes found"
        else
            echo "Killed $KILLED duplicate process(es)"
        fi
        ;;
        
    *)
        echo "Trading Bot Control Script"
        echo ""
        echo "Usage: $0 {start|stop|status|restart|logs|check-duplicates|kill-duplicates}"
        echo ""
        echo "Commands:"
        echo "  start            - Start the trading bot (via systemd)"
        echo "  stop             - Stop all trading bot instances"
        echo "  status           - Show detailed status"
        echo "  restart          - Stop and start the bot"
        echo "  logs             - Follow live logs"
        echo "  check-duplicates - Check for duplicate instances (for cron)"
        echo "  kill-duplicates  - Kill duplicate instances, keep systemd one"
        exit 1
        ;;
esac
