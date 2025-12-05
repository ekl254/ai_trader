#!/bin/bash
# Trading Bot Control Script
# Controls the ai_trader Docker container

set -e

CONTAINER_NAME="ai_trader"
PROJECT_DIR="/root/ai_trader"

cd "$PROJECT_DIR" || exit 1

# Function to check if container is running
is_container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if trading bot is running inside container
is_bot_running_in_container() {
    if is_container_running; then
        docker top "$CONTAINER_NAME" 2>/dev/null | grep -q "src.main.*continuous"
    else
        return 1
    fi
}

case "$1" in
    start)
        if is_container_running; then
            echo "Container '$CONTAINER_NAME' is already running"
            docker top "$CONTAINER_NAME"
            exit 0
        fi
        
        echo "Starting ai_trader container..."
        docker-compose up -d
        sleep 3
        
        if is_container_running; then
            echo "Container started successfully"
            docker top "$CONTAINER_NAME"
        else
            echo "ERROR: Failed to start container"
            docker-compose logs --tail=20
            exit 1
        fi
        ;;
        
    stop)
        if ! is_container_running; then
            echo "Container '$CONTAINER_NAME' is not running"
            exit 0
        fi
        
        echo "Stopping ai_trader container..."
        docker-compose stop
        echo "Container stopped"
        ;;
        
    restart)
        echo "Restarting ai_trader container..."
        docker-compose restart
        sleep 3
        
        if is_container_running; then
            echo "Container restarted successfully"
            docker top "$CONTAINER_NAME"
        else
            echo "ERROR: Failed to restart container"
            exit 1
        fi
        ;;
        
    status)
        echo "=== AI Trader Status ==="
        echo ""
        
        if is_container_running; then
            echo "Container: RUNNING"
            echo ""
            echo "Processes:"
            docker top "$CONTAINER_NAME"
            echo ""
            echo "Container info:"
            docker ps --filter "name=$CONTAINER_NAME" --format "  ID: {{.ID}}\n  Image: {{.Image}}\n  Created: {{.CreatedAt}}\n  Status: {{.Status}}"
        else
            echo "Container: STOPPED"
        fi
        ;;
        
    logs)
        echo "Showing live logs (Ctrl+C to exit)..."
        docker-compose logs -f --tail=100
        ;;
        
    rebuild)
        echo "Rebuilding and restarting container..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        sleep 3
        
        if is_container_running; then
            echo "Container rebuilt and started successfully"
            docker top "$CONTAINER_NAME"
        else
            echo "ERROR: Failed to start container after rebuild"
            exit 1
        fi
        ;;
        
    shell)
        if ! is_container_running; then
            echo "Container is not running"
            exit 1
        fi
        echo "Opening shell in container..."
        docker exec -it "$CONTAINER_NAME" /bin/bash
        ;;
        
    *)
        echo "AI Trader Control Script (Docker)"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|rebuild|shell}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the container"
        echo "  stop     - Stop the container"
        echo "  restart  - Restart the container"
        echo "  status   - Show container and process status"
        echo "  logs     - Follow live container logs"
        echo "  rebuild  - Rebuild and restart the container"
        echo "  shell    - Open a shell inside the container"
        exit 1
        ;;
esac
