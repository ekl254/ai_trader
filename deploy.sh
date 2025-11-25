#!/bin/bash

# VPS Details
VPS_USER="root"
VPS_IP="69.62.64.51"
VPS_DIR="/root/ai_trader"

echo "🚀 Deploying to VPS ($VPS_IP)..."

# 1. Sync Files
# Exclude venv, git, logs, data (keep production data), and cache
echo "📦 Syncing files..."
rsync -avz \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'logs' \
    --exclude 'data' \
    --exclude '.DS_Store' \
    --exclude '.env' \
    ./ $VPS_USER@$VPS_IP:$VPS_DIR/

# 2. Rebuild and Restart Docker
echo "🔄 Rebuilding and restarting container..."
ssh $VPS_USER@$VPS_IP "cd $VPS_DIR && docker compose up -d --build"

echo "✅ Deployment complete!"
echo "📊 Dashboard: http://$VPS_IP:8082"
echo "📝 Logs: ssh $VPS_USER@$VPS_IP 'cd $VPS_DIR && docker compose logs -f'"
