# 🚀 Deploying AI Trader to VPS

This guide explains how to deploy your AI Trading Bot to a Virtual Private Server (VPS) so it runs continuously, 24/7.

## Prerequisites

- A VPS (e.g., DigitalOcean, AWS, Linode, Vultr)
  - **OS**: Ubuntu 22.04 LTS (Recommended)
  - **Specs**: Minimum 2GB RAM (4GB recommended for FinBERT), 1 vCPU
- SSH access to your VPS

## Step 1: Install Docker on VPS

Connect to your VPS via SSH and run the following commands to install Docker:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoids using sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt install -y docker-compose-plugin
```

## Step 2: Upload Code to VPS

You can use `scp` (Secure Copy) to upload your project folder from your local machine to the VPS.

Run this command **from your local machine** (replace `user@your-vps-ip` with your actual details):

```bash
# Copy the entire ai_trader folder to VPS
scp -r /Users/enocklangat/Documents/AI/ai_trader user@your-vps-ip:~/ai_trader
```

*Alternatively, if you use GitHub:*
1. Push your code to a private GitHub repo.
2. `git clone` it on your VPS.

## Step 3: Configure Environment

1. SSH into your VPS:
   ```bash
   ssh user@your-vps-ip
   ```

2. Navigate to the folder:
   ```bash
   cd ~/ai_trader
   ```

3. Ensure your `.env` file is present and has the correct API keys.
   ```bash
   nano .env
   ```
   *(Paste your keys if they aren't there)*

## Step 4: Build and Run

We use Docker Compose to build the container and run it in the background.

```bash
# Build and start the container in detached mode (-d)
docker compose up -d --build
```

## Step 5: Verify Deployment

1. **Check Status**:
   ```bash
   docker compose ps
   ```
   You should see `ai_trader` listed as `Up`.

2. **View Logs**:
   ```bash
   docker compose logs -f
   ```
   This will show you the live logs. Press `Ctrl+C` to exit logs (the bot keeps running).

3. **Access Dashboard**:
   Open your browser and go to:
   `http://your-vps-ip:8080`

## Managing the Bot

- **Stop the bot**:
  ```bash
  docker compose down
  ```

- **Restart the bot**:
  ```bash
  docker compose restart
  ```

- **Update code**:
  1. Upload new files (or `git pull`).
  2. Rebuild and restart:
     ```bash
     docker compose up -d --build
     ```

## 💡 Why Docker?
- **Reliability**: If the bot crashes, Docker automatically restarts it (`restart: unless-stopped`).
- **Consistency**: Runs exactly the same as on your machine.
- **Cleanliness**: Doesn't clutter your VPS with Python libraries.
