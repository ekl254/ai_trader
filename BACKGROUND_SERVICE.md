# AI Trading Bot - Background Service Setup

The trading bot now runs as an **independent background service** that operates continuously when the market is open, **even when the dashboard is closed**.

---

## 🚀 Quick Start

### Local Machine (macOS)

**Start the bot:**
```bash
./start_trading.sh
```

**Check status:**
```bash
./status_trading.sh
```

**Stop the bot:**
```bash
./stop_trading.sh
```

---

## 📋 Available Commands

### 1. **Start Trading Bot**
```bash
./start_trading.sh
```
- Starts continuous trading in the background
- Creates PID file to track the process
- Logs to `logs/trading.log`

### 2. **Check Status**
```bash
./status_trading.sh
```
Shows:
- ✅ Running status
- Process ID (PID)
- Memory usage
- Recent trading activity

### 3. **Stop Trading Bot**
```bash
./stop_trading.sh
```
- Gracefully stops the trading bot
- Cleans up PID file
- Force kills if needed

### 4. **View Live Logs**
```bash
tail -f logs/trading.log
```

---

## 🔄 Auto-Start on System Boot (macOS)

To have the trading bot start automatically when you log in:

```bash
# Install LaunchAgent
cp com.aitrader.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.aitrader.bot.plist
```

**Verify it's installed:**
```bash
launchctl list | grep aitrader
```

**Uninstall (if needed):**
```bash
launchctl unload ~/Library/LaunchAgents/com.aitrader.bot.plist
rm ~/Library/LaunchAgents/com.aitrader.bot.plist
```

---

## 🖥️ VPS Deployment (Linux with systemd)

The VPS is already configured with systemd for automatic startup.

### Service Commands

```bash
# Check status
sudo systemctl status ai-trader

# Start service
sudo systemctl start ai-trader

# Stop service
sudo systemctl stop ai-trader

# Restart service
sudo systemctl restart ai-trader

# View logs
sudo journalctl -u ai-trader -f

# Or view file logs
tail -f /root/ai_trader/logs/trading.log
```

### Enable/Disable Auto-Start

```bash
# Enable auto-start on boot
sudo systemctl enable ai-trader

# Disable auto-start
sudo systemctl disable ai-trader
```

---

## 🤔 How It Works

### Independent Operation
- Trading bot runs as a **separate process** from the dashboard
- Dashboard is only for monitoring and manual controls
- Bot continues running even if:
  - Dashboard is closed
  - Browser is closed
  - You log out (if LaunchAgent/systemd is configured)

### Market Hours Detection
- Bot checks if market is open every 30 seconds
- Automatically starts trading when market opens
- Stops scanning when market closes (but process stays alive)

### Scanning Schedule
- **Full universe scan**: Every 15 minutes
- **Position management**: Every 2 minutes (stop losses, take profits)
- **Max new positions per scan**: 3

### Crash Recovery
- **macOS**: LaunchAgent can restart on failure (set `KeepAlive` to true)
- **VPS**: systemd automatically restarts on failure (`Restart=on-failure`)

---

## 📁 File Locations

```
ai_trader/
├── start_trading.sh          # Start script
├── stop_trading.sh           # Stop script
├── status_trading.sh         # Status checker
├── trading_bot.pid           # Process ID (auto-generated)
├── com.aitrader.bot.plist    # macOS LaunchAgent
├── ai-trader.service         # Linux systemd service
└── logs/
    ├── trading.log           # Main trading log
    ├── launchd.log          # macOS startup logs
    └── trading_error.log    # VPS error logs
```

---

## ⚠️ Important Notes

1. **Only ONE instance should run at a time**
   - Scripts prevent multiple instances
   - Check status before starting manually

2. **Dashboard is separate**
   - Dashboard runs on port 8082
   - You can start/stop dashboard independently:
     ```bash
     python web/dashboard.py
     ```

3. **Logs grow over time**
   - Monitor `logs/trading.log` size
   - Consider log rotation for production

4. **Stop before system updates**
   - Run `./stop_trading.sh` before:
     - System updates
     - Python updates
     - Code changes (restart after)

---

## 🐛 Troubleshooting

### Bot won't start
```bash
# Check if already running
./status_trading.sh

# Check for errors
tail -50 logs/trading.log
```

### Stale PID file
```bash
# Clean up and restart
rm trading_bot.pid
./start_trading.sh
```

### Bot not trading
```bash
# Check market status
./status_trading.sh

# Verify credentials in .env
cat .env | grep ALPACA
```

### High CPU/Memory usage
```bash
# Check process stats
./status_trading.sh

# Restart if needed
./stop_trading.sh
./start_trading.sh
```

---

## 🎯 Summary

**Before (Dashboard-dependent):**
- ❌ Had to click "Start Continuous Trading" in dashboard
- ❌ Stopped when dashboard closed
- ❌ No auto-start on boot

**After (Independent Service):**
- ✅ Runs automatically in background
- ✅ Survives dashboard close/browser close
- ✅ Auto-starts on system boot (if configured)
- ✅ systemd manages it on VPS
- ✅ Simple start/stop/status commands

---

## 📊 Monitoring

**Local:**
- Dashboard: http://localhost:8082 (admin/admin)
- Status: `./status_trading.sh`

**VPS:**
- Dashboard: http://69.62.64.51:8082 (admin/admin) *[needs firewall rule]*
- Status: `sudo systemctl status ai-trader`

Both show real-time positions, orders, and trading activity!
