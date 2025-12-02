# 📊 Web Dashboard - Complete Guide

## 🚀 ONE-COMMAND START

```bash
cd /Users/enocklangat/Documents/AI/ai_trader
./start_dashboard.sh
```

Then open your browser to: **http://localhost:5000**

That's it! Everything is controlled from the web interface - no terminal commands needed!

---

## ✨ Dashboard Features

### 1. **Real-Time Market Status**
- Live market open/closed indicator
- Next open/close times
- Trading system status (idle/active)

### 2. **Portfolio Overview (Auto-Refreshing)**
- Portfolio value
- Buying power
- Cash balance
- Today's P&L with percentage

### 3. **Live Positions**
- All open positions with real-time prices
- Profit/loss for each position
- Entry prices and current values

### 4. **Recent Orders**
- Order history
- Fill status
- Execution prices

### 5. **Live Activity Log**
- Real-time trading decisions
- System events
- Error tracking

### 6. **One-Click Trading Controls**
- 🚀 **Start Trading Scan** - Analyzes stocks and places orders
- ⚙️ **Check Stop Losses** - Manages existing positions
- 🔻 **Close All Positions** - Emergency close all (EOD)
- 🔄 **Refresh Data** - Manual refresh (auto-refreshes every 5 seconds)

---

## 📖 How to Use

### Step 1: Start the Dashboard

```bash
./start_dashboard.sh
```

You'll see:
```
============================================================
🚀 AI Trading Dashboard Starting
============================================================

📊 Dashboard URL: http://localhost:5000
🌐 Network URL:   http://0.0.0.0:5000

✨ Features:
   • Live market status
   • Portfolio overview
   • Real-time positions
   • Order history
   • Live trading logs
   • One-click trading controls

Press Ctrl+C to stop
============================================================
```

### Step 2: Open Your Browser

Go to: **http://localhost:5000**

### Step 3: Use the Dashboard

**When Market Opens (9:30 AM EST):**

1. Click **🚀 Start Trading Scan**
   - System analyzes 20 liquid stocks
   - Calculates scores (technical + fundamental + sentiment)
   - Automatically places orders for top 5 stocks
   - Watch the live log for details!

2. Monitor your positions in real-time
   - See profit/loss update live
   - Watch for stop loss triggers

3. Click **⚙️ Check Stop Losses** anytime
   - Checks all positions
   - Closes losing positions (2% stop loss)
   - Takes profits (6% take profit)

4. At end of day, click **🔻 Close All Positions**
   - Closes everything (intraday trading only)

---

## 🎨 Dashboard Layout

```
┌─────────────────────────────────────────────────┐
│  🤖 AI Trading Dashboard                        │
│  Market: 🟢 OPEN  │  Trading: ● Idle            │
│  [Start Scan] [Check Stops] [Close All] [↻]    │
└─────────────────────────────────────────────────┘

┌────────────┐ ┌────────────┐ ┌─────────┐ ┌──────┐
│ Portfolio  │ │ Buying     │ │  Cash   │ │ P&L  │
│ $100,000   │ │ Power      │ │ $50,000 │ │+$500 │
└────────────┘ └────────────┘ └─────────┘ └──────┘

┌─────────────────────────────────────────────────┐
│ 📊 Open Positions                               │
│ AAPL  10 shares  $150 → $155  +$50 (+3.3%)     │
│ MSFT  5 shares   $300 → $310  +$50 (+3.3%)     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 📝 Recent Orders                                │
│ AAPL  BUY  10  MARKET  FILLED  $150.00         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 📋 Live Activity Log                            │
│ 9:31 AM - buy_executed: AAPL 10 shares @ $150  │
│ 9:30 AM - trading_session_started               │
└─────────────────────────────────────────────────┘
```

---

## ⚡ Quick Actions

### Morning Routine (9:30 AM)
1. Open dashboard: http://localhost:5000
2. Click **🚀 Start Trading Scan**
3. Watch the log as it analyzes and trades!

### During the Day
- Dashboard auto-refreshes every 5 seconds
- Click **⚙️ Check Stop Losses** every hour
- Monitor positions for profit/loss

### End of Day (3:00 PM)
- Click **🔻 Close All Positions**
- All positions close automatically

---

## 🔧 Advanced Usage

### Access from Other Devices

The dashboard is available on your local network:
- Find your IP: `ifconfig | grep inet`
- Access from phone/tablet: `http://YOUR_IP:5000`

### Keep Running in Background

```bash
# Run in background (optional)
nohup ./start_dashboard.sh > dashboard.log 2>&1 &

# Check if running
ps aux | grep dashboard

# Stop it
pkill -f dashboard.py
```

### Auto-Start on Mac Login

1. Open **System Preferences** → **Users & Groups**
2. Click **Login Items**
3. Click **+** and add `start_dashboard.sh`

---

## 🛟 Troubleshooting

### Dashboard Won't Start

```bash
# Make sure venv is activated
cd /Users/enocklangat/Documents/AI/ai_trader
source venv/bin/activate

# Check if Flask is installed
python -c "import flask; print('Flask OK')"

# If not, install it
pip install flask flask-cors
```

### Port 5000 Already in Use

Edit `web/dashboard.py` line 248:
```python
app.run(host="0.0.0.0", port=5001, debug=False)  # Change port to 5001
```

### Can't Click Buttons

- Check browser console for errors (F12)
- Make sure JavaScript is enabled
- Try a different browser (Chrome/Firefox)

### "Trading Already Running"

- Wait for current scan to finish (takes 2-5 minutes)
- Or refresh the page

---

## 📊 What Happens When You Click "Start Trading Scan"?

1. **Loads Stock Universe** (20 liquid S&P 500 stocks)
2. **For Each Stock:**
   - Fetches 60 days of price data
   - Calculates technical indicators (RSI, MACD, Bollinger Bands)
   - Gets fundamentals from EODHD (P/E, ROE, Debt/Equity)
   - Analyzes recent news sentiment
   - Combines into composite score (0-100)
3. **Selects Top 5** stocks with score ≥70
4. **Calculates Position Sizes** using 2% risk rule
5. **Executes Orders** on Alpaca Paper Trading
6. **Logs Everything** to activity log

All visible in real-time on the dashboard!

---

## 🎯 Dashboard Benefits

✅ **No Terminal Needed** - Everything from web browser  
✅ **Real-Time Updates** - Auto-refreshes every 5 seconds  
✅ **One-Click Trading** - Start scans with a button click  
✅ **Live Monitoring** - Watch positions update live  
✅ **Full Audit Trail** - See every decision in the log  
✅ **Mobile Friendly** - Access from phone/tablet  
✅ **Beautiful UI** - Dark theme, clean design  

---

## 🚀 Ready to Trade!

1. **Start Dashboard**: `./start_dashboard.sh`
2. **Open Browser**: http://localhost:5000
3. **Click Button**: 🚀 Start Trading Scan
4. **Watch Magic**: Live updates in real-time!

**That's it! No terminal commands needed - everything is in the browser!** 🎉

---

## 🌅 Premarket Features

### Premarket Scanner
The bot automatically scans for premarket opportunities between 4:00 AM - 9:30 AM ET:
- Identifies stocks with significant overnight gaps
- Detects unusual premarket volume activity
- Queues top candidates for market open execution

### Dashboard Premarket Section
The main dashboard displays:
- **Premarket Candidates**: Top 10 stocks queued for market open
- **Performance Stats**: Historical premarket trade analytics

### Premarket API Endpoints

#### GET /api/premarket/candidates
Returns current premarket candidates queued for market open.

**Response:**
```json
{
  "scan_time": "2025-12-02T08:30:00-05:00",
  "candidates": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "score": 85.5,
      "price": 142.50,
      "gap_pct": 2.3,
      "volume_ratio": 1.8
    }
  ],
  "count": 10,
  "is_today": true
}
```

#### GET /api/premarket/history
Returns historical premarket performance data (last 30 days).

**Query Parameters:**
- cputime         unlimited
filesize        unlimited
datasize        unlimited
stacksize       7MB
coredumpsize    0kB
addressspace    unlimited
memorylocked    unlimited
maxproc         2666
descriptors     unlimited (optional): Number of days to return (default: 30)

**Response:**
```json
[
  {
    "date": "2025-12-02",
    "scan_time": "2025-12-02T08:30:00-05:00",
    "candidates_count": 15,
    "executed": [
      {
        "symbol": "NVDA",
        "entry_price": 142.50,
        "exit_price": 145.20,
        "profit_loss_pct": 1.89
      }
    ]
  }
]
```

#### GET /api/premarket/stats
Returns aggregated premarket performance statistics.

**Response:**
```json
{
  "total_days": 30,
  "total_executed": 45,
  "total_closed": 40,
  "win_count": 28,
  "loss_count": 12,
  "win_rate": 70.0,
  "avg_profit_pct": 1.25,
  "total_profit_pct": 50.0,
  "best_trade": {
    "symbol": "TSLA",
    "profit_pct": 5.2,
    "date": "2025-11-25"
  },
  "worst_trade": {
    "symbol": "AMD",
    "profit_pct": -2.1,
    "date": "2025-11-20"
  }
}
```

### Data Files
-  - Current day's premarket candidates
-  - Historical premarket performance (90 days retention)

---

## Premarket Features

### Premarket Scanner
The bot automatically scans for premarket opportunities between 4:00 AM - 9:30 AM ET:
- Identifies stocks with significant overnight gaps
- Detects unusual premarket volume activity
- Queues top candidates for market open execution

### Dashboard Premarket Section
The main dashboard displays:
- **Premarket Candidates**: Top 10 stocks queued for market open
- **Performance Stats**: Historical premarket trade analytics

### Premarket API Endpoints

#### GET /api/premarket/candidates
Returns current premarket candidates queued for market open.

**Response:**
```json
{
  "scan_time": "2025-12-02T08:30:00-05:00",
  "candidates": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "score": 85.5,
      "price": 142.50,
      "gap_pct": 2.3,
      "volume_ratio": 1.8
    }
  ],
  "count": 10,
  "is_today": true
}
```

#### GET /api/premarket/history
Returns historical premarket performance data (last 30 days).

**Query Parameters:**
- limit (optional): Number of days to return (default: 30)

**Response:**
```json
[
  {
    "date": "2025-12-02",
    "scan_time": "2025-12-02T08:30:00-05:00",
    "candidates_count": 15,
    "executed": [
      {
        "symbol": "NVDA",
        "entry_price": 142.50,
        "exit_price": 145.20,
        "profit_loss_pct": 1.89
      }
    ]
  }
]
```

#### GET /api/premarket/stats
Returns aggregated premarket performance statistics.

**Response:**
```json
{
  "total_days": 30,
  "total_executed": 45,
  "total_closed": 40,
  "win_count": 28,
  "loss_count": 12,
  "win_rate": 70.0,
  "avg_profit_pct": 1.25,
  "total_profit_pct": 50.0,
  "best_trade": {
    "symbol": "TSLA",
    "profit_pct": 5.2,
    "date": "2025-11-25"
  },
  "worst_trade": {
    "symbol": "AMD",
    "profit_pct": -2.1,
    "date": "2025-11-20"
  }
}
```

### Data Files
- data/premarket_candidates.json - Current day premarket candidates
- data/premarket_history.json - Historical premarket performance (90 days retention)
