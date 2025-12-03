# Continuous Trading System

## 🔄 How It Works Now

The system now runs **fully automatically** during market hours with continuous buying and selling:

### What Changed:

#### Before (Manual):
- ❌ Click "Start Trading Scan" → Scan 20 stocks → Place trades → **STOP**
- ❌ Had to click again for another scan
- ❌ Limited to 20 stocks
- ❌ Position management only on manual click

#### After (Continuous):
- ✅ Click "Start Continuous Trading" → **Runs all day automatically**
- ✅ Scans **ALL S&P 500 stocks** (not just 20)
- ✅ Automatically buys when opportunities appear
- ✅ Automatically sells on stop-loss or take-profit
- ✅ Keeps running until market closes or you click "Stop"

---

## 🎯 Trading Schedule

### Full Universe Scan (Every 15 Minutes)
- **What**: Scans ALL S&P 500 stocks in your universe
- **When**: Every 15 minutes during market hours
- **Action**: Identifies high-scoring opportunities
- **Buys**: Up to 3 new positions per scan (if slots available)
- **Limit**: Maximum 10 total positions (configurable)

### Position Management (Every 2 Minutes)
- **What**: Checks all open positions
- **When**: Every 2 minutes
- **Action**: 
  - Triggers stop-loss at -2% loss
  - Takes profit at +6% gain
  - Monitors for exit signals
- **Result**: Automatically sells positions when conditions met

### Background Loop (Every 30 Seconds)
- Checks if market is still open
- Verifies system health
- Coordinates scan and management timing

---

## 📊 Trading Flow

```
Market Opens (9:30 AM)
    ↓
Click "Start Continuous Trading"
    ↓
╔════════════════════════════════════╗
║   CONTINUOUS LOOP BEGINS           ║
╠════════════════════════════════════╣
║                                    ║
║  Every 15 minutes:                 ║
║  ├─ Scan ALL S&P 500 stocks       ║
║  ├─ Score each ticker              ║
║  ├─ Find top opportunities         ║
║  └─ Buy up to 3 new positions     ║
║                                    ║
║  Every 2 minutes:                  ║
║  ├─ Check all positions            ║
║  ├─ Trigger stop-loss (-2%)       ║
║  ├─ Take profit (+6%)             ║
║  └─ Close positions as needed     ║
║                                    ║
║  Every 30 seconds:                 ║
║  └─ Verify market is still open   ║
║                                    ║
╚════════════════════════════════════╝
    ↓
Market Closes (4:00 PM) → Auto Stops
```

---

## 🎮 Dashboard Controls

### Start Continuous Trading Button
**Action**: Launches continuous trading loop
- Scans full S&P 500 universe
- Buys when good opportunities appear
- Sells when stop-loss or take-profit hit
- Runs until market closes or manually stopped

**What You'll See**:
```
Status: "Continuous Trading Active" (pulsing green indicator)
Logs: Scan updates every 15 minutes
Logs: Position checks every 2 minutes
Positions: Automatically updated as trades execute
```

### Stop Trading Button
**Action**: Gracefully stops continuous trading
- Completes current operation
- Stops scanning for new positions
- Keeps existing positions open
- Can be restarted anytime

### Close All Positions Button
**Action**: Emergency close (end of day)
- Immediately closes all open positions
- Market orders for instant execution
- Use at end of day or emergency stop

---

## 📈 Position Management

### Automatic Buying
- **Check**: Every scan (15 minutes)
- **Condition**: Stock score ≥ 55, technical ≥ 40, sentiment ≥ 40
- **Limit**: Max 3 new positions per scan
- **Max Total**: 10 positions at once
- **Size**: 2% of portfolio per trade (with 10% max position size cap)

### Automatic Selling
- **Check**: Every 2 minutes
- **Stop Loss**: -2% from entry → Sell immediately
- **Take Profit**: +6% from entry → Sell immediately
- **Position Review**: Continuous monitoring

### Smart Slot Management
- **Available Slots**: Calculated dynamically
- **Example**: If 7 positions open, 3 slots available
- **Behavior**: Only buys into available slots
- **Result**: Never exceeds max position limit

---

## 📋 Scan Behavior

### Universe Coverage
```
Total Stocks: ~500 S&P 500 companies
Scan Rate: ~0.3 seconds per stock
Full Scan Time: ~2.5 minutes
Frequency: Every 15 minutes
Daily Scans: ~25 full universe scans
```

### API Rate Limits
- **Delay**: 0.3 seconds between stocks
- **Purpose**: Respect API rate limits
- **Result**: No API throttling issues

### Error Handling
- **Failed Scan**: Logs error, continues to next stock
- **API Error**: Retries after 1 minute
- **Market Closed**: Stops automatically

---

## 🔍 What You'll See

### Dashboard Updates
```
8:45 AM - Market opens soon
9:30 AM - Click "Start Continuous Trading"
9:30 AM - [LOG] Continuous trading started, universe_size: 500
9:31 AM - [LOG] Starting full universe scan #1, symbols: 500
9:33 AM - [LOG] Buy candidate found: AAPL, score: 67.5
9:33 AM - [LOG] Buy executed: AAPL, shares: 50, price: $178.23
9:35 AM - [LOG] Running position management check
9:45 AM - [LOG] Starting full universe scan #2, symbols: 500
9:47 AM - [LOG] Buy candidate found: MSFT, score: 71.2
...continues all day...
3:00 PM - [LOG] Take profit triggered: AAPL, gain: +6.2%
3:00 PM - [LOG] Sell executed: AAPL, shares: 50, profit: $662
4:00 PM - [LOG] Market closed, stopping trading
4:00 PM - [LOG] Continuous trading stopped, total_scans: 25
```

### Analysis Page Updates
- Shows ALL scanned tickers
- Updated every scan
- Filters: Traded, Rejected, High Score, Low Score
- Timestamps show when each ticker was analyzed

---

## ⚙️ Configuration

### Scan Intervals
```python
# src/main.py:132-154
scan_interval = 900 seconds  # 15 minutes
manage_interval = 120 seconds  # 2 minutes
check_interval = 30 seconds  # 30 seconds
```

### Position Limits
```python
# config/config.py:34-37
max_positions = 10  # Maximum open positions
risk_per_trade = 0.02  # 2% per trade
max_position_size = 0.10  # 10% max per position
```

### Thresholds
```python
# config/config.py:40-41
min_composite_score = 72.5  # Raised for higher quality entries
min_factor_score = 40.0
```

---

## 🚀 How to Use

### 1. Start the Dashboard
```bash
./start_dashboard.sh
```
Open: http://localhost:8080

### 2. Wait for Market Open
- Dashboard shows: "Market: 🔴 CLOSED"
- Wait until: "Market: 🟢 OPEN"

### 3. Start Continuous Trading
- Click: **"🚀 Start Continuous Trading"**
- Watch status change to: **"Continuous Trading Active"**
- See green pulsing indicator

### 4. Monitor Progress
- **Dashboard**: Live positions, P&L, orders
- **Analysis Page**: All scanned tickers with scores
- **Activity Log**: Every buy, sell, scan update

### 5. End of Day
- Option A: Let it auto-stop when market closes
- Option B: Click **"⏸️ Stop Trading"** manually
- Option C: Click **"🔻 Close All Positions"** to close everything

---

## 📊 Expected Results

### During Market Hours
- **Scans per day**: ~25 full universe scans
- **Stocks analyzed**: ~12,500 ticker analyses per day
- **New positions**: Up to 75 opportunities identified (3 per scan)
- **Actual trades**: Limited by available slots and capital
- **Position checks**: ~200 stop-loss/take-profit checks per day

### Example Day
```
9:30 AM - Started with $100K
10:00 AM - 5 positions, $50K deployed
12:00 PM - Sold AAPL (+6%), bought TSLA
2:00 PM - 8 positions, $80K deployed
3:00 PM - Sold 2 positions (1 profit, 1 stop-loss)
4:00 PM - Day ends, 6 positions open
Result: +2.3% day, 15 trades executed
```

---

## 🛡️ Safety Features

### Automatic Protections
- ✅ Stops when market closes
- ✅ Never exceeds max positions
- ✅ Respects API rate limits
- ✅ Position size caps at 10%
- ✅ Stop-loss on every position
- ✅ Error recovery and retries

### Manual Controls
- ⏸️ Stop button (graceful shutdown)
- 🔻 Emergency close all button
- 🔄 Manual refresh button
- 📊 Real-time monitoring

---

## 💡 Tips

### For Best Results
1. **Start Early**: Click "Start" right at 9:30 AM
2. **Monitor Dashboard**: Check occasionally, don't need to watch constantly
3. **Review Analysis**: Use Analysis page to see why trades were made
4. **End of Day**: Let it auto-close positions or close manually before 4 PM

### For Testing
1. **Paper Account**: All trades are paper (no real money)
2. **Lower Thresholds**: Currently set to 55/40 for more trades
3. **Watch Logs**: Activity log shows all decisions
4. **Adjust Settings**: Modify `config/config.py` as needed

---

## 🎯 Summary

**Before**: Manual clicks, limited scans, one-time execution
**After**: Fully automatic, continuous operation, all-day trading

**Key Benefits**:
- 🔄 Runs continuously without manual intervention
- 📊 Scans ALL S&P 500 stocks every 15 minutes
- 💰 Automatically buys high-scoring opportunities
- 🎯 Automatically sells on stops/profits
- 🛡️ Built-in safety limits and controls
- 📈 Maximizes trading opportunities throughout the day

**One Click = All Day Trading!** 🚀
