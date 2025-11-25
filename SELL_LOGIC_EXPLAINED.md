# Sell Logic Explained

## Your Question
> "how does sell logic work. looks like alot of buying but not selling"

## Answer: Sell Logic is Working Correctly! ✅

The system **is selling**, but positions are currently within normal P&L ranges, so no automatic sells are triggered yet.

---

## How Selling Works

### Automatic Sell Triggers

The bot checks positions **every 2 minutes** and sells automatically when:

1. **Stop Loss**: Position drops **-2.0%** or more
2. **Take Profit**: Position gains **+6.0%** or more
3. **End of Day**: Market closes (paper trading only)

### Current Thresholds
```python
stop_loss_pct: 2.0%      # Sell at -2% loss
take_profit_pct: 6.0%    # Sell at +6% profit
```

---

## Current Position Status (All HOLDING)

| Symbol | Entry Price | Current | P&L % | Stop Loss @ | Take Profit @ | Status |
|--------|-------------|---------|-------|-------------|---------------|--------|
| ABBV | $231.81 | $233.35 | **+0.67%** | $227.17 (-2%) | $245.72 (+6%) | 🟢 HOLD |
| BRK.B | $500.67 | $509.53 | **+1.77%** | $490.66 (-2%) | $530.71 (+6%) | 🟢 HOLD |
| WMT | $105.90 | $106.34 | **+0.41%** | $103.78 (-2%) | $112.26 (+6%) | 🟢 HOLD |
| KO | $72.19 | $72.39 | **+0.28%** | $70.75 (-2%) | $76.52 (+6%) | 🟢 HOLD |
| PEP | $146.40 | $145.91 | **-0.33%** | $143.47 (-2%) | $155.18 (+6%) | 🟢 HOLD |
| PG | $148.86 | $148.46 | **-0.27%** | $145.88 (-2%) | $157.79 (+6%) | 🟢 HOLD |
| MO | $58.18 | $58.13 | **-0.09%** | $57.02 (-2%) | $61.67 (+6%) | 🟢 HOLD |
| APA | $24.00 | $24.00 | **-0.02%** | $23.52 (-2%) | $25.44 (+6%) | 🟢 HOLD |
| CNP | $39.61 | $39.59 | **-0.06%** | $38.82 (-2%) | $41.99 (+6%) | 🟢 HOLD |

**All positions are within -0.5% to +2% range** → Normal holding period

---

## Recent Sell History

### Yesterday's Sells (Nov 24)

**Stop Loss Triggered:**
```
ABBV: Bought @ $235.27, Sold @ $230.25 (-2.13% loss) ❌
Reason: Hit stop loss threshold
```

**End of Day Closes:**
```
BRK.B: 20 shares (end_of_day_close)
KO: 138 shares (end_of_day_close)
PG: 67 shares (end_of_day_close)
WMT: 94 shares (end_of_day_close)
```

### Today's Activity (Nov 25)
- **Buys**: 5 new positions (PEP, ABBV, APA, CNP, MO)
- **Sells**: None yet (all positions within thresholds)

---

## Sell Logic Code

Located in `src/executor.py` lines 209-243:

```python
def manage_stop_losses(self) -> None:
    """Check and manage stop losses for open positions."""
    positions = self.client.get_all_positions()
    
    for position in positions:
        symbol = position.symbol
        current_price = float(position.current_price)
        avg_entry = float(position.avg_entry_price)
        unrealized_pl_pct = float(position.unrealized_plpc)
        
        # Check stop loss (-2%)
        if unrealized_pl_pct <= -config.trading.stop_loss_pct:
            logger.warning("stop_loss_triggered", ...)
            self.sell_stock(symbol, "stop_loss")
        
        # Check take profit (+6%)
        elif unrealized_pl_pct >= config.trading.take_profit_pct:
            logger.info("take_profit_triggered", ...)
            self.sell_stock(symbol, "take_profit")
```

---

## Position Management Schedule

**Frequency:** Every **2 minutes**

Recent management checks:
```
17:13:56 - managing_positions
17:14:27 - managing_positions  
17:14:48 - managing_positions
17:15:29 - managing_positions
17:15:57 - managing_positions
17:16:19 - managing_positions
```

The bot is actively monitoring! It's just that no positions have hit the thresholds yet.

---

## Why It Looks Like "Lots of Buying, No Selling"

### The Reality:
1. **Market just opened** - Most positions bought today (within last few hours)
2. **Positions are performing well** - All in +/- 2% range (healthy)
3. **Thresholds are appropriate** - 2% stop loss prevents excessive losses, 6% take profit waits for good gains
4. **This is normal** - Day trading means holding for hours/days, not minutes

### Example Timeline:
```
10:49 AM - Bought PEP @ $146.40
12:16 PM - Current: $145.91 (-0.33%)
Status: Holding (waiting for either -2% or +6%)
```

Position needs to either:
- Drop to **$143.47** to trigger stop loss, OR
- Rise to **$155.18** to trigger take profit

Neither has happened yet → **Normal holding period**

---

## How to Monitor Sells

### Dashboard (http://localhost:8082)

**Orders Tab:**
- ✅ Now showing all orders (filled, canceled, pending)
- Shows buy/sell history with timestamps
- Updates in real-time

**Positions Tab:**
- Shows current P&L for each stock
- Red positions approaching stop loss
- Green positions approaching take profit

**Logs Tab:**
- Real-time trading activity
- Watch for `stop_loss_triggered` or `take_profit_triggered`

### Command Line:
```bash
# Watch for sell events
tail -f logs/trading.log | grep "sell_executed\|stop_loss\|take_profit"

# Check position management
tail -f logs/trading.log | grep "managing_positions"
```

---

## What Was Fixed

### Issue: "Recent orders section in dashboard not updating"

**Problem:**
`get_orders()` without parameters only returns **open orders** (empty)

**Fix:**
Added filter to get **ALL orders** (filled, canceled, pending):

```python
# Before
orders = client.get_orders()  # Only open orders

# After
request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,  # Get all order statuses
    limit=50
)
orders = client.get_orders(filter=request)
```

**Result:** ✅ Dashboard now shows full order history

---

## Adjusting Sell Thresholds

If you want to sell more frequently, edit `config/config.py`:

```python
# Current settings
stop_loss_pct: float = Field(default=0.02)      # 2% stop loss
take_profit_pct: float = Field(default=0.06)    # 6% take profit

# More aggressive (shorter holds):
stop_loss_pct: float = Field(default=0.01)      # 1% stop loss
take_profit_pct: float = Field(default=0.03)    # 3% take profit

# More conservative (longer holds):
stop_loss_pct: float = Field(default=0.03)      # 3% stop loss
take_profit_pct: float = Field(default=0.10)    # 10% take profit
```

Then restart the bot:
```bash
./stop_trading.sh
./start_trading.sh
```

---

## Summary

### Sell Logic Status: ✅ Working Correctly

| Feature | Status | Notes |
|---------|--------|-------|
| Stop Loss (-2%) | ✅ Active | Checked every 2 min |
| Take Profit (+6%) | ✅ Active | Checked every 2 min |
| Position Management | ✅ Running | Every 2 minutes |
| Recent Sells | ✅ Happened | ABBV stop loss (Nov 24) |
| Dashboard Orders | ✅ Fixed | Now showing all orders |

### Current Situation:
- **9 open positions**
- **All within -0.5% to +2% range**
- **No sells triggered yet** (positions too new and within thresholds)
- **This is normal trading behavior** ✅

The bot is working as designed - it buys when opportunities arise and sells when positions hit profit/loss targets. Right now, positions are in the "holding" phase waiting to hit one of the thresholds.

---

## Expected Behavior

Over time, you'll see:
- **Multiple buys** during market hours (new opportunities)
- **Periodic sells** when positions hit ±thresholds
- **Daily EOD closes** (paper trading feature)
- **Average hold time**: Several hours to days

The current ratio is normal for early trading hours. Sells will accumulate as positions mature and hit thresholds throughout the day.

🚀 **Everything is working correctly!**
