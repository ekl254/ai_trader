# Portfolio Rebalancing Feature

## Overview

The rebalancing feature automatically replaces weaker-performing positions with better opportunities when all position slots are full. This ensures your portfolio is always optimized with the highest-scoring stocks.

## How It Works

### 1. Trigger Conditions

Rebalancing activates when **all** of these conditions are met:

- `enable_rebalancing = True` in config
- All position slots are full (e.g., 10/10 positions occupied)
- System finds better buy candidates during a scan
- Weakest position meets minimum hold time requirement

### 2. Scoring Process

Every 15 minutes during market scans:

1. **Scan Universe**: Evaluate all 100 top S&P 500 stocks
2. **Rescore Existing Positions**: Re-evaluate all current holdings
3. **Compare Scores**: Find weakest position vs best new candidate
4. **Check Threshold**: New candidate must score `rebalance_score_diff` points higher
5. **Execute Swap**: Sell weak position, buy strong candidate

### 3. Configuration

Located in `config/config.py`:

```python
# Rebalancing Settings
enable_rebalancing: bool = True              # Enable/disable feature
rebalance_score_diff: float = 10.0           # Min score difference to trigger swap
rebalance_min_hold_time: int = 30            # Min hold time (minutes) before rebalancing
```

## Example Scenario

### Portfolio State (10/10 slots full)

| Symbol | Score | Hold Time |
|--------|-------|-----------|
| AAPL   | 72.3  | 2h 15m    |
| MSFT   | 68.5  | 1h 45m    |
| GOOGL  | 65.2  | 3h 10m    |
| **PEP**| **52.1** | **45m** |  ← Weakest position
| NVDA   | 71.8  | 1h 20m    |
| ...    | ...   | ...       |

### New Scan Results

| Symbol | Score |
|--------|-------|
| **TSLA** | **75.3** | ← Best new candidate
| AMD    | 68.9  |
| NFLX   | 63.4  |

### Rebalancing Decision

- **Weakest Position**: PEP (52.1)
- **Best Candidate**: TSLA (75.3)
- **Score Difference**: 75.3 - 52.1 = **23.2 points**
- **Threshold**: 10.0 points
- **Hold Time**: 45 minutes (>= 30 minutes ✓)

**Result**: ✅ **SWAP TRIGGERED**

**Actions**:
1. Sell PEP (full position)
2. Wait 2 seconds for settlement
3. Buy TSLA (using freed capital)

## Logging

All rebalancing events are logged with full context:

```json
{
  "event": "rebalancing_opportunity_found",
  "old_symbol": "PEP",
  "old_score": 52.1,
  "new_symbol": "TSLA",
  "new_score": 75.3,
  "score_diff": 23.2
}
```

```json
{
  "event": "position_swap_completed",
  "old_symbol": "PEP",
  "new_symbol": "TSLA",
  "new_score": 75.3
}
```

Check logs at: `logs/trading.log`

## Safety Features

### 1. Minimum Hold Time

Prevents excessive churn by requiring positions to be held for at least 30 minutes (configurable).

**Why?**
- Avoids rapid buying/selling
- Reduces transaction costs
- Prevents pattern day trading violations

### 2. Score Threshold

Only swaps if new candidate is **significantly** better (default: 10+ points higher).

**Why?**
- Ensures meaningful improvement
- Avoids marginal swaps
- Reduces unnecessary trading

### 3. Atomic Swaps

Rebalancing is atomic - either both sell and buy succeed, or neither happens.

**Error Handling**:
- If sell fails → Log error, keep existing position
- If buy fails → Log error, position slot freed (will fill on next scan)

## Code Location

### Files Modified

1. **`src/strategy.py`** (lines 211-241)
   - Added `rescore_positions()` method
   - Evaluates existing holdings using same scoring logic

2. **`src/executor.py`** (lines 246-295)
   - Added `swap_position()` method
   - Handles atomic sell/buy operations

3. **`src/main.py`** (lines 72-180)
   - Added rebalancing logic to `scan_and_trade()`
   - Rescores positions when slots full
   - Executes swaps when criteria met

4. **`config/config.py`** (lines 47-54)
   - Added configuration parameters

## Disabling Rebalancing

To disable rebalancing, set in `.env` or directly in config:

```python
enable_rebalancing = False
```

System will then:
- Skip rescoring existing positions
- Exit early when all slots full
- Only trade when slots become available via stops/EOD closes

## Performance Impact

### API Usage

**With Rebalancing ON (all slots full)**:
- Universe scan: 100 API calls (bars data)
- Position rescoring: 10 API calls (bars data)
- **Total**: ~110 calls per 15-minute scan

**With Rebalancing OFF**:
- No scans when slots full
- **Total**: 0 calls

### Recommendation

- **Enable** during market hours when seeking optimal portfolio
- **Disable** if approaching API rate limits
- **Adjust** `rebalance_score_diff` higher to reduce swap frequency

## Testing

To test rebalancing logic:

1. Fill all position slots (buy 10 stocks)
2. Wait 30+ minutes
3. Let system scan - watch for lower-scoring positions
4. Monitor logs for rebalancing events

**Test Mode**:
```python
# Lower thresholds for testing
rebalance_score_diff = 5.0    # Easier to trigger
rebalance_min_hold_time = 5   # 5 minutes instead of 30
```

## FAQ

**Q: Will rebalancing trigger stop losses?**  
A: No. Rebalancing only evaluates scores. Stop losses (-2%) and take profits (+6%) are checked separately every 2 minutes.

**Q: Can I rebalance specific stocks?**  
A: Not directly. System automatically selects weakest position. To prevent rebalancing a stock, you'd need to manually manage it outside the bot.

**Q: What if rebalancing happens during a dip?**  
A: Position rescoring uses the same multi-factor analysis (technical + sentiment + fundamental). If a stock is temporarily dipping but fundamentals/sentiment remain strong, it will maintain a good score.

**Q: Does rebalancing affect day trading limits?**  
A: Yes. Each swap counts as both a sell and a buy. Monitor your day trade count if using a margin account <$25k.

## Advanced Features (NEW!)

### ✅ Actual Entry Time Tracking

Positions are now tracked with **exact entry times** from order fills:
- Stored in `data/position_tracking.json`
- Used for accurate hold duration calculations
- Falls back to conservative estimate if data unavailable

### ✅ Rebalancing Cooldown Period

Prevents excessive trading with a cooldown mechanism:
```python
rebalance_cooldown_minutes = 60  # Max 1 swap per hour
```

**How it works:**
- After each swap, system enters cooldown period
- New swaps blocked until cooldown expires
- Logged clearly: `rebalancing_in_cooldown`

### ✅ Position Locking

**Lock positions** to prevent them from being rebalanced:

```bash
# Lock a position
python manage_positions.py lock AAPL

# Unlock a position
python manage_positions.py unlock AAPL

# List locked positions
python manage_positions.py locked

# Unlock all
python manage_positions.py unlock-all
```

**When to lock:**
- Strong conviction holds (e.g., long-term winners)
- Tax loss harvesting timing
- Specific sector exposure maintenance

### ✅ Partial Position Rebalancing

Sell only a **portion** of weak positions:
```python
rebalance_partial_sell_pct = 0.5  # Sell 50%, keep 50%
```

**Benefits:**
- Gradual position rotation
- Maintain some exposure to recovering stocks
- Lower risk if rebalancing decision was premature

**Example:**
- Old position: 100 shares AAPL (score: 52)
- New candidate: TSLA (score: 75)
- Partial sell: Sell 50 shares AAPL, buy TSLA
- Result: 50 AAPL + new TSLA position

### ✅ Position Management CLI

**New tool:** `manage_positions.py`

```bash
# List all positions with lock status
python manage_positions.py list

# Show rebalancing history
python manage_positions.py history

# Show rebalancing statistics
python manage_positions.py stats
```

**Stats include:**
- Total rebalances
- Average score improvement
- Min/max score improvements
- Last rebalance timestamp
- Cooldown status

## Configuration Reference

All settings in `config/config.py`:

```python
# Core rebalancing
enable_rebalancing: bool = True
rebalance_score_diff: float = 10.0           # Min score difference
rebalance_min_hold_time: int = 30            # Minutes

# NEW: Cooldown & partial selling
rebalance_cooldown_minutes: int = 60         # Minutes between swaps
rebalance_partial_sell_pct: float = 0.0      # 0.0 = full position, 0.5 = 50%
```

## Position Tracking File

Located at `data/position_tracking.json`:

```json
{
  "positions": {
    "AAPL": {
      "entry_time": "2025-11-25T14:30:00+00:00",
      "entry_price": 195.50,
      "score": 72.3,
      "reason": "new_position"
    }
  },
  "locked_positions": ["BRK.B", "MSFT"],
  "rebalancing_history": [
    {
      "timestamp": "2025-11-25T15:45:00+00:00",
      "old_symbol": "PEP",
      "new_symbol": "TSLA",
      "old_score": 52.1,
      "new_score": 75.3,
      "score_diff": 23.2
    }
  ],
  "last_rebalance_time": "2025-11-25T15:45:00+00:00"
}
```

## Summary

Rebalancing keeps your portfolio optimized by automatically upgrading weaker positions with stronger opportunities. Configure thresholds to match your trading style - aggressive (low threshold) or conservative (high threshold).

**Recommended Settings:**

**Conservative (Low frequency, high conviction)**
- Score difference: 15.0
- Min hold time: 60 minutes
- Cooldown: 120 minutes
- Partial sell: 0.0 (full swaps only)

**Balanced (Default)**
- Score difference: 10.0
- Min hold time: 30 minutes
- Cooldown: 60 minutes
- Partial sell: 0.0

**Aggressive (High frequency, opportunistic)**
- Score difference: 7.0
- Min hold time: 15 minutes
- Cooldown: 30 minutes
- Partial sell: 0.5 (gradual rotation)
