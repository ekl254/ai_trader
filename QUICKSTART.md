# 🚀 Quick Start Guide

## ✅ System Status: READY TO TRADE!

Your AI trading system is installed and working!

## Usage

### Simple Commands (Using run.sh)

```bash
# Run a trading scan
./run.sh scan

# Manage existing positions (check stop losses)
./run.sh manage

# End-of-day close all positions
./run.sh eod
```

### Alternative (Manual Method)

```bash
# Activate environment
source venv/bin/activate

# Set Python path and run
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/main.py scan
```

## What Just Happened?

When you ran `./run.sh scan`, the system:

1. ✅ Loaded your API credentials from `.env`
2. ✅ Connected to Alpaca Paper Trading
3. ✅ Checked if market is open (currently CLOSED - it's Sunday)
4. ✅ Loaded stock universe (20 liquid stocks as fallback)
5. ✅ Logged everything to `logs/trading.log`

## View Logs in Real-Time

```bash
# Pretty print JSON logs
tail -f logs/trading.log | jq .

# Or just tail
tail -f logs/trading.log
```

## Test During Market Hours

The system detected the market is closed. To test when the market opens:

**Market Hours (EST):**
- Monday-Friday: 9:30 AM - 4:00 PM
- Pre-market: 4:00 AM - 9:30 AM
- After-hours: 4:00 PM - 8:00 PM

Run this on a weekday during market hours:

```bash
./run.sh scan
```

The system will:
1. Load S&P 500 stocks (20 liquid stocks)
2. Analyze each with technical + fundamental + sentiment
3. Score each stock (0-100)
4. Buy top 5 stocks with score ≥70
5. Calculate position sizes using 2% risk rule
6. Execute orders on Alpaca Paper Trading

## Check Your Alpaca Account

Visit: https://app.alpaca.markets/paper/dashboard/overview

You'll see:
- Account balance
- Open positions
- Order history
- P&L

## Next Steps

### 1. Test Connection (Right Now)

```bash
./run.sh scan
```

This works even when market is closed - it just won't execute trades.

### 2. Monitor Logs

```bash
tail -f logs/trading.log | jq .
```

### 3. Schedule Automated Trading

```bash
crontab -e
```

Add:
```cron
# Morning scan at 10 AM EST (market open)
0 10 * * 1-5 cd /Users/enocklangat/Documents/AI/ai_trader && ./run.sh scan

# Manage positions every hour
0 * * * 1-5 cd /Users/enocklangat/Documents/AI/ai_trader && ./run.sh manage

# Close all positions at 3 PM EST
0 15 * * 1-5 cd /Users/enocklangat/Documents/AI/ai_trader && ./run.sh eod
```

## System Features

### Risk Management ✅
- 2% risk per trade (enforced)
- 10% max position size
- Automatic stop losses
- Max 10 concurrent positions

### Analysis ✅
- **Technical**: RSI, MACD, Bollinger Bands, Volume
- **Fundamental**: P/E Ratio, ROE, Debt/Equity
- **Sentiment**: AI-powered news analysis

### APIs ✅
- Alpaca Paper Trading (configured)
- EODHD for fundamentals & news (configured)
- Aggressive caching to stay within limits

## Troubleshooting

### "Market Closed"
Normal! The market is closed on weekends and outside 9:30 AM - 4:00 PM EST weekdays.

### "HTTP Error 403" for S&P 500
Wikipedia sometimes blocks scrapers. The system uses a fallback list of 20 liquid stocks.

### Want to customize?

Edit `config/config.py`:
- Change risk per trade
- Adjust scoring thresholds
- Modify technical indicators

## Files You Might Want to Check

- **Logs**: `logs/trading.log` - Every decision logged
- **Config**: `.env` - Your API credentials
- **Cache**: `data/cache/` - Cached API responses
- **Code**: `src/` - All trading logic

## Example Log Output

```json
{
  "event": "buy_executed",
  "symbol": "AAPL",
  "shares": 65,
  "price": 150.0,
  "position_value": 9750.0,
  "stop_loss": 147.0,
  "timestamp": "2025-11-25T10:00:00Z"
}
```

## Safety Reminders

✅ Paper trading only (no real money)
✅ All trades validated before execution
✅ Complete audit trail
✅ Stop losses enforced

## Ready to Go Live?

**DON'T!** Test thoroughly with paper trading first:
1. Run for at least 2 weeks
2. Monitor all trades
3. Understand the strategy
4. Adjust parameters
5. Backtest extensively

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./run.sh scan` | Scan and trade |
| `./run.sh manage` | Check stops/exits |
| `./run.sh eod` | Close all positions |
| `tail -f logs/trading.log` | Watch logs |

**Need Help?** Check `README.md` or `AGENTS.md`

**Happy Trading! 📈**
