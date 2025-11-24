# AI-Driven Algorithmic Trading System

Fully automated trading system using Alpaca Paper Trading with multi-factor analysis (technical, fundamental, sentiment).

## Features

- **Multi-Factor Strategy**: Technical indicators + FinBERT sentiment analysis
- **FinBERT Sentiment**: AI model trained specifically on financial news
- **Simplified Stack**: Alpaca for trading/data + NewsAPI for news
- **Risk Management**: Strict 2% risk per trade, automated stop losses
- **Paper Trading**: Safe testing with Alpaca paper account
- **Structured Logging**: Full audit trail of all decisions
- **S&P 500 Universe**: Trade liquid, high-quality stocks
- **Continuous Trading**: Automated scanning and trading during market hours
- **Web Dashboard**: Beautiful real-time monitoring interface

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

Your `.env` file has been created with your API credentials. The system is ready to use!

### 3. Test Installation

```bash
# Run tests
pytest tests/

# Check code quality
black .
ruff check .
mypy src/
```

## Usage

### Run Trading Scan

```bash
python src/main.py scan
```

### Manage Existing Positions

```bash
python src/main.py manage
```

### End-of-Day Close

```bash
python src/main.py eod
```

### Schedule with Cron

```bash
# Edit crontab
crontab -e

# Add these lines (adjust paths):
0 10 * * 1-5 cd /Users/enocklangat/Documents/AI/ai_trader && /Users/enocklangat/Documents/AI/ai_trader/venv/bin/python src/main.py scan
0 15 * * 1-5 cd /Users/enocklangat/Documents/AI/ai_trader && /Users/enocklangat/Documents/AI/ai_trader/venv/bin/python src/main.py eod
```

## Architecture

```
src/
├── main.py           # Main trading engine
├── strategy.py       # Multi-factor scoring
├── data_provider.py  # EODHD + Alpaca data
├── risk_manager.py   # Position sizing & validation
├── executor.py       # Order execution
├── sentiment.py      # News sentiment analysis
├── universe.py       # S&P 500 stock list
└── logger.py         # Structured logging

config/
└── config.py         # Configuration management

tests/
└── test_*.py         # Unit tests
```

## Risk Management

- **2% Risk Rule**: Never risk more than 2% of portfolio on single trade
- **10% Position Limit**: No position exceeds 10% of portfolio
- **Stop Loss**: Automatic 2% stop loss on all positions
- **Take Profit**: 6% profit target (3:1 reward/risk)
- **Max Positions**: Limit to 10 concurrent positions

## Data Sources

### Alpaca (Trading & Market Data)
- Paper trading with $100K virtual account
- Real-time market data included
- No rate limits on paper trading
- Historical price bars for technical analysis

### NewsAPI (Sentiment Analysis)
- Free tier: 100 requests per day
- Real-time news from 80,000+ sources
- Enhanced sentiment quality with FinBERT

### FinBERT Sentiment Model
- BERT model fine-tuned on financial news
- Understands financial terminology (earnings, volatility, etc.)
- More accurate sentiment than generic models
- 438MB model, cached after first download

## Monitoring

View logs in real-time:

```bash
tail -f logs/trading.log | jq .
```

## Safety Features

- Paper trading only (no real money)
- Comprehensive validation before every trade
- All decisions logged with reasoning
- Position limits enforced
- Market hours checking
- API error handling with retries

## Troubleshooting

**Issue**: Import errors
```bash
pip install -r requirements.txt --upgrade
```

**Issue**: API authentication failed
```bash
# Check .env file has correct credentials
cat .env
```

**Issue**: EODHD rate limit
```bash
# Clear cache to force fresh data
rm -rf data/cache/
```

## Development

Follow guidelines in `AGENTS.md` for:
- Code style (Black, isort)
- Type hints (mypy)
- Testing (pytest)
- Linting (ruff)

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Test connection**: `python -c "from src.logger import logger; logger.info('test')"`
3. **Run first scan**: `python src/main.py scan`
4. **Monitor logs**: `tail -f logs/trading.log`

## Disclaimer

This is for educational purposes only. Paper trading does not guarantee real trading performance. Never trade with money you can't afford to lose.
