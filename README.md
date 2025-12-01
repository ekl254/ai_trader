# AI-Driven Algorithmic Trading System

Fully automated trading system using Alpaca Paper Trading with multi-factor analysis, performance tracking, and strategy optimization.

## 🚀 Features

### Core Trading
- **Multi-Factor Strategy**: Technical indicators (50%) + FinBERT sentiment (50%)
- **FinBERT Sentiment**: AI model trained specifically on financial news
- **Market Regime Detection**: Pauses trading in strong bear markets (SPY trend analysis)
- **Auto-Restart Mode**: Bot runs 24/7, automatically starts trading when market opens
- **Risk Management**: Strict 2% risk per trade, automated stop losses
- **Paper Trading**: Safe testing with Alpaca paper account
- **S&P 500 Universe**: Trade liquid, high-quality stocks (100 most liquid)

### Advanced Features
- **Performance Tracking**: Complete trade history with ML-ready data export
- **Strategy Optimizer**: Data-driven threshold and weight optimization
- **Smart Rebalancing**: Replace weak positions with stronger opportunities
- **Intelligent Reasons**: LLM-enhanced decision explanations
- **Real-time Dashboard**: Beautiful web interface with live updates
- **Sentiment Caching**: 30-minute TTL cache reduces NewsAPI calls
- **Market Regime Filter**: Detects bull/bear/neutral markets using SPY
- **EMA Trend Confirmation**: 20/50 EMA crossover for trend validation

### Dashboard Pages
- **Main Dashboard**: Portfolio overview, positions, orders
- **Analysis**: Scan results with smart qualification reasons
- **Performance**: Win rate, Sharpe ratio, score correlation analysis
- **Optimizer**: Automated strategy recommendations

## 📊 Current Performance (Sample Data)

- **Win Rate**: 50% (30 trades)
- **Profit Factor**: 1.64
- **Sharpe Ratio**: 0.79
- **Score Correlation**: 0.616 (moderate-strong positive)
- **Optimized Threshold**: 72.5 (raised from 55.0)

## 🎯 Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file with your API credentials:

```bash
# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

# NewsAPI (Free tier: 100 requests/day)
NEWS_API_KEY=your_key_here

# Optional: Ollama for LLM reasons
OLLAMA_URL=http://your-vps:11434
```

### 3. Start the System

```bash
# Start dashboard (port 8080)
python3 web/dashboard.py &

# Start trading bot (auto-restart mode)
./bot_control.sh start

# Bot will automatically:
# - Wait for market open (9:30 AM ET)
# - Scan 100 stocks every 15 minutes
# - Execute trades when score ≥ 72.5
# - Manage positions every 2 minutes
```

## 🎮 Usage

### Bot Control

```bash
# Check status
./bot_control.sh status

# Stop bot
./bot_control.sh stop

# Restart bot
./bot_control.sh restart

# View live logs
tail -f logs/trading.log | jq .
```

### Manual Trading

```bash
# One-time scan
python src/main.py scan

# Manage positions
python src/main.py manage

# End-of-day close
python src/main.py eod
```

### Dashboard Access

```bash
# Open dashboard
open http://localhost:8080

# Pages:
# /              - Main dashboard
# /analysis      - Scan results with smart reasons
# /performance   - Trade analytics and metrics
# /optimizer     - Strategy recommendations
```

## 📁 Architecture

```
ai_trader/
├── src/
│   ├── main.py                    # Trading engine (auto-restart mode)
│   ├── strategy.py                # Multi-factor scoring (technical + sentiment)
│   ├── executor.py                # Order execution
│   ├── performance_tracker.py     # Trade history & analytics
│   ├── strategy_optimizer.py      # Automated optimization
│   ├── llm_reason_generator.py    # Smart decision explanations
│   ├── sentiment.py               # FinBERT sentiment analysis (with caching)
│   ├── market_regime.py           # SPY-based market regime detection
│   ├── newsapi_client.py          # News fetching with company name mapping
│   ├── universe.py                # S&P 500 with liquidity filtering
│   └── position_tracker.py        # Position management
│
├── web/
│   ├── dashboard.py               # Flask web server
│   └── templates/
│       ├── dashboard.html         # Main UI
│       ├── analysis.html          # Scan results
│       ├── performance.html       # Analytics
│       └── optimizer.html         # Recommendations
│
├── config/
│   └── config.py                  # Configuration (threshold: 72.5)
│
├── tests/
│   ├── test_strategy.py           # Strategy scoring tests
│   ├── test_executor.py           # Order execution tests
│   ├── test_sentiment.py          # Sentiment analysis tests
│   └── test_risk_manager.py       # Risk management tests
│
├── data/
│   └── performance_history.json   # Trade history (30 trades)
│
└── bot_control.sh                 # Bot lifecycle management
```

## 🛡️ Risk Management

- **2% Risk Rule**: Never risk more than 2% of portfolio on single trade
- **10% Position Limit**: No position exceeds 10% of portfolio
- **Stop Loss**: Automatic 2% stop loss on all positions
- **Take Profit**: 6% profit target (3:1 reward/risk)
- **Max Positions**: Limit to 10 concurrent positions
- **Optimized Threshold**: 72.5 minimum composite score

## 📈 Data Sources

### Alpaca (Trading & Market Data)
- Paper trading with $100K virtual account
- Real-time market data included
- Historical price bars for technical analysis
- No rate limits on paper trading

### NewsAPI (Sentiment Analysis)
- Free tier: 100 requests per day
- Real-time news from 80,000+ sources
- Enhanced sentiment quality with FinBERT

### FinBERT Sentiment Model
- BERT model fine-tuned on financial news
- Understands financial terminology
- More accurate sentiment than generic models
- 438MB model, cached after first download

## 🔍 Monitoring

### View Logs

```bash
# Trading activity
tail -f logs/trading.log | jq .

# Bot startup/shutdown
tail -f logs/bot_stdout.log

# Dashboard access
tail -f logs/dashboard.log
```

### Performance Metrics

```bash
# Generate performance report
python3 test_performance_tracker.py

# View in dashboard
open http://localhost:8080/performance
```

## 🚀 Advanced Features

### Strategy Optimizer

The optimizer analyzes historical performance and recommends:
- Optimal score thresholds
- Component weight adjustments
- Win rate improvements
- Sharpe ratio optimization

```bash
# Access optimizer
open http://localhost:8080/optimizer
```

### Performance Tracking

Every trade is automatically recorded with:
- Entry/exit scores and prices
- Hold duration and profit/loss
- News sentiment at entry
- Exit reason (stop loss, target profit, etc.)

Data is stored in `data/performance_history.json` for ML analysis.

### Smart Reasons

Analysis page shows intelligent explanations:
- **Qualified**: "✓ Strong technicals, positive sentiment"
- **Rejected**: "✗ Composite 17 pts below threshold"

Powered by smart fallback logic with optional LLM enhancement.

## 🔧 Configuration

### Adjust Strategy (config/config.py)

```python
# Current optimized values
min_composite_score: 72.5  # Raised from 55.0
min_factor_score: 40.0

# Weights (simplified - no dead-weight fundamental)
technical: 0.50  # RSI, MACD, Bollinger, Volume, EMA
sentiment: 0.50  # FinBERT news sentiment
```

### Technical Indicators
- **RSI**: Oversold/overbought detection with gradient scoring
- **MACD**: Signal crossover with histogram strength
- **Bollinger Bands**: Dynamic column detection for price position
- **Volume**: Price-direction weighted (bullish confirmation)
- **EMA Crossover**: 20/50 EMA trend confirmation

### Market Regime Detection
The system monitors SPY to detect market conditions:
- **Strong Bull**: SMA20 > SMA50 > SMA200, price above SMA20
- **Bull**: Price above SMA50
- **Neutral**: Mixed signals
- **Bear**: Price below SMA50
- **Strong Bear**: Trading paused (all SMAs declining)

### Enable LLM Reasons

```python
# Edit src/llm_reason_generator.py line 23
self.use_llm = True  # Enable cloud model

# Restart dashboard
./bot_control.sh restart
```

## 🐛 Troubleshooting

**Issue**: Bot not trading
```bash
# Check threshold - stocks must score ≥ 72.5
# View analysis page to see scores
open http://localhost:8080/analysis
```

**Issue**: Dashboard not loading
```bash
# Check if running
ps aux | grep dashboard.py

# Restart
pkill -f dashboard.py
python3 web/dashboard.py &
```

**Issue**: News API rate limit
```bash
# You've hit 100 requests/day limit
# Bot defaults to sentiment=50 (neutral)
# Upgrade at https://newsapi.org/pricing
```

## 📝 Development

Follow guidelines in `AGENTS.md` for:
- Code style (Black, isort)
- Type hints (mypy)
- Testing (pytest)
- Linting (ruff)

## 🎓 Next Steps

1. **Monitor Performance**: Wait for real trades (market open)
2. **Review Analytics**: Check /performance page after trades
3. **Run Optimizer**: Get recommendations with more data
4. **Adjust Strategy**: Use optimizer insights to improve

## ⚠️ Disclaimer

This is for educational purposes only. Paper trading does not guarantee real trading performance. Never trade with money you can't afford to lose.

## 📚 Documentation

- `START_HERE.md` - Quick start guide
- `WEB_DASHBOARD.md` - Dashboard features
- `CONTINUOUS_TRADING.md` - Auto-trading setup
- `REBALANCING.md` - Position replacement strategy
- `AGENTS.md` - Development guidelines
