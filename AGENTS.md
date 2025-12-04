# Agent Guidelines for AI Trading System

## Build/Test Commands
- **Run all tests**: `pytest tests/`
- **Single test file**: `pytest tests/test_risk_manager.py`
- **Single test**: `pytest tests/test_risk_manager.py::test_position_size_calculation -v`
- **Lint**: `ruff check .` (auto-fix: `ruff check --fix .`)
- **Format**: `black . && isort .`
- **Type check**: `mypy src/`
- **Run bot**: `PYTHONPATH=/root/ai_trader python src/main.py continuous --auto-restart`
- **Run in Docker**: `docker-compose up -d`

## Code Style
- **Imports**: stdlib -> third-party -> local (isort profile=black, line_length=88)
- **Formatting**: Black defaults (88 chars), double quotes
- **Types**: Type hints required on all functions; use `Optional[]`, `Dict[]`, `List[]`
- **Naming**: snake_case (funcs/vars), PascalCase (classes), UPPER_CASE (constants)
- **Error Handling**: Always `except Exception as e:`, log with `logger.error("event", error=str(e))`
- **Docstrings**: Google style, required for public functions
- **Config**: Use `config.config` for settings, env vars via `python-dotenv`

## Trading System Rules

### Risk Management
- **Position Sizing**: Validate via `risk_manager.calculate_position_size()` before trades
- **2% Risk Rule**: Never risk more than 2% of portfolio on single trade
- **10% Position Limit**: No position exceeds 10% of portfolio value
- **Cash Validation**: Always check actual cash, NOT buying power (prevents margin)
- **5% Cash Buffer**: Maintain minimum 5% cash reserve before new trades

### Account Health
- **Health Check**: Use `risk_manager.check_account_health()` before trading operations
- **Block New Buys**: Use `risk_manager.should_block_new_buys()` to prevent trades when unhealthy
- **Forced Reduction**: If cash goes negative, sell weakest positions via `get_positions_to_reduce()`
- **Position Limits**: 7 positions in neutral regime, up to 10 in bull market

### Trade Execution
- **Validate First**: Call `risk_manager.validate_trade(symbol, shares, price)` before execution
- **Swap Logic**: Use sell-first approach in `executor.swap_position()` for rebalancing
- **Pending Buys**: Track with `risk_manager.mark_pending_buy()` and `clear_pending_buy()`
- **Audit Trail**: Log all trade decisions with reasoning via structlog

### API Integration
- **Trading**: Alpaca API for orders, positions, account data
- **Sentiment**: NewsAPI + FinBERT for news sentiment analysis
- **Testing**: Mock all external APIs with `pytest-mock`

## Key Files

### Core Trading
- `src/main.py` - Trading engine, continuous mode, health checks
- `src/executor.py` - Order execution, swap logic, stop loss management
- `src/strategy.py` - Multi-factor scoring (technical + sentiment)
- `src/risk_manager.py` - Account health, position sizing, trade validation

### Position Management
- `src/position_sizer.py` - Dynamic sizing with conviction adjustments
- `src/position_tracker.py` - Local position tracking with scores
- `src/market_regime.py` - SPY-based regime detection

### Analysis
- `src/sentiment.py` - FinBERT sentiment with caching
- `src/performance_tracker.py` - Trade history and analytics
- `src/strategy_optimizer.py` - Automated threshold optimization

### Infrastructure
- `src/metrics.py` - Prometheus metrics for monitoring
- `src/secrets_manager.py` - Secure credential handling
- `src/logger.py` - Structured logging configuration
- `src/watchdog.py` - Hang detection, heartbeat monitoring, and auto-recovery

## Constants Reference

```python
# risk_manager.py
MIN_CASH_PCT = 0.05              # 5% minimum cash reserve
MARGIN_WARNING_THRESHOLD = 0.10  # Warn below 10% cash
POSITION_OVERLOAD_THRESHOLD = 1.2  # Critical at 120% of max positions

# config.py
max_position_size = 0.10         # 10% max per position
max_positions = 10               # Absolute maximum positions
min_composite_score = 72.5       # Minimum score to buy
```

## Common Patterns

### Checking Account Health
```python
from src.risk_manager import risk_manager

health = risk_manager.check_account_health()
if not health["healthy"]:
    logger.warning("account_unhealthy", warnings=health["warnings"])
    # Handle reduction or block trades
```

### Validating a Trade
```python
is_valid, reason = risk_manager.validate_trade(symbol, shares, price)
if not is_valid:
    logger.warning("trade_rejected", symbol=symbol, reason=reason)
    return False
```

### Position Reduction
```python
to_reduce = risk_manager.get_positions_to_reduce(target_reduction=3)
for pos in to_reduce:
    executor.sell_stock(pos["symbol"], "forced_reduction")
```

### Using Watchdog for Long Operations
```python
from src.watchdog import watchdog, OperationTimeout

# Update heartbeat regularly in long-running loops
for symbol in symbols:
    watchdog.heartbeat(f"scanning_{symbol}")
    process_symbol(symbol)

# Wrap API calls in timeout protection
try:
    with OperationTimeout(seconds=30, operation="api_call"):
        result = api.call()
except TimeoutError:
    logger.error("api_call_timeout")
```
