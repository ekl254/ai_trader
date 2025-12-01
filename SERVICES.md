# AI Trader Services

## URLs

| Service | URL |
|---------|-----|
| **AI Trader Dashboard** | https://trader.enocklangat.com/ |
| AI Trader (IP fallback) | http://69.62.64.51/trader/ |
| Stocks App | https://stocks.enocklangat.com/ |
| Home Assistant | http://69.62.64.51:8080/ |
| Glance | http://69.62.64.51:8081/ |

## Dashboard Pages

- **Dashboard**: https://trader.enocklangat.com/ - Main trading view with positions, orders, logs
- **Analysis**: https://trader.enocklangat.com/analysis - Real-time stock scoring
- **Performance**: https://trader.enocklangat.com/performance - Trade history and metrics
- **Optimizer**: https://trader.enocklangat.com/optimizer - Strategy optimization and weight tuning

## Starting Services

```bash
# Start AI Trader bot
cd /root/ai_trader && ./start_trading.sh

# Start Dashboard (if not running)
cd /root/ai_trader && source venv/bin/activate && python web/dashboard.py &

# Check status
./status_trading.sh
```

## Internal Ports

| Service | Internal Port |
|---------|---------------|
| AI Trader Dashboard | 8082 |
| Ollama LLM | 11434 |
| Stocks App Backend | 3002 |
