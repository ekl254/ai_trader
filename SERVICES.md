# Services Documentation

## Server: 69.62.64.51

---

## AI Trading System (This Project)

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **AI Trader Dashboard** | 8085 (nginx) -> 8082 (Flask) | http://69.62.64.51:8085/ | Main trading dashboard with real-time portfolio, positions, analysis |
| Dashboard - Analysis | 8085 | http://69.62.64.51:8085/analysis | Stock scoring and analysis page |
| Dashboard - Performance | 8085 | http://69.62.64.51:8085/performance | Trading performance metrics |
| Dashboard - Optimizer | 8085 | http://69.62.64.51:8085/optimizer | Strategy optimization with weight simulator |

### AI Trader Management Commands
```bash
# Start/Stop Trading Bot
./bot_control.sh start
./bot_control.sh stop
./bot_control.sh status

# Start Dashboard (if not running)
cd /root/ai_trader && source venv/bin/activate && nohup python web/dashboard.py > logs/dashboard.log 2>&1 &

# Check Dashboard Logs
tail -f /root/ai_trader/logs/dashboard.log

# Check Trading Logs
tail -f /root/ai_trader/logs/trading.log
```

---

## Other Services

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Home Assistant** | 8080 (nginx) -> 100.118.62.40:8123 | http://69.62.64.51:8080/ | Home automation (proxied via Tailscale) |
| **Glance** | 8081 (nginx) -> 100.118.62.40:8090 | http://69.62.64.51:8081/ | Dashboard/homepage (proxied via Tailscale) |
| **n8n** | 443 (HTTPS) | https://n8n.enocklangat.com/ | Workflow automation |
| **Stocks App** | 80/443 | http://69.62.64.51/ or https://stocks.enocklangat.com/ | Stock tracking web app (React + Node.js API on port 3002) |
| **ntopng** | 3001 (direct), 80 (nginx) | http://ntopng.srv754009.local/ | Network traffic monitoring |
| **Ollama** | 11434 | http://localhost:11434/ | Local LLM inference |

---

## Internal Services (localhost only)

| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL | 5432 | Database server |
| Redis | 6379 | In-memory cache/message broker |
| Stocks API Backend | 3002 | Node.js API for stocks app |

---

## Nginx Configuration Files

| Config File | Service |
|-------------|---------|
| `/etc/nginx/sites-enabled/ai-trader` | AI Trader Dashboard (8085 -> 8082) |
| `/etc/nginx/sites-enabled/homeassistant` | Home Assistant (8080 -> Tailscale) |
| `/etc/nginx/sites-enabled/glance` | Glance Dashboard (8081 -> Tailscale) |
| `/etc/nginx/sites-enabled/n8n.conf` | n8n (443 HTTPS) |
| `/etc/nginx/sites-enabled/stocks-app` | Stocks App (80 HTTP) |
| `/etc/nginx/sites-enabled/stocks.enocklangat.com` | Stocks App (443 HTTPS) |
| `/etc/nginx/sites-enabled/ntopng` | ntopng (80) |

---

## Port Summary

| Port | Service | Protocol |
|------|---------|----------|
| 22 | SSH | TCP |
| 80 | Nginx (stocks-app, ntopng, n8n redirect) | HTTP |
| 443 | Nginx (n8n, stocks.enocklangat.com) | HTTPS |
| 3001 | ntopng | HTTP |
| 3002 | Stocks App API (internal) | HTTP |
| 5432 | PostgreSQL (localhost) | TCP |
| 5678 | n8n (Docker) | HTTP |
| 6379 | Redis (localhost) | TCP |
| 8080 | Home Assistant (nginx proxy) | HTTP |
| 8081 | Glance (nginx proxy) | HTTP |
| 8082 | AI Trader Flask (internal) | HTTP |
| 8085 | AI Trader Dashboard (nginx proxy) | HTTP |
| 11434 | Ollama LLM | HTTP |

---

## Quick Access URLs

### Production
- **AI Trader**: http://69.62.64.51:8085/
- **Home Assistant**: http://69.62.64.51:8080/
- **Glance**: http://69.62.64.51:8081/
- **Stocks App**: https://stocks.enocklangat.com/
- **n8n**: https://n8n.enocklangat.com/

### Management
```bash
# Restart nginx
sudo systemctl reload nginx

# Check nginx config
nginx -t

# View all nginx sites
ls -la /etc/nginx/sites-enabled/

# Check what's running on ports
netstat -tlnp | grep LISTEN
```

---

## Docker Containers

```bash
# List running containers
docker ps

# n8n container management
docker restart n8n
docker logs n8n
```

---

*Last updated: December 1, 2025*
