#!/usr/bin/env python3
"""Web dashboard for AI Trading System."""

import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.trading.client import TradingClient

from config.config import config
from src.llm_reason_generator import get_llm_reason_generator  # type: ignore
from src.market_regime import market_regime_detector  # type: ignore
from src.newsapi_client import newsapi_client
from src.auto_learner import auto_learner  # type: ignore
from src.performance_tracker import PerformanceTracker  # type: ignore
from src.position_tracker import PositionTracker
from src.strategy_optimizer import StrategyOptimizer

app = Flask(__name__)
app.secret_key = os.getenv("DASHBOARD_SECRET_KEY", secrets.token_hex(32))

# Authentication credentials from environment
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "langatenock@gmail.com")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "kibetekl")


def login_required(f):
    """Decorator to require login for routes."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not DASHBOARD_PASSWORD:
            # No password set, allow access (for development)
            return f(*args, **kwargs)
        if not session.get("authenticated"):
            # Check if this is an API request (return JSON 401) or page request (redirect)
            is_api_request = (
                request.is_json
                or request.path.startswith("/api/")
                or request.headers.get("Accept", "").startswith("application/json")
                or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            )
            if is_api_request:
                return (
                    jsonify({"error": "Authentication required", "redirect": "/login"}),
                    401,
                )
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if not DASHBOARD_PASSWORD:
        # No password configured, redirect to dashboard
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD:
            session["authenticated"] = True
            session["username"] = username
            return redirect(url_for("index"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Logout and clear session."""
    session.clear()
    return redirect(url_for("login"))


# Initialize LLM reason generator (connects to VPS Ollama)
llm_generator = get_llm_reason_generator()

# Store trading process
trading_process = None
trading_running = False


def get_market_status() -> dict[str, Any]:
    """Get current market status."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        clock = client.get_clock()

        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }
    except Exception as e:
        return {"error": str(e)}


def get_account_info() -> dict[str, Any]:
    """Get account information."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        account = client.get_account()

        equity = float(account.equity) if account.equity else 0
        last_equity = float(account.last_equity) if account.last_equity else equity

        return {
            "portfolio_value": (
                float(account.portfolio_value) if account.portfolio_value else 0
            ),
            "buying_power": float(account.buying_power) if account.buying_power else 0,
            "cash": float(account.cash) if account.cash else 0,
            "equity": equity,
            "last_equity": last_equity,
            "profit_loss": equity - last_equity,
            "profit_loss_pct": (
                ((equity - last_equity) / last_equity * 100) if last_equity > 0 else 0
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def get_positions() -> list[dict[str, Any]]:
    """Get current positions with rebalancing metadata."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        positions = client.get_all_positions()
        tracker = PositionTracker()

        result = []
        for pos in positions:
            symbol = pos.symbol

            # Get position data from tracker
            position_data = tracker.data.get("positions", {}).get(symbol)

            # Calculate hold duration in minutes
            hold_duration_min = None
            entry_score = None

            if position_data:
                entry_time_str = position_data.get("entry_time")
                if entry_time_str:
                    from datetime import datetime

                    entry_time = datetime.fromisoformat(entry_time_str)
                    hold_duration_min = int(
                        (datetime.now(entry_time.tzinfo) - entry_time).total_seconds()
                        / 60
                    )
                entry_score = position_data.get("score")

            result.append(
                {
                    "symbol": symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": (
                        float(pos.current_price) if pos.current_price else 0
                    ),
                    "market_value": float(pos.market_value) if pos.market_value else 0,
                    "cost_basis": float(pos.cost_basis) if pos.cost_basis else 0,
                    "unrealized_pl": (
                        float(pos.unrealized_pl) if pos.unrealized_pl else 0
                    ),
                    "unrealized_plpc": (
                        float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
                    ),
                    "side": pos.side,
                    "is_locked": tracker.is_locked(symbol),
                    "hold_duration_min": hold_duration_min,
                    "entry_score": entry_score,
                }
            )

        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_recent_orders() -> list[dict[str, Any]]:
    """Get last 20 orders (all statuses)."""
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )

        # Get last 20 orders regardless of status
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=20,
        )
        orders = client.get_orders(filter=request)

        result = []
        for order in list(orders):
            result.append(
                {
                    "id": str(order.id),
                    "symbol": order.symbol,
                    "qty": float(order.qty) if order.qty else 0,
                    "side": order.side.value if order.side else "unknown",
                    "type": order.type.value if order.type else "unknown",
                    "status": order.status.value if order.status else "unknown",
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": (
                        float(order.filled_avg_price) if order.filled_avg_price else 0
                    ),
                    "created_at": str(order.created_at),
                }
            )

        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_recent_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent log entries."""
    try:
        log_file = Path(__file__).parent.parent / config.logging.log_file

        if not log_file.exists():
            return []

        logs = []
        with open(log_file) as f:
            # Read last N lines
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return list(reversed(logs))  # Most recent first
    except Exception as e:
        return [{"error": str(e)}]


def get_analysis_data() -> list[dict[str, Any]]:
    """Get ticker analysis data from recent scans only (last 2 hours)."""
    try:
        from datetime import datetime, timedelta

        log_file = Path(__file__).parent.parent / config.logging.log_file

        if not log_file.exists():
            return []

        # Only show scans from last 2 hours (using UTC for comparison)
        cutoff_time = datetime.now(UTC) - timedelta(hours=2)

        analyses = []
        with open(log_file) as f:
            for line in f:
                try:
                    log = json.loads(line.strip())

                    # Look for symbol_scored events
                    if log.get("event") == "symbol_scored":
                        timestamp_str = log.get("timestamp", "")
                        if timestamp_str:
                            # Parse timestamp (ISO format with Z = UTC)
                            log_time = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )

                            # Skip if too old (compare in UTC)
                            if log_time < cutoff_time:
                                continue

                        symbol = log.get("symbol")
                        composite = log.get("composite", 0)
                        technical = log.get("technical", 0)
                        fundamental = log.get("fundamental", 0)
                        sentiment = log.get("sentiment", 0)

                        # Generate LLM-based reason
                        reason = llm_generator.generate_reason(
                            symbol=symbol,
                            composite=composite,
                            technical=technical,
                            fundamental=fundamental,
                            sentiment=sentiment,
                            thresholds={
                                "min_composite": config.trading.min_composite_score,
                                "min_factor": config.trading.min_factor_score,
                            },
                        )

                        analyses.append(
                            {
                                "symbol": symbol,
                                "composite_score": round(composite, 2),
                                "technical_score": round(technical, 2),
                                "fundamental_score": round(fundamental, 2),
                                "sentiment_score": round(sentiment, 2),
                                "timestamp": timestamp_str,
                                "traded": composite
                                >= config.trading.min_composite_score
                                and technical >= config.trading.min_factor_score
                                and sentiment >= config.trading.min_factor_score,
                                "reason": reason,
                            }
                        )

                except (json.JSONDecodeError, ValueError):
                    continue

        # Return most recent first
        return list(reversed(analyses))
    except Exception as e:
        return [{"error": str(e)}]


def _get_trade_reason(
    composite: float, technical: float, fundamental: float, sentiment: float
) -> str:
    """Generate human-readable trade decision reason."""
    min_composite = config.trading.min_composite_score
    min_factor = config.trading.min_factor_score

    if (
        composite >= min_composite
        and technical >= min_factor
        and sentiment >= min_factor
    ):
        return "✓ QUALIFIED - Meets all criteria"

    reasons = []
    if composite < min_composite:
        reasons.append(f"Composite score too low ({composite:.1f} < {min_composite})")
    if technical < min_factor:
        reasons.append(f"Technical score too low ({technical:.1f} < {min_factor})")
    if sentiment < min_factor:
        reasons.append(f"Sentiment score too low ({sentiment:.1f} < {min_factor})")

    return "✗ REJECTED - " + "; ".join(reasons)


@app.route("/")
@login_required
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


def is_bot_running() -> bool:
    """Check if trading bot is running via Docker, systemd service, or PID file."""
    try:
        # Check if running inside Docker container (bot runs as main process)
        if os.path.exists("/.dockerenv"):
            # Inside Docker - check if main trading process exists
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "python.*main.py"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                pass
            # Alternative: check if continuous trading started in logs
            log_file = Path(__file__).parent.parent / "logs" / "trading.log"
            if log_file.exists():
                try:
                    content = log_file.read_text()
                    if "continuous_trading_started" in content:
                        return True
                except Exception:
                    pass
            return True  # If in Docker, assume bot is running as main process

        # Outside Docker: check systemd service (preferred method)
        result = subprocess.run(
            ["systemctl", "is-active", "ai-trader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip() == "active":
            return True

        # Fallback: check PID file for manual runs
        pid_file = Path(__file__).parent.parent / "logs" / "bot.pid"
        if not pid_file.exists():
            return False

        pid = int(pid_file.read_text().strip())
        # Check if process is actually running

        try:
            os.kill(pid, 0)  # Signal 0 checks if process exists
            return True
        except OSError:
            # Process not found, clean up stale PID file
            pid_file.unlink(missing_ok=True)
            return False
    except Exception:
        return False


def get_nyse_time() -> dict[str, Any]:
    """Get current NYSE time."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore

    nyse_tz = ZoneInfo("America/New_York")
    nyse_now = datetime.now(nyse_tz)

    return {
        "time": nyse_now.strftime("%I:%M:%S %p"),
        "date": nyse_now.strftime("%a, %b %d, %Y"),
        "datetime": nyse_now.isoformat(),
        "timezone": "ET",
    }


@app.route("/api/status")
@login_required
def api_status():
    """API endpoint for market status with regime info."""
    # Get regime data
    try:
        regime_data = market_regime_detector.get_market_regime()
        regime_info = {
            "regime": regime_data.get("regime", "unknown"),
            "stop_loss_pct": regime_data.get("stop_loss_pct", 0.03),
            "take_profit_pct": regime_data.get("take_profit_pct", 0.08),
            "min_score": regime_data.get("min_score", 70),
            "momentum_20d": regime_data.get("momentum_20d", 0),
            "volatility": regime_data.get("volatility", 0),
            "recommendation": regime_data.get("recommendation", ""),
        }
    except Exception as e:
        regime_info = {"regime": "error", "error": str(e)}

    return jsonify(
        {
            "market": get_market_status(),
            "account": get_account_info(),
            "regime": regime_info,
            "trading_running": is_bot_running(),
            "nyse_time": get_nyse_time(),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/positions")
@login_required
def api_positions():
    """API endpoint for positions."""
    return jsonify(get_positions())


@app.route("/api/orders")
@login_required
def api_orders():
    """API endpoint for orders."""
    return jsonify(get_recent_orders())


@app.route("/api/symbol/<symbol>/details")
@login_required
def api_generic_symbol_details(symbol: str):
    """Get price and market data for any symbol (not necessarily a position)."""
    try:
        from src.data_provider import alpaca_provider

        symbol = symbol.upper()

        # Get latest trade price
        latest = alpaca_provider.get_latest_trade(symbol)
        current_price = latest["price"]

        # Get recent bars for OHLC data (days=5 to ensure we get today + yesterday)
        bars = alpaca_provider.get_bars(symbol, days=5)

        if bars is not None and len(bars) >= 1:
            today_bar = bars.iloc[-1]
            open_price = float(today_bar["open"])
            day_high = float(today_bar["high"])
            day_low = float(today_bar["low"])

            # Previous close from yesterday's bar or today's open
            if len(bars) >= 2:
                prev_close = float(bars.iloc[-2]["close"])
            else:
                prev_close = open_price
        else:
            # Fallback values
            open_price = current_price
            day_high = current_price
            day_low = current_price
            prev_close = current_price

        # Calculate day change
        day_change = current_price - prev_close
        day_change_pct = (
            ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        )

        # Calculate position in day range (0-100%)
        day_range = day_high - day_low
        if day_range > 0:
            day_range_position = ((current_price - day_low) / day_range) * 100
        else:
            day_range_position = 50  # If no range, put in middle

        day_range_position = max(0, min(100, day_range_position))

        return jsonify(
            {
                "symbol": symbol,
                "current_price": current_price,
                "open_price": open_price,
                "day_high": day_high,
                "day_low": day_low,
                "prev_close": prev_close,
                "day_change": round(day_change, 2),
                "day_change_pct": round(day_change_pct, 2),
                "day_range_position": round(day_range_position, 1),
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/regime")
@login_required
def api_regime():
    """API endpoint for current market regime and parameters."""
    try:
        regime_data = market_regime_detector.get_market_regime()
        return jsonify(
            {
                "regime": regime_data.get("regime", "unknown"),
                "spy_price": regime_data.get("spy_price", 0),
                "spy_ema20": regime_data.get("spy_ema20", 0),
                "spy_ema50": regime_data.get("spy_ema50", 0),
                "momentum_20d": regime_data.get("momentum_20d", 0),
                "volatility": regime_data.get("volatility", 0),
                "stop_loss_pct": regime_data.get("stop_loss_pct", 0.03),
                "take_profit_pct": regime_data.get("take_profit_pct", 0.08),
                "min_score": regime_data.get("min_score", 70),
                "max_positions": regime_data.get("max_positions_override", 10),
                "should_trade": regime_data.get("should_trade", True),
                "recommendation": regime_data.get("recommendation", ""),
                "timestamp": regime_data.get("timestamp", ""),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
@login_required
def api_logs():
    """API endpoint for logs."""
    limit = int(request.args.get("limit", 50))
    return jsonify(get_recent_logs(limit))


@app.route("/analysis")
@login_required
def analysis_page():
    """Analysis page showing ticker scores."""
    return render_template("analysis.html")


@app.route("/performance")
@login_required
def performance_page():
    """Performance analytics page."""
    return render_template("performance.html")


@app.route("/api/analysis")
@login_required
def api_analysis() -> Any:
    """Get recent ticker analysis."""
    data = get_analysis_data()
    return jsonify(data)


@app.route("/api/analysis/<symbol>/details")
@login_required
def api_symbol_details(symbol: str) -> Any:
    """Get detailed analysis for a specific symbol showing actual scan data and reasoning."""
    try:
        import json
        from pathlib import Path

        log_file = Path(__file__).parent.parent / config.logging.log_file

        if not log_file.exists():
            return jsonify({"error": "Log file not found"}), 404

        # Find the most recent complete analysis for this symbol
        symbol_data = {
            "symbol": symbol.upper(),
            "score_data": None,
            "news_data": None,
            "articles": [],
        }

        # Read logs backward to find most recent data
        with open(log_file) as f:
            lines = f.readlines()

        # Process from newest to oldest
        for line in reversed(lines):
            try:
                log = json.loads(line.strip())

                if log.get("symbol") != symbol.upper():
                    continue

                # Get score breakdown
                if (
                    log.get("event") == "symbol_scored"
                    and not symbol_data["score_data"]
                ):
                    symbol_data["score_data"] = {
                        "timestamp": log.get("timestamp"),
                        "composite": log.get("composite"),
                        "technical": log.get("technical"),
                        "fundamental": log.get("fundamental"),
                        "sentiment": log.get("sentiment"),
                    }

                # Get news sentiment data
                if (
                    log.get("event") == "news_sentiment_analyzed"
                    and not symbol_data["news_data"]
                ):
                    symbol_data["news_data"] = {
                        "articles_count": log.get("articles_count"),
                        "sentiments_count": log.get("sentiments_count"),
                        "avg_sentiment": log.get("avg_sentiment"),
                        "score": log.get("score"),
                    }

                # Get articles fetched
                if (
                    log.get("event") == "newsapi_articles_fetched"
                    and len(symbol_data["articles"]) == 0
                ):
                    # Try to find corresponding articles in nearby logs
                    article_count = log.get("count", 0)
                    symbol_data["articles_info"] = f"{article_count} articles fetched"

                # Stop if we have all data
                if symbol_data["score_data"] and symbol_data["news_data"]:
                    break

            except (json.JSONDecodeError, KeyError):
                continue

        if not symbol_data["score_data"]:
            return (
                jsonify(
                    {
                        "error": f"No recent analysis found for {symbol}. Symbol may not have been scanned yet."
                    }
                ),
                404,
            )

        # Build reasoning
        score_data = symbol_data["score_data"]
        news_data = symbol_data.get("news_data", {})

        reasoning = []

        # Technical analysis reasoning
        tech_score = score_data["technical"]
        if tech_score >= 60:
            reasoning.append(
                f"✅ Strong technical indicators (RSI, MACD, Bollinger Bands) = {tech_score:.1f}/100"
            )
        elif tech_score >= 40:
            reasoning.append(f"⚠️ Moderate technical indicators = {tech_score:.1f}/100")
        else:
            reasoning.append(f"❌ Weak technical indicators = {tech_score:.1f}/100")

        # Sentiment reasoning
        sent_score = score_data["sentiment"]
        if news_data:
            articles = news_data.get("articles_count", 0)
            avg_sent = news_data.get("avg_sentiment", 0)
            if sent_score >= 60:
                reasoning.append(
                    f"✅ Positive news sentiment from {articles} articles (FinBERT: {avg_sent:.3f}) = {sent_score:.1f}/100"
                )
            elif sent_score >= 40:
                reasoning.append(
                    f"⚠️ Neutral news sentiment from {articles} articles (FinBERT: {avg_sent:.3f}) = {sent_score:.1f}/100"
                )
            else:
                reasoning.append(
                    f"❌ Negative news sentiment from {articles} articles (FinBERT: {avg_sent:.3f}) = {sent_score:.1f}/100"
                )
        else:
            if sent_score == 50.0:
                reasoning.append(
                    f"⚠️ No news available, using neutral sentiment = {sent_score:.1f}/100"
                )
            else:
                reasoning.append(f"📰 Sentiment analysis = {sent_score:.1f}/100")

        # Fundamental reasoning
        fund_score = score_data["fundamental"]
        reasoning.append(
            f"ℹ️ Fundamentals = {fund_score:.1f}/100 (using neutral default)"
        )

        # Composite calculation
        composite = score_data["composite"]
        reasoning.append("")
        reasoning.append("**Final Composite Score Calculation:**")
        reasoning.append(
            f"({tech_score:.1f} × 40%) + ({sent_score:.1f} × 30%) + ({fund_score:.1f} × 30%) = **{composite:.1f}/100**"
        )
        reasoning.append("")

        # Decision logic
        if composite >= 55.0:
            if tech_score >= 40.0 and sent_score >= 40.0:
                reasoning.append(
                    "✅ **QUALIFIED FOR TRADING** - All criteria met (composite ≥55, technical ≥40, sentiment ≥40)"
                )
            else:
                if tech_score < 40.0:
                    reasoning.append(
                        f"❌ **REJECTED** - Technical score too low ({tech_score:.1f} < 40)"
                    )
                if sent_score < 40.0:
                    reasoning.append(
                        f"❌ **REJECTED** - Sentiment score too low ({sent_score:.1f} < 40)"
                    )
        else:
            reasoning.append(
                f"❌ **REJECTED** - Composite score too low ({composite:.1f} < 55)"
            )

        return jsonify(
            {
                "symbol": symbol.upper(),
                "timestamp": score_data["timestamp"],
                "scores": {
                    "composite": round(composite, 2),
                    "technical": round(tech_score, 2),
                    "fundamental": round(fund_score, 2),
                    "sentiment": round(sent_score, 2),
                },
                "news_analysis": news_data,
                "reasoning": reasoning,
                "decision_logic": {
                    "threshold_composite": 55.0,
                    "threshold_technical": 40.0,
                    "threshold_sentiment": 40.0,
                    "qualified": composite >= 55.0
                    and tech_score >= 40.0
                    and sent_score >= 40.0,
                },
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/trade/start", methods=["POST"])
@login_required
def start_trading():
    """Start continuous trading."""
    if is_bot_running():
        return jsonify({"status": "already_running"})

    try:
        project_root = Path(__file__).parent.parent
        bot_script = project_root / "bot_control.sh"

        result = subprocess.run(
            [str(bot_script), "start"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return jsonify({"status": "started", "message": result.stdout.strip()})
        else:
            return jsonify({"status": "error", "message": result.stderr.strip()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/trade/stop", methods=["POST"])
@login_required
def stop_trading():
    """Stop continuous trading."""
    if not is_bot_running():
        return jsonify({"status": "not_running"})

    try:
        project_root = Path(__file__).parent.parent
        bot_script = project_root / "bot_control.sh"

        result = subprocess.run(
            [str(bot_script), "stop"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return jsonify({"status": "stopped", "message": result.stdout.strip()})
        else:
            return jsonify({"status": "error", "message": result.stderr.strip()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/trade/eod", methods=["POST"])
@login_required
def end_of_day_close():
    """Close all positions."""
    try:
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            [str(project_root / "run.sh"), "eod"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        return jsonify({"status": "completed", "output": result.stdout})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/trade/manage", methods=["POST"])
@login_required
def manage_positions():
    """Manage positions (check stop losses)."""
    try:
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            [str(project_root / "run.sh"), "manage"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        return jsonify({"status": "completed", "output": result.stdout})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/rebalancing/stats")
@login_required
def api_rebalancing_stats():
    """Get rebalancing statistics."""
    try:
        tracker = PositionTracker()
        stats = tracker.get_rebalancing_stats()
        last_rebalance = tracker.get_last_rebalance_time()

        return jsonify(
            {
                "total_rebalances": stats["total_rebalances"],
                "avg_score_improvement": stats["avg_score_improvement"],
                "min_score_improvement": stats["min_score_improvement"],
                "max_score_improvement": stats["max_score_improvement"],
                "can_rebalance_now": tracker.can_rebalance_now(
                    config.trading.rebalance_cooldown_minutes
                ),
                "last_rebalance_time": (
                    last_rebalance.isoformat() if last_rebalance else None
                ),
                "cooldown_minutes": config.trading.rebalance_cooldown_minutes,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rebalancing/history")
@login_required
def api_rebalancing_history():
    """Get rebalancing history."""
    try:
        tracker = PositionTracker()
        limit = int(request.args.get("limit", 20))
        history = tracker.get_rebalancing_history(limit=limit)

        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions/<symbol>/details")
@login_required
def api_position_details(symbol: str):
    """Get detailed information for a specific position."""
    try:
        from datetime import datetime

        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )

        # Get current position from Alpaca
        positions = client.get_all_positions()
        position = None
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                position = pos
                break

        if not position:
            return jsonify({"error": f"Position {symbol} not found"}), 404

        # Get tracker data for entry metadata
        tracker = PositionTracker()
        position_data = tracker.data.get("positions", {}).get(symbol.upper(), {})

        # Calculate hold duration
        entry_time_str = position_data.get("entry_time")
        hold_duration_str = "-"
        entry_datetime = None
        if entry_time_str:
            entry_datetime = datetime.fromisoformat(entry_time_str)
            now = datetime.now(UTC)
            duration = now - entry_datetime
            days = duration.days
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            if days > 0:
                hold_duration_str = f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                hold_duration_str = f"{hours}h {minutes}m"
            else:
                hold_duration_str = f"{minutes}m"

        # Get score breakdown
        score_breakdown = position_data.get("score_breakdown", {})
        entry_score = position_data.get("score", 0)
        entry_reason = position_data.get("reason", "unknown")

        # Calculate current P&L metrics
        entry_price = float(position.avg_entry_price)
        current_price = (
            float(position.current_price) if position.current_price else entry_price
        )
        unrealized_pl = float(position.unrealized_pl) if position.unrealized_pl else 0
        unrealized_plpc = (
            float(position.unrealized_plpc) * 100 if position.unrealized_plpc else 0
        )

        # Calculate stop loss and take profit levels based on config
        # Note: stop_loss_pct and take_profit_pct are already decimals (0.03 = 3%)
        stop_loss_pct = config.trading.stop_loss_pct
        take_profit_pct = config.trading.take_profit_pct
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)

        # Distance to stop/target
        pct_to_stop = ((current_price - stop_loss_price) / current_price) * 100
        pct_to_target = ((take_profit_price - current_price) / current_price) * 100

        # Build exit plan explanation
        exit_plan = []
        exit_plan.append(
            f"**Stop Loss**: ${stop_loss_price:.2f} ({stop_loss_pct * 100:.0f}% below entry)"
        )
        exit_plan.append(f"  → Currently {pct_to_stop:.1f}% above stop loss")
        exit_plan.append(
            f"**Take Profit**: ${take_profit_price:.2f} ({take_profit_pct * 100:.0f}% above entry)"
        )
        exit_plan.append(f"  → Currently {pct_to_target:.1f}% from target")
        exit_plan.append("")
        exit_plan.append("**Exit Conditions:**")
        exit_plan.append("• Price hits stop loss → Automatic sell")
        exit_plan.append("• Price hits take profit → Automatic sell")
        exit_plan.append("• Score drops significantly → May rebalance (if unlocked)")
        exit_plan.append("• Better opportunity found → May rebalance (if unlocked)")

        # Build entry reasoning
        entry_reasoning = []
        if entry_reason == "new_position":
            entry_reasoning.append("📊 **Entered as new position**")
        elif entry_reason == "rebalancing":
            entry_reasoning.append(
                "🔄 **Entered via rebalancing** (replaced lower-scoring position)"
            )
        elif entry_reason == "unknown":
            entry_reasoning.append(
                "📊 **Legacy position** (entered before tracking was added)"
            )
        else:
            entry_reasoning.append(f"📊 **Entry type:** {entry_reason}")

        # Check if score_breakdown has actual values
        has_breakdown = score_breakdown and any(v > 0 for v in score_breakdown.values())

        if has_breakdown:
            entry_reasoning.append("")
            entry_reasoning.append("**Score Breakdown at Entry:**")
            tech = score_breakdown.get("technical", 0)
            sent = score_breakdown.get("sentiment", 0)
            fund = score_breakdown.get("fundamental", 0)
            entry_reasoning.append(f"• Technical: {tech:.1f}/100")
            entry_reasoning.append(f"• Sentiment: {sent:.1f}/100")
            entry_reasoning.append(f"• Fundamental: {fund:.1f}/100")
            entry_reasoning.append(f"• **Composite: {entry_score:.1f}/100**")
        elif entry_score > 0:
            entry_reasoning.append("")
            entry_reasoning.append(f"**Entry Score:** {entry_score:.1f}/100")
            entry_reasoning.append(
                "(Detailed breakdown not available for this position)"
            )

        # News sentiment at entry
        news_sentiment = position_data.get("news_sentiment")
        news_count = position_data.get("news_count")
        if news_sentiment is not None:
            entry_reasoning.append("")
            entry_reasoning.append(
                f"**News at Entry:** {news_count or 0} articles, sentiment: {news_sentiment:.3f}"
            )

        return jsonify(
            {
                "symbol": symbol.upper(),
                "qty": float(position.qty),
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": (
                    float(position.market_value) if position.market_value else 0
                ),
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "entry_time": entry_time_str,
                "entry_time_formatted": (
                    entry_datetime.strftime("%Y-%m-%d %H:%M UTC")
                    if entry_datetime
                    else "-"
                ),
                "hold_duration": hold_duration_str,
                "entry_score": entry_score,
                "score_breakdown": score_breakdown,
                "entry_reasoning": entry_reasoning,
                "exit_plan": exit_plan,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "pct_to_stop": pct_to_stop,
                "pct_to_target": pct_to_target,
                "is_locked": tracker.is_locked(symbol.upper()),
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/news/<symbol>")
@login_required
def api_symbol_news(symbol: str):
    """Get latest news for a symbol."""
    try:
        days_back = request.args.get("days", 7, type=int)
        max_articles = request.args.get("limit", 10, type=int)

        # Fetch news from Alpaca
        articles = newsapi_client.get_stock_news(
            symbol=symbol.upper(), days_back=days_back, max_articles=max_articles
        )

        # Format articles for frontend
        formatted_articles = []
        for article in articles:
            # Parse and format the date
            published_at = article.get("publishedAt", "")
            try:
                if published_at:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    formatted_date = dt.strftime("%b %d, %Y %H:%M")
                else:
                    formatted_date = "Unknown"
            except Exception:
                formatted_date = (
                    published_at[:16] if len(published_at) > 16 else published_at
                )

            formatted_articles.append(
                {
                    "title": article.get("title", "No title"),
                    "description": (
                        article.get("description", "")[:200] + "..."
                        if len(article.get("description", "")) > 200
                        else article.get("description", "")
                    ),
                    "url": article.get("url", ""),
                    "source": article.get("source", "Unknown"),
                    "publishedAt": formatted_date,
                    "symbols": article.get("symbols", []),
                }
            )

        return jsonify(
            {
                "symbol": symbol.upper(),
                "count": len(formatted_articles),
                "articles": formatted_articles,
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/positions/lock/<symbol>", methods=["POST"])
@login_required
def api_lock_position(symbol: str):
    """Lock a position to prevent rebalancing."""
    try:
        tracker = PositionTracker()
        tracker.lock_position(symbol.upper())

        return jsonify(
            {
                "status": "success",
                "symbol": symbol.upper(),
                "locked": True,
                "message": f"Position {symbol.upper()} locked successfully",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/positions/unlock/<symbol>", methods=["POST"])
@login_required
def api_unlock_position(symbol: str):
    """Unlock a position to allow rebalancing."""
    try:
        tracker = PositionTracker()
        tracker.unlock_position(symbol.upper())

        return jsonify(
            {
                "status": "success",
                "symbol": symbol.upper(),
                "locked": False,
                "message": f"Position {symbol.upper()} unlocked successfully",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/performance/stats")
@login_required
def api_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        performance_tracker = PerformanceTracker()
        days = request.args.get("days", type=int)

        metrics = performance_tracker.get_performance_metrics(days=days)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/score-analysis")
@login_required
def api_score_analysis():
    """Get score correlation analysis."""
    try:
        performance_tracker = PerformanceTracker()
        analysis = performance_tracker.get_score_correlation_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/exit-analysis")
@login_required
def api_exit_analysis():
    """Get exit reason analysis."""
    try:
        performance_tracker = PerformanceTracker()
        analysis = performance_tracker.get_exit_reason_analysis()
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/recent-trades")
@login_required
def api_recent_trades():
    """Get recent trades."""
    try:
        performance_tracker = PerformanceTracker()
        limit = request.args.get("limit", 20, type=int)
        trades = performance_tracker.get_recent_trades(limit=limit)
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/daily")
@login_required
def api_daily_performance():
    """Get daily performance stats."""
    try:
        performance_tracker = PerformanceTracker()
        days = request.args.get("days", 30, type=int)
        daily_stats = performance_tracker.get_daily_performance(days=days)
        return jsonify(daily_stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimizer/analyze")
@login_required
def api_optimizer_analyze():
    """Get strategy optimization analysis and recommendations."""
    try:
        optimizer = StrategyOptimizer()
        result = optimizer.analyze_and_recommend()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/optimizer")
@login_required
def optimizer_page():
    """Strategy optimizer page."""
    return render_template("optimizer.html")


# ============ AUTO-LEARNING ENDPOINTS ============


@app.route("/api/learning/stats")
@login_required
def api_learning_stats():
    """Get auto-learning system statistics."""
    try:
        stats = auto_learner.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/history")
@login_required
def api_learning_history():
    """Get parameter change history (audit trail)."""
    try:
        limit = request.args.get("limit", 20, type=int)
        history = auto_learner.get_parameter_history(limit=limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/equity-history")
@login_required
def api_learning_equity_history():
    """Get equity history for charting."""
    try:
        days = request.args.get("days", 30, type=int)
        history = auto_learner.get_equity_history(days=days)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/toggle", methods=["POST"])
@login_required
def api_learning_toggle():
    """Enable or disable auto-learning."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        enabled = data.get("enabled")
        if enabled is None:
            return jsonify({"error": "Missing 'enabled' field"}), 400

        if enabled:
            auto_learner.enable_learning()
            message = "Auto-learning enabled"
        else:
            auto_learner.disable_learning()
            message = "Auto-learning disabled (optimization still runs but won't apply changes)"

        return jsonify(
            {
                "status": "success",
                "message": message,
                "learning_enabled": auto_learner.state.get("learning_enabled", True),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning/force-optimize", methods=["POST"])
@login_required
def api_learning_force_optimize():
    """Force an optimization cycle regardless of trade count."""
    try:
        result = auto_learner.force_optimization()
        return jsonify(result)
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/config/weights", methods=["GET"])
@login_required
def api_get_weights():
    """Get current score weights and threshold."""
    try:
        return jsonify(
            {
                "weights": {
                    "technical": config.trading.weight_technical,
                    "sentiment": config.trading.weight_sentiment,
                    "fundamental": config.trading.weight_fundamental,
                },
                "threshold": config.trading.min_composite_score,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/weights", methods=["POST"])
@login_required
def api_update_weights():
    """Update score weights and threshold.

    Expects JSON body:
    {
        "technical": 0.50,
        "sentiment": 0.50,
        "fundamental": 0.00,
        "threshold": 72.5
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract values
        tech_weight = data.get("technical")
        sent_weight = data.get("sentiment")
        fund_weight = data.get("fundamental")
        threshold = data.get("threshold")

        # Validate weights sum to 1.0
        if (
            tech_weight is not None
            and sent_weight is not None
            and fund_weight is not None
        ):
            weight_sum = tech_weight + sent_weight + fund_weight
            if abs(weight_sum - 1.0) > 0.01:
                return (
                    jsonify(
                        {
                            "error": f"Weights must sum to 100%. Current sum: {weight_sum * 100:.1f}%"
                        }
                    ),
                    400,
                )

            # Validate individual weights
            for name, val in [
                ("technical", tech_weight),
                ("sentiment", sent_weight),
                ("fundamental", fund_weight),
            ]:
                if val < 0 or val > 1:
                    return (
                        jsonify({"error": f"{name} weight must be between 0 and 1"}),
                        400,
                    )

        # Validate threshold
        if threshold is not None:
            if threshold < 0 or threshold > 100:
                return jsonify({"error": "Threshold must be between 0 and 100"}), 400

        # Update config values in memory
        if tech_weight is not None:
            config.trading.weight_technical = tech_weight
        if sent_weight is not None:
            config.trading.weight_sentiment = sent_weight
        if fund_weight is not None:
            config.trading.weight_fundamental = fund_weight
        if threshold is not None:
            config.trading.min_composite_score = threshold

        # Save to .env file for persistence across restarts
        env_file = Path(__file__).parent.parent / ".env"
        env_updates = {}

        if tech_weight is not None:
            env_updates["WEIGHT_TECHNICAL"] = str(tech_weight)
        if sent_weight is not None:
            env_updates["WEIGHT_SENTIMENT"] = str(sent_weight)
        if fund_weight is not None:
            env_updates["WEIGHT_FUNDAMENTAL"] = str(fund_weight)
        if threshold is not None:
            env_updates["MIN_COMPOSITE_SCORE"] = str(threshold)

        if env_updates:
            _update_env_file(env_file, env_updates)

        return jsonify(
            {
                "status": "success",
                "message": "Configuration updated successfully",
                "weights": {
                    "technical": config.trading.weight_technical,
                    "sentiment": config.trading.weight_sentiment,
                    "fundamental": config.trading.weight_fundamental,
                },
                "threshold": config.trading.min_composite_score,
                "note": "Changes applied immediately. Bot restart not required.",
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def _update_env_file(env_file: Path, updates: dict[str, str]) -> None:
    """Update or add environment variables in .env file."""
    # Read existing content
    existing_lines = []
    if env_file.exists():
        with open(env_file) as f:
            existing_lines = f.readlines()

    # Track which keys we've updated
    updated_keys = set()
    new_lines = []

    for line in existing_lines:
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith("#"):
            # Check if this line contains a key we want to update
            for key, value in updates.items():
                if line_stripped.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    updated_keys.add(key)
                    break
            else:
                # Line doesn't match any update key, keep as is
                new_lines.append(line)
        else:
            # Empty line or comment, keep as is
            new_lines.append(line)

    # Add any keys that weren't already in the file
    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}\n")

    # Write back
    with open(env_file, "w") as f:
        f.writelines(new_lines)


# ============ PREMARKET ENDPOINTS ============

PREMARKET_CANDIDATES_FILE = (
    Path(__file__).parent.parent / "data" / "premarket_candidates.json"
)
PREMARKET_HISTORY_FILE = (
    Path(__file__).parent.parent / "data" / "premarket_history.json"
)


@app.route("/api/premarket/candidates")
@login_required
def api_premarket_candidates():
    """Get current premarket candidates queued for market open."""
    try:
        if not PREMARKET_CANDIDATES_FILE.exists():
            return jsonify(
                {
                    "scan_time": None,
                    "candidates": [],
                    "count": 0,
                    "message": "No premarket scan has been run yet",
                }
            )

        with open(PREMARKET_CANDIDATES_FILE) as f:
            data = json.load(f)

        # Check if scan is from today
        scan_time = data.get("scan_time")
        is_today = False
        if scan_time:
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo

            et = ZoneInfo("America/New_York")
            scan_dt = datetime.fromisoformat(scan_time)
            today = datetime.now(et).date()
            is_today = scan_dt.date() == today

        return jsonify(
            {
                "scan_time": scan_time,
                "candidates": data.get("candidates", [])[:20],  # Return top 20
                "count": data.get("count", 0),
                "is_today": is_today,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/premarket/history")
@login_required
def api_premarket_history():
    """Get historical premarket performance data."""
    try:
        if not PREMARKET_HISTORY_FILE.exists():
            return jsonify([])

        with open(PREMARKET_HISTORY_FILE) as f:
            history = json.load(f)

        # Return most recent first
        limit = request.args.get("limit", 30, type=int)
        return jsonify(list(reversed(history[-limit:])))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/premarket/stats")
@login_required
def api_premarket_stats():
    """Get aggregated premarket performance statistics."""
    try:
        if not PREMARKET_HISTORY_FILE.exists():
            return jsonify(
                {
                    "total_days": 0,
                    "total_executed": 0,
                    "total_closed": 0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0,
                    "avg_profit_pct": 0,
                    "total_profit_pct": 0,
                    "best_trade": None,
                    "worst_trade": None,
                }
            )

        with open(PREMARKET_HISTORY_FILE) as f:
            history = json.load(f)

        total_days = len(history)
        total_executed = 0
        total_closed = 0
        win_count = 0
        loss_count = 0
        total_profit_pct = 0
        best_trade = None
        worst_trade = None
        best_pct = float("-inf")
        worst_pct = float("inf")

        for record in history:
            for trade in record.get("executed", []):
                total_executed += 1

                if "profit_loss_pct" in trade:
                    total_closed += 1
                    pct = trade["profit_loss_pct"]
                    total_profit_pct += pct

                    if pct >= 0:
                        win_count += 1
                    else:
                        loss_count += 1

                    if pct > best_pct:
                        best_pct = pct
                        best_trade = {
                            "symbol": trade["symbol"],
                            "profit_pct": pct,
                            "date": record["date"],
                        }

                    if pct < worst_pct:
                        worst_pct = pct
                        worst_trade = {
                            "symbol": trade["symbol"],
                            "profit_pct": pct,
                            "date": record["date"],
                        }

        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
        avg_profit_pct = total_profit_pct / total_closed if total_closed > 0 else 0

        return jsonify(
            {
                "total_days": total_days,
                "total_executed": total_executed,
                "total_closed": total_closed,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_rate, 1),
                "avg_profit_pct": round(avg_profit_pct, 2),
                "total_profit_pct": round(total_profit_pct, 2),
                "best_trade": best_trade,
                "worst_trade": worst_trade,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ POSITION SIZING ENDPOINTS ============


@app.route("/api/position-sizing/info")
@login_required
def api_position_sizing_info():
    """Get current position sizing parameters and portfolio status."""
    try:
        from src.position_sizer import position_sizer

        portfolio_info = position_sizer.get_portfolio_info()
        should_deploy, deploy_reason = position_sizer.should_deploy_cash(portfolio_info)

        # Get regime data for max positions
        regime_data = market_regime_detector.get_market_regime()
        base_max = regime_data.get("max_positions_override", 10)

        # Calculate adjusted max positions
        adjusted_max, max_reason = (
            position_sizer.get_max_positions_with_cash_deployment(
                base_max=base_max,
                portfolio_info=portfolio_info,
                qualified_count=10,  # Estimate
            )
        )

        return jsonify(
            {
                "portfolio": {
                    "value": portfolio_info["portfolio_value"],
                    "cash": portfolio_info["cash"],
                    "cash_pct": round(
                        portfolio_info["cash"]
                        / portfolio_info["portfolio_value"]
                        * 100,
                        1,
                    ),
                    "invested": portfolio_info["invested_value"],
                    "position_count": portfolio_info["position_count"],
                },
                "sizing_params": {
                    "min_position_pct": position_sizer.MIN_POSITION_PCT * 100,
                    "max_position_pct": position_sizer.MAX_POSITION_PCT * 100,
                    "min_cash_reserve_pct": position_sizer.MIN_CASH_RESERVE_PCT * 100,
                    "cash_deploy_trigger_pct": position_sizer.CASH_DEPLOYMENT_TRIGGER_PCT
                    * 100,
                    "max_new_positions_per_day": position_sizer.MAX_NEW_POSITIONS_PER_DAY,
                },
                "cash_deployment": {
                    "should_deploy": should_deploy,
                    "reason": deploy_reason,
                    "base_max_positions": base_max,
                    "adjusted_max_positions": adjusted_max,
                    "adjustment_reason": max_reason,
                },
                "conviction_tiers": [
                    {"min_score": tier[0], "multiplier": tier[1]}
                    for tier in position_sizer.CONVICTION_TIERS
                ],
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/position-sizing/calculate/<symbol>")
@login_required
def api_calculate_position_size(symbol: str):
    """Calculate position size for a specific symbol."""
    try:
        from src.data_provider import alpaca_provider
        from src.position_sizer import position_sizer
        from src.strategy import strategy

        # Get current price
        latest = alpaca_provider.get_latest_trade(symbol.upper())
        current_price = latest["price"]

        # Get score (or use provided score parameter)
        score = request.args.get("score", type=float)
        if score is None:
            _should_buy, score, _reasoning = strategy.should_buy(symbol.upper())

        # Calculate position size
        result = position_sizer.calculate_position_size(
            symbol=symbol.upper(),
            current_price=current_price,
            composite_score=score,
        )

        return jsonify(
            {
                "symbol": result.symbol,
                "current_price": current_price,
                "score": result.conviction_score,
                "recommended_size": result.recommended_size,
                "recommended_shares": result.recommended_shares,
                "conviction_multiplier": result.conviction_multiplier,
                "volatility_multiplier": result.volatility_multiplier,
                "base_size": result.base_size,
                "capped": result.capped,
                "cap_reason": result.cap_reason,
                "rationale": result.rationale,
            }
        )
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ============ MONITORING AND METRICS ENDPOINTS ============


@app.route("/prom")
def metrics_endpoint():
    """Prometheus metrics endpoint (no auth required for monitoring)."""
    from src.metrics import trading_metrics

    return (
        trading_metrics.get_metrics(),
        200,
        {"Content-Type": "text/plain; charset=utf-8"},
    )


@app.route("/health")
def health_check():
    """Health check endpoint for load balancers."""
    try:
        # Check if we can connect to Alpaca
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        client.get_account()
        status = "healthy"
        code = 200
    except Exception:
        status = "unhealthy"
        code = 503

    return (
        jsonify(
            {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
            }
        ),
        code,
    )


@app.route("/api/watchdog/status")
@login_required
def api_watchdog_status():
    """Get watchdog/heartbeat status for the trading bot."""
    try:
        from src.watchdog import check_health_file

        health = check_health_file()
        return jsonify(health)
    except Exception as e:
        return jsonify({"error": str(e), "is_healthy": False}), 500


@app.route("/api/bot/health")
def api_bot_health():
    """
    Bot health check endpoint for external monitoring (Docker, systemd, etc.).

    Returns 200 if bot is healthy, 503 if unhealthy.
    No authentication required for monitoring tools.
    """
    try:
        from src.watchdog import check_health_file

        health = check_health_file()

        if health.get("is_healthy", False):
            return jsonify(health), 200
        else:
            return jsonify(health), 503
    except Exception as e:
        return jsonify({"error": str(e), "is_healthy": False}), 503


@app.route("/ready")
def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if all required services are available
    checks = {
        "alpaca": False,
        "market_data": False,
    }

    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        client.get_account()
        checks["alpaca"] = True
    except Exception:
        pass

    try:
        from src.data_provider import alpaca_provider

        alpaca_provider.get_latest_trade("SPY")
        checks["market_data"] = True
    except Exception:
        pass

    all_ready = all(checks.values())

    return jsonify(
        {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }
    ), (200 if all_ready else 503)


@app.route("/monitoring")
@login_required
def metrics_view():
    """Metrics dashboard page."""
    return render_template("metrics.html")


@app.route("/grafana")
@login_required
def grafana_view():
    """Grafana-style metrics dashboard."""
    return render_template("grafana.html")


@app.route("/api/trading-data")
@login_required
def api_metrics():
    """JSON API endpoint for metrics data."""
    import time

    from src.metrics import trading_metrics

    # Get all metrics as a dictionary
    with trading_metrics._lock:
        metrics = trading_metrics._metrics.copy()
        uptime = time.time() - trading_metrics._start_time

    return jsonify(
        {
            # System
            "uptime_seconds": uptime,
            "bot_status": metrics.get("bot_status", "unknown"),
            "market_regime": metrics.get("market_regime", "unknown"),
            # Counters
            "trades_total": metrics.get("trades_total", 0),
            "trades_buy_total": metrics.get("trades_buy_total", 0),
            "trades_sell_total": metrics.get("trades_sell_total", 0),
            "orders_submitted_total": metrics.get("orders_submitted_total", 0),
            "orders_filled_total": metrics.get("orders_filled_total", 0),
            "orders_rejected_total": metrics.get("orders_rejected_total", 0),
            "orders_cancelled_total": metrics.get("orders_cancelled_total", 0),
            "errors_total": metrics.get("errors_total", 0),
            "api_calls_total": metrics.get("api_calls_total", 0),
            "scans_total": metrics.get("scans_total", 0),
            # Gauges
            "portfolio_value": metrics.get("portfolio_value", 0),
            "buying_power": metrics.get("buying_power", 0),
            "cash_balance": metrics.get("cash_balance", 0),
            "positions_count": metrics.get("positions_count", 0),
            "unrealized_pnl": metrics.get("unrealized_pnl", 0),
            "realized_pnl_today": metrics.get("realized_pnl_today", 0),
            "daily_pnl_percent": metrics.get("daily_pnl_percent", 0),
            "win_rate": metrics.get("win_rate", 0),
            "active_orders_count": metrics.get("active_orders_count", 0),
            # Trading specific
            "last_scan_duration_seconds": metrics.get("last_scan_duration_seconds", 0),
            "last_trade_timestamp": metrics.get("last_trade_timestamp", 0),
            "avg_trade_score": metrics.get("avg_trade_score", 0),
            "avg_position_hold_minutes": metrics.get("avg_position_hold_minutes", 0),
            "avg_trade_pnl_percent": metrics.get("avg_trade_pnl_percent", 0),
        }
    )


# ============ MAIN ENTRY POINT ============

if __name__ == "__main__":
    print("=" * 60)
    print("AI Trading Dashboard Starting")
    print("=" * 60)
    print()
    print("Dashboard URL: http://localhost:8080")
    print("Network URL:   http://0.0.0.0:8080")
    print()
    print("Features:")
    print("   - Live market status")
    print("   - Portfolio overview")
    print("   - Real-time positions")
    print("   - Order history")
    print("   - Live trading logs")
    print("   - One-click trading controls")
    print("   - Position sizing calculator")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    app.run(host="0.0.0.0", port=8082, debug=False)
