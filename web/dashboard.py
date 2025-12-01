#!/usr/bin/env python3
"""Web dashboard for AI Trading System."""

import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, redirect, url_for, session

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.trading.client import TradingClient
from config.config import config
from src.position_tracker import PositionTracker
from src.performance_tracker import PerformanceTracker  # type: ignore
from src.strategy_optimizer import StrategyOptimizer  # type: ignore
from src.llm_reason_generator import get_llm_reason_generator  # type: ignore

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
                return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
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


def get_market_status() -> Dict[str, Any]:
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


def get_account_info() -> Dict[str, Any]:
    """Get account information."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        account = client.get_account()

        equity = float(account.equity) if account.equity else 0
        last_equity = float(account.last_equity) if account.last_equity else equity

        return {
            "portfolio_value": float(account.portfolio_value)
            if account.portfolio_value
            else 0,
            "buying_power": float(account.buying_power) if account.buying_power else 0,
            "cash": float(account.cash) if account.cash else 0,
            "equity": equity,
            "last_equity": last_equity,
            "profit_loss": equity - last_equity,
            "profit_loss_pct": ((equity - last_equity) / last_equity * 100)
            if last_equity > 0
            else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def get_positions() -> List[Dict[str, Any]]:
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
                        (datetime.now() - entry_time).total_seconds() / 60
                    )
                entry_score = position_data.get("score")

            result.append(
                {
                    "symbol": symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price)
                    if pos.current_price
                    else 0,
                    "market_value": float(pos.market_value) if pos.market_value else 0,
                    "cost_basis": float(pos.cost_basis) if pos.cost_basis else 0,
                    "unrealized_pl": float(pos.unrealized_pl)
                    if pos.unrealized_pl
                    else 0,
                    "unrealized_plpc": float(pos.unrealized_plpc) * 100
                    if pos.unrealized_plpc
                    else 0,
                    "side": pos.side,
                    "is_locked": tracker.is_locked(symbol),
                    "hold_duration_min": hold_duration_min,
                    "entry_score": entry_score,
                }
            )

        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_recent_orders() -> List[Dict[str, Any]]:
    """Get recent orders."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        orders = client.get_orders()

        result = []
        for order in list(orders)[:20]:
            result.append(
                {
                    "id": str(order.id),
                    "symbol": order.symbol,
                    "qty": float(order.qty) if order.qty else 0,
                    "side": order.side.value if order.side else "unknown",
                    "type": order.type.value if order.type else "unknown",
                    "status": order.status.value if order.status else "unknown",
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": float(order.filled_avg_price)
                    if order.filled_avg_price
                    else 0,
                    "created_at": str(order.created_at),
                }
            )

        return result
    except Exception as e:
        return [{"error": str(e)}]


def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    try:
        log_file = Path(__file__).parent.parent / config.logging.log_file

        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r") as f:
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


def get_analysis_data() -> List[Dict[str, Any]]:
    """Get ticker analysis data from recent scans only (last 2 hours)."""
    try:
        from datetime import datetime, timedelta, timezone

        log_file = Path(__file__).parent.parent / config.logging.log_file

        if not log_file.exists():
            return []

        # Only show scans from last 2 hours (using UTC for comparison)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)

        analyses = []
        with open(log_file, "r") as f:
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
    """Check if trading bot is running via systemd service or PID file."""
    try:
        # First check systemd service (preferred method)
        result = subprocess.run(
            ["systemctl", "is-active", "ai-trader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip() == "active":
            return True
        
        # Fallback: check PID file for manual runs
        pid_file = Path(__file__).parent.parent / "logs" / "bot.pid"
        if not pid_file.exists():
            return False

        pid = int(pid_file.read_text().strip())
        # Check if process is actually running
        import os
        import signal

        try:
            os.kill(pid, 0)  # Signal 0 checks if process exists
            return True
        except OSError:
            # Process not found, clean up stale PID file
            pid_file.unlink(missing_ok=True)
            return False
    except Exception:
        return False


def get_nyse_time() -> Dict[str, Any]:
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
    """API endpoint for market status."""
    return jsonify(
        {
            "market": get_market_status(),
            "account": get_account_info(),
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
        with open(log_file, "r") as f:
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
            return jsonify(
                {
                    "error": f"No recent analysis found for {symbol}. Symbol may not have been scanned yet."
                }
            ), 404

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
        reasoning.append(f"**Final Composite Score Calculation:**")
        reasoning.append(
            f"({tech_score:.1f} × 40%) + ({sent_score:.1f} × 30%) + ({fund_score:.1f} × 30%) = **{composite:.1f}/100**"
        )
        reasoning.append("")

        # Decision logic
        if composite >= 55.0:
            if tech_score >= 40.0 and sent_score >= 40.0:
                reasoning.append(
                    f"✅ **QUALIFIED FOR TRADING** - All criteria met (composite ≥55, technical ≥40, sentiment ≥40)"
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
                "last_rebalance_time": last_rebalance.isoformat()
                if last_rebalance
                else None,
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
        from datetime import datetime, timezone

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
            now = datetime.now(timezone.utc)
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
        stop_loss_pct = config.trading.stop_loss_pct
        take_profit_pct = config.trading.take_profit_pct
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)

        # Distance to stop/target
        pct_to_stop = ((current_price - stop_loss_price) / current_price) * 100
        pct_to_target = ((take_profit_price - current_price) / current_price) * 100

        # Build exit plan explanation
        exit_plan = []
        exit_plan.append(
            f"**Stop Loss**: ${stop_loss_price:.2f} ({stop_loss_pct}% below entry)"
        )
        exit_plan.append(f"  → Currently {pct_to_stop:.1f}% above stop loss")
        exit_plan.append(
            f"**Take Profit**: ${take_profit_price:.2f} ({take_profit_pct}% above entry)"
        )
        exit_plan.append(f"  → Currently {pct_to_target:.1f}% from target")
        exit_plan.append("")
        exit_plan.append("**Exit Conditions:**")
        exit_plan.append("• Price hits stop loss → Automatic sell")
        exit_plan.append("• Price hits take profit → Automatic sell")
        exit_plan.append(f"• Score drops significantly → May rebalance (if unlocked)")
        exit_plan.append(f"• Better opportunity found → May rebalance (if unlocked)")

        # Build entry reasoning
        entry_reasoning = []
        if entry_reason == "new_position":
            entry_reasoning.append("📊 **Entered as new position**")
        elif entry_reason == "rebalancing":
            entry_reasoning.append(
                "🔄 **Entered via rebalancing** (replaced lower-scoring position)"
            )
        else:
            entry_reasoning.append(f"📊 **Entry type:** {entry_reason}")

        if score_breakdown:
            entry_reasoning.append("")
            entry_reasoning.append("**Score Breakdown at Entry:**")
            tech = score_breakdown.get("technical", 0)
            sent = score_breakdown.get("sentiment", 0)
            fund = score_breakdown.get("fundamental", 0)
            entry_reasoning.append(f"• Technical: {tech:.1f}/100")
            entry_reasoning.append(f"• Sentiment: {sent:.1f}/100")
            entry_reasoning.append(f"• Fundamental: {fund:.1f}/100")
            entry_reasoning.append(f"• **Composite: {entry_score:.1f}/100**")

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
                "market_value": float(position.market_value)
                if position.market_value
                else 0,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "entry_time": entry_time_str,
                "entry_time_formatted": entry_datetime.strftime("%Y-%m-%d %H:%M UTC")
                if entry_datetime
                else "-",
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


@app.route("/api/performance/metrics")
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
        if tech_weight is not None and sent_weight is not None and fund_weight is not None:
            weight_sum = tech_weight + sent_weight + fund_weight
            if abs(weight_sum - 1.0) > 0.01:
                return jsonify(
                    {
                        "error": f"Weights must sum to 100%. Current sum: {weight_sum * 100:.1f}%"
                    }
                ), 400

            # Validate individual weights
            for name, val in [("technical", tech_weight), ("sentiment", sent_weight), ("fundamental", fund_weight)]:
                if val < 0 or val > 1:
                    return jsonify({"error": f"{name} weight must be between 0 and 1"}), 400

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


def _update_env_file(env_file: Path, updates: Dict[str, str]) -> None:
    """Update or add environment variables in .env file."""
    # Read existing content
    existing_lines = []
    if env_file.exists():
        with open(env_file, "r") as f:
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


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI Trading Dashboard Starting")
    print("=" * 60)
    print()
    print("📊 Dashboard URL: http://localhost:8080")
    print("🌐 Network URL:   http://0.0.0.0:8080")
    print()
    print("✨ Features:")
    print("   • Live market status")
    print("   • Portfolio overview")
    print("   • Real-time positions")
    print("   • Order history")
    print("   • Live trading logs")
    print("   • One-click trading controls")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    app.run(host="0.0.0.0", port=8082, debug=False)
