#!/usr/bin/env python3
"""Web dashboard for AI Trading System."""

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.trading.client import TradingClient
from config.config import config

app = Flask(__name__)

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
            "portfolio_value": float(account.portfolio_value) if account.portfolio_value else 0,
            "buying_power": float(account.buying_power) if account.buying_power else 0,
            "cash": float(account.cash) if account.cash else 0,
            "equity": equity,
            "last_equity": last_equity,
            "profit_loss": equity - last_equity,
            "profit_loss_pct": ((equity - last_equity) / last_equity * 100) if last_equity > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def get_positions() -> List[Dict[str, Any]]:
    """Get current positions."""
    try:
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        positions = client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price) if pos.current_price else 0,
                "market_value": float(pos.market_value) if pos.market_value else 0,
                "cost_basis": float(pos.cost_basis) if pos.cost_basis else 0,
                "unrealized_pl": float(pos.unrealized_pl) if pos.unrealized_pl else 0,
                "unrealized_plpc": float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0,
                "side": pos.side,
            })
        
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
            result.append({
                "id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty) if order.qty else 0,
                "side": order.side.value if order.side else "unknown",
                "type": order.type.value if order.type else "unknown",
                "status": order.status.value if order.status else "unknown",
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                "created_at": str(order.created_at),
            })
        
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
                            log_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            
                            # Skip if too old (compare in UTC)
                            if log_time < cutoff_time:
                                continue
                        
                        symbol = log.get("symbol")
                        composite = log.get("composite", 0)
                        technical = log.get("technical", 0)
                        fundamental = log.get("fundamental", 0)
                        sentiment = log.get("sentiment", 0)
                        
                        analyses.append({
                            "symbol": symbol,
                            "composite_score": round(composite, 2),
                            "technical_score": round(technical, 2),
                            "fundamental_score": round(fundamental, 2),
                            "sentiment_score": round(sentiment, 2),
                            "timestamp": timestamp_str,
                            "traded": composite >= 55.0 and technical >= 40.0 and sentiment >= 40.0,
                            "reason": _get_trade_reason(composite, technical, fundamental, sentiment)
                        })
                        
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Return most recent first
        return list(reversed(analyses))
    except Exception as e:
        return [{"error": str(e)}]


def _get_trade_reason(composite: float, technical: float, fundamental: float, sentiment: float) -> str:
    """Generate human-readable trade decision reason."""
    if composite >= 55.0 and technical >= 40.0 and sentiment >= 40.0:
        return "✓ TRADED - Meets all criteria"
    
    reasons = []
    if composite < 55.0:
        reasons.append(f"Composite score too low ({composite:.1f} < 55.0)")
    if technical < 40.0:
        reasons.append(f"Technical score too low ({technical:.1f} < 40.0)")
    if sentiment < 40.0:
        reasons.append(f"Sentiment score too low ({sentiment:.1f} < 40.0)")
    
    return "✗ NOT TRADED - " + "; ".join(reasons)


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    """API endpoint for market status."""
    global trading_running
    return jsonify({
        "market": get_market_status(),
        "account": get_account_info(),
        "trading_running": trading_running,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/positions")
def api_positions():
    """API endpoint for positions."""
    return jsonify(get_positions())


@app.route("/api/orders")
def api_orders():
    """API endpoint for orders."""
    return jsonify(get_recent_orders())


@app.route("/api/logs")
def api_logs():
    """API endpoint for logs."""
    limit = int(request.args.get("limit", 50))
    return jsonify(get_recent_logs(limit))


@app.route("/analysis")
def analysis_page():
    """Analysis page showing ticker scores."""
    return render_template("analysis.html")


@app.route("/api/analysis")
def api_analysis():
    """API endpoint for analysis data."""
    return jsonify(get_analysis_data())


@app.route("/api/trade/start", methods=["POST"])
def start_trading():
    """Start continuous trading."""
    global trading_process, trading_running
    
    if trading_running:
        return jsonify({"status": "already_running"})
    
    try:
        # Run continuous trading in background
        project_root = Path(__file__).parent.parent
        
        def run_continuous():
            global trading_running
            trading_running = True
            subprocess.run(
                [str(project_root / "run.sh"), "continuous"],
                cwd=str(project_root),
            )
            trading_running = False
        
        thread = threading.Thread(target=run_continuous, daemon=True)
        thread.start()
        
        return jsonify({"status": "started", "mode": "continuous"})
    except Exception as e:
        trading_running = False
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/trade/stop", methods=["POST"])
def stop_trading():
    """Stop continuous trading."""
    global trading_process, trading_running
    
    if not trading_running:
        return jsonify({"status": "not_running"})
    
    # The continuous loop will stop when market closes or on next check
    trading_running = False
    return jsonify({"status": "stopping"})


@app.route("/api/trade/eod", methods=["POST"])
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
    
    app.run(host="0.0.0.0", port=8080, debug=False)
