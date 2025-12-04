#!/usr/bin/env python3
"""Prometheus metrics for the AI Trading System.

Provides metrics for monitoring trading performance, system health,
and operational insights via Prometheus-compatible endpoints.
"""

import threading
import time
from typing import Any


class TradingMetrics:
    """Prometheus-style metrics collector for trading system."""

    def __init__(self) -> None:
        """Initialize metrics storage."""
        self._lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            # Counters
            "trades_total": 0,
            "trades_buy_total": 0,
            "trades_sell_total": 0,
            "orders_submitted_total": 0,
            "orders_filled_total": 0,
            "orders_rejected_total": 0,
            "orders_cancelled_total": 0,
            "errors_total": 0,
            "api_calls_total": 0,
            "scans_total": 0,
            # Gauges
            "portfolio_value": 0.0,
            "buying_power": 0.0,
            "cash_balance": 0.0,
            "positions_count": 0,
            "unrealized_pnl": 0.0,
            "realized_pnl_today": 0.0,
            "daily_pnl_percent": 0.0,
            "win_rate": 0.0,
            "active_orders_count": 0,
            # Trading specific
            "last_scan_duration_seconds": 0.0,
            "last_trade_timestamp": 0,
            "market_regime": "unknown",
            "bot_status": "stopped",
            # Histograms (simplified as averages)
            "avg_trade_score": 0.0,
            "avg_position_hold_minutes": 0.0,
            "avg_trade_pnl_percent": 0.0,
        }
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        """Increment a counter metric."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name] += value

    def set(self, name: str, value: Any) -> None:
        """Set a gauge metric."""
        with self._lock:
            self._metrics[name] = value

    def get(self, name: str) -> Any:
        """Get a metric value."""
        with self._lock:
            return self._metrics.get(name)

    def record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: float,
        score: float | None = None,
    ) -> None:
        """Record a trade execution."""
        with self._lock:
            self._metrics["trades_total"] += 1
            if side.lower() == "buy":
                self._metrics["trades_buy_total"] += 1
            else:
                self._metrics["trades_sell_total"] += 1
            self._metrics["last_trade_timestamp"] = int(time.time())

            if score is not None:
                # Update rolling average score
                total = self._metrics["trades_total"]
                old_avg = self._metrics["avg_trade_score"]
                self._metrics["avg_trade_score"] = (
                    old_avg * (total - 1) + score
                ) / total

    def record_order(self, status: str) -> None:
        """Record an order status."""
        with self._lock:
            self._metrics["orders_submitted_total"] += 1
            if status == "filled":
                self._metrics["orders_filled_total"] += 1
            elif status == "rejected":
                self._metrics["orders_rejected_total"] += 1
            elif status == "cancelled":
                self._metrics["orders_cancelled_total"] += 1

    def record_error(self, error_type: str = "unknown") -> None:
        """Record an error occurrence."""
        with self._lock:
            self._metrics["errors_total"] += 1

    def record_scan(self, duration_seconds: float) -> None:
        """Record a market scan."""
        with self._lock:
            self._metrics["scans_total"] += 1
            self._metrics["last_scan_duration_seconds"] = duration_seconds

    def update_portfolio(
        self,
        portfolio_value: float,
        buying_power: float,
        cash: float,
        positions_count: int,
        unrealized_pnl: float,
    ) -> None:
        """Update portfolio metrics."""
        with self._lock:
            self._metrics["portfolio_value"] = portfolio_value
            self._metrics["buying_power"] = buying_power
            self._metrics["cash_balance"] = cash
            self._metrics["positions_count"] = positions_count
            self._metrics["unrealized_pnl"] = unrealized_pnl

    def get_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        with self._lock:
            uptime = time.time() - self._start_time
            lines = [
                "# HELP ai_trader_uptime_seconds Time since metrics collector started",
                "# TYPE ai_trader_uptime_seconds gauge",
                f"ai_trader_uptime_seconds {uptime:.2f}",
                "",
                "# HELP ai_trader_trades_total Total number of trades executed",
                "# TYPE ai_trader_trades_total counter",
                f'ai_trader_trades_total{{side="buy"}} {self._metrics["trades_buy_total"]}',
                f'ai_trader_trades_total{{side="sell"}} {self._metrics["trades_sell_total"]}',
                "",
                "# HELP ai_trader_orders_total Total number of orders by status",
                "# TYPE ai_trader_orders_total counter",
                f'ai_trader_orders_total{{status="submitted"}} {self._metrics["orders_submitted_total"]}',
                f'ai_trader_orders_total{{status="filled"}} {self._metrics["orders_filled_total"]}',
                f'ai_trader_orders_total{{status="rejected"}} {self._metrics["orders_rejected_total"]}',
                f'ai_trader_orders_total{{status="cancelled"}} {self._metrics["orders_cancelled_total"]}',
                "",
                "# HELP ai_trader_errors_total Total number of errors",
                "# TYPE ai_trader_errors_total counter",
                f"ai_trader_errors_total {self._metrics['errors_total']}",
                "",
                "# HELP ai_trader_scans_total Total number of market scans",
                "# TYPE ai_trader_scans_total counter",
                f"ai_trader_scans_total {self._metrics['scans_total']}",
                "",
                "# HELP ai_trader_portfolio_value Current portfolio value in USD",
                "# TYPE ai_trader_portfolio_value gauge",
                f"ai_trader_portfolio_value {self._metrics['portfolio_value']:.2f}",
                "",
                "# HELP ai_trader_buying_power Available buying power in USD",
                "# TYPE ai_trader_buying_power gauge",
                f"ai_trader_buying_power {self._metrics['buying_power']:.2f}",
                "",
                "# HELP ai_trader_cash_balance Cash balance in USD",
                "# TYPE ai_trader_cash_balance gauge",
                f"ai_trader_cash_balance {self._metrics['cash_balance']:.2f}",
                "",
                "# HELP ai_trader_positions_count Number of open positions",
                "# TYPE ai_trader_positions_count gauge",
                f"ai_trader_positions_count {self._metrics['positions_count']}",
                "",
                "# HELP ai_trader_unrealized_pnl Unrealized profit/loss in USD",
                "# TYPE ai_trader_unrealized_pnl gauge",
                f"ai_trader_unrealized_pnl {self._metrics['unrealized_pnl']:.2f}",
                "",
                "# HELP ai_trader_daily_pnl_percent Daily P&L percentage",
                "# TYPE ai_trader_daily_pnl_percent gauge",
                f"ai_trader_daily_pnl_percent {self._metrics['daily_pnl_percent']:.4f}",
                "",
                "# HELP ai_trader_win_rate Trading win rate percentage",
                "# TYPE ai_trader_win_rate gauge",
                f"ai_trader_win_rate {self._metrics['win_rate']:.2f}",
                "",
                "# HELP ai_trader_avg_trade_score Average composite score of trades",
                "# TYPE ai_trader_avg_trade_score gauge",
                f"ai_trader_avg_trade_score {self._metrics['avg_trade_score']:.2f}",
                "",
                "# HELP ai_trader_scan_duration_seconds Duration of last market scan",
                "# TYPE ai_trader_scan_duration_seconds gauge",
                f"ai_trader_scan_duration_seconds {self._metrics['last_scan_duration_seconds']:.2f}",
                "",
                "# HELP ai_trader_last_trade_timestamp Unix timestamp of last trade",
                "# TYPE ai_trader_last_trade_timestamp gauge",
                f"ai_trader_last_trade_timestamp {self._metrics['last_trade_timestamp']}",
                "",
                "# HELP ai_trader_info Trading bot information",
                "# TYPE ai_trader_info gauge",
                f'ai_trader_info{{version="1.0.0",regime="{self._metrics["market_regime"]}",status="{self._metrics["bot_status"]}"}} 1',
            ]
            return "\n".join(lines) + "\n"


# Global metrics instance
trading_metrics = TradingMetrics()
