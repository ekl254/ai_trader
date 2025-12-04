#!/usr/bin/env python3
"""Performance tracking and analytics for trading system."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import logger


class PerformanceTracker:
    """Track and analyze trading performance with detailed metrics."""

    def __init__(self, data_file: str = "data/performance_history.json"):
        """Initialize performance tracker."""
        self.data_file = Path(data_file)
        self.data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        """Load performance data from disk."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error("failed_to_load_performance_data", error=str(e))
                return self._create_empty_data()
        return self._create_empty_data()

    def _create_empty_data(self) -> dict[str, Any]:
        """Create empty performance data structure."""
        return {
            "trades": [],
            "daily_stats": {},
            "ml_training_data": [],
            "version": "1.0",
        }

    def _save_data(self) -> None:
        """Save performance data to disk."""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.error("failed_to_save_performance_data", error=str(e))

    def record_trade(
        self,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        entry_score: dict[str, float],
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        quantity: float,
        news_sentiment: float | None = None,
        news_count: int | None = None,
    ) -> None:
        """
        Record a completed trade with all context.

        Args:
            symbol: Stock symbol
            entry_time: When position was opened
            entry_price: Entry price per share
            entry_score: Dict with composite, technical, sentiment, fundamental scores
            exit_time: When position was closed
            exit_price: Exit price per share
            exit_reason: Reason for exit (stop_loss, rebalancing, eod, target_profit)
            quantity: Number of shares
            news_sentiment: Sentiment score at entry (optional)
            news_count: Number of news articles at entry (optional)
        """
        hold_duration = (exit_time - entry_time).total_seconds() / 60  # minutes
        cost = entry_price * quantity
        revenue = exit_price * quantity
        profit_loss = revenue - cost
        profit_loss_pct = (profit_loss / cost) * 100 if cost > 0 else 0

        trade_data = {
            "symbol": symbol,
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            "entry_score": entry_score,
            "exit_time": exit_time.isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "hold_duration_min": round(hold_duration, 2),
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 4),
            "news_sentiment_at_entry": news_sentiment,
            "news_count_at_entry": news_count,
            "trade_date": entry_time.date().isoformat(),
        }

        self.data["trades"].append(trade_data)
        self._update_daily_stats(trade_data)
        self._save_data()

        logger.info(
            "trade_recorded",
            symbol=symbol,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            hold_duration_min=hold_duration,
        )

    def _update_daily_stats(self, trade_data: dict) -> None:
        """Update daily performance statistics."""
        date = trade_data["trade_date"]

        if date not in self.data["daily_stats"]:
            self.data["daily_stats"][date] = {
                "trades": 0,
                "profit_loss": 0,
                "winners": 0,
                "losers": 0,
            }

        stats = self.data["daily_stats"][date]
        stats["trades"] += 1
        stats["profit_loss"] += trade_data["profit_loss"]

        if trade_data["profit_loss"] > 0:
            stats["winners"] += 1
        else:
            stats["losers"] += 1

    def get_performance_metrics(self, days: int | None = None) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            days: Limit to last N days (None = all time)

        Returns:
            Dict with performance metrics
        """
        trades = self.data["trades"]

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
            trades = [t for t in trades if t["trade_date"] >= cutoff]

        if not trades:
            return {
                "total_trades": 0,
                "message": "No trades recorded yet",
            }

        profitable = [t for t in trades if t["profit_loss"] > 0]
        losing = [t for t in trades if t["profit_loss"] <= 0]

        total_profit = sum(t["profit_loss"] for t in profitable)
        total_loss = sum(abs(t["profit_loss"]) for t in losing)
        net_profit = total_profit - total_loss

        # Calculate Sharpe ratio (annualized)
        returns = [t["profit_loss_pct"] for t in trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        sharpe = (
            (avg_return / (std_return + 0.0001)) * np.sqrt(252) if std_return else 0
        )

        # Max drawdown
        cumulative_pl = np.cumsum([t["profit_loss"] for t in trades])
        running_max = np.maximum.accumulate(cumulative_pl)
        drawdown = running_max - cumulative_pl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(profitable),
            "losing_trades": len(losing),
            "win_rate": round((len(profitable) / len(trades)) * 100, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_profit": round(net_profit, 2),
            "avg_profit_per_winning_trade": (
                round(total_profit / len(profitable), 2) if profitable else 0
            ),
            "avg_loss_per_losing_trade": (
                round(total_loss / len(losing), 2) if losing else 0
            ),
            "profit_factor": (
                round(total_profit / total_loss, 2) if total_loss > 0 else float("inf")
            ),
            "avg_hold_time_min": round(
                np.mean([t["hold_duration_min"] for t in trades]), 2
            ),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 2),
            "best_trade": max(trades, key=lambda t: t["profit_loss"]),
            "worst_trade": min(trades, key=lambda t: t["profit_loss"]),
            "avg_return_pct": round(avg_return, 4),
        }

    def get_score_correlation_analysis(self) -> dict[str, Any]:
        """Analyze correlation between entry scores and profitability."""
        trades = self.data["trades"]

        if len(trades) < 10:
            return {
                "status": "insufficient_data",
                "min_required": 10,
                "current": len(trades),
            }

        scores = [t["entry_score"]["composite"] for t in trades]
        profits = [t["profit_loss_pct"] for t in trades]

        correlation = np.corrcoef(scores, profits)[0, 1]
        # Handle NaN correlation (shouldn't happen with composite scores, but just in case)
        if np.isnan(correlation):
            correlation = 0.0

        # Analyze by score ranges
        ranges = {
            "55-60": [t for t in trades if 55 <= t["entry_score"]["composite"] < 60],
            "60-65": [t for t in trades if 60 <= t["entry_score"]["composite"] < 65],
            "65-70": [t for t in trades if 65 <= t["entry_score"]["composite"] < 70],
            "70-75": [t for t in trades if 70 <= t["entry_score"]["composite"] < 75],
            "75+": [t for t in trades if t["entry_score"]["composite"] >= 75],
        }

        range_analysis = {}
        for range_name, range_trades in ranges.items():
            if range_trades:
                winners = [t for t in range_trades if t["profit_loss"] > 0]
                range_analysis[range_name] = {
                    "count": len(range_trades),
                    "avg_profit_pct": round(
                        np.mean([t["profit_loss_pct"] for t in range_trades]), 4
                    ),
                    "win_rate": round((len(winners) / len(range_trades)) * 100, 2),
                    "total_profit": round(
                        sum(t["profit_loss"] for t in range_trades), 2
                    ),
                }

        # Analyze individual score components
        component_correlations = {}
        for component in ["technical", "sentiment", "fundamental"]:
            comp_scores = [t["entry_score"].get(component, 50) for t in trades]
            # Check if there's any variance in the scores
            if np.std(comp_scores) < 0.001:  # No variance
                component_correlations[component] = None
            else:
                comp_corr = np.corrcoef(comp_scores, profits)[0, 1]
                # Handle NaN results
                if np.isnan(comp_corr):
                    component_correlations[component] = None
                else:
                    component_correlations[component] = round(comp_corr, 3)

        interpretation = (
            "Strong positive"
            if correlation > 0.7
            else (
                "Moderate positive"
                if correlation > 0.4
                else (
                    "Weak positive"
                    if correlation > 0.2
                    else (
                        "Weak negative"
                        if correlation > -0.2
                        else (
                            "Moderate negative"
                            if correlation > -0.4
                            else "Strong negative"
                        )
                    )
                )
            )
        )

        return {
            "overall_correlation": round(correlation, 3),
            "interpretation": interpretation,
            "score_ranges": range_analysis,
            "component_correlations": component_correlations,
            "recommendation": self._generate_score_recommendations(
                correlation, component_correlations, range_analysis
            ),
        }

    def _generate_score_recommendations(
        self,
        overall_corr: float,
        component_corrs: dict[str, float],
        range_analysis: dict,
    ) -> list[str]:
        """Generate actionable recommendations based on score analysis."""
        recommendations = []

        # Overall score threshold
        if overall_corr > 0.5:
            recommendations.append(
                "✅ Score system is working well - higher scores correlate with better profits"
            )

            # Find best performing range
            best_range = max(
                range_analysis.items(), key=lambda x: x[1]["avg_profit_pct"]
            )
            if best_range[0] != "55-60":
                recommendations.append(
                    f"💡 Consider increasing minimum score threshold to {best_range[0].split('-')[0]} (best performing range)"
                )
        else:
            recommendations.append(
                "⚠️ Weak correlation between scores and profits - review scoring methodology"
            )

        # Component analysis (filter out None values)
        valid_component_corrs = {
            k: v for k, v in component_corrs.items() if v is not None
        }

        if valid_component_corrs:
            best_component = max(valid_component_corrs.items(), key=lambda x: abs(x[1]))
            worst_component = min(
                valid_component_corrs.items(), key=lambda x: abs(x[1])
            )

            if abs(best_component[1]) > 0.5:
                recommendations.append(
                    f"✅ {best_component[0].capitalize()} score is most predictive ({best_component[1]:.2f} correlation)"
                )

            if abs(worst_component[1]) < 0.2:
                recommendations.append(
                    f"❌ {worst_component[0].capitalize()} score shows weak correlation ({worst_component[1]:.2f}) - consider reducing its weight"
                )

        # Check for components with no variance
        none_components = [k for k, v in component_corrs.items() if v is None]
        if none_components:
            recommendations.append(
                f"⚠️ {', '.join(c.capitalize() for c in none_components)} shows no variance (all same value) - consider removing or diversifying"
            )

        return recommendations

    def get_exit_reason_analysis(self) -> dict[str, Any]:
        """Analyze performance by exit reason."""
        trades = self.data["trades"]

        if not trades:
            return {"message": "No trades recorded"}

        exit_reasons = {}
        for trade in trades:
            reason = trade["exit_reason"]
            if reason not in exit_reasons:
                exit_reasons[reason] = []
            exit_reasons[reason].append(trade)

        analysis = {}
        for reason, reason_trades in exit_reasons.items():
            winners = [t for t in reason_trades if t["profit_loss"] > 0]
            analysis[reason] = {
                "count": len(reason_trades),
                "win_rate": round((len(winners) / len(reason_trades)) * 100, 2),
                "avg_profit": round(
                    np.mean([t["profit_loss"] for t in reason_trades]), 2
                ),
                "avg_profit_pct": round(
                    np.mean([t["profit_loss_pct"] for t in reason_trades]), 4
                ),
                "total_profit": round(sum(t["profit_loss"] for t in reason_trades), 2),
            }

        return analysis

    def get_recent_trades(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most recent trades."""
        trades = sorted(
            self.data["trades"], key=lambda t: t["entry_time"], reverse=True
        )
        return trades[:limit]

    def get_daily_performance(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily performance statistics for last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        daily_stats = []
        for date, stats in sorted(self.data["daily_stats"].items()):
            if date >= cutoff:
                daily_stats.append(
                    {
                        "date": date,
                        "trades": stats["trades"],
                        "profit_loss": round(stats["profit_loss"], 2),
                        "winners": stats["winners"],
                        "losers": stats["losers"],
                        "win_rate": (
                            round((stats["winners"] / stats["trades"]) * 100, 2)
                            if stats["trades"] > 0
                            else 0
                        ),
                    }
                )

        return daily_stats

    def export_ml_training_data(self) -> list[dict[str, Any]]:
        """Export trades in ML-ready format."""
        ml_data = []

        for trade in self.data["trades"]:
            entry_time = datetime.fromisoformat(trade["entry_time"])

            ml_data.append(
                {
                    # Features (inputs)
                    "composite_score": trade["entry_score"]["composite"],
                    "technical_score": trade["entry_score"]["technical"],
                    "sentiment_score": trade["entry_score"]["sentiment"],
                    "fundamental_score": trade["entry_score"]["fundamental"],
                    "news_sentiment": trade.get("news_sentiment_at_entry", 0),
                    "news_count": trade.get("news_count_at_entry", 0),
                    "hour_of_day": entry_time.hour,
                    "day_of_week": entry_time.weekday(),
                    # Targets (outputs to predict)
                    "profit_loss_pct": trade["profit_loss_pct"],
                    "was_profitable": 1 if trade["profit_loss"] > 0 else 0,
                    "hold_duration_min": trade["hold_duration_min"],
                    "exit_reason": trade["exit_reason"],
                }
            )

        return ml_data
