#!/usr/bin/env python3
"""Automated strategy optimization based on historical performance."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from config.config import config
from src.logger import logger
from src.performance_tracker import PerformanceTracker


class StrategyOptimizer:
    """Automatically optimize trading strategy parameters based on historical performance."""

    def __init__(self):
        """Initialize strategy optimizer."""
        self.performance_tracker = PerformanceTracker()
        # Read weights from config instead of hardcoded values
        self.current_weights = {
            "technical": config.trading.weight_technical,
            "sentiment": config.trading.weight_sentiment,
            "fundamental": config.trading.weight_fundamental,
        }
        self.current_threshold = config.trading.min_composite_score
        self.min_trades_for_optimization = 30

    def analyze_and_recommend(self) -> Dict[str, Any]:
        """
        Analyze historical performance and recommend optimizations.

        Returns:
            Dict with analysis results and recommendations
        """
        logger.info("strategy_optimization_started")

        # Get performance metrics
        metrics = self.performance_tracker.get_performance_metrics()

        if metrics.get("total_trades", 0) < self.min_trades_for_optimization:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {self.min_trades_for_optimization} trades for optimization",
                "current_trades": metrics.get("total_trades", 0),
                "trades_needed": self.min_trades_for_optimization
                - metrics.get("total_trades", 0),
                "min_required": self.min_trades_for_optimization,
            }

        # Optimize score weights
        optimal_weights, weight_improvement = self._optimize_score_weights()

        # Optimize threshold
        optimal_threshold, threshold_improvement = self._optimize_threshold()

        # Get score correlation analysis
        score_analysis = self.performance_tracker.get_score_correlation_analysis()

        # Calculate expected win rates for current and optimal configs
        current_expected_wr = self._calculate_expected_win_rate(
            self.current_weights, self.current_threshold
        )
        optimal_expected_wr = self._calculate_expected_win_rate(
            optimal_weights, optimal_threshold
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_weights,
            weight_improvement,
            optimal_threshold,
            threshold_improvement,
            score_analysis,
            metrics,
        )

        result = {
            "status": "analysis_complete",
            "current_performance": {
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "total_trades": metrics["total_trades"],
            },
            "current_config": {
                "weights": self.current_weights,
                "threshold": self.current_threshold,
                "expected_win_rate": current_expected_wr,
            },
            "optimal_config": {
                "weights": optimal_weights,
                "threshold": optimal_threshold,
                "expected_win_rate": optimal_expected_wr,
            },
            "projected_improvement": {
                "weight_optimization": weight_improvement,
                "threshold_optimization": threshold_improvement,
                "combined_estimate": (weight_improvement + threshold_improvement) / 2,
            },
            "recommendations": recommendations,
            "score_analysis": score_analysis,
        }

        logger.info(
            "strategy_optimization_complete",
            projected_improvement=result["projected_improvement"]["combined_estimate"],
        )

        return result

    def _calculate_expected_win_rate(
        self, weights: Dict[str, float], threshold: float
    ) -> Dict[str, Any]:
        """
        Calculate expected win rate for given weights and threshold.

        Args:
            weights: Score component weights
            threshold: Minimum composite score threshold

        Returns:
            Dict with expected win rate and supporting stats
        """
        trades = self.performance_tracker.data.get("trades", [])

        if len(trades) < 10:
            return {
                "win_rate": None,
                "sample_size": len(trades),
                "avg_profit_pct": None,
                "message": "Insufficient data"
            }

        # Recalculate composite scores with given weights
        qualifying_trades = []
        for trade in trades:
            entry_score = trade.get("entry_score", {})
            new_composite = (
                entry_score.get("technical", 50) * weights.get("technical", 0.4)
                + entry_score.get("sentiment", 50) * weights.get("sentiment", 0.3)
                + entry_score.get("fundamental", 50) * weights.get("fundamental", 0.3)
            )

            if new_composite >= threshold:
                qualifying_trades.append({
                    "profit_loss": trade.get("profit_loss", 0),
                    "profit_loss_pct": trade.get("profit_loss_pct", 0),
                    "composite_score": new_composite,
                })

        if len(qualifying_trades) < 5:
            return {
                "win_rate": None,
                "sample_size": len(qualifying_trades),
                "avg_profit_pct": None,
                "message": "Too few qualifying trades"
            }

        # Calculate win rate
        winners = [t for t in qualifying_trades if t["profit_loss"] > 0]
        win_rate = (len(winners) / len(qualifying_trades)) * 100

        # Calculate average profit
        avg_profit_pct = np.mean([t["profit_loss_pct"] for t in qualifying_trades])

        # Calculate expected value per trade
        avg_winner = np.mean([t["profit_loss_pct"] for t in winners]) if winners else 0
        losers = [t for t in qualifying_trades if t["profit_loss"] <= 0]
        avg_loser = np.mean([t["profit_loss_pct"] for t in losers]) if losers else 0

        expected_value = (win_rate / 100 * avg_winner) + ((100 - win_rate) / 100 * avg_loser)

        return {
            "win_rate": round(win_rate, 1),
            "sample_size": len(qualifying_trades),
            "avg_profit_pct": round(avg_profit_pct, 2),
            "avg_winner_pct": round(avg_winner, 2),
            "avg_loser_pct": round(avg_loser, 2),
            "expected_value_pct": round(expected_value, 2),
        }

    def _optimize_score_weights(self) -> Tuple[Dict[str, float], float]:
        """
        Find optimal score component weights using historical data.

        Returns:
            Tuple of (optimal_weights_dict, projected_improvement_pct)
        """
        trades = self.performance_tracker.data["trades"]

        if len(trades) < self.min_trades_for_optimization:
            return self.current_weights, 0.0

        def objective(weights):
            """Minimize negative Sharpe ratio."""
            technical_w, sentiment_w, fundamental_w = weights

            # Recalculate composite scores with new weights
            recalculated_trades = []
            for trade in trades:
                new_score = (
                    trade["entry_score"]["technical"] * technical_w
                    + trade["entry_score"]["sentiment"] * sentiment_w
                    + trade["entry_score"]["fundamental"] * fundamental_w
                )

                # Only include trades that would still pass threshold
                if new_score >= self.current_threshold:
                    recalculated_trades.append(trade["profit_loss_pct"])

            if len(recalculated_trades) < 5:
                return 1000  # Penalty for eliminating too many trades

            # Calculate Sharpe ratio
            mean_return = np.mean(recalculated_trades)
            std_return = np.std(recalculated_trades)
            sharpe = (
                (mean_return / (std_return + 0.0001)) * np.sqrt(252)
                if std_return > 0
                else 0
            )

            return -sharpe  # Minimize negative = maximize Sharpe

        # Constraint: weights must sum to 1.0
        constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

        # Bounds: each weight between 0 and 0.7
        bounds = [(0.0, 0.7), (0.0, 0.7), (0.0, 0.7)]

        # Starting point: current weights
        x0 = [
            self.current_weights["technical"],
            self.current_weights["sentiment"],
            self.current_weights["fundamental"],
        ]

        result = minimize(
            objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP"
        )

        optimal_weights = {
            "technical": round(result.x[0], 3),
            "sentiment": round(result.x[1], 3),
            "fundamental": round(result.x[2], 3),
        }

        # Calculate improvement
        current_sharpe = -objective(x0)
        optimal_sharpe = -result.fun
        improvement_pct = (
            ((optimal_sharpe - current_sharpe) / abs(current_sharpe)) * 100
            if current_sharpe != 0
            else 0
        )

        logger.info(
            "score_weights_optimized",
            optimal_weights=optimal_weights,
            improvement_pct=improvement_pct,
        )

        return optimal_weights, improvement_pct

    def _optimize_threshold(self) -> Tuple[float, float]:
        """
        Find optimal minimum score threshold.

        Returns:
            Tuple of (optimal_threshold, projected_improvement_pct)
        """
        trades = self.performance_tracker.data["trades"]

        if len(trades) < self.min_trades_for_optimization:
            return self.current_threshold, 0.0

        # Test different thresholds
        thresholds = np.arange(50, 85, 2.5)  # Test 50, 52.5, 55, ..., 82.5
        results = []

        for threshold in thresholds:
            qualifying_trades = [
                t for t in trades if t["entry_score"]["composite"] >= threshold
            ]

            if len(qualifying_trades) < 10:
                continue  # Need minimum trades

            win_rate = (
                len([t for t in qualifying_trades if t["profit_loss"] > 0])
                / len(qualifying_trades)
                * 100
            )
            avg_profit_pct = np.mean([t["profit_loss_pct"] for t in qualifying_trades])
            sharpe = self._calculate_sharpe(
                [t["profit_loss_pct"] for t in qualifying_trades]
            )

            results.append(
                {
                    "threshold": threshold,
                    "win_rate": win_rate,
                    "avg_profit_pct": avg_profit_pct,
                    "sharpe": sharpe,
                    "trade_count": len(qualifying_trades),
                    "score": sharpe * 0.5
                    + (win_rate / 100) * 0.3
                    + (avg_profit_pct / 100) * 0.2,  # Combined score
                }
            )

        if not results:
            return self.current_threshold, 0.0

        # Find best threshold
        best = max(results, key=lambda x: x["score"])
        optimal_threshold = best["threshold"]

        # Calculate improvement
        current_result = next(
            (r for r in results if r["threshold"] == self.current_threshold), None
        )
        if current_result:
            improvement_pct = (
                (best["sharpe"] - current_result["sharpe"])
                / abs(current_result["sharpe"])
                * 100
                if current_result["sharpe"] != 0
                else 0
            )
        else:
            improvement_pct = 0.0

        logger.info(
            "threshold_optimized",
            optimal_threshold=optimal_threshold,
            improvement_pct=improvement_pct,
        )

        return optimal_threshold, improvement_pct

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return (mean_return / std_return) * np.sqrt(252)

    def _generate_recommendations(
        self,
        optimal_weights: Dict[str, float],
        weight_improvement: float,
        optimal_threshold: float,
        threshold_improvement: float,
        score_analysis: Dict,
        metrics: Dict,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []

        # Weight recommendations
        if weight_improvement > 5:
            recommendations.append(
                {
                    "type": "weights",
                    "priority": "high",
                    "title": "Optimize Score Component Weights",
                    "description": f"Adjusting weights could improve Sharpe ratio by {weight_improvement:.1f}%",
                    "current": self.current_weights,
                    "recommended": optimal_weights,
                    "changes": [
                        f"Technical: {self.current_weights['technical']:.1%} → {optimal_weights['technical']:.1%}",
                        f"Sentiment: {self.current_weights['sentiment']:.1%} → {optimal_weights['sentiment']:.1%}",
                        f"Fundamental: {self.current_weights['fundamental']:.1%} → {optimal_weights['fundamental']:.1%}",
                    ],
                }
            )

        # Threshold recommendations
        if threshold_improvement > 3:
            recommendations.append(
                {
                    "type": "threshold",
                    "priority": "high",
                    "title": "Adjust Minimum Score Threshold",
                    "description": f"Raising threshold could improve Sharpe ratio by {threshold_improvement:.1f}%",
                    "current": self.current_threshold,
                    "recommended": optimal_threshold,
                    "impact": f"May reduce trade frequency but improve win rate",
                }
            )

        # Component-specific recommendations
        if score_analysis.get("component_correlations"):
            corrs = score_analysis["component_correlations"]

            # Filter out None values (components with no variance)
            valid_corrs = {k: v for k, v in corrs.items() if v is not None}

            if valid_corrs:
                # Identify weakest component
                weakest = min(valid_corrs.items(), key=lambda x: abs(x[1]))
                if abs(weakest[1]) < 0.2:
                    recommendations.append(
                        {
                            "type": "component",
                            "priority": "medium",
                            "title": f"Consider Reducing {weakest[0].capitalize()} Weight",
                            "description": f"{weakest[0].capitalize()} shows weak correlation ({weakest[1]:.2f}) with profitability",
                            "suggestion": f"Reduce {weakest[0]} weight and redistribute to stronger components",
                        }
                    )

            # Check for components with no variance
            none_corrs = [k for k, v in corrs.items() if v is None]
            if none_corrs:
                recommendations.append(
                    {
                        "type": "component",
                        "priority": "low",
                        "title": f"No Variance in {', '.join(c.capitalize() for c in none_corrs)} Scores",
                        "description": f"These components show no variation across trades (all same value)",
                        "suggestion": f"Consider removing or diversifying {', '.join(none_corrs)} analysis",
                    }
                )

        # Win rate recommendations
        if metrics["win_rate"] < 55:
            recommendations.append(
                {
                    "type": "win_rate",
                    "priority": "high",
                    "title": "Improve Win Rate",
                    "description": f"Current win rate ({metrics['win_rate']:.1f}%) is below target",
                    "suggestions": [
                        "Increase minimum score threshold",
                        "Review exit strategy (stop losses may be too tight)",
                        "Consider adding filters for market conditions",
                    ],
                }
            )

        # Profit factor recommendations
        if metrics["profit_factor"] < 1.5:
            recommendations.append(
                {
                    "type": "profit_factor",
                    "priority": "medium",
                    "title": "Improve Profit Factor",
                    "description": f"Current profit factor ({metrics['profit_factor']:.2f}) suggests winners aren't significantly larger than losers",
                    "suggestions": [
                        "Let winners run longer (reduce premature exits)",
                        "Tighten entry criteria to improve quality",
                        "Review position sizing strategy",
                    ],
                }
            )

        return recommendations

    def apply_optimizations(
        self, apply_weights: bool = False, apply_threshold: bool = False
    ) -> Dict[str, Any]:
        """
        Apply recommended optimizations to config.

        Args:
            apply_weights: Whether to update score weights
            apply_threshold: Whether to update minimum threshold

        Returns:
            Dict with applied changes
        """
        if not apply_weights and not apply_threshold:
            return {"status": "no_changes_requested"}

        analysis = self.analyze_and_recommend()

        if analysis["status"] != "analysis_complete":
            return analysis

        changes = []

        if apply_weights:
            optimal_weights = analysis["optimal_config"]["weights"]
            # Note: Would need to update config file or database
            # For now, just log the recommendation
            changes.append(
                {
                    "parameter": "score_weights",
                    "old_value": self.current_weights,
                    "new_value": optimal_weights,
                    "status": "recommendation_only",
                    "note": "Manual config update required in config/config.py",
                }
            )

        if apply_threshold:
            optimal_threshold = analysis["optimal_config"]["threshold"]
            changes.append(
                {
                    "parameter": "composite_score_threshold",
                    "old_value": self.current_threshold,
                    "new_value": optimal_threshold,
                    "status": "recommendation_only",
                    "note": "Manual config update required in config/config.py",
                }
            )

        logger.info(
            "optimization_recommendations_generated", changes_count=len(changes)
        )

        return {"status": "recommendations_generated", "changes": changes}
