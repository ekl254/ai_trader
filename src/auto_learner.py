#!/usr/bin/env python3
"""Auto-learning module that optimizes trading parameters based on performance."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.config import config
from src.logger import logger
from src.performance_tracker import PerformanceTracker
from src.strategy_optimizer import StrategyOptimizer


class AutoLearner:
    """
    Automated learning system that:
    1. Tracks completed trades
    2. Runs optimization after N trades
    3. Auto-applies parameter changes when confidence is high
    4. Maintains audit trail of all changes
    """

    # Configuration
    TRADES_BEFORE_OPTIMIZATION = 10  # Run optimizer every N new trades
    MIN_TRADES_FOR_LEARNING = 30  # Minimum trades before auto-applying changes
    MIN_IMPROVEMENT_PCT = 10.0  # Minimum projected improvement to auto-apply
    MAX_WEIGHT_CHANGE = 0.15  # Maximum single weight change allowed (15%)
    MAX_THRESHOLD_CHANGE = 5.0  # Maximum threshold change allowed (5 points)

    def __init__(self, data_file: str = "data/learning_state.json"):
        """Initialize auto-learner."""
        self.data_file = Path(data_file)
        self.performance_tracker = PerformanceTracker()
        self.optimizer = StrategyOptimizer()
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load learning state from disk."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error("failed_to_load_learning_state", error=str(e))

        return self._create_empty_state()

    def _create_empty_state(self) -> dict[str, Any]:
        """Create empty learning state."""
        return {
            "version": "1.0",
            "last_optimization_at": None,
            "trades_since_last_optimization": 0,
            "total_optimizations_run": 0,
            "total_changes_applied": 0,
            "parameter_history": [],  # Audit trail
            "daily_equity": [],  # Daily equity snapshots
            "learning_enabled": True,
            "current_parameters": {
                "weight_technical": config.trading.weight_technical,
                "weight_sentiment": config.trading.weight_sentiment,
                "weight_fundamental": config.trading.weight_fundamental,
                "min_composite_score": config.trading.min_composite_score,
            },
        }

    def _save_state(self) -> None:
        """Save learning state to disk."""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error("failed_to_save_learning_state", error=str(e))

    def record_trade_completed(self) -> dict[str, Any] | None:
        """
        Called after each completed trade. Checks if optimization should run.

        Returns:
            Dict with optimization results if run, None otherwise
        """
        self.state["trades_since_last_optimization"] += 1
        self._save_state()

        trades_pending = self.state["trades_since_last_optimization"]
        logger.info(
            "trade_recorded_for_learning",
            trades_since_optimization=trades_pending,
            threshold=self.TRADES_BEFORE_OPTIMIZATION,
        )

        # Check if we should run optimization
        if trades_pending >= self.TRADES_BEFORE_OPTIMIZATION:
            return self.run_optimization_cycle()

        return None

    def run_optimization_cycle(self) -> dict[str, Any]:
        """
        Run a full optimization cycle.

        Returns:
            Dict with optimization results and any applied changes
        """
        logger.info("auto_learning_optimization_started")

        # Get current trade count
        metrics = self.performance_tracker.get_performance_metrics()
        total_trades = metrics.get("total_trades", 0)

        result: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_trades": total_trades,
            "optimization_run": False,
            "changes_applied": [],
            "recommendations": [],
        }

        # Check minimum trades requirement
        if total_trades < self.MIN_TRADES_FOR_LEARNING:
            result["status"] = "insufficient_data"
            result["message"] = (
                f"Need {self.MIN_TRADES_FOR_LEARNING} trades, have {total_trades}"
            )
            logger.info(
                "auto_learning_skipped_insufficient_data",
                total_trades=total_trades,
                required=self.MIN_TRADES_FOR_LEARNING,
            )
            return result

        # Run optimizer
        try:
            analysis = self.optimizer.analyze_and_recommend()
        except Exception as e:
            logger.error("auto_learning_optimization_failed", error=str(e))
            result["status"] = "error"
            result["error"] = str(e)
            return result

        if analysis.get("status") != "analysis_complete":
            result["status"] = analysis.get("status", "unknown")
            result["message"] = analysis.get("message", "Optimization incomplete")
            return result

        result["optimization_run"] = True
        result["current_performance"] = analysis.get("current_performance", {})
        result["projected_improvement"] = analysis.get("projected_improvement", {})

        # Update state
        self.state["last_optimization_at"] = datetime.now(UTC).isoformat()
        self.state["trades_since_last_optimization"] = 0
        self.state["total_optimizations_run"] += 1

        # Check if we should auto-apply changes
        if self.state.get("learning_enabled", True):
            changes = self._evaluate_and_apply_changes(analysis)
            result["changes_applied"] = changes

        # Store recommendations for dashboard
        result["recommendations"] = analysis.get("recommendations", [])

        self._save_state()

        logger.info(
            "auto_learning_optimization_complete",
            changes_applied=len(result["changes_applied"]),
            projected_improvement=result["projected_improvement"].get(
                "combined_estimate", 0
            ),
        )

        return result

    def _evaluate_and_apply_changes(
        self, analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Evaluate optimization results and apply changes if criteria met.

        Args:
            analysis: Results from strategy optimizer

        Returns:
            List of applied changes
        """
        changes_applied = []
        projected = analysis.get("projected_improvement", {})
        optimal = analysis.get("optimal_config", {})
        current = analysis.get("current_config", {})

        # Check weight optimization
        weight_improvement = projected.get("weight_optimization", 0)
        if weight_improvement >= self.MIN_IMPROVEMENT_PCT:
            weight_change = self._apply_weight_changes(
                current.get("weights", {}),
                optimal.get("weights", {}),
                weight_improvement,
            )
            if weight_change:
                changes_applied.append(weight_change)

        # Check threshold optimization
        threshold_improvement = projected.get("threshold_optimization", 0)
        if threshold_improvement >= self.MIN_IMPROVEMENT_PCT:
            threshold_change = self._apply_threshold_change(
                current.get("threshold", 72.5),
                optimal.get("threshold", 72.5),
                threshold_improvement,
            )
            if threshold_change:
                changes_applied.append(threshold_change)

        return changes_applied

    def _apply_weight_changes(
        self,
        current_weights: dict[str, float],
        optimal_weights: dict[str, float],
        improvement_pct: float,
    ) -> dict[str, Any] | None:
        """Apply weight changes if within safety bounds."""
        # Calculate changes
        changes = {}
        for key in ["technical", "sentiment", "fundamental"]:
            current = current_weights.get(key, 0.33)
            optimal = optimal_weights.get(key, 0.33)
            diff = optimal - current

            # Cap the change
            if abs(diff) > self.MAX_WEIGHT_CHANGE:
                # Apply partial change in the right direction
                capped = self.MAX_WEIGHT_CHANGE if diff > 0 else -self.MAX_WEIGHT_CHANGE
                changes[key] = current + capped
                logger.info(
                    "weight_change_capped",
                    component=key,
                    requested=diff,
                    capped_to=capped,
                )
            else:
                changes[key] = optimal

        # Normalize to sum to 1.0
        total = sum(changes.values())
        if total > 0:
            changes = {k: v / total for k, v in changes.items()}

        # Apply to config
        config.trading.weight_technical = changes["technical"]
        config.trading.weight_sentiment = changes["sentiment"]
        config.trading.weight_fundamental = changes["fundamental"]

        # Update .env file for persistence
        self._update_env_weights(changes)

        # Record in audit trail
        change_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "weights",
            "previous": current_weights,
            "new": changes,
            "improvement_pct": improvement_pct,
            "reason": "auto_optimization",
        }
        self.state["parameter_history"].append(change_record)
        self.state["total_changes_applied"] += 1
        self.state["current_parameters"]["weight_technical"] = changes["technical"]
        self.state["current_parameters"]["weight_sentiment"] = changes["sentiment"]
        self.state["current_parameters"]["weight_fundamental"] = changes["fundamental"]

        logger.info(
            "weights_auto_updated",
            technical=changes["technical"],
            sentiment=changes["sentiment"],
            fundamental=changes["fundamental"],
            improvement_pct=improvement_pct,
        )

        return change_record

    def _apply_threshold_change(
        self,
        current_threshold: float,
        optimal_threshold: float,
        improvement_pct: float,
    ) -> dict[str, Any] | None:
        """Apply threshold change if within safety bounds."""
        diff = optimal_threshold - current_threshold

        # Cap the change
        if abs(diff) > self.MAX_THRESHOLD_CHANGE:
            capped = (
                self.MAX_THRESHOLD_CHANGE if diff > 0 else -self.MAX_THRESHOLD_CHANGE
            )
            new_threshold = current_threshold + capped
            logger.info(
                "threshold_change_capped",
                requested=diff,
                capped_to=capped,
            )
        else:
            new_threshold = optimal_threshold

        # Apply to config
        config.trading.min_composite_score = new_threshold

        # Update .env file for persistence
        self._update_env_threshold(new_threshold)

        # Record in audit trail
        change_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "threshold",
            "previous": current_threshold,
            "new": new_threshold,
            "improvement_pct": improvement_pct,
            "reason": "auto_optimization",
        }
        self.state["parameter_history"].append(change_record)
        self.state["total_changes_applied"] += 1
        self.state["current_parameters"]["min_composite_score"] = new_threshold

        logger.info(
            "threshold_auto_updated",
            previous=current_threshold,
            new=new_threshold,
            improvement_pct=improvement_pct,
        )

        return change_record

    def _update_env_weights(self, weights: dict[str, float]) -> None:
        """Update weights in .env file for persistence."""
        env_file = Path(__file__).parent.parent / ".env"
        self._update_env_file(
            env_file,
            {
                "WEIGHT_TECHNICAL": str(round(weights["technical"], 3)),
                "WEIGHT_SENTIMENT": str(round(weights["sentiment"], 3)),
                "WEIGHT_FUNDAMENTAL": str(round(weights["fundamental"], 3)),
            },
        )

    def _update_env_threshold(self, threshold: float) -> None:
        """Update threshold in .env file for persistence."""
        env_file = Path(__file__).parent.parent / ".env"
        self._update_env_file(
            env_file,
            {"MIN_COMPOSITE_SCORE": str(round(threshold, 1))},
        )

    def _update_env_file(self, env_file: Path, updates: dict[str, str]) -> None:
        """Update or add environment variables in .env file."""
        existing_lines = []
        if env_file.exists():
            with open(env_file) as f:
                existing_lines = f.readlines()

        updated_keys = set()
        new_lines = []

        for line in existing_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith("#"):
                matched = False
                for key, value in updates.items():
                    if line_stripped.startswith(f"{key}="):
                        new_lines.append(f"{key}={value}\n")
                        updated_keys.add(key)
                        matched = True
                        break
                if not matched:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Add keys that weren't in the file
        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")

        with open(env_file, "w") as f:
            f.writelines(new_lines)

    def record_daily_equity(self, equity: float, cash: float) -> None:
        """
        Record daily equity snapshot for drawdown analysis.

        Args:
            equity: Total portfolio equity
            cash: Cash balance
        """
        today = datetime.now(UTC).date().isoformat()

        # Check if we already have an entry for today
        existing = next(
            (e for e in self.state["daily_equity"] if e["date"] == today),
            None,
        )

        if existing:
            # Update existing entry
            existing["equity"] = equity
            existing["cash"] = cash
            existing["updated_at"] = datetime.now(UTC).isoformat()
        else:
            # Add new entry
            self.state["daily_equity"].append(
                {
                    "date": today,
                    "equity": equity,
                    "cash": cash,
                    "recorded_at": datetime.now(UTC).isoformat(),
                }
            )

        # Keep only last 365 days
        if len(self.state["daily_equity"]) > 365:
            self.state["daily_equity"] = self.state["daily_equity"][-365:]

        self._save_state()

        logger.info(
            "daily_equity_recorded",
            date=today,
            equity=equity,
            cash=cash,
        )

    def get_equity_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get equity history for the last N days."""
        return self.state["daily_equity"][-days:]

    def get_parameter_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get parameter change history (audit trail)."""
        return list(reversed(self.state["parameter_history"][-limit:]))

    def get_learning_stats(self) -> dict[str, Any]:
        """Get learning system statistics."""
        equity_data = self.state["daily_equity"]

        # Calculate max drawdown if we have equity history
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        if len(equity_data) > 1:
            equities = [e["equity"] for e in equity_data]
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                drawdown = peak - eq
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct

        return {
            "learning_enabled": self.state.get("learning_enabled", True),
            "total_optimizations_run": self.state.get("total_optimizations_run", 0),
            "total_changes_applied": self.state.get("total_changes_applied", 0),
            "trades_since_last_optimization": self.state.get(
                "trades_since_last_optimization", 0
            ),
            "trades_until_next_optimization": max(
                0,
                self.TRADES_BEFORE_OPTIMIZATION
                - self.state.get("trades_since_last_optimization", 0),
            ),
            "last_optimization_at": self.state.get("last_optimization_at"),
            "current_parameters": self.state.get("current_parameters", {}),
            "equity_history_days": len(equity_data),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "parameter_changes_count": len(self.state.get("parameter_history", [])),
        }

    def enable_learning(self) -> None:
        """Enable auto-learning."""
        self.state["learning_enabled"] = True
        self._save_state()
        logger.info("auto_learning_enabled")

    def disable_learning(self) -> None:
        """Disable auto-learning (optimization still runs but doesn't apply changes)."""
        self.state["learning_enabled"] = False
        self._save_state()
        logger.info("auto_learning_disabled")

    def force_optimization(self) -> dict[str, Any]:
        """Force an optimization cycle regardless of trade count."""
        logger.info("forced_optimization_requested")
        return self.run_optimization_cycle()


# Global instance
auto_learner = AutoLearner()
