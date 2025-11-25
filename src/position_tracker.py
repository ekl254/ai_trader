"""Track position entry times, rebalancing history, and locked positions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.logger import logger


class PositionTracker:
    """Tracks position metadata including entry times, locks, and rebalancing history."""

    def __init__(self, data_file: str = "data/position_tracking.json") -> None:
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load tracking data from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("failed_to_load_position_tracking", error=str(e))
                return self._initialize_data()
        return self._initialize_data()

    def _initialize_data(self) -> Dict[str, Any]:
        """Initialize empty tracking data structure."""
        return {
            "positions": {},  # {symbol: {entry_time, entry_price, score, reason}}
            "locked_positions": [],  # [symbol1, symbol2, ...]
            "rebalancing_history": [],  # [{timestamp, old_symbol, new_symbol, reason, ...}]
            "last_rebalance_time": None,
        }

    def _save_data(self) -> None:
        """Save tracking data to file."""
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            logger.info("position_tracking_saved", file=str(self.data_file))
        except Exception as e:
            logger.error("failed_to_save_position_tracking", error=str(e))

    def track_entry(
        self, symbol: str, entry_price: float, score: float, reason: str
    ) -> None:
        """
        Track a new position entry.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            score: Composite score at entry
            reason: Entry reason (e.g., "new_position", "rebalancing")
        """
        entry_time = datetime.now(timezone.utc)

        self.data["positions"][symbol] = {
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            "score": score,
            "reason": reason,
        }

        self._save_data()

        logger.info(
            "position_entry_tracked",
            symbol=symbol,
            entry_time=entry_time.isoformat(),
            entry_price=entry_price,
            score=score,
        )

    def track_exit(self, symbol: str, exit_reason: str) -> None:
        """
        Track a position exit.

        Args:
            symbol: Stock symbol
            exit_reason: Reason for exit (e.g., "stop_loss", "take_profit", "rebalancing")
        """
        if symbol in self.data["positions"]:
            position_data = self.data["positions"].pop(symbol)

            logger.info(
                "position_exit_tracked",
                symbol=symbol,
                exit_reason=exit_reason,
                hold_duration=self._calculate_hold_duration(
                    position_data["entry_time"]
                ),
            )

            self._save_data()

    def track_rebalance(
        self,
        old_symbol: str,
        new_symbol: str,
        old_score: float,
        new_score: float,
        score_diff: float,
    ) -> None:
        """
        Track a rebalancing event.

        Args:
            old_symbol: Symbol being sold
            new_symbol: Symbol being bought
            old_score: Score of old position
            new_score: Score of new position
            score_diff: Score difference
        """
        rebalance_time = datetime.now(timezone.utc)

        event = {
            "timestamp": rebalance_time.isoformat(),
            "old_symbol": old_symbol,
            "new_symbol": new_symbol,
            "old_score": old_score,
            "new_score": new_score,
            "score_diff": score_diff,
        }

        self.data["rebalancing_history"].append(event)
        self.data["last_rebalance_time"] = rebalance_time.isoformat()

        self._save_data()

        logger.info(
            "rebalancing_event_tracked",
            old_symbol=old_symbol,
            new_symbol=new_symbol,
            score_diff=score_diff,
        )

    def get_entry_time(self, symbol: str) -> Optional[datetime]:
        """
        Get entry time for a position.

        Args:
            symbol: Stock symbol

        Returns:
            Entry time as datetime, or None if not tracked
        """
        if symbol in self.data["positions"]:
            entry_time_str = self.data["positions"][symbol]["entry_time"]
            return datetime.fromisoformat(entry_time_str)
        return None

    def get_hold_duration_seconds(self, symbol: str) -> float:
        """
        Get hold duration in seconds.

        Args:
            symbol: Stock symbol

        Returns:
            Hold duration in seconds, or 0 if not tracked
        """
        entry_time = self.get_entry_time(symbol)
        if entry_time:
            return (datetime.now(timezone.utc) - entry_time).total_seconds()
        return 0.0

    def _calculate_hold_duration(self, entry_time_str: str) -> float:
        """Calculate hold duration in seconds from ISO string."""
        entry_time = datetime.fromisoformat(entry_time_str)
        return (datetime.now(timezone.utc) - entry_time).total_seconds()

    def is_locked(self, symbol: str) -> bool:
        """
        Check if a position is locked from rebalancing.

        Args:
            symbol: Stock symbol

        Returns:
            True if locked, False otherwise
        """
        return symbol in self.data["locked_positions"]

    def lock_position(self, symbol: str) -> None:
        """
        Lock a position to prevent rebalancing.

        Args:
            symbol: Stock symbol
        """
        if symbol not in self.data["locked_positions"]:
            self.data["locked_positions"].append(symbol)
            self._save_data()
            logger.info("position_locked", symbol=symbol)

    def unlock_position(self, symbol: str) -> None:
        """
        Unlock a position to allow rebalancing.

        Args:
            symbol: Stock symbol
        """
        if symbol in self.data["locked_positions"]:
            self.data["locked_positions"].remove(symbol)
            self._save_data()
            logger.info("position_unlocked", symbol=symbol)

    def get_locked_positions(self) -> List[str]:
        """Get list of locked positions."""
        return self.data["locked_positions"].copy()

    def get_last_rebalance_time(self) -> Optional[datetime]:
        """
        Get timestamp of last rebalancing event.

        Returns:
            Last rebalance time, or None if never rebalanced
        """
        last_time = self.data.get("last_rebalance_time")
        if last_time:
            return datetime.fromisoformat(last_time)
        return None

    def can_rebalance_now(self, cooldown_minutes: int) -> bool:
        """
        Check if rebalancing is allowed (cooldown check).

        Args:
            cooldown_minutes: Minimum minutes between rebalancing

        Returns:
            True if rebalancing allowed, False if still in cooldown
        """
        last_rebalance = self.get_last_rebalance_time()
        if last_rebalance is None:
            return True

        elapsed_seconds = (datetime.now(timezone.utc) - last_rebalance).total_seconds()
        elapsed_minutes = elapsed_seconds / 60

        return elapsed_minutes >= cooldown_minutes

    def get_rebalancing_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent rebalancing history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of rebalancing events (most recent first)
        """
        history = self.data["rebalancing_history"]
        return list(reversed(history[-limit:]))

    def get_rebalancing_stats(self) -> Dict[str, Any]:
        """
        Get rebalancing statistics.

        Returns:
            Dict with total_rebalances, avg_score_improvement, etc.
        """
        history = self.data["rebalancing_history"]

        if not history:
            return {
                "total_rebalances": 0,
                "avg_score_improvement": 0.0,
                "min_score_improvement": 0.0,
                "max_score_improvement": 0.0,
            }

        score_diffs = [event["score_diff"] for event in history]

        return {
            "total_rebalances": len(history),
            "avg_score_improvement": sum(score_diffs) / len(score_diffs),
            "min_score_improvement": min(score_diffs),
            "max_score_improvement": max(score_diffs),
        }


# Global tracker instance
position_tracker = PositionTracker()
