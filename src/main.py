"""Main trading engine."""

import json
import os
import sys
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Clock, Position

from config.config import config
from src.data_provider import alpaca_provider
from src.executor import executor
from src.logger import logger
from src.market_regime import market_regime_detector
from src.position_sizer import position_sizer
from src.position_tracker import position_tracker
from src.risk_manager import risk_manager
from src.strategy import strategy
from src.universe import filter_liquid_stocks, get_sp500_symbols
from src.watchdog import OperationTimeout, watchdog

# File paths for premarket data persistence
PREMARKET_CANDIDATES_FILE = (
    Path(__file__).parent.parent / "data" / "premarket_candidates.json"
)
PREMARKET_HISTORY_FILE = (
    Path(__file__).parent.parent / "data" / "premarket_history.json"
)


def save_premarket_candidates(
    candidates: list[dict[str, Any]], scan_time: datetime
) -> None:
    """Save premarket candidates to file for dashboard visibility."""
    data = {
        "scan_time": scan_time.isoformat(),
        "candidates": candidates,
        "count": len(candidates),
    }
    try:
        PREMARKET_CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREMARKET_CANDIDATES_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("premarket_candidates_saved", count=len(candidates))
    except Exception as e:
        logger.error("failed_to_save_premarket_candidates", error=str(e))


def load_premarket_candidates() -> dict[str, Any]:
    """Load premarket candidates from file."""
    try:
        if PREMARKET_CANDIDATES_FILE.exists():
            with open(PREMARKET_CANDIDATES_FILE) as f:
                result: dict[str, Any] = json.load(f)
                return result
    except Exception as e:
        logger.error("failed_to_load_premarket_candidates", error=str(e))
    return {"scan_time": None, "candidates": [], "count": 0}


def record_premarket_execution(
    date: str,
    candidates: list[dict[str, Any]],
    executed: list[dict[str, Any]],
    regime: str,
) -> None:
    """Record premarket scan and execution results for historical tracking."""
    try:
        # Load existing history
        history = []
        if PREMARKET_HISTORY_FILE.exists():
            with open(PREMARKET_HISTORY_FILE) as f:
                history = json.load(f)

        # Create record for this day
        record = {
            "date": date,
            "scan_time": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
            "regime": regime,
            "total_candidates": len(candidates),
            "top_candidates": [
                {"symbol": c["symbol"], "score": c["score"], "price": c.get("price", 0)}
                for c in candidates[:10]
            ],
            "executed": executed,
            "execution_count": len(executed),
        }

        # Check if we already have a record for today (update it)
        existing_idx = None
        for i, h in enumerate(history):
            if h.get("date") == date:
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing record
            history[existing_idx] = record
        else:
            # Add new record
            history.append(record)

        # Keep last 90 days of history
        history = history[-90:]

        # Save
        with open(PREMARKET_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("premarket_history_recorded", date=date, executed=len(executed))
    except Exception as e:
        logger.error("failed_to_record_premarket_history", error=str(e))


def update_premarket_performance(
    symbol: str, entry_price: float, exit_price: float, exit_reason: str
) -> None:
    """Update premarket trade performance when a position is closed."""
    try:
        if not PREMARKET_HISTORY_FILE.exists():
            return

        with open(PREMARKET_HISTORY_FILE) as f:
            history = json.load(f)

        # Find the most recent record where this symbol was executed
        for record in reversed(history):
            for executed in record.get("executed", []):
                if executed.get("symbol") == symbol and "exit_price" not in executed:
                    # Update with exit data
                    executed["exit_price"] = exit_price
                    executed["exit_reason"] = exit_reason
                    executed["exit_time"] = datetime.now(
                        pytz.timezone("US/Eastern")
                    ).isoformat()
                    executed["profit_loss"] = exit_price - entry_price
                    executed["profit_loss_pct"] = (
                        (exit_price - entry_price) / entry_price
                    ) * 100

                    # Save updated history
                    with open(PREMARKET_HISTORY_FILE, "w") as f:
                        json.dump(history, f, indent=2)

                    logger.info(
                        "premarket_performance_updated",
                        symbol=symbol,
                        profit_loss_pct=executed["profit_loss_pct"],
                    )
                    return
    except Exception as e:
        logger.error("failed_to_update_premarket_performance", error=str(e))


class TradingEngine:
    """Main trading engine orchestrator."""

    def __init__(self) -> None:
        self.universe = self._load_universe()
        self.premarket_candidates: list[
            dict[str, Any]
        ] = []  # Store premarket scan results
        self.last_premarket_scan: datetime | None = None

        # Sync position tracking with Alpaca on startup
        self._sync_position_tracking()

        logger.info("trading_engine_initialized", universe_size=len(self.universe))

    def _sync_position_tracking(self) -> None:
        """Sync position tracking data with actual Alpaca positions."""
        try:
            client = TradingClient(
                config.alpaca.api_key,
                config.alpaca.secret_key,
                paper=True,
            )
            positions = client.get_all_positions()

            # Clear any stale pending buys from previous session
            risk_manager.clear_all_pending_buys()

            if positions:
                assert isinstance(positions, list)
                sync_result = position_tracker.sync_with_alpaca(positions)
                if sync_result["added"] or sync_result["removed"]:
                    logger.info(
                        "position_tracking_synced_on_startup",
                        added=sync_result["added"],
                        removed=sync_result["removed"],
                    )
        except Exception as e:
            logger.warning("position_sync_failed_on_startup", error=str(e))

    def _load_universe(self) -> list[str]:
        """Load and filter stock universe."""
        symbols = get_sp500_symbols()
        liquid_symbols = filter_liquid_stocks(symbols)
        logger.info("universe_loaded", total=len(symbols), liquid=len(liquid_symbols))
        return liquid_symbols

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            with OperationTimeout(seconds=30, operation="check_market_status"):
                client = TradingClient(
                    config.alpaca.api_key,
                    config.alpaca.secret_key,
                    paper=True,
                )
                clock: Clock = client.get_clock()
                return bool(clock.is_open)

        except TimeoutError:
            logger.error("market_status_check_timeout")
            return False
        except Exception as e:
            logger.error("failed_to_check_market_status", error=str(e))
            return False

    def is_premarket_hours(self) -> bool:
        """Check if we're in premarket hours (4:00 AM - 9:30 AM ET)."""
        try:
            et = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            current_time = now.time()

            # Premarket: 4:00 AM to 9:30 AM ET (weekdays only)
            premarket_start = dt_time(4, 0)
            premarket_end = dt_time(9, 30)

            # Check if it's a weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            return premarket_start <= current_time < premarket_end
        except Exception as e:
            logger.error("failed_to_check_premarket_hours", error=str(e))
            return False

    def get_minutes_until_market_open(self) -> int:
        """Get minutes until market opens (9:30 AM ET)."""
        try:
            et = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

            if now >= market_open:
                return 0

            delta = market_open - now
            return int(delta.total_seconds() / 60)
        except Exception:
            return 0

    def run_premarket_scan(self) -> list[dict[str, Any]]:
        """Run premarket analysis to identify top candidates for market open.

        This scans the universe and scores stocks using available premarket data.
        Results are stored and ready for immediate execution at market open.
        """
        logger.info("premarket_scan_started", universe_size=len(self.universe))

        # Get regime data for adaptive parameters
        regime_data = market_regime_detector.get_market_regime()
        regime = regime_data.get("regime", "neutral")
        regime_min_score = regime_data.get("min_score", 70)

        logger.info(
            "premarket_regime_check",
            regime=regime,
            min_score=regime_min_score,
        )

        candidates: list[dict[str, Any]] = []
        scanned = 0
        errors = 0

        for i, symbol in enumerate(self.universe):
            try:
                # Log progress every 50 symbols
                if (i + 1) % 50 == 0:
                    logger.info(
                        "premarket_scan_progress",
                        progress=f"{i + 1}/{len(self.universe)}",
                        candidates_found=len(candidates),
                    )

                should_buy, score, reasoning = strategy.should_buy(symbol)
                scanned += 1

                if should_buy:
                    # Get current/premarket price
                    try:
                        latest = alpaca_provider.get_latest_trade(symbol)
                        price = latest["price"]
                    except Exception:
                        price = 0  # Will get fresh price at market open

                    candidates.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "reasoning": reasoning,
                            "price": price,
                            "scanned_at": datetime.now(
                                pytz.timezone("US/Eastern")
                            ).isoformat(),
                        }
                    )
                    logger.info(
                        "premarket_candidate_found",
                        symbol=symbol,
                        score=score,
                    )

                # Small delay to avoid rate limits
                time.sleep(0.2)

            except Exception as e:
                errors += 1
                if errors <= 5:  # Only log first 5 errors
                    logger.warning("premarket_scan_error", symbol=symbol, error=str(e))
                continue

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Store results in memory and file
        self.premarket_candidates = candidates
        self.last_premarket_scan = datetime.now(pytz.timezone("US/Eastern"))

        # Save to file for dashboard visibility
        save_premarket_candidates(candidates, self.last_premarket_scan)

        logger.info(
            "premarket_scan_completed",
            scanned=scanned,
            candidates=len(candidates),
            errors=errors,
            top_5=[c["symbol"] for c in candidates[:5]],
        )

        return candidates

    def execute_premarket_candidates(self, max_positions: int = 3) -> int:
        """Execute trades for premarket candidates at market open.

        Returns number of trades executed.
        """
        if not self.premarket_candidates:
            logger.info("no_premarket_candidates_to_execute")
            return 0

        # Check how many slots available
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        positions: list[Position] = client.get_all_positions()
        current_positions = len(positions)
        available_slots = max(0, config.trading.max_positions - current_positions)

        if available_slots == 0:
            logger.info("no_slots_for_premarket_candidates")
            return 0

        trades_to_execute = min(
            available_slots, max_positions, len(self.premarket_candidates)
        )
        executed = 0
        executed_trades = []

        # Get regime for historical record
        regime_data = market_regime_detector.get_market_regime()
        regime = regime_data.get("regime", "unknown")

        logger.info(
            "executing_premarket_candidates",
            candidates=len(self.premarket_candidates),
            slots_available=available_slots,
            will_execute=trades_to_execute,
        )

        for candidate in self.premarket_candidates[:trades_to_execute]:
            try:
                # Get fresh price at market open
                latest = alpaca_provider.get_latest_trade(candidate["symbol"])
                current_price = latest["price"]

                success = executor.buy_stock(
                    candidate["symbol"],
                    candidate["score"],
                    candidate["reasoning"],
                    current_price,
                )

                if success:
                    executed += 1
                    executed_trades.append(
                        {
                            "symbol": candidate["symbol"],
                            "score": candidate["score"],
                            "premarket_price": candidate["price"],
                            "execution_price": current_price,
                            "execution_time": datetime.now(
                                pytz.timezone("US/Eastern")
                            ).isoformat(),
                        }
                    )
                    logger.info(
                        "premarket_candidate_executed",
                        symbol=candidate["symbol"],
                        score=candidate["score"],
                        premarket_price=candidate["price"],
                        execution_price=current_price,
                    )
                    time.sleep(1)

            except Exception as e:
                logger.error(
                    "premarket_execution_failed",
                    symbol=candidate["symbol"],
                    error=str(e),
                )

        # Record to history
        today = datetime.now(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d")
        record_premarket_execution(
            date=today,
            candidates=self.premarket_candidates,
            executed=executed_trades,
            regime=regime,
        )

        # Clear candidates after execution (but keep file for dashboard until next scan)
        self.premarket_candidates = []

        logger.info("premarket_execution_completed", trades_executed=executed)
        return executed

    def scan_and_trade(self, max_new_positions: int = 5) -> None:
        """Scan universe and execute trades (with optional rebalancing)."""
        logger.info("starting_market_scan", universe_size=len(self.universe))
        watchdog.heartbeat("scan_and_trade_started")

        # Check market regime and get adaptive trading parameters
        regime_data = market_regime_detector.get_market_regime()
        regime = regime_data.get("regime", "neutral")
        should_trade = regime_data.get("should_trade", True)
        position_multiplier = regime_data.get("position_size_multiplier", 1.0)
        max_positions_override = regime_data.get("max_positions_override")

        # Get regime-adaptive parameters
        regime_stop_loss = regime_data.get("stop_loss_pct", 0.03)
        regime_take_profit = regime_data.get("take_profit_pct", 0.08)
        regime_min_score = regime_data.get("min_score", 70)

        # Set regime parameters in risk manager
        from src.risk_manager import risk_manager

        risk_manager.set_regime_parameters(
            {
                "regime": regime,
                "stop_loss_pct": regime_stop_loss,
                "take_profit_pct": regime_take_profit,
                "min_score": regime_min_score,
                "position_multiplier": position_multiplier,
                "max_positions": max_positions_override,
            }
        )

        logger.info(
            "market_regime_check",
            regime=regime,
            should_trade=should_trade,
            position_multiplier=position_multiplier,
            stop_loss_pct=f"{regime_stop_loss:.1%}",
            take_profit_pct=f"{regime_take_profit:.1%}",
            min_score=regime_min_score,
            recommendation=regime_data.get("recommendation"),
        )

        # Skip trading in strong bear markets
        if not should_trade:
            logger.warning(
                "trading_paused_due_to_regime",
                regime=regime,
                recommendation=regime_data.get("recommendation"),
            )
            return

        # Check how many open slots we have
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        positions: list[Position] = client.get_all_positions()
        current_positions = len(positions)

        # Use regime-adjusted max positions if available
        base_max_positions = max_positions_override or config.trading.max_positions

        # Get portfolio info for dynamic position sizing
        portfolio_info = position_sizer.get_portfolio_info()

        logger.info(
            "portfolio_status",
            portfolio_value=portfolio_info["portfolio_value"],
            cash=portfolio_info["cash"],
            cash_pct=f"{(portfolio_info['cash'] / portfolio_info['portfolio_value'] * 100):.1f}%",
            positions=portfolio_info["position_count"],
        )

        # Rescore existing positions if rebalancing enabled and all slots full
        position_scores: dict[str, dict[str, Any]] = {}
        position_entry_times: dict[str, datetime] = {}

        available_slots = max(0, base_max_positions - current_positions)

        if config.trading.enable_rebalancing and available_slots == 0:
            logger.info(
                "rebalancing_enabled_rescoring_positions", count=current_positions
            )

            # Get position symbols
            position_symbols = [pos.symbol for pos in positions]

            # Get tracked entry times (or use conservative fallback)
            for pos in positions:
                tracked_entry = position_tracker.get_entry_time(pos.symbol)
                if tracked_entry:
                    position_entry_times[pos.symbol] = tracked_entry
                else:
                    # Fallback: assume position is old enough to rebalance
                    position_entry_times[pos.symbol] = datetime.min.replace(tzinfo=UTC)
                    logger.warning(
                        "no_tracked_entry_time",
                        symbol=pos.symbol,
                        fallback="using_datetime_min",
                    )

            # Rescore positions
            position_scores = strategy.rescore_positions(position_symbols)

            # Log rescoring results
            for symbol, data in position_scores.items():
                hold_duration_min = (
                    position_tracker.get_hold_duration_seconds(symbol) / 60
                )
                logger.info(
                    "position_score",
                    symbol=symbol,
                    score=data["score"],
                    hold_duration_min=round(hold_duration_min, 1),
                )

        # If no slots available and rebalancing disabled, check cash deployment
        if available_slots == 0 and not config.trading.enable_rebalancing:
            # Check if we should deploy excess cash
            should_deploy, deploy_reason = position_sizer.should_deploy_cash(
                portfolio_info
            )
            if not should_deploy:
                logger.info("no_available_position_slots", cash_status=deploy_reason)
                return
            else:
                logger.info("cash_deployment_triggered", reason=deploy_reason)

        buy_candidates = []

        # Scan ALL universe (not just 20)
        for i, symbol in enumerate(self.universe):
            try:
                # Update heartbeat every 50 symbols to prevent timeout
                if i % 50 == 0:
                    watchdog.heartbeat(f"scanning_symbol_{i}/{len(self.universe)}")

                logger.info(
                    "scanning_symbol",
                    symbol=symbol,
                    progress=f"{i + 1}/{len(self.universe)}",
                )

                should_buy, score, reasoning = strategy.should_buy(symbol)

                if should_buy:
                    # Get current price
                    latest = alpaca_provider.get_latest_trade(symbol)
                    buy_candidates.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "reasoning": reasoning,
                            "price": latest["price"],
                        }
                    )
                    logger.info("buy_candidate_found", symbol=symbol, score=score)

                # Small delay to avoid rate limits
                time.sleep(0.3)

            except Exception as e:
                logger.error("scan_failed", symbol=symbol, error=str(e))
                continue

        # Sort by score and execute top trades
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)

        logger.info("scan_complete", candidates=len(buy_candidates))
        watchdog.heartbeat("scan_complete_checking_rebalancing")

        # Check for rebalancing opportunities if enabled and no slots available
        if (
            config.trading.enable_rebalancing
            and available_slots == 0
            and len(buy_candidates) > 0
        ):
            logger.info("checking_rebalancing_opportunities")

            # Check cooldown period
            if not position_tracker.can_rebalance_now(
                config.trading.rebalance_cooldown_minutes
            ):
                last_rebalance = position_tracker.get_last_rebalance_time()
                logger.info(
                    "rebalancing_in_cooldown",
                    last_rebalance=(
                        last_rebalance.isoformat() if last_rebalance else None
                    ),
                    cooldown_minutes=config.trading.rebalance_cooldown_minutes,
                )
                # Don't return - check cash deployment instead
            else:
                current_time = datetime.now(UTC)
                min_hold_seconds = config.trading.rebalance_min_hold_time * 60

                # Get locked positions
                locked_positions = position_tracker.get_locked_positions()
                if locked_positions:
                    logger.info("locked_positions", symbols=locked_positions)

                # Find weakest position that can be rebalanced
                weakest_symbol = None
                weakest_score = float("inf")

                for symbol, data in position_scores.items():
                    # Skip locked positions
                    if position_tracker.is_locked(symbol):
                        logger.info("position_locked_skip_rebalancing", symbol=symbol)
                        continue

                    score = data["score"]
                    entry_time = position_entry_times.get(
                        symbol, datetime.min.replace(tzinfo=UTC)
                    )
                    hold_duration = (current_time - entry_time).total_seconds()

                    # Check if position has been held long enough
                    if hold_duration >= min_hold_seconds and score < weakest_score:
                        weakest_score = score
                        weakest_symbol = symbol

                # Check if best candidate is significantly better
                if weakest_symbol and len(buy_candidates) > 0:
                    best_candidate = buy_candidates[0]
                    score_diff = best_candidate["score"] - weakest_score

                    if score_diff >= config.trading.rebalance_score_diff:
                        logger.info(
                            "rebalancing_opportunity_found",
                            old_symbol=weakest_symbol,
                            old_score=weakest_score,
                            new_symbol=best_candidate["symbol"],
                            new_score=best_candidate["score"],
                            score_diff=score_diff,
                        )

                        # Determine if partial or full rebalancing
                        partial_pct = config.trading.rebalance_partial_sell_pct

                        # Execute the swap (partial or full)
                        success = executor.swap_position(
                            weakest_symbol,
                            best_candidate["symbol"],
                            weakest_score,
                            best_candidate["score"],
                            best_candidate["reasoning"],
                            best_candidate["price"],
                            partial_sell_pct=partial_pct if partial_pct > 0 else 1.0,
                        )

                        if success:
                            logger.info("rebalancing_completed")
                            return  # Exit after rebalancing
                    else:
                        logger.info(
                            "no_rebalancing_needed",
                            score_diff=score_diff,
                            threshold=config.trading.rebalance_score_diff,
                        )

        # === DYNAMIC CASH DEPLOYMENT ===
        # Check if we should deploy excess cash by adding positions beyond regime limit
        adjusted_max_positions, max_reason = (
            position_sizer.get_max_positions_with_cash_deployment(
                base_max=base_max_positions,
                portfolio_info=portfolio_info,
                qualified_count=len(buy_candidates),
            )
        )

        logger.info(
            "position_limit_check",
            base_max=base_max_positions,
            adjusted_max=adjusted_max_positions,
            reason=max_reason,
            current_positions=current_positions,
        )

        # Recalculate available slots with adjusted max
        available_slots = max(0, adjusted_max_positions - current_positions)

        if available_slots == 0 and len(buy_candidates) > 0:
            logger.info(
                "no_slots_after_cash_deployment_check",
                base_max=base_max_positions,
                adjusted_max=adjusted_max_positions,
                current=current_positions,
            )
            return

        # === BATCH POSITION SIZING ===
        # Use dynamic position sizer for batch sizing
        if buy_candidates and available_slots > 0:
            size_results = position_sizer.calculate_batch_sizes(
                candidates=buy_candidates[
                    : available_slots * 2
                ],  # Consider 2x candidates for flexibility
                portfolio_info=portfolio_info,
                max_positions=adjusted_max_positions,
            )

            logger.info(
                "batch_sizing_completed",
                candidates_considered=min(len(buy_candidates), available_slots * 2),
                positions_sized=len(size_results),
            )

            # Execute trades with dynamic sizes
            trades_executed = 0
            for size_result in size_results:
                if trades_executed >= max_new_positions:
                    break

                # Find the candidate data for this symbol
                candidate = next(
                    (c for c in buy_candidates if c["symbol"] == size_result.symbol),
                    None,
                )
                if not candidate:
                    continue

                try:
                    logger.info(
                        "executing_sized_trade",
                        symbol=size_result.symbol,
                        size=size_result.recommended_size,
                        shares=size_result.recommended_shares,
                        conviction_mult=size_result.conviction_multiplier,
                        volatility_mult=size_result.volatility_multiplier,
                    )

                    success = executor.buy_stock_with_size(
                        symbol=candidate["symbol"],
                        score=candidate["score"],
                        reasoning=candidate["reasoning"],
                        current_price=candidate["price"],
                        size_result=size_result,
                    )

                    if success:
                        trades_executed += 1
                        time.sleep(1)  # Brief pause between orders

                except Exception as e:
                    logger.error(
                        "trade_execution_failed",
                        symbol=candidate["symbol"],
                        error=str(e),
                    )

            logger.info("trading_completed", trades_executed=trades_executed)

    def manage_positions(self) -> None:
        """Manage existing positions (stops, take profits, forced reductions)."""
        logger.info("managing_positions")

        # First, check account health and handle any critical issues
        self._check_and_handle_account_health()

        # Then run normal stop loss/take profit management
        executor.manage_stop_losses()

    def _check_and_handle_account_health(self) -> None:
        """Check account health and take corrective action if needed."""
        from src.risk_manager import risk_manager

        health = risk_manager.check_account_health()

        if health.get("healthy", True):
            return  # All good

        # Log the issues
        logger.warning(
            "account_health_issues_detected",
            warnings=health.get("warnings", []),
            actions_needed=health.get("actions_needed", []),
            cash=health.get("cash", 0),
            position_count=health.get("position_count", 0),
            max_positions=health.get("max_positions", 10),
        )

        # If we have negative cash or severe position overload, force reduction
        cash = health.get("cash", 0)
        position_count = health.get("position_count", 0)
        max_positions = health.get("max_positions", 10)

        # Calculate how many positions to reduce
        positions_over = max(0, position_count - max_positions)

        # Calculate cash shortfall - we need 5% minimum cash reserve
        portfolio_value = health.get("portfolio_value", 100000)
        cash_pct = health.get("cash_pct", 0)
        min_cash_pct = 0.05  # 5% minimum

        cash_needed = 0
        positions_to_sell_for_cash = 0

        if cash < 0:
            # Negative cash - need to cover it plus buffer
            cash_needed = abs(cash) + 5000  # Add $5K buffer
        elif cash_pct < min_cash_pct:
            # Below 5% minimum - need to sell to restore reserve
            target_cash = portfolio_value * min_cash_pct
            cash_needed = target_cash - cash
            logger.info(
                "cash_recovery_needed",
                current_cash=cash,
                target_cash=target_cash,
                cash_needed=cash_needed,
            )

        if cash_needed > 0:
            # Calculate how many positions to sell based on average position value
            avg_position_value = portfolio_value / max(position_count, 1)
            positions_to_sell_for_cash = int(cash_needed / avg_position_value) + 1

        # Total positions to reduce: max of (over limit) or (needed for cash recovery)
        target_reduction = max(positions_over, positions_to_sell_for_cash)

        if target_reduction > 0:
            logger.warning(
                "forced_position_reduction_needed",
                positions_over=positions_over,
                cash_needed=cash_needed,
                positions_to_sell_for_cash=positions_to_sell_for_cash,
                target_reduction=target_reduction,
            )

            # Get weakest positions to sell
            to_reduce = risk_manager.get_positions_to_reduce(
                target_reduction=target_reduction
            )

            if to_reduce:
                for pos in to_reduce:
                    symbol = pos["symbol"]
                    score = pos.get("score", 0)

                    logger.info(
                        "forced_selling_weak_position",
                        symbol=symbol,
                        score=score,
                        reason="account_health_recovery",
                    )

                    success = executor.sell_stock(
                        symbol, "forced_reduction_account_health"
                    )

                    if success:
                        logger.info("forced_sell_completed", symbol=symbol)
                        # Wait for order to process
                        import time

                        time.sleep(2)

                        # Check if we've recovered enough
                        new_health = risk_manager.check_account_health()
                        if new_health.get("healthy", False):
                            logger.info("account_health_recovered")
                            break
                    else:
                        logger.error("forced_sell_failed", symbol=symbol)

    def end_of_day_close(self) -> None:
        """Close all positions at end of day."""
        logger.info("end_of_day_routine")
        executor.close_all_positions()

    def run_trading_session(self) -> None:
        """Run a complete trading session."""
        logger.info("trading_session_started")

        if not self.is_market_open():
            logger.info("market_closed")
            return

        try:
            # Morning scan and initial trades
            self.scan_and_trade()

            # Manage positions throughout the day
            self.manage_positions()

            logger.info("trading_session_completed")

        except Exception as e:
            logger.error("trading_session_failed", error=str(e))

    def run_continuous_trading(self, auto_restart: bool = True) -> None:
        """Run continuous trading throughout market hours.

        Strategy:
        - Premarket (4:00-9:30 AM ET): Scan universe and queue top candidates
        - Market open: Execute premarket candidates immediately
        - During market: Full universe scan every 15 minutes
        - Position management every 2 minutes (stop losses, take profits)
        - If auto_restart=True, waits for market to open instead of exiting

        Safeguards:
        - Watchdog monitors for hangs and triggers restart if unresponsive
        - Heartbeat updates every loop iteration
        - Timeout protection on critical operations
        - Automatic recovery from API failures

        Args:
            auto_restart: If True, bot waits for market open and auto-restarts
        """
        logger.info(
            "continuous_trading_started",
            universe_size=len(self.universe),
            auto_restart=auto_restart,
        )

        # Start watchdog monitoring
        watchdog.start_monitoring()
        watchdog.heartbeat("trading_engine_starting")

        # Set up restart callback for watchdog
        def restart_callback() -> None:
            """Called by watchdog when a hang is detected."""
            logger.warning("watchdog_triggered_restart")
            # Force a new scan by resetting timers
            nonlocal last_scan_time, last_manage_time
            last_scan_time = None
            last_manage_time = None

        watchdog.set_restart_callback(restart_callback)

        scan_count = 0
        last_scan_time = None
        last_manage_time = None
        last_premarket_scan_time = None
        executed_premarket_today = False
        last_trading_day = None
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                # Update heartbeat at start of each loop iteration
                watchdog.heartbeat("main_loop_iteration")

                current_time = datetime.now(pytz.timezone("US/Eastern"))
                current_date = current_time.date()

                # Reset daily flags at midnight
                if last_trading_day != current_date:
                    executed_premarket_today = False
                    last_premarket_scan_time = None
                    self.premarket_candidates = []
                    last_trading_day = current_date
                    consecutive_errors = 0  # Reset error count on new day
                    logger.info("new_trading_day", date=str(current_date))
                    watchdog.heartbeat("new_trading_day_reset")

                # Check if market is open
                if self.is_market_open():
                    # === MARKET IS OPEN ===
                    watchdog.heartbeat("market_open_processing")

                    # Execute premarket candidates at market open (once per day)
                    if not executed_premarket_today and self.premarket_candidates:
                        logger.info(
                            "market_opened_executing_premarket_candidates",
                            candidates=len(self.premarket_candidates),
                        )
                        watchdog.heartbeat("executing_premarket_candidates")
                        try:
                            with OperationTimeout(
                                seconds=300, operation="premarket_execution"
                            ):
                                trades = self.execute_premarket_candidates(
                                    max_positions=3
                                )
                                executed_premarket_today = True
                                if trades > 0:
                                    last_scan_time = (
                                        current_time  # Skip immediate rescan
                                    )
                        except TimeoutError:
                            logger.error("premarket_execution_timeout")
                            executed_premarket_today = True  # Don't retry

                    # Full universe scan every 15 minutes
                    if (
                        last_scan_time is None
                        or (current_time - last_scan_time).total_seconds() >= 900
                    ):
                        logger.info(
                            "starting_full_universe_scan",
                            scan_number=scan_count + 1,
                            symbols=len(self.universe),
                        )
                        watchdog.heartbeat("starting_universe_scan")
                        try:
                            with OperationTimeout(
                                seconds=600, operation="universe_scan"
                            ):
                                self.scan_and_trade(max_new_positions=3)
                                last_scan_time = current_time
                                scan_count += 1
                                consecutive_errors = 0  # Reset on success
                        except TimeoutError:
                            logger.error(
                                "universe_scan_timeout",
                                scan_number=scan_count + 1,
                            )
                            last_scan_time = current_time  # Prevent immediate retry
                            consecutive_errors += 1
                        watchdog.heartbeat("universe_scan_complete")

                    # Manage positions every 2 minutes
                    if (
                        last_manage_time is None
                        or (current_time - last_manage_time).total_seconds() >= 120
                    ):
                        logger.info("running_position_management_check")
                        watchdog.heartbeat("managing_positions")
                        try:
                            with OperationTimeout(
                                seconds=120, operation="position_management"
                            ):
                                self.manage_positions()
                                last_manage_time = current_time
                        except TimeoutError:
                            logger.error("position_management_timeout")
                            last_manage_time = current_time

                    # Sleep for 30 seconds before next check
                    watchdog.heartbeat("sleeping_30s")
                    time.sleep(30)

                elif self.is_premarket_hours():
                    # === PREMARKET HOURS (4:00 AM - 9:30 AM ET) ===
                    watchdog.heartbeat("premarket_hours_processing")
                    minutes_to_open = self.get_minutes_until_market_open()

                    # Run premarket scan once, about 30-60 minutes before market open
                    # or if we haven't scanned today
                    should_scan = (
                        last_premarket_scan_time is None
                        or (current_time - last_premarket_scan_time).total_seconds()
                        >= 3600  # Re-scan every hour
                    )

                    if should_scan and minutes_to_open <= 90:  # Within 90 mins of open
                        logger.info(
                            "running_premarket_analysis",
                            minutes_to_open=minutes_to_open,
                        )
                        watchdog.heartbeat("running_premarket_scan")
                        try:
                            with OperationTimeout(
                                seconds=600, operation="premarket_scan"
                            ):
                                self.run_premarket_scan()
                                last_premarket_scan_time = current_time
                        except TimeoutError:
                            logger.error("premarket_scan_timeout")
                            last_premarket_scan_time = current_time

                        # Log top candidates
                        if self.premarket_candidates:
                            logger.info(
                                "premarket_top_candidates",
                                count=len(self.premarket_candidates),
                                top_5=[
                                    (c["symbol"], c["score"])
                                    for c in self.premarket_candidates[:5]
                                ],
                                minutes_to_open=minutes_to_open,
                            )
                    else:
                        logger.info(
                            "premarket_waiting",
                            minutes_to_open=minutes_to_open,
                            has_candidates=len(self.premarket_candidates) > 0,
                            next_scan_in=(
                                "60 min"
                                if last_premarket_scan_time
                                else f"{max(0, 90 - minutes_to_open)} min"
                            ),
                        )

                    # Check more frequently as we approach market open
                    if minutes_to_open <= 5:
                        watchdog.heartbeat("premarket_waiting_near_open")
                        time.sleep(30)  # Check every 30 seconds near open
                    elif minutes_to_open <= 15:
                        watchdog.heartbeat("premarket_waiting_15min")
                        time.sleep(60)  # Check every minute
                    else:
                        watchdog.heartbeat("premarket_waiting")
                        time.sleep(300)  # Check every 5 minutes

                else:
                    # === MARKET CLOSED (not premarket) ===
                    if auto_restart:
                        watchdog.heartbeat("market_closed_waiting")
                        logger.info(
                            "market_closed_waiting",
                            check_interval_min=5,
                            current_time=current_time.strftime("%H:%M ET"),
                        )
                        time.sleep(300)  # Wait 5 minutes, then check again
                        continue
                    else:
                        logger.info("market_closed_stopping_trading")
                        break

                # Check for too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "too_many_consecutive_errors",
                        count=consecutive_errors,
                        max_allowed=max_consecutive_errors,
                    )
                    # Force exit - systemd will restart us
                    watchdog.stop_monitoring()
                    os._exit(1)

            except KeyboardInterrupt:
                logger.info("continuous_trading_interrupted")
                watchdog.stop_monitoring()
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    "continuous_trading_error",
                    error=str(e),
                    consecutive_errors=consecutive_errors,
                )
                watchdog.heartbeat(f"error_recovery_{consecutive_errors}")
                time.sleep(60)  # Wait 1 minute before retrying

        watchdog.stop_monitoring()
        logger.info("continuous_trading_stopped", total_scans=scan_count)

    def run_eod_routine(self) -> None:
        """Run end-of-day routine."""
        logger.info("eod_routine_started")

        try:
            self.end_of_day_close()
            logger.info("eod_routine_completed")

        except Exception as e:
            logger.error("eod_routine_failed", error=str(e))


def main() -> None:
    """Main entry point."""
    logger.info("ai_trader_starting")

    engine = TradingEngine()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "scan":
            engine.run_trading_session()
        elif command == "continuous":
            # Check for auto-restart flag
            auto_restart = "--auto-restart" in sys.argv
            engine.run_continuous_trading(auto_restart=auto_restart)
        elif command == "eod":
            engine.run_eod_routine()
        elif command == "manage":
            engine.manage_positions()
        elif command == "premarket":
            # Run premarket scan manually
            logger.info("manual_premarket_scan_requested")
            candidates = engine.run_premarket_scan()
            print("\n=== Premarket Scan Results ===")
            print(f"Total candidates: {len(candidates)}")
            print("\nTop 10 candidates:")
            for i, c in enumerate(candidates[:10], 1):
                print(f"  {i}. {c['symbol']}: score={c['score']:.1f}")
        else:
            logger.error("unknown_command", command=command)
            print("Usage: python src/main.py [scan|continuous|eod|manage|premarket]")
            print("  continuous --auto-restart: Run 24/7 with premarket analysis")
            print("  premarket: Run premarket scan manually")
    else:
        # Default: run trading session
        engine.run_trading_session()


if __name__ == "__main__":
    main()
