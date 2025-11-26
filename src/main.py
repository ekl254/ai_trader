"""Main trading engine."""

import sys
import time
from datetime import datetime, time as dt_time, timezone
from typing import List, Dict, Any

import pytz  # type: ignore
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position, Clock

from config.config import config
from src.data_provider import alpaca_provider
from src.executor import executor
from src.logger import logger
from src.strategy import strategy
from src.universe import get_sp500_symbols, filter_liquid_stocks
from src.position_tracker import position_tracker  # type: ignore


class TradingEngine:
    """Main trading engine orchestrator."""

    def __init__(self) -> None:
        self.universe = self._load_universe()
        logger.info("trading_engine_initialized", universe_size=len(self.universe))

    def _load_universe(self) -> List[str]:
        """Load and filter stock universe."""
        symbols = get_sp500_symbols()
        liquid_symbols = filter_liquid_stocks(symbols)
        logger.info("universe_loaded", total=len(symbols), liquid=len(liquid_symbols))
        return liquid_symbols

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            client = TradingClient(
                config.alpaca.api_key,
                config.alpaca.secret_key,
                paper=True,
            )
            clock: Clock = client.get_clock()  # type: ignore
            return bool(clock.is_open)

        except Exception as e:
            logger.error("failed_to_check_market_status", error=str(e))
            return False

    def scan_and_trade(self, max_new_positions: int = 5) -> None:
        """Scan universe and execute trades (with optional rebalancing)."""
        logger.info("starting_market_scan", universe_size=len(self.universe))

        # Check how many open slots we have
        client = TradingClient(
            config.alpaca.api_key, config.alpaca.secret_key, paper=True
        )
        positions: List[Position] = client.get_all_positions()  # type: ignore
        current_positions = len(positions)
        max_positions = config.trading.max_positions
        available_slots = max(0, max_positions - current_positions)

        logger.info(
            "position_check",
            current=current_positions,
            max=max_positions,
            available=available_slots,
        )

        # Rescore existing positions if rebalancing enabled and all slots full
        position_scores: Dict[str, Dict[str, Any]] = {}
        position_entry_times: Dict[str, datetime] = {}

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
                    position_entry_times[pos.symbol] = datetime.min.replace(
                        tzinfo=timezone.utc
                    )
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

        # If no slots available and rebalancing disabled, exit early
        if available_slots == 0 and not config.trading.enable_rebalancing:
            logger.info("no_available_position_slots")
            return

        buy_candidates = []

        # Scan ALL universe (not just 20)
        for i, symbol in enumerate(self.universe):
            try:
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
                    last_rebalance=last_rebalance.isoformat()
                    if last_rebalance
                    else None,
                    cooldown_minutes=config.trading.rebalance_cooldown_minutes,
                )
                return

            current_time = datetime.now(timezone.utc)
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
                    symbol, datetime.min.replace(tzinfo=timezone.utc)
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

            return  # No slots and no rebalancing done, exit

        # Execute trades for available slots
        trades_to_execute = min(available_slots, max_new_positions, len(buy_candidates))

        for candidate in buy_candidates[:trades_to_execute]:
            try:
                success = executor.buy_stock(
                    candidate["symbol"],
                    candidate["score"],
                    candidate["reasoning"],
                    candidate["price"],
                )

                if success:
                    time.sleep(1)  # Brief pause between orders

            except Exception as e:
                logger.error(
                    "trade_execution_failed", symbol=candidate["symbol"], error=str(e)
                )

    def manage_positions(self) -> None:
        """Manage existing positions (stops, take profits)."""
        logger.info("managing_positions")
        executor.manage_stop_losses()

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
        - Full universe scan every 15 minutes (looking for new opportunities)
        - Position management every 2 minutes (stop losses, take profits)
        - Automatically buy when slots available
        - Automatically sell when conditions met
        - If auto_restart=True, waits for market to open instead of exiting

        Args:
            auto_restart: If True, bot waits for market open and auto-restarts
        """
        logger.info(
            "continuous_trading_started",
            universe_size=len(self.universe),
            auto_restart=auto_restart,
        )

        scan_count = 0
        last_scan_time = None
        last_manage_time = None

        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    if auto_restart:
                        logger.info(
                            "market_closed_waiting_for_open", check_interval_min=5
                        )
                        time.sleep(300)  # Wait 5 minutes, then check again
                        continue
                    else:
                        logger.info("market_closed_stopping_trading")
                        break

                current_time = datetime.now(pytz.timezone("US/Eastern"))

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
                    self.scan_and_trade(
                        max_new_positions=3
                    )  # Max 3 new positions per scan
                    last_scan_time = current_time
                    scan_count += 1

                # Manage positions every 2 minutes (sell on stops/profits)
                if (
                    last_manage_time is None
                    or (current_time - last_manage_time).total_seconds() >= 120
                ):
                    logger.info("running_position_management_check")
                    self.manage_positions()
                    last_manage_time = current_time

                # Sleep for 30 seconds before next check
                time.sleep(30)

            except KeyboardInterrupt:
                logger.info("continuous_trading_interrupted")
                break
            except Exception as e:
                logger.error("continuous_trading_error", error=str(e))
                time.sleep(60)  # Wait 1 minute before retrying

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
        else:
            logger.error("unknown_command", command=command)
            print("Usage: python src/main.py [scan|continuous|eod|manage]")
            print("  continuous --auto-restart: Run 24/7, auto-start when market opens")
    else:
        # Default: run trading session
        engine.run_trading_session()


if __name__ == "__main__":
    main()
