"""Main trading engine."""

import sys
import time
from datetime import datetime, time as dt_time
from typing import List

import pytz  # type: ignore

from config.config import config
from src.data_provider import alpaca_provider
from src.executor import executor
from src.logger import logger
from src.strategy import strategy
from src.universe import get_sp500_symbols, filter_liquid_stocks


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
            from alpaca.trading.client import TradingClient
            
            client = TradingClient(
                config.alpaca.api_key,
                config.alpaca.secret_key,
                paper=True,
            )
            clock = client.get_clock()
            return clock.is_open
            
        except Exception as e:
            logger.error("failed_to_check_market_status", error=str(e))
            return False

    def scan_and_trade(self, max_new_positions: int = 5) -> None:
        """Scan universe and execute trades."""
        logger.info("starting_market_scan", universe_size=len(self.universe))
        
        # Check how many open slots we have
        from alpaca.trading.client import TradingClient
        client = TradingClient(config.alpaca.api_key, config.alpaca.secret_key, paper=True)
        current_positions = len(client.get_all_positions())
        max_positions = config.trading.max_positions
        available_slots = max(0, max_positions - current_positions)
        
        logger.info("position_check", current=current_positions, max=max_positions, available=available_slots)
        
        if available_slots == 0:
            logger.info("no_available_position_slots")
            return
        
        buy_candidates = []
        
        # Scan ALL universe (not just 20)
        for i, symbol in enumerate(self.universe):
            try:
                logger.info("scanning_symbol", symbol=symbol, progress=f"{i+1}/{len(self.universe)}")
                
                should_buy, score, reasoning = strategy.should_buy(symbol)
                
                if should_buy:
                    # Get current price
                    latest = alpaca_provider.get_latest_trade(symbol)
                    buy_candidates.append({
                        "symbol": symbol,
                        "score": score,
                        "reasoning": reasoning,
                        "price": latest["price"],
                    })
                    logger.info("buy_candidate_found", symbol=symbol, score=score)
                
                # Small delay to avoid rate limits
                time.sleep(0.3)
                
            except Exception as e:
                logger.error("scan_failed", symbol=symbol, error=str(e))
                continue
        
        # Sort by score and execute top trades
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info("scan_complete", candidates=len(buy_candidates))
        
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
                logger.error("trade_execution_failed", symbol=candidate["symbol"], error=str(e))

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

    def run_continuous_trading(self) -> None:
        """Run continuous trading throughout market hours.
        
        Strategy:
        - Full universe scan every 15 minutes (looking for new opportunities)
        - Position management every 2 minutes (stop losses, take profits)
        - Automatically buy when slots available
        - Automatically sell when conditions met
        """
        logger.info("continuous_trading_started", universe_size=len(self.universe))
        
        scan_count = 0
        last_scan_time = None
        last_manage_time = None
        
        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    logger.info("market_closed_stopping_trading")
                    break
                
                current_time = datetime.now(pytz.timezone('US/Eastern'))
                
                # Full universe scan every 15 minutes
                if last_scan_time is None or (current_time - last_scan_time).total_seconds() >= 900:
                    logger.info("starting_full_universe_scan", scan_number=scan_count + 1, symbols=len(self.universe))
                    self.scan_and_trade(max_new_positions=3)  # Max 3 new positions per scan
                    last_scan_time = current_time
                    scan_count += 1
                
                # Manage positions every 2 minutes (sell on stops/profits)
                if last_manage_time is None or (current_time - last_manage_time).total_seconds() >= 120:
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
            engine.run_continuous_trading()
        elif command == "eod":
            engine.run_eod_routine()
        elif command == "manage":
            engine.manage_positions()
        else:
            logger.error("unknown_command", command=command)
            print("Usage: python src/main.py [scan|continuous|eod|manage]")
    else:
        # Default: run trading session
        engine.run_trading_session()


if __name__ == "__main__":
    main()
