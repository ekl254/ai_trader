"""Trade execution and order management."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
from alpaca.trading.models import Order, Position
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from config.config import config
from src.logger import log_trade_decision, logger
from src.performance_tracker import PerformanceTracker  # type: ignore
from src.position_tracker import position_tracker  # type: ignore
from src.risk_manager import risk_manager
from src.position_sizer import position_sizer, PositionSizeResult
from src.clients import get_trading_client, trading_circuit_breaker


class TradeExecutor:
    """Executes trades and manages orders."""

    def __init__(self) -> None:
        self.client = get_trading_client()
        self.performance_tracker = PerformanceTracker()

    def place_market_order(
        self, symbol: str, qty: int, side: OrderSide
    ) -> Optional[str]:
        """
        Place a market order.

        Returns:
            Order ID if successful, None otherwise
        """
        # Check circuit breaker
        if not trading_circuit_breaker.can_proceed():
            logger.error(
                "order_blocked_by_circuit_breaker",
                symbol=symbol,
                status=trading_circuit_breaker.get_status(),
            )
            return None
        
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )

            order: Order = self.client.submit_order(order_data)  # type: ignore

            logger.info(
                "order_placed",
                order_id=str(order.id),
                symbol=symbol,
                qty=qty,
                side=side.value,
                type="market",
            )

            trading_circuit_breaker.record_success()
            return str(order.id)

        except Exception as e:
            trading_circuit_breaker.record_failure()
            logger.error(
                "order_failed",
                symbol=symbol,
                qty=qty,
                side=side.value,
                error=str(e),
            )
            return None

    def wait_for_order_fill(
        self, order_id: str, timeout_seconds: int = 30, poll_interval: float = 0.5
    ) -> Optional[Order]:
        """
        Wait for an order to be filled.
        
        Args:
            order_id: The order ID to check
            timeout_seconds: Maximum time to wait
            poll_interval: Time between status checks
            
        Returns:
            The filled order, or None if not filled within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                order = self.client.get_order_by_id(order_id)
                
                if order.status == OrderStatus.FILLED:
                    logger.info(
                        "order_filled",
                        order_id=order_id,
                        symbol=order.symbol,
                        filled_qty=order.filled_qty,
                        filled_avg_price=order.filled_avg_price,
                    )
                    return order
                
                if order.status in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                    logger.warning(
                        "order_not_filled",
                        order_id=order_id,
                        status=str(order.status),
                        symbol=order.symbol,
                    )
                    return None
                
                # Order still pending, wait and retry
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error("order_status_check_failed", order_id=order_id, error=str(e))
                time.sleep(poll_interval)
        
        logger.warning("order_fill_timeout", order_id=order_id, timeout=timeout_seconds)
        return None

    def place_limit_order(
        self, symbol: str, qty: int, side: OrderSide, limit_price: float
    ) -> Optional[str]:
        """
        Place a limit order.

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )

            order: Order = self.client.submit_order(order_data)  # type: ignore

            logger.info(
                "order_placed",
                order_id=str(order.id),
                symbol=symbol,
                qty=qty,
                side=side.value,
                type="limit",
                limit_price=limit_price,
            )

            return str(order.id)

        except Exception as e:
            logger.error(
                "order_failed",
                symbol=symbol,
                qty=qty,
                side=side.value,
                limit_price=limit_price,
                error=str(e),
            )
            return None

    def buy_stock(
        self, symbol: str, score: float, reasoning: Dict, current_price: float,
        size_override: Optional[PositionSizeResult] = None,
        skip_tracking: bool = False
    ) -> bool:
        """
        Execute a buy order with full risk management and dynamic sizing.

        Args:
            symbol: Stock symbol
            score: Composite score
            reasoning: Reasoning dict with score breakdown
            current_price: Current stock price
            size_override: Optional pre-calculated position size
            skip_tracking: If True, skip position tracking (used by swap_position)

        Returns:
            True if order placed successfully
        """
        logger.info("attempting_buy", symbol=symbol, score=score, price=current_price)

        # Check if we can open position
        if not risk_manager.can_open_position(symbol, score=score):
            logger.info("cannot_open_position", symbol=symbol)
            return False

        # Mark as pending to prevent double buys
        risk_manager.mark_pending_buy(symbol)

        # Use dynamic position sizing
        if size_override:
            size_result = size_override
        else:
            size_result = position_sizer.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                composite_score=score,
            )
        
        shares = size_result.recommended_shares

        if shares <= 0:
            logger.warning("invalid_position_size", symbol=symbol)
            return False

        # Validate trade using risk manager
        is_valid, reason = risk_manager.validate_trade(symbol, shares, current_price)
        if not is_valid:
            logger.warning("trade_validation_failed", symbol=symbol, reason=reason)
            return False

        # Log trade decision with sizing rationale
        log_trade_decision(logger, symbol, "BUY", reasoning, score)
        logger.info(
            "position_sizing_applied",
            symbol=symbol,
            recommended_size=size_result.recommended_size,
            shares=shares,
            conviction_mult=size_result.conviction_multiplier,
            volatility_mult=size_result.volatility_multiplier,
            rationale=size_result.rationale,
        )

        # Execute order
        order_id = self.place_market_order(symbol, shares, OrderSide.BUY)

        if order_id:
            # Wait for order fill before proceeding
            filled_order = self.wait_for_order_fill(order_id, timeout_seconds=30)
            
            if filled_order is None:
                logger.error(
                    "buy_order_not_filled",
                    order_id=order_id,
                    symbol=symbol,
                    shares=shares,
                )
                # Order wasn't filled - clear pending marker and fail
                risk_manager.clear_pending_buy(symbol)
                return False
            
            # Order filled - now safe to clear pending and track
            risk_manager.clear_pending_buy(symbol)
            
            # Record entry for daily limit tracking
            position_sizer.record_entry(symbol)
            
            # Use actual fill price if available
            actual_price = float(filled_order.filled_avg_price or current_price)
            actual_shares = int(filled_order.filled_qty or shares)
            
            # Track position entry with detailed score breakdown and sizing info
            # Skip if called from swap_position (which handles its own tracking)
            if not skip_tracking:
                position_tracker.track_entry(
                symbol=symbol,
                entry_price=actual_price,
                score=score,
                reason="new_position",
                score_breakdown={
                    "technical": reasoning.get("technical", {}).get("total", 0),
                    "sentiment": reasoning.get("sentiment", {}).get("total", 0),
                    "fundamental": reasoning.get("fundamental", {}).get("total", 0),
                },
                news_sentiment=reasoning.get("sentiment", {}).get("total"),
                news_count=None,  # News count not tracked in current sentiment structure
                )

            logger.info(
                "buy_executed",
                symbol=symbol,
                shares=actual_shares,
                price=actual_price,
                position_value=actual_shares * actual_price,
                conviction_multiplier=size_result.conviction_multiplier,
                volatility_multiplier=size_result.volatility_multiplier,
                order_id=order_id,
            )
            return True

        # Order failed - clear pending marker
        risk_manager.clear_pending_buy(symbol)
        return False

    def buy_stock_with_size(
        self, 
        symbol: str, 
        score: float, 
        reasoning: Dict, 
        current_price: float,
        size_result: PositionSizeResult
    ) -> bool:
        """
        Execute a buy order with a pre-calculated position size.
        
        This is used by main.py when batch sizing is already done.
        """
        return self.buy_stock(
            symbol=symbol,
            score=score,
            reasoning=reasoning,
            current_price=current_price,
            size_override=size_result,
            skip_tracking=False
        )

    def sell_stock(self, symbol: str, reason: str, sell_pct: float = 1.0) -> bool:
        """
        Sell entire or partial position in a stock.

        Args:
            symbol: Stock symbol
            reason: Reason for selling
            sell_pct: Percentage of position to sell (0.0-1.0, default 1.0 = 100%)

        Returns:
            True if order placed successfully
        """
        try:
            positions = risk_manager.get_current_positions()

            if symbol not in positions:
                logger.warning("no_position_to_sell", symbol=symbol)
                return False

            total_qty = int(positions[symbol])
            qty_to_sell = int(total_qty * sell_pct)

            if qty_to_sell <= 0:
                logger.warning(
                    "invalid_sell_quantity",
                    symbol=symbol,
                    total_qty=total_qty,
                    sell_pct=sell_pct,
                    qty_to_sell=qty_to_sell,
                )
                return False

            logger.info(
                "attempting_sell",
                symbol=symbol,
                qty=qty_to_sell,
                total_qty=total_qty,
                sell_pct=sell_pct,
                reason=reason,
            )

            order_id = self.place_market_order(symbol, qty_to_sell, OrderSide.SELL)

            if order_id:
                # Get position info before exit
                position_data = position_tracker.data.get("positions", {}).get(symbol)

                # Track position exit only if selling entire position
                if sell_pct >= 1.0:
                    position_tracker.track_exit(symbol=symbol, exit_reason=reason)

                    # Record complete trade for performance tracking
                    if position_data:
                        try:
                            # Get current position details for exit price
                            position = self.client.get_open_position(symbol)
                            exit_price = (
                                float(position.current_price)
                                if position.current_price
                                else 0
                            )

                            # Parse entry data
                            entry_time = datetime.fromisoformat(
                                position_data.get(
                                    "entry_time", datetime.now().isoformat()
                                )
                            )
                            entry_price = position_data.get("entry_price", 0)
                            entry_score = position_data.get("score_breakdown", {})

                            # Record the trade
                            self.performance_tracker.record_trade(
                                symbol=symbol,
                                entry_time=entry_time,
                                entry_price=entry_price,
                                entry_score={
                                    "composite": position_data.get("score", 0),
                                    "technical": entry_score.get("technical", 0),
                                    "sentiment": entry_score.get("sentiment", 0),
                                    "fundamental": entry_score.get("fundamental", 0),
                                },
                                exit_time=datetime.now(),
                                exit_price=exit_price,
                                exit_reason=reason,
                                quantity=qty_to_sell,
                                news_sentiment=position_data.get("news_sentiment"),
                                news_count=position_data.get("news_count"),
                            )
                        except Exception as e:
                            logger.error(
                                "failed_to_record_trade_performance",
                                symbol=symbol,
                                error=str(e),
                            )

                logger.info(
                    "sell_executed",
                    symbol=symbol,
                    qty=qty_to_sell,
                    total_qty=total_qty,
                    sell_pct=sell_pct,
                    partial=sell_pct < 1.0,
                    reason=reason,
                    order_id=order_id,
                )
                return True

            return False

        except Exception as e:
            logger.error("sell_failed", symbol=symbol, error=str(e))
            return False

    def close_all_positions(self) -> None:
        """Close all open positions (end of day)."""
        try:
            positions = risk_manager.get_current_positions()

            logger.info("closing_all_positions", count=len(positions))

            for symbol in positions:
                self.sell_stock(symbol, "end_of_day_close")

            logger.info("all_positions_closed")

        except Exception as e:
            logger.error("failed_to_close_all_positions", error=str(e))

    def manage_stop_losses(self) -> None:
        """Check and manage stop losses for open positions."""
        try:
            positions: List[Position] = self.client.get_all_positions()  # type: ignore

            for position in positions:
                symbol = position.symbol
                current_price = float(position.current_price or 0)
                avg_entry = float(position.avg_entry_price)
                unrealized_pl_pct = float(position.unrealized_plpc or 0)

                # Get regime-adjusted thresholds
                stop_loss_pct = risk_manager.get_stop_loss_pct()
                take_profit_pct = risk_manager.get_take_profit_pct()

                # Check stop loss
                if unrealized_pl_pct <= -stop_loss_pct:
                    logger.warning(
                        "stop_loss_triggered",
                        symbol=symbol,
                        entry=avg_entry,
                        current=current_price,
                        loss_pct=unrealized_pl_pct,
                        threshold=stop_loss_pct,
                    )
                    self.sell_stock(symbol, "stop_loss")

                # Check take profit
                elif unrealized_pl_pct >= take_profit_pct:
                    logger.info(
                        "take_profit_triggered",
                        symbol=symbol,
                        entry=avg_entry,
                        current=current_price,
                        profit_pct=unrealized_pl_pct,
                        threshold=take_profit_pct,
                    )
                    self.sell_stock(symbol, "take_profit")

        except Exception as e:
            logger.error("failed_to_manage_stops", error=str(e))

    def swap_position(
        self,
        old_symbol: str,
        new_symbol: str,
        old_score: float,
        new_score: float,
        new_reasoning: Dict,
        new_price: float,
        partial_sell_pct: float = 1.0,
    ) -> bool:
        """
        Swap one position for another (rebalancing).
        
        IMPORTANT: This uses SELL-FIRST logic to ensure we free up cash
        before attempting to buy. This prevents the scenario where we're
        at max positions and can't execute the swap.

        Args:
            old_symbol: Symbol to sell
            new_symbol: Symbol to buy
            old_score: Score of old symbol
            new_score: Score of new symbol
            new_reasoning: Reasoning for new symbol
            new_price: Current price of new symbol
            partial_sell_pct: Percentage of old position to sell (0.0-1.0, default 1.0)

        Returns:
            True if swap executed successfully
        """
        new_symbol = new_symbol.upper()
        old_symbol = old_symbol.upper()
        
        # Pre-check: Is the new symbol already held?
        # (Don't check position limits - we're going to sell first!)
        current_positions = risk_manager.get_current_positions()
        if new_symbol in current_positions:
            logger.warning(
                "swap_aborted_already_holding_target",
                old_symbol=old_symbol,
                new_symbol=new_symbol,
            )
            return False
        
        # Check if there's a pending buy for the target symbol
        if risk_manager.is_pending_buy(new_symbol):
            logger.warning(
                "swap_aborted_pending_buy_exists",
                old_symbol=old_symbol,
                new_symbol=new_symbol,
            )
            return False

        score_diff = new_score - old_score
        is_partial = partial_sell_pct < 1.0

        logger.info(
            "attempting_position_swap_sell_first",
            old_symbol=old_symbol,
            new_symbol=new_symbol,
            old_score=old_score,
            new_score=new_score,
            score_diff=score_diff,
            partial=is_partial,
            sell_pct=partial_sell_pct if is_partial else 1.0,
        )

        # === SELL FIRST ===
        # This frees up both a position slot AND cash for the buy
        sell_success = self.sell_stock(
            old_symbol, "rebalancing", sell_pct=partial_sell_pct
        )

        if not sell_success:
            logger.error("swap_failed_on_sell", old_symbol=old_symbol)
            return False

        # Wait for sell order to fill with proper verification
        # Give more time for the order to process
        max_retries = 5
        for attempt in range(max_retries):
            time.sleep(1)  # Check every second
            updated_positions = risk_manager.get_current_positions()
            
            if partial_sell_pct >= 1.0:
                # Full sell - position should be gone
                if old_symbol not in updated_positions:
                    logger.info("swap_sell_confirmed", old_symbol=old_symbol, attempt=attempt + 1)
                    break
            else:
                # Partial sell - harder to verify, just wait a bit more
                if attempt >= 2:
                    break
        else:
            if partial_sell_pct >= 1.0 and old_symbol in updated_positions:
                logger.warning(
                    "swap_sell_may_not_have_completed",
                    old_symbol=old_symbol,
                    still_in_positions=True,
                    attempts=max_retries,
                )
                # Continue anyway - the order was placed

        # === THEN BUY ===
        # Mark as pending to prevent race conditions
        risk_manager.mark_pending_buy(new_symbol)
        
        try:
            # Use dynamic sizing - skip the can_open_position check since we just sold
            size_result = position_sizer.calculate_position_size(
                symbol=new_symbol,
                current_price=new_price,
                composite_score=new_score,
            )
            
            if size_result.recommended_shares <= 0:
                logger.error("swap_failed_invalid_size", new_symbol=new_symbol)
                risk_manager.clear_pending_buy(new_symbol)
                return False
            
            # Validate the trade
            is_valid, reason = risk_manager.validate_trade(
                new_symbol, size_result.recommended_shares, new_price
            )
            if not is_valid:
                logger.error("swap_failed_validation", new_symbol=new_symbol, reason=reason)
                risk_manager.clear_pending_buy(new_symbol)
                return False
            
            # Log trade decision
            log_trade_decision(logger, new_symbol, "BUY", new_reasoning, new_score)
            
            # Execute the buy order
            order_id = self.place_market_order(
                new_symbol, size_result.recommended_shares, OrderSide.BUY
            )
            
            if not order_id:
                logger.error("swap_failed_on_buy_order", new_symbol=new_symbol)
                risk_manager.clear_pending_buy(new_symbol)
                return False
            
            # Success! Clear pending and track
            risk_manager.clear_pending_buy(new_symbol)
            position_sizer.record_entry(new_symbol)
            
            # Track the new position with rebalancing reason
            position_tracker.track_entry(
                symbol=new_symbol,
                entry_price=new_price,
                score=new_score,
                reason="rebalancing",
                score_breakdown={
                    "technical": new_reasoning.get("technical", {}).get("total", 0),
                    "sentiment": new_reasoning.get("sentiment", {}).get("total", 0),
                    "fundamental": new_reasoning.get("fundamental", {}).get("total", 0),
                },
                news_sentiment=new_reasoning.get("sentiment", {}).get("total"),
                news_count=None,
            )
            
            logger.info(
                "swap_buy_executed",
                symbol=new_symbol,
                shares=size_result.recommended_shares,
                price=new_price,
                order_id=order_id,
            )
            
        except Exception as e:
            logger.error("swap_buy_exception", new_symbol=new_symbol, error=str(e))
            risk_manager.clear_pending_buy(new_symbol)
            return False

        # Track rebalancing event AFTER successful swap
        position_tracker.track_rebalance(
            old_symbol=old_symbol,
            new_symbol=new_symbol,
            old_score=old_score,
            new_score=new_score,
            score_diff=score_diff,
        )

        logger.info(
            "position_swap_completed",
            old_symbol=old_symbol,
            new_symbol=new_symbol,
            old_score=old_score,
            new_score=new_score,
            score_diff=score_diff,
        )

        return True


executor = TradeExecutor()
