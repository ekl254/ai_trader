"""Trade execution and order management."""

from datetime import datetime
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

from config.config import config
from src.logger import logger, log_trade_decision
from src.risk_manager import risk_manager


class TradeExecutor:
    """Executes trades and manages orders."""

    def __init__(self) -> None:
        self.client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper=True,
        )

    def place_market_order(
        self, symbol: str, qty: int, side: OrderSide
    ) -> Optional[str]:
        """
        Place a market order.
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.client.submit_order(order_data)
            
            logger.info(
                "order_placed",
                order_id=order.id,
                symbol=symbol,
                qty=qty,
                side=side.value,
                type="market",
            )
            
            return str(order.id)
            
        except Exception as e:
            logger.error(
                "order_failed",
                symbol=symbol,
                qty=qty,
                side=side.value,
                error=str(e),
            )
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
            
            order = self.client.submit_order(order_data)
            
            logger.info(
                "order_placed",
                order_id=order.id,
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
        self, symbol: str, score: float, reasoning: Dict, current_price: float
    ) -> bool:
        """
        Execute a buy order with full risk management.
        
        Returns:
            True if order placed successfully
        """
        logger.info("attempting_buy", symbol=symbol, score=score, price=current_price)
        
        # Check if we can open position
        if not risk_manager.can_open_position(symbol):
            logger.info("cannot_open_position", symbol=symbol)
            return False
        
        # Calculate position size
        position_info = risk_manager.calculate_position_size(symbol, current_price)
        shares = position_info["shares"]
        
        if shares <= 0:
            logger.warning("invalid_position_size", symbol=symbol)
            return False
        
        # Validate trade
        is_valid, reason = risk_manager.validate_trade(symbol, shares, current_price)
        if not is_valid:
            logger.warning("trade_validation_failed", symbol=symbol, reason=reason)
            return False
        
        # Log trade decision
        log_trade_decision(logger, symbol, "BUY", reasoning, score)
        
        # Execute order
        order_id = self.place_market_order(symbol, shares, OrderSide.BUY)
        
        if order_id:
            logger.info(
                "buy_executed",
                symbol=symbol,
                shares=shares,
                price=current_price,
                position_value=position_info["position_value"],
                stop_loss=position_info["stop_loss_price"],
                order_id=order_id,
            )
            return True
        
        return False

    def sell_stock(self, symbol: str, reason: str) -> bool:
        """
        Sell entire position in a stock.
        
        Returns:
            True if order placed successfully
        """
        try:
            positions = risk_manager.get_current_positions()
            
            if symbol not in positions:
                logger.warning("no_position_to_sell", symbol=symbol)
                return False
            
            qty = int(positions[symbol])
            
            logger.info("attempting_sell", symbol=symbol, qty=qty, reason=reason)
            
            order_id = self.place_market_order(symbol, qty, OrderSide.SELL)
            
            if order_id:
                logger.info(
                    "sell_executed",
                    symbol=symbol,
                    qty=qty,
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
            positions = self.client.get_all_positions()
            
            for position in positions:
                symbol = position.symbol
                current_price = float(position.current_price)
                avg_entry = float(position.avg_entry_price)
                unrealized_pl_pct = float(position.unrealized_plpc)
                
                # Check stop loss
                if unrealized_pl_pct <= -config.trading.stop_loss_pct:
                    logger.warning(
                        "stop_loss_triggered",
                        symbol=symbol,
                        entry=avg_entry,
                        current=current_price,
                        loss_pct=unrealized_pl_pct,
                    )
                    self.sell_stock(symbol, "stop_loss")
                
                # Check take profit
                elif unrealized_pl_pct >= config.trading.take_profit_pct:
                    logger.info(
                        "take_profit_triggered",
                        symbol=symbol,
                        entry=avg_entry,
                        current=current_price,
                        profit_pct=unrealized_pl_pct,
                    )
                    self.sell_stock(symbol, "take_profit")
                
        except Exception as e:
            logger.error("failed_to_manage_stops", error=str(e))


executor = TradeExecutor()
