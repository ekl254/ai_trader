"""Risk management and position sizing."""

from typing import Dict, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from config.config import config
from src.logger import logger


class RiskManager:
    """Manages risk and position sizing."""

    def __init__(self) -> None:
        self.config = config.trading
        self.client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper=True,
        )

    def get_account_value(self) -> float:
        """Get current account value."""
        try:
            account = self.client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error("failed_to_get_account_value", error=str(e))
            return self.config.portfolio_value

    def get_buying_power(self) -> float:
        """Get available buying power."""
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error("failed_to_get_buying_power", error=str(e))
            return 0.0

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions as {symbol: quantity}."""
        try:
            positions = self.client.get_all_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error("failed_to_get_positions", error=str(e))
            return {}

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate position size based on 2% risk rule.
        
        Args:
            symbol: Stock symbol
            entry_price: Intended entry price
            stop_loss_price: Stop loss price (if None, uses 2% below entry)
        
        Returns:
            Dict with shares, position_value, risk_amount
        """
        account_value = self.get_account_value()
        
        # Calculate risk amount (2% of account)
        risk_amount = account_value * self.config.risk_per_trade
        
        # Calculate stop loss if not provided
        if stop_loss_price is None:
            stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share <= 0:
            logger.warning(
                "invalid_risk_calculation",
                symbol=symbol,
                entry=entry_price,
                stop=stop_loss_price,
            )
            return {"shares": 0, "position_value": 0, "risk_amount": 0}
        
        # Calculate shares based on risk
        shares = risk_amount / risk_per_share
        shares = int(shares)  # Round down to whole shares
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Apply max position size constraint
        max_position_value = account_value * self.config.max_position_size
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
            logger.info(
                "position_size_capped",
                symbol=symbol,
                original_shares=int(risk_amount / risk_per_share),
                capped_shares=shares,
            )
        
        # Check buying power
        buying_power = self.get_buying_power()
        if position_value > buying_power:
            shares = int(buying_power / entry_price)
            position_value = shares * entry_price
            logger.warning(
                "insufficient_buying_power",
                symbol=symbol,
                required=position_value,
                available=buying_power,
                adjusted_shares=shares,
            )
        
        result = {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "stop_loss_price": stop_loss_price,
            "risk_reward_ratio": (entry_price * (1 + self.config.take_profit_pct) - entry_price) / risk_per_share if risk_per_share > 0 else 0,
        }
        
        logger.info(
            "position_size_calculated",
            symbol=symbol,
            entry_price=entry_price,
            **result,
        )
        
        return result

    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        current_positions = self.get_current_positions()
        
        # Check if already holding
        if symbol in current_positions:
            logger.info("already_holding_position", symbol=symbol)
            return False
        
        # Check max positions
        if len(current_positions) >= self.config.max_positions:
            logger.info(
                "max_positions_reached",
                current=len(current_positions),
                max=self.config.max_positions,
            )
            return False
        
        return True

    def validate_trade(
        self, symbol: str, shares: int, price: float
    ) -> Tuple[bool, str]:
        """
        Validate a trade before execution.
        
        Returns:
            (is_valid, reason)
        """
        if shares <= 0:
            return False, "Invalid share quantity"
        
        position_value = shares * price
        account_value = self.get_account_value()
        
        # Check position size limits
        position_pct = position_value / account_value
        if position_pct > self.config.max_position_size:
            return False, f"Position size {position_pct:.1%} exceeds max {self.config.max_position_size:.1%}"
        
        # Check buying power
        buying_power = self.get_buying_power()
        if position_value > buying_power:
            return False, f"Insufficient buying power: ${buying_power:.2f} < ${position_value:.2f}"
        
        return True, "Trade validated"


risk_manager = RiskManager()
