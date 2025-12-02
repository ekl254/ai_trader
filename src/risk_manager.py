"""Risk management and position sizing with regime-adaptive parameters."""

from typing import Dict, Optional, Tuple

from alpaca.trading.client import TradingClient

from config.config import config
from src.logger import logger


class RiskManager:
    """Manages risk and position sizing with regime-adaptive parameters."""

    def __init__(self) -> None:
        self.config = config.trading
        self.client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper=True,
        )
        self._regime_params: Optional[Dict] = None

    def set_regime_parameters(self, regime_params: Dict) -> None:
        """
        Set regime-specific trading parameters.
        
        Args:
            regime_params: Dict with stop_loss_pct, take_profit_pct, etc.
        """
        self._regime_params = regime_params
        logger.info(
            "regime_parameters_set",
            regime=regime_params.get("regime", "unknown"),
            stop_loss=regime_params.get("stop_loss_pct"),
            take_profit=regime_params.get("take_profit_pct"),
            min_score=regime_params.get("min_score"),
        )

    def get_stop_loss_pct(self) -> float:
        """Get current stop loss percentage (regime-adjusted or default)."""
        if self._regime_params and "stop_loss_pct" in self._regime_params:
            return self._regime_params["stop_loss_pct"]
        return self.config.stop_loss_pct

    def get_take_profit_pct(self) -> float:
        """Get current take profit percentage (regime-adjusted or default)."""
        if self._regime_params and "take_profit_pct" in self._regime_params:
            return self._regime_params["take_profit_pct"]
        return self.config.take_profit_pct

    def get_min_score(self) -> float:
        """Get current minimum score threshold (regime-adjusted or default)."""
        if self._regime_params and "min_score" in self._regime_params:
            return self._regime_params["min_score"]
        return self.config.min_composite_score

    def get_position_multiplier(self) -> float:
        """Get position size multiplier for current regime."""
        if self._regime_params and "position_multiplier" in self._regime_params:
            return self._regime_params["position_multiplier"]
        return 1.0

    def get_max_positions(self) -> int:
        """Get max positions for current regime."""
        if self._regime_params and "max_positions" in self._regime_params:
            return self._regime_params["max_positions"]
        return self.config.max_positions

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
        Calculate position size based on risk rule with regime-adaptive parameters.

        Args:
            symbol: Stock symbol
            entry_price: Intended entry price
            stop_loss_price: Stop loss price (if None, uses regime-adjusted %)

        Returns:
            Dict with shares, position_value, risk_amount, stop/take prices
        """
        account_value = self.get_account_value()

        # Calculate risk amount (2% of account, adjusted by regime multiplier)
        base_risk = account_value * self.config.risk_per_trade
        risk_amount = base_risk * self.get_position_multiplier()

        # Get regime-adjusted stop loss
        stop_loss_pct = self.get_stop_loss_pct()
        take_profit_pct = self.get_take_profit_pct()

        # Calculate stop loss price if not provided
        if stop_loss_price is None:
            stop_loss_price = entry_price * (1 - stop_loss_pct)

        # Calculate take profit price
        take_profit_price = entry_price * (1 + take_profit_pct)

        # Calculate risk per share
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0:
            logger.warning(
                "invalid_risk_calculation",
                symbol=symbol,
                entry=entry_price,
                stop=stop_loss_price,
            )
            return {
                "shares": 0, 
                "position_value": 0, 
                "risk_amount": 0,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
            }

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
            "stop_loss_pct": stop_loss_pct,
            "take_profit_price": take_profit_price,
            "take_profit_pct": take_profit_pct,
            "risk_reward_ratio": (take_profit_price - entry_price) / risk_per_share if risk_per_share > 0 else 0,
            "regime": self._regime_params.get("regime", "default") if self._regime_params else "default",
        }

        logger.info(
            "position_size_calculated",
            symbol=symbol,
            entry_price=entry_price,
            regime=result["regime"],
            stop_loss_pct=f"{stop_loss_pct:.1%}",
            take_profit_pct=f"{take_profit_pct:.1%}",
            shares=shares,
            position_value=position_value,
        )

        return result

    def can_open_position(self, symbol: str, score: float = 0) -> bool:
        """
        Check if we can open a new position.
        
        Args:
            symbol: Stock symbol
            score: Composite score (optional) - high scores can exceed regime limits
            
        Returns:
            True if position can be opened
        """
        current_positions = self.get_current_positions()

        # Check if already holding
        if symbol in current_positions:
            logger.info("already_holding_position", symbol=symbol)
            return False

        # Get regime-adjusted max and absolute max from config
        regime_max = self.get_max_positions()
        absolute_max = self.config.max_positions  # Hard cap from config (usually 10)
        current_count = len(current_positions)
        
        # Dynamic position limit based on score:
        # - Score >= 80: Can go up to absolute max (high conviction)
        # - Score >= 75: Can exceed regime limit by 2
        # - Score >= 70: Can exceed regime limit by 1
        # - Below 70: Stick to regime limit
        if score >= 80:
            effective_max = absolute_max
        elif score >= 75:
            effective_max = min(regime_max + 2, absolute_max)
        elif score >= 70:
            effective_max = min(regime_max + 1, absolute_max)
        else:
            effective_max = regime_max
        
        if current_count >= effective_max:
            logger.info(
                "max_positions_reached",
                current=current_count,
                max=effective_max,
                regime_max=regime_max,
                absolute_max=absolute_max,
                score=score,
                regime=self._regime_params.get("regime", "default") if self._regime_params else "default",
            )
            return False
        
        # Log when we're exceeding regime limit due to high score
        if current_count >= regime_max and current_count < effective_max:
            logger.info(
                "exceeding_regime_limit_high_score",
                symbol=symbol,
                score=score,
                current=current_count,
                regime_max=regime_max,
                effective_max=effective_max,
            )

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
            return (
                False,
                f"Position size {position_pct:.1%} exceeds max {self.config.max_position_size:.1%}",
            )

        # Check buying power
        buying_power = self.get_buying_power()
        if position_value > buying_power:
            return (
                False,
                f"Insufficient buying power: ${buying_power:.2f} < ${position_value:.2f}",
            )

        return True, "Trade validated"


risk_manager = RiskManager()
