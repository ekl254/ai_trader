"""Risk management and position sizing with regime-adaptive parameters."""

from typing import Dict, Optional, Tuple, Set

from alpaca.trading.client import TradingClient

from config.config import config
from src.logger import logger


class RiskManager:
    """Manages risk and position sizing with regime-adaptive parameters."""
    
    # Safety thresholds
    MIN_CASH_PCT = 0.05  # Minimum 5% cash reserve
    MARGIN_WARNING_THRESHOLD = 0.10  # Warn if cash < 10% of portfolio (early warning)
    POSITION_OVERLOAD_THRESHOLD = 1.2  # Warn if positions > 120% of max

    def __init__(self) -> None:
        self.config = config.trading
        self.client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper=True,
        )
        self._regime_params: Optional[Dict] = None
        # Track pending buys to prevent race conditions with Alpaca API
        self._pending_buys: Set[str] = set()
        self._margin_warning_logged = False
        self._overload_warning_logged = False

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

    def mark_pending_buy(self, symbol: str) -> None:
        """Mark a symbol as having a pending buy order.
        
        This prevents race conditions where we might try to buy the same
        stock twice before the first order shows up in Alpaca positions.
        """
        self._pending_buys.add(symbol.upper())
        logger.info("marked_pending_buy", symbol=symbol)

    def clear_pending_buy(self, symbol: str) -> None:
        """Clear a pending buy marker (called after order fills or fails)."""
        self._pending_buys.discard(symbol.upper())
        logger.info("cleared_pending_buy", symbol=symbol)

    def clear_all_pending_buys(self) -> None:
        """Clear all pending buy markers (called on session start/sync)."""
        self._pending_buys.clear()
        logger.info("cleared_all_pending_buys")

    def is_pending_buy(self, symbol: str) -> bool:
        """Check if a symbol has a pending buy order."""
        return symbol.upper() in self._pending_buys

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

    def get_cash(self) -> float:
        """Get available cash (not margin/buying power)."""
        try:
            account = self.client.get_account()
            return float(account.cash)
        except Exception as e:
            logger.error("failed_to_get_cash", error=str(e))
            return 0.0

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions as {symbol: quantity}."""
        try:
            positions = self.client.get_all_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error("failed_to_get_positions", error=str(e))
            return {}

    def check_account_health(self) -> Dict:
        """
        Check account health and return status with any warnings.
        
        Returns dict with:
            - healthy: bool
            - cash_pct: float
            - position_count: int
            - max_positions: int
            - warnings: List[str]
            - actions_needed: List[str]
        """
        try:
            account = self.client.get_account()
            positions = self.client.get_all_positions()
            
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            cash_pct = cash / portfolio_value if portfolio_value > 0 else 0
            position_count = len(positions)
            max_positions = self.get_max_positions()
            
            warnings = []
            actions_needed = []
            healthy = True
            
            # Check for negative cash (margin usage)
            if cash < 0:
                healthy = False
                warnings.append(f"CRITICAL: Negative cash balance (${cash:,.2f}) - using margin!")
                actions_needed.append("Sell positions to restore positive cash balance")
                if not self._margin_warning_logged:
                    logger.error(
                        "margin_warning_negative_cash",
                        cash=cash,
                        portfolio_value=portfolio_value,
                        position_count=position_count,
                    )
                    self._margin_warning_logged = True
            elif cash_pct < self.MARGIN_WARNING_THRESHOLD:
                warnings.append(f"Low cash reserve ({cash_pct:.1%}) - risk of margin call")
                if not self._margin_warning_logged:
                    logger.warning(
                        "low_cash_warning",
                        cash=cash,
                        cash_pct=cash_pct,
                        threshold=self.MARGIN_WARNING_THRESHOLD,
                    )
                    self._margin_warning_logged = True
            else:
                self._margin_warning_logged = False
            
            # Check for position overload
            if position_count > max_positions:
                overload_pct = position_count / max_positions
                if overload_pct >= self.POSITION_OVERLOAD_THRESHOLD:
                    healthy = False
                    warnings.append(f"CRITICAL: {position_count} positions vs {max_positions} max ({overload_pct:.0%})")
                    excess = position_count - max_positions
                    actions_needed.append(f"Sell {excess} position(s) to get back under limit")
                else:
                    warnings.append(f"Position overload: {position_count}/{max_positions}")
                
                if not self._overload_warning_logged:
                    logger.warning(
                        "position_overload_warning",
                        current=position_count,
                        max=max_positions,
                        excess=position_count - max_positions,
                    )
                    self._overload_warning_logged = True
            else:
                self._overload_warning_logged = False
            
            return {
                "healthy": healthy,
                "cash": cash,
                "cash_pct": cash_pct,
                "portfolio_value": portfolio_value,
                "position_count": position_count,
                "max_positions": max_positions,
                "warnings": warnings,
                "actions_needed": actions_needed,
            }
            
        except Exception as e:
            logger.error("account_health_check_failed", error=str(e))
            return {
                "healthy": False,
                "warnings": [f"Failed to check account health: {str(e)}"],
                "actions_needed": [],
            }

    def get_positions_to_reduce(self, target_reduction: int = 0) -> list:
        """
        Identify positions that should be sold to reduce position count.
        
        If target_reduction is 0, calculates based on being over max.
        Returns list of symbols sorted by priority (weakest first).
        
        Args:
            target_reduction: Number of positions to reduce (0 = auto-calculate)
            
        Returns:
            List of dicts with symbol, score, reason for each position to sell
        """
        try:
            positions = self.client.get_all_positions()
            max_positions = self.get_max_positions()
            current_count = len(positions)
            
            if target_reduction <= 0:
                target_reduction = max(0, current_count - max_positions)
            
            if target_reduction <= 0:
                return []
            
            # Get position tracker for scores
            try:
                from src.position_tracker import position_tracker
                tracked_data = position_tracker.data.get("positions", {})
            except Exception:
                tracked_data = {}
            
            # Build list of positions with their data
            position_list = []
            for pos in positions:
                symbol = pos.symbol
                tracked = tracked_data.get(symbol, {})
                score = tracked.get("score", 50)  # Default to neutral score
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price or entry_price)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                position_list.append({
                    "symbol": symbol,
                    "score": score,
                    "pnl_pct": pnl_pct,
                    "market_value": float(pos.market_value),
                    "reason": tracked.get("reason", "unknown"),
                })
            
            # Sort by score (lowest first) - these are candidates for selling
            position_list.sort(key=lambda x: x["score"])
            
            # Return the weakest positions up to target_reduction
            to_reduce = position_list[:target_reduction]
            
            logger.info(
                "positions_identified_for_reduction",
                target_reduction=target_reduction,
                candidates=[p["symbol"] for p in to_reduce],
            )
            
            return to_reduce
            
        except Exception as e:
            logger.error("failed_to_identify_positions_to_reduce", error=str(e))
            return []

    def should_block_new_buys(self) -> Tuple[bool, str]:
        """
        Check if new buys should be blocked due to account health issues.
        
        Returns:
            (should_block, reason)
        """
        health = self.check_account_health()
        
        # Block if not healthy (negative cash or severe overload)
        if not health.get("healthy", True):
            reasons = health.get("warnings", ["Unknown issue"])
            return True, f"Account unhealthy: {'; '.join(reasons)}"
        
        # Block if over position limit (even if not critical)
        position_count = health.get("position_count", 0)
        max_positions = health.get("max_positions", 10)
        if position_count >= max_positions:
            return True, f"At position limit ({position_count}/{max_positions})"
        
        return False, "OK"

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

        # Check CASH available (maintain MIN_CASH_PCT buffer)
        cash = self.get_cash()
        min_cash_buffer = account_value * self.MIN_CASH_PCT
        available_cash = max(0, cash - min_cash_buffer)
        
        if position_value > available_cash:
            shares = int(available_cash / entry_price)
            position_value = shares * entry_price
            logger.warning(
                "insufficient_cash_for_position",
                symbol=symbol,
                required=position_value,
                available_cash=available_cash,
                cash=cash,
                min_buffer=min_cash_buffer,
                adjusted_shares=shares,
            )

        # Also check buying power as absolute limit
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

    def can_open_position(self, symbol: str, score: float = 0, skip_health_check: bool = False) -> bool:
        """
        Check if we can open a new position.
        
        Args:
            symbol: Stock symbol
            score: Composite score (optional) - high scores can exceed regime limits
            skip_health_check: If True, skip account health check (used by swap_position)
            
        Returns:
            True if position can be opened
        """
        symbol = symbol.upper()
        
        # === SAFETY CHECK: Account health ===
        if not skip_health_check:
            should_block, block_reason = self.should_block_new_buys()
            if should_block:
                logger.info(
                    "new_buy_blocked_account_health",
                    symbol=symbol,
                    reason=block_reason,
                )
                return False
        
        current_positions = self.get_current_positions()

        # Check if already holding
        if symbol in current_positions:
            logger.info("already_holding_position", symbol=symbol)
            return False

        # Check if there's a pending buy for this symbol (prevents double buys)
        if self.is_pending_buy(symbol):
            logger.info("pending_buy_exists", symbol=symbol)
            return False

        # Also check position tracker as another source of truth
        try:
            from src.position_tracker import position_tracker
            if symbol in position_tracker.data.get("positions", {}):
                logger.info("position_tracked_locally", symbol=symbol)
                return False
        except Exception:
            pass  # If tracker not available, continue with Alpaca check

        # Get regime-adjusted max and absolute max from config
        regime_max = self.get_max_positions()
        absolute_max = self.config.max_positions  # Hard cap from config (usually 10)
        current_count = len(current_positions) + len(self._pending_buys)
        
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
                pending_buys=len(self._pending_buys),
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

        # Check CASH (not buying power) to prevent margin usage
        # We want to maintain at least MIN_CASH_PCT buffer after the trade
        cash = self.get_cash()
        min_cash_buffer = account_value * self.MIN_CASH_PCT
        available_for_trade = cash - min_cash_buffer
        
        if position_value > available_for_trade:
            return (
                False,
                f"Insufficient cash: ${cash:.2f} (need ${position_value:.2f} + ${min_cash_buffer:.0f} buffer)",
            )
        
        # Also check against buying power as a safety net
        buying_power = self.get_buying_power()
        if position_value > buying_power:
            return (
                False,
                f"Insufficient buying power: ${buying_power:.2f} < ${position_value:.2f}",
            )

        return True, "Trade validated"


risk_manager = RiskManager()
