"""
Dynamic Position Sizing Module

Implements conviction-weighted position sizing with:
- Score-based conviction multipliers
- Volatility adjustments
- Dynamic cash deployment
- Risk management caps
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.clients import get_data_client, get_trading_client
from src.logger import logger


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    symbol: str
    recommended_size: float  # Dollar amount
    recommended_shares: int
    conviction_score: float  # 0-100
    conviction_multiplier: float  # 0.8-1.5
    volatility_multiplier: float  # 0.7-1.3
    base_size: float
    rationale: list[str]
    capped: bool  # Was size capped by min/max?
    cap_reason: str | None


class DynamicPositionSizer:
    """
    Calculates optimal position sizes based on:
    1. Conviction (composite score)
    2. Volatility (ATR-based)
    3. Available capital
    4. Risk management rules
    """

    # Position size limits as percentage of portfolio
    MIN_POSITION_PCT = 0.05  # 5% minimum
    MAX_POSITION_PCT = 0.10  # 10% maximum (aligned with config.max_position_size)

    # Target cash reserve - unified with risk_manager.MIN_CASH_PCT
    MIN_CASH_RESERVE_PCT = 0.05  # Keep at least 5% cash (matches risk_manager)
    CASH_DEPLOYMENT_TRIGGER_PCT = 0.15  # Deploy when cash > 15%

    # Conviction multipliers based on composite score
    CONVICTION_TIERS = [
        (90, 1.5),  # Score 90+: 1.5x
        (85, 1.4),  # Score 85-90: 1.4x
        (80, 1.2),  # Score 80-85: 1.2x
        (75, 1.0),  # Score 75-80: 1.0x (baseline)
        (70, 0.85),  # Score 70-75: 0.85x
        (0, 0.7),  # Below 70: 0.7x (minimum)
    ]

    # Volatility adjustment ranges
    VOL_LOW_THRESHOLD = 0.015  # Daily vol < 1.5% = low
    VOL_HIGH_THRESHOLD = 0.035  # Daily vol > 3.5% = high
    VOL_LOW_MULTIPLIER = 1.2  # Size up for low vol
    VOL_HIGH_MULTIPLIER = 0.75  # Size down for high vol

    # Sector concentration limit
    MAX_SECTOR_PCT = 0.30  # Max 30% in any sector

    # Daily limits
    MAX_NEW_POSITIONS_PER_DAY = 4

    def __init__(self):
        self.trading_client = get_trading_client()
        self.data_client = get_data_client()
        self._daily_entries: dict[str, int] = {}  # date -> count
        self._volatility_cache: dict[str, tuple[float, datetime]] = {}

    def get_conviction_multiplier(self, score: float) -> tuple[float, str]:
        """Map composite score to conviction multiplier."""
        for threshold, multiplier in self.CONVICTION_TIERS:
            if score >= threshold:
                if score >= 90:
                    reason = f"Exceptional score ({score:.1f}) -> 1.5x size"
                elif score >= 85:
                    reason = f"Very high conviction ({score:.1f}) -> 1.4x size"
                elif score >= 80:
                    reason = f"High conviction ({score:.1f}) -> 1.2x size"
                elif score >= 75:
                    reason = f"Standard conviction ({score:.1f}) -> 1.0x size"
                elif score >= 70:
                    reason = f"Moderate conviction ({score:.1f}) -> 0.85x size"
                else:
                    reason = f"Low conviction ({score:.1f}) -> 0.7x size"
                return multiplier, reason
        return 0.7, f"Minimum conviction ({score:.1f})"

    def get_volatility(self, symbol: str) -> float | None:
        """Calculate 20-day volatility. Uses cache to avoid repeated API calls."""
        # Check cache (valid for 1 hour)
        if symbol in self._volatility_cache:
            cached_vol, cached_time = self._volatility_cache[symbol]
            if datetime.now() - cached_time < timedelta(hours=1):
                return cached_vol

        try:
            end = datetime.now()
            start = end - timedelta(days=30)

            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end
            )
            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars or len(bars[symbol]) < 10:
                return None

            prices = [float(bar.close) for bar in bars[symbol]]
            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]

            if not returns:
                return None

            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance**0.5

            # Cache result
            self._volatility_cache[symbol] = (volatility, datetime.now())

            return volatility

        except Exception as e:
            logger.warning("volatility_calculation_failed", symbol=symbol, error=str(e))
            return None

    def get_volatility_multiplier(self, symbol: str) -> tuple[float, str]:
        """Adjust position size based on volatility."""
        volatility = self.get_volatility(symbol)

        if volatility is None:
            return 1.0, "Volatility unknown, using 1.0x"

        vol_pct = volatility * 100

        if volatility < self.VOL_LOW_THRESHOLD:
            return (
                self.VOL_LOW_MULTIPLIER,
                f"Low volatility ({vol_pct:.1f}% daily) -> 1.2x size",
            )
        elif volatility > self.VOL_HIGH_THRESHOLD:
            return (
                self.VOL_HIGH_MULTIPLIER,
                f"High volatility ({vol_pct:.1f}% daily) -> 0.75x size",
            )
        else:
            # Linear interpolation between thresholds
            vol_range = self.VOL_HIGH_THRESHOLD - self.VOL_LOW_THRESHOLD
            vol_position = (volatility - self.VOL_LOW_THRESHOLD) / vol_range
            multiplier = self.VOL_LOW_MULTIPLIER - vol_position * (
                self.VOL_LOW_MULTIPLIER - self.VOL_HIGH_MULTIPLIER
            )
            return (
                multiplier,
                f"Medium volatility ({vol_pct:.1f}% daily) -> {multiplier:.2f}x size",
            )

    def get_portfolio_info(self) -> dict:
        """Get current portfolio value and cash."""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()

            return {
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "position_count": len(positions),
                "invested_value": sum(float(p.market_value) for p in positions),
            }
        except Exception as e:
            logger.error("portfolio_info_failed", error=str(e))
            return {
                "portfolio_value": 100000,
                "cash": 30000,
                "buying_power": 30000,
                "position_count": 7,
                "invested_value": 70000,
            }

    def should_deploy_cash(self, portfolio_info: dict) -> tuple[bool, str]:
        """Determine if we should deploy idle cash beyond normal regime limits."""
        cash_pct = portfolio_info["cash"] / portfolio_info["portfolio_value"]

        if cash_pct > self.CASH_DEPLOYMENT_TRIGGER_PCT:
            excess_cash = portfolio_info["cash"] - (
                portfolio_info["portfolio_value"] * self.MIN_CASH_RESERVE_PCT
            )
            return (
                True,
                f"Cash at {cash_pct*100:.1f}% (>${excess_cash:,.0f} deployable)",
            )

        return False, f"Cash at {cash_pct*100:.1f}% - within normal range"

    def get_max_positions_with_cash_deployment(
        self, base_max: int, portfolio_info: dict, qualified_count: int
    ) -> tuple[int, str]:
        """Calculate adjusted max positions considering cash deployment."""
        should_deploy, _deploy_reason = self.should_deploy_cash(portfolio_info)

        if not should_deploy:
            return base_max, f"Using regime max: {base_max}"

        # Calculate how many extra positions we can add
        cash_pct = portfolio_info["cash"] / portfolio_info["portfolio_value"]
        excess_cash_pct = cash_pct - self.MIN_CASH_RESERVE_PCT

        # Each position is roughly 8-12% of portfolio, use 10% as estimate
        extra_positions = int(excess_cash_pct / 0.10)
        extra_positions = min(extra_positions, 3)  # Cap at 3 extra
        extra_positions = min(
            extra_positions, qualified_count
        )  # Don't exceed qualified candidates

        new_max = base_max + extra_positions
        absolute_max = 12  # Never exceed 12 positions total
        new_max = min(new_max, absolute_max)

        if extra_positions > 0:
            reason = f"Cash deployment: {base_max} + {extra_positions} extra = {new_max} max positions"
        else:
            reason = f"Using regime max: {base_max}"

        return new_max, reason

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        composite_score: float,
        portfolio_info: dict | None = None,
        target_positions: int = 10,
    ) -> PositionSizeResult:
        """Calculate optimal position size for a stock."""
        if portfolio_info is None:
            portfolio_info = self.get_portfolio_info()

        portfolio_value = portfolio_info["portfolio_value"]
        rationale = []

        # 1. Calculate base size
        base_size = portfolio_value / target_positions
        rationale.append(
            f"Base size: ${base_size:,.0f} (portfolio/${target_positions} positions)"
        )

        # 2. Apply conviction multiplier
        conviction_mult, conviction_reason = self.get_conviction_multiplier(
            composite_score
        )
        rationale.append(conviction_reason)

        # 3. Apply volatility multiplier
        vol_mult, vol_reason = self.get_volatility_multiplier(symbol)
        rationale.append(vol_reason)

        # 4. Calculate raw size
        raw_size = base_size * conviction_mult * vol_mult
        rationale.append(
            f"Raw size: ${base_size:,.0f} x {conviction_mult:.2f} x {vol_mult:.2f} = ${raw_size:,.0f}"
        )

        # 5. Apply caps
        min_size = portfolio_value * self.MIN_POSITION_PCT
        max_size = portfolio_value * self.MAX_POSITION_PCT

        capped = False
        cap_reason = None

        if raw_size < min_size:
            final_size = min_size
            capped = True
            cap_reason = f"Increased to minimum ({self.MIN_POSITION_PCT*100:.0f}%)"
            rationale.append(f"Capped UP to min: ${final_size:,.0f}")
        elif raw_size > max_size:
            final_size = max_size
            capped = True
            cap_reason = f"Reduced to maximum ({self.MAX_POSITION_PCT*100:.0f}%)"
            rationale.append(f"Capped DOWN to max: ${final_size:,.0f}")
        else:
            final_size = raw_size
            rationale.append(f"Final size: ${final_size:,.0f} (within limits)")

        # 6. Ensure we don't exceed available cash (with minimum buffer)
        available_cash = portfolio_info["cash"]
        min_cash_buffer = (
            portfolio_info["portfolio_value"] * 0.05
        )  # Keep 5% cash buffer
        usable_cash = max(0, available_cash - min_cash_buffer)

        if usable_cash <= 0:
            logger.warning(
                "no_cash_available_for_position",
                symbol=symbol,
                cash=available_cash,
                min_buffer=min_cash_buffer,
            )
            return PositionSizeResult(
                symbol=symbol,
                recommended_size=0,
                recommended_shares=0,
                conviction_score=composite_score,
                conviction_multiplier=conviction_mult,
                volatility_multiplier=vol_mult,
                base_size=base_size,
                capped=True,
                cap_reason="No cash available (negative or below buffer)",
                rationale=[*rationale, "BLOCKED: No usable cash available"],
            )

        if final_size > usable_cash:
            final_size = usable_cash * 0.95  # Leave small buffer
            capped = True
            cap_reason = f"Limited by available cash (${available_cash:,.0f} - ${min_cash_buffer:,.0f} buffer = ${usable_cash:,.0f})"
            rationale.append(f"Reduced to available cash: ${final_size:,.0f}")

        # 7. Calculate shares
        shares = int(final_size / current_price)
        actual_size = shares * current_price

        if shares < 1:
            shares = 1
            actual_size = current_price
            rationale.append(f"Minimum 1 share: ${actual_size:,.0f}")

        return PositionSizeResult(
            symbol=symbol,
            recommended_size=actual_size,
            recommended_shares=shares,
            conviction_score=composite_score,
            conviction_multiplier=conviction_mult,
            volatility_multiplier=vol_mult,
            base_size=base_size,
            rationale=rationale,
            capped=capped,
            cap_reason=cap_reason,
        )

    def calculate_batch_sizes(
        self,
        candidates: list[dict],
        portfolio_info: dict | None = None,
        max_positions: int = 10,
    ) -> list[PositionSizeResult]:
        """Calculate position sizes for multiple candidates."""
        if portfolio_info is None:
            portfolio_info = self.get_portfolio_info()

        results = []
        remaining_cash = portfolio_info["cash"] - (
            portfolio_info["portfolio_value"] * self.MIN_CASH_RESERVE_PCT
        )
        remaining_cash = max(0, remaining_cash)

        current_positions = portfolio_info["position_count"]
        slots_available = max_positions - current_positions

        # Check daily entry limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_entries = self._daily_entries.get(today, 0)
        daily_remaining = self.MAX_NEW_POSITIONS_PER_DAY - daily_entries

        slots_available = min(slots_available, daily_remaining)

        logger.info(
            "batch_sizing_started",
            candidates=len(candidates),
            remaining_cash=remaining_cash,
            slots_available=slots_available,
            daily_entries_today=daily_entries,
        )

        # Sort by score descending - prioritize best opportunities
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("score", 0), reverse=True
        )

        for candidate in sorted_candidates:
            if len(results) >= slots_available:
                break

            if (
                remaining_cash
                < portfolio_info["portfolio_value"] * self.MIN_POSITION_PCT
            ):
                logger.info(
                    "batch_sizing_stopped", reason="insufficient_remaining_cash"
                )
                break

            symbol = candidate["symbol"]
            price = candidate["price"]
            score = candidate["score"]

            # Calculate size
            size_result = self.calculate_position_size(
                symbol=symbol,
                current_price=price,
                composite_score=score,
                portfolio_info=portfolio_info,
                target_positions=max_positions,
            )

            # Check if we can afford it
            if size_result.recommended_size <= remaining_cash:
                results.append(size_result)
                remaining_cash -= size_result.recommended_size
                logger.info(
                    "position_sized",
                    symbol=symbol,
                    size=size_result.recommended_size,
                    shares=size_result.recommended_shares,
                    conviction=size_result.conviction_multiplier,
                    remaining_cash=remaining_cash,
                )
            else:
                logger.info(
                    "position_skipped_insufficient_cash",
                    symbol=symbol,
                    needed=size_result.recommended_size,
                    available=remaining_cash,
                )

        return results

    def record_entry(self, symbol: str):
        """Record a position entry for daily limit tracking."""
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_entries[today] = self._daily_entries.get(today, 0) + 1
        logger.info(
            "entry_recorded", symbol=symbol, daily_count=self._daily_entries[today]
        )


# Global instance
position_sizer = DynamicPositionSizer()
