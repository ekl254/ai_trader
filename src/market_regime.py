"""Market regime detection for adaptive trading with dynamic parameters."""

from datetime import UTC, datetime
from enum import Enum

from src.logger import logger


class MarketRegime(Enum):
    """Market regime classifications."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


# Regime-specific trading parameters (optimized via backtesting study)
REGIME_PARAMETERS = {
    MarketRegime.STRONG_BULL: {
        "stop_loss_pct": 0.05,  # 5% - wider stops in strong trends
        "take_profit_pct": 0.12,  # 12% - let winners run
        "min_score": 60,  # Lower threshold - ride the trend
        "position_multiplier": 1.0,
        "max_positions": 10,
        "should_trade": True,
        "description": "Strong uptrend - wider stops, aggressive targets",
    },
    MarketRegime.BULL: {
        "stop_loss_pct": 0.04,  # 4% - moderately wide
        "take_profit_pct": 0.10,  # 10% - good targets
        "min_score": 65,  # Moderate threshold
        "position_multiplier": 1.0,
        "max_positions": 10,
        "should_trade": True,
        "description": "Bullish trend - normal trading with trend",
    },
    MarketRegime.NEUTRAL: {
        # Allow full positions in neutral - we have cash to deploy
        "stop_loss_pct": 0.03,  # 3% - tighter for choppy markets
        "take_profit_pct": 0.06,  # 6% - take profits quicker
        "min_score": 70,  # Higher threshold - be selective
        "position_multiplier": 0.75,
        "max_positions": 7,
        "should_trade": True,
        "description": "Sideways market - be selective, tighter stops",
    },
    MarketRegime.BEAR: {
        "stop_loss_pct": 0.02,  # 2% - tight stops
        "take_profit_pct": 0.05,  # 5% - quick profits
        "min_score": 75,  # High threshold - only best setups
        "position_multiplier": 0.5,
        "max_positions": 5,
        "should_trade": True,
        "description": "Bearish trend - defensive, tight stops",
    },
    MarketRegime.STRONG_BEAR: {
        "stop_loss_pct": 0.02,  # 2% - very tight
        "take_profit_pct": 0.04,  # 4% - take any profit
        "min_score": 80,  # Very high threshold
        "position_multiplier": 0.25,
        "max_positions": 3,
        "should_trade": False,  # Avoid new longs
        "description": "Strong downtrend - capital preservation mode",
    },
    MarketRegime.HIGH_VOLATILITY: {
        "stop_loss_pct": 0.04,  # 4% - wider for vol spikes
        "take_profit_pct": 0.08,  # 8% - moderate targets
        "min_score": 72,  # Elevated threshold
        "position_multiplier": 0.5,
        "max_positions": 5,
        "should_trade": True,
        "description": "High volatility - reduced size, wider stops",
    },
}


class MarketRegimeDetector:
    """Detects current market regime using SPY and VIX indicators."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[datetime, dict]] = {}
        self._cache_ttl = 900  # 15 minutes

    def get_market_regime(self) -> dict:
        """
        Analyze current market regime using SPY trend and VIX levels.

        Returns:
            Dict with regime, trading parameters, and recommendations
        """
        cache_key = "market_regime"
        now = datetime.now(UTC)

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (now - cached_time).total_seconds() < self._cache_ttl:
                return cached_data

        try:
            from src.data_provider import alpaca_provider

            # Get SPY data for trend analysis
            spy_df = alpaca_provider.get_bars("SPY", days=60)

            if len(spy_df) < 50:
                return self._default_regime("insufficient_data")

            # Calculate SPY indicators
            close = spy_df["close"]

            # EMAs for trend
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()

            latest_close = close.iloc[-1]
            latest_ema20 = ema_20.iloc[-1]
            latest_ema50 = ema_50.iloc[-1]

            # Calculate momentum (20-day return)
            momentum_20d = (latest_close - close.iloc[-20]) / close.iloc[-20] * 100

            # Calculate volatility (20-day standard deviation annualized)
            returns = close.pct_change().dropna()
            volatility = returns.tail(20).std() * (252**0.5) * 100

            # Determine regime
            regime = self._classify_regime(
                latest_close, latest_ema20, latest_ema50, momentum_20d, volatility
            )

            # Get regime-specific parameters
            params = REGIME_PARAMETERS[regime]

            result = {
                "regime": regime.value,
                "regime_enum": regime,
                # Market data
                "spy_price": round(latest_close, 2),
                "spy_ema20": round(latest_ema20, 2),
                "spy_ema50": round(latest_ema50, 2),
                "momentum_20d": round(momentum_20d, 2),
                "volatility": round(volatility, 2),
                # Trading parameters (regime-adaptive)
                "stop_loss_pct": params["stop_loss_pct"],
                "take_profit_pct": params["take_profit_pct"],
                "min_score": params["min_score"],
                "position_size_multiplier": params["position_multiplier"],
                "max_positions_override": params["max_positions"],
                "should_trade": params["should_trade"],
                "recommendation": params["description"],
                "timestamp": now.isoformat(),
            }

            # Cache result
            self._cache[cache_key] = (now, result)

            logger.info(
                "market_regime_detected",
                regime=regime.value,
                momentum=momentum_20d,
                volatility=volatility,
                stop_loss=params["stop_loss_pct"],
                take_profit=params["take_profit_pct"],
            )

            return result

        except Exception as e:
            logger.error("market_regime_detection_failed", error=str(e))
            return self._default_regime(str(e))

    def _classify_regime(
        self,
        price: float,
        ema20: float,
        ema50: float,
        momentum: float,
        volatility: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators."""

        # High volatility override (VIX proxy via realized vol)
        if volatility > 30:
            return MarketRegime.HIGH_VOLATILITY

        # Trend classification
        above_ema20 = price > ema20
        above_ema50 = price > ema50
        ema20_above_ema50 = ema20 > ema50

        if above_ema20 and above_ema50 and ema20_above_ema50:
            if momentum > 5:
                return MarketRegime.STRONG_BULL
            return MarketRegime.BULL

        elif not above_ema20 and not above_ema50 and not ema20_above_ema50:
            if momentum < -5:
                return MarketRegime.STRONG_BEAR
            return MarketRegime.BEAR

        return MarketRegime.NEUTRAL

    def get_trading_parameters(self) -> dict:
        """
        Get current regime-adjusted trading parameters.

        Returns:
            Dict with stop_loss_pct, take_profit_pct, min_score
        """
        regime_data = self.get_market_regime()
        return {
            "regime": regime_data["regime"],
            "stop_loss_pct": regime_data["stop_loss_pct"],
            "take_profit_pct": regime_data["take_profit_pct"],
            "min_score": regime_data["min_score"],
            "position_multiplier": regime_data["position_size_multiplier"],
            "max_positions": regime_data["max_positions_override"],
            "recommendation": regime_data["recommendation"],
        }

    def _default_regime(self, reason: str) -> dict:
        """Return default neutral regime on error."""
        params = REGIME_PARAMETERS[MarketRegime.NEUTRAL]
        return {
            "regime": MarketRegime.NEUTRAL.value,
            "error": reason,
            "stop_loss_pct": params["stop_loss_pct"],
            "take_profit_pct": params["take_profit_pct"],
            "min_score": params["min_score"],
            "position_size_multiplier": params["position_multiplier"],
            "max_positions_override": params["max_positions"],
            "should_trade": True,
            "recommendation": f"Using default neutral settings due to: {reason}",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def should_enter_trade(self) -> tuple[bool, str]:
        """
        Quick check if market conditions allow new trades.

        Returns:
            (should_trade, reason)
        """
        regime = self.get_market_regime()

        if not regime.get("should_trade", True):
            return (
                False,
                f"Market regime ({regime['regime']}) suggests avoiding new positions",
            )

        return True, regime.get("recommendation", "Conditions acceptable")


# Global instance
market_regime_detector = MarketRegimeDetector()
