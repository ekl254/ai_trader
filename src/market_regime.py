"""Market regime detection for adaptive trading."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional
from enum import Enum

import pandas as pd

from src.logger import logger


class MarketRegime(Enum):
    """Market regime classifications."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


class MarketRegimeDetector:
    """Detects current market regime using SPY and VIX indicators."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}
        self._cache_ttl = 900  # 15 minutes

    def get_market_regime(self) -> Dict:
        """
        Analyze current market regime using SPY trend and VIX levels.

        Returns:
            Dict with regime, confidence, and trading adjustment recommendations
        """
        cache_key = "market_regime"
        now = datetime.now(timezone.utc)

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

            # Calculate position adjustment factor
            adjustment = self._calculate_adjustment(regime, volatility)

            result = {
                "regime": regime.value,
                "spy_price": round(latest_close, 2),
                "spy_ema20": round(latest_ema20, 2),
                "spy_ema50": round(latest_ema50, 2),
                "momentum_20d": round(momentum_20d, 2),
                "volatility": round(volatility, 2),
                "position_size_multiplier": adjustment["position_multiplier"],
                "should_trade": adjustment["should_trade"],
                "max_positions_override": adjustment["max_positions"],
                "recommendation": adjustment["recommendation"],
                "timestamp": now.isoformat(),
            }

            # Cache result
            self._cache[cache_key] = (now, result)

            logger.info(
                "market_regime_detected",
                regime=regime.value,
                momentum=momentum_20d,
                volatility=volatility,
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

    def _calculate_adjustment(self, regime: MarketRegime, volatility: float) -> Dict:
        """Calculate trading adjustments based on regime."""

        adjustments = {
            MarketRegime.STRONG_BULL: {
                "position_multiplier": 1.0,
                "should_trade": True,
                "max_positions": None,  # Use default
                "recommendation": "Full trading - strong uptrend confirmed",
            },
            MarketRegime.BULL: {
                "position_multiplier": 1.0,
                "should_trade": True,
                "max_positions": None,
                "recommendation": "Normal trading - bullish trend",
            },
            MarketRegime.NEUTRAL: {
                "position_multiplier": 0.75,
                "should_trade": True,
                "max_positions": 7,
                "recommendation": "Reduced exposure - mixed signals",
            },
            MarketRegime.BEAR: {
                "position_multiplier": 0.5,
                "should_trade": True,
                "max_positions": 5,
                "recommendation": "Defensive mode - bearish trend, reduce size",
            },
            MarketRegime.STRONG_BEAR: {
                "position_multiplier": 0.25,
                "should_trade": False,
                "max_positions": 3,
                "recommendation": "Capital preservation - avoid new longs",
            },
            MarketRegime.HIGH_VOLATILITY: {
                "position_multiplier": 0.5,
                "should_trade": True,
                "max_positions": 5,
                "recommendation": "High volatility - reduce position sizes",
            },
        }

        return adjustments.get(regime, adjustments[MarketRegime.NEUTRAL])

    def _default_regime(self, reason: str) -> Dict:
        """Return default neutral regime on error."""
        return {
            "regime": MarketRegime.NEUTRAL.value,
            "error": reason,
            "position_size_multiplier": 0.75,
            "should_trade": True,
            "max_positions_override": 7,
            "recommendation": f"Using default settings due to: {reason}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def should_enter_trade(self) -> Tuple[bool, str]:
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
