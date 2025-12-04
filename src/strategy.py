"""Trading strategy with multi-factor scoring."""

from typing import cast

import pandas as pd
import pandas_ta as ta  # type: ignore

from config.config import config
from src.data_provider import alpaca_provider
from src.logger import logger
from src.sentiment import get_news_sentiment


class TradingStrategy:
    """Multi-factor trading strategy."""

    def __init__(self) -> None:
        self.config = config.trading

    def calculate_technical_score(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Calculate technical analysis score (0-100)."""
        if len(df) < 30:
            return 0.0, {"error": "insufficient_data"}

        # Calculate indicators - cast to Series for proper typing
        close_series = cast(pd.Series, df["close"])
        df["RSI"] = ta.rsi(close_series, length=self.config.rsi_period)
        macd = ta.macd(close_series)
        if macd is not None:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
        bbands = ta.bbands(close_series)
        if bbands is not None:
            # Dynamically find BB column names (pandas-ta naming can vary)
            bb_cols = bbands.columns.tolist()
            bb_upper_col = (
                [c for c in bb_cols if c.startswith("BBU_")][0]
                if any(c.startswith("BBU_") for c in bb_cols)
                else None
            )
            bb_lower_col = (
                [c for c in bb_cols if c.startswith("BBL_")][0]
                if any(c.startswith("BBL_") for c in bb_cols)
                else None
            )
            bb_mid_col = (
                [c for c in bb_cols if c.startswith("BBM_")][0]
                if any(c.startswith("BBM_") for c in bb_cols)
                else None
            )

            if bb_upper_col:
                df["BB_upper"] = bbands[bb_upper_col]
            if bb_lower_col:
                df["BB_lower"] = bbands[bb_lower_col]
            if bb_mid_col:
                df["BB_mid"] = bbands[bb_mid_col]

        latest = df.iloc[-1]

        # RSI Score (0-100)
        rsi_value = latest.get("RSI")
        if pd.isna(rsi_value):
            rsi_score = 50.0
        elif rsi_value < self.config.rsi_oversold:
            rsi_score = 100.0  # Oversold = buy opportunity
        elif rsi_value > self.config.rsi_overbought:
            rsi_score = 0.0  # Overbought = sell signal
        else:
            # Scale 30-70 range to 50-100 (prefer middle range)
            rsi_score = 100 - abs(rsi_value - 50) * 2

        # MACD Score - gradient based on histogram strength
        macd_value = latest.get("MACD")
        macd_signal = latest.get("MACD_signal")
        if pd.isna(macd_value) or pd.isna(macd_signal):
            macd_score = 50.0
        else:
            macd_histogram = macd_value - macd_signal
            # Scale histogram to score: positive = bullish, negative = bearish
            # Typical histogram range is roughly -2 to +2 for most stocks
            if macd_histogram > 0:
                # Bullish: score 50-100 based on strength
                macd_score = min(100, 50 + (macd_histogram * 25))
            else:
                # Bearish: score 0-50 based on strength
                macd_score = max(0, 50 + (macd_histogram * 25))

        # Bollinger Bands Score
        close = latest["close"]
        bb_upper = latest.get("BB_upper")
        bb_lower = latest.get("BB_lower")
        latest.get("BB_mid")

        if pd.isna(bb_upper) or pd.isna(bb_lower):
            bb_score = 50.0
        elif close < bb_lower:
            bb_score = 100.0  # Below lower band = oversold
        elif close > bb_upper:
            bb_score = 0.0  # Above upper band = overbought
        else:
            # Score based on position within bands
            bb_range = bb_upper - bb_lower
            position = (close - bb_lower) / bb_range if bb_range > 0 else 0.5
            bb_score = (1 - abs(position - 0.5) * 2) * 100

        # Volume Score - contextualized with price direction
        avg_volume = df["volume"].tail(20).mean()
        current_volume = latest["volume"]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Check if price is up or down today
        prev_close = df["close"].iloc[-2] if len(df) > 1 else latest["close"]
        price_change_pct = (
            (latest["close"] - prev_close) / prev_close if prev_close > 0 else 0
        )

        # High volume on up days is bullish, high volume on down days is bearish
        if price_change_pct > 0:
            # Up day with volume = bullish confirmation
            volume_score = min(100, 50 + (volume_ratio - 1) * 30)
        elif price_change_pct < -0.005:  # Down more than 0.5%
            # Down day with high volume = bearish (selling pressure)
            volume_score = max(0, 50 - (volume_ratio - 1) * 30)
        else:
            # Flat day - neutral volume score
            volume_score = 50.0

        # Add trend confirmation via EMA
        df["EMA_20"] = ta.ema(close_series, length=20)
        df["EMA_50"] = ta.ema(close_series, length=50)
        ema_20 = latest.get("EMA_20")
        ema_50 = latest.get("EMA_50")

        # Trend score: above both EMAs = bullish, below both = bearish
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            if close > ema_20 > ema_50:
                trend_score = 80.0  # Strong uptrend
            elif close > ema_20 and close > ema_50:
                trend_score = 65.0  # Moderate uptrend
            elif close < ema_20 < ema_50:
                trend_score = 20.0  # Strong downtrend
            elif close < ema_20 and close < ema_50:
                trend_score = 35.0  # Moderate downtrend
            else:
                trend_score = 50.0  # Mixed/consolidation
        else:
            trend_score = 50.0

        # Weighted average - added trend component
        technical_score = (
            rsi_score * 0.25
            + macd_score * 0.25
            + bb_score * 0.20
            + volume_score * 0.15
            + trend_score * 0.15
        )

        details = {
            "rsi": float(rsi_value) if not pd.isna(rsi_value) else None,
            "rsi_score": rsi_score,
            "macd": float(macd_value) if not pd.isna(macd_value) else None,
            "macd_signal": float(macd_signal) if not pd.isna(macd_signal) else None,
            "macd_histogram": (
                float(macd_value - macd_signal)
                if not pd.isna(macd_value) and not pd.isna(macd_signal)
                else None
            ),
            "macd_score": macd_score,
            "bb_score": bb_score,
            "bb_upper": float(bb_upper) if not pd.isna(bb_upper) else None,
            "bb_lower": float(bb_lower) if not pd.isna(bb_lower) else None,
            "volume_ratio": round(volume_ratio, 2),
            "volume_score": volume_score,
            "price_change_pct": round(price_change_pct * 100, 2),
            "ema_20": float(ema_20) if not pd.isna(ema_20) else None,
            "ema_50": float(ema_50) if not pd.isna(ema_50) else None,
            "trend_score": trend_score,
            "total": technical_score,
        }

        return technical_score, details

    def calculate_fundamental_score(self, symbol: str) -> tuple[float, dict]:
        """Calculate fundamental analysis score (neutral - not using external fundamental data)."""
        # Simplified: no external fundamental data, always neutral
        return 50.0, {"total": 50.0, "note": "Using technical + sentiment only"}

    def calculate_sentiment_score(self, symbol: str) -> tuple[float, dict]:
        """Calculate news sentiment score (0-100)."""
        # Sentiment analysis (using NewsAPI)
        try:
            sentiment_score = get_news_sentiment(symbol, max_articles=20)
            logger.info(
                "sentiment_score_calculated", symbol=symbol, score=sentiment_score
            )

            details = {
                "total": sentiment_score,
            }

            return sentiment_score, details

        except Exception as e:
            logger.error("sentiment_analysis_failed", symbol=symbol, error=str(e))
            return 50.0, {"total": 50.0, "error": str(e)}

    def score_symbol(self, symbol: str) -> tuple[float, dict]:
        """Calculate composite score for a symbol."""
        logger.info("scoring_symbol", symbol=symbol)

        # Get price data
        df = alpaca_provider.get_bars(symbol, days=60)

        # Calculate all scores
        technical_score, technical_details = self.calculate_technical_score(df)
        fundamental_score, fundamental_details = self.calculate_fundamental_score(
            symbol
        )
        sentiment_score, sentiment_details = self.calculate_sentiment_score(symbol)

        # Weighted composite score using configurable weights
        composite_score = (
            technical_score * self.config.weight_technical
            + sentiment_score * self.config.weight_sentiment
            + fundamental_score * self.config.weight_fundamental
        )

        reasoning = {
            "technical": technical_details,
            "fundamental": fundamental_details,
            "sentiment": sentiment_details,
            "composite": composite_score,
        }

        logger.info(
            "symbol_scored",
            symbol=symbol,
            composite=composite_score,
            technical=technical_score,
            fundamental=fundamental_score,
            sentiment=sentiment_score,
        )

        return composite_score, reasoning

    def should_buy(self, symbol: str) -> tuple[bool, float, dict]:
        """Determine if symbol should be bought."""
        score, reasoning = self.score_symbol(symbol)

        # Check composite score (regime-adjusted)
        from src.risk_manager import risk_manager

        min_score = risk_manager.get_min_score()
        if score < min_score:
            return False, score, reasoning

        # Check technical (required)
        if reasoning["technical"]["total"] < self.config.min_factor_score:
            return False, score, reasoning

        # Sentiment required if fundamentals failed (API error)
        fundamental_failed = "error" in reasoning["fundamental"]
        if fundamental_failed:
            # If fundamentals unavailable, require strong technical + sentiment
            if reasoning["sentiment"]["total"] < self.config.min_factor_score:
                return False, score, reasoning
        else:
            # All factors must meet threshold when available
            if reasoning["fundamental"]["total"] < self.config.min_factor_score:
                return False, score, reasoning
            if reasoning["sentiment"]["total"] < self.config.min_factor_score:
                return False, score, reasoning

        return True, score, reasoning

    def rescore_positions(self, symbols: list[str]) -> dict[str, dict]:
        """
        Rescore existing positions to evaluate rebalancing opportunities.

        Args:
            symbols: List of symbols to rescore

        Returns:
            Dict mapping symbol to score and reasoning
        """
        position_scores: dict[str, dict] = {}

        for symbol in symbols:
            try:
                score, reasoning = self.score_symbol(symbol)
                position_scores[symbol] = {
                    "score": score,
                    "reasoning": reasoning,
                }
                logger.info("position_rescored", symbol=symbol, score=score)
            except Exception as e:
                logger.error("rescore_failed", symbol=symbol, error=str(e))
                # Assign low score if rescoring fails
                position_scores[symbol] = {
                    "score": 0.0,
                    "reasoning": {"error": str(e)},
                }

        return position_scores


strategy = TradingStrategy()
