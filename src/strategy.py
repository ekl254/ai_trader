"""Trading strategy with multi-factor scoring."""

from typing import Dict, List, Tuple

import pandas as pd
import pandas_ta as ta  # type: ignore

from config.config import config
from src.data_provider import alpaca_provider
from src.logger import logger
from src.sentiment import analyze_sentiment, get_news_sentiment


class TradingStrategy:
    """Multi-factor trading strategy."""

    def __init__(self) -> None:
        self.config = config.trading

    def calculate_technical_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate technical analysis score (0-100)."""
        if len(df) < 30:
            return 0.0, {"error": "insufficient_data"}
        
        # Calculate indicators
        df["RSI"] = ta.rsi(df["close"], length=self.config.rsi_period)
        macd = ta.macd(df["close"])
        if macd is not None:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
        bbands = ta.bbands(df["close"])
        if bbands is not None:
            # pandas-ta returns column names with double std parameter
            df["BB_upper"] = bbands["BBU_5_2.0_2.0"]
            df["BB_lower"] = bbands["BBL_5_2.0_2.0"]
            df["BB_mid"] = bbands["BBM_5_2.0_2.0"]
        
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
        
        # MACD Score
        macd_value = latest.get("MACD")
        macd_signal = latest.get("MACD_signal")
        if pd.isna(macd_value) or pd.isna(macd_signal):
            macd_score = 50.0
        elif macd_value > macd_signal:
            macd_score = 75.0  # Bullish
        else:
            macd_score = 25.0  # Bearish
        
        # Bollinger Bands Score
        close = latest["close"]
        bb_upper = latest.get("BB_upper")
        bb_lower = latest.get("BB_lower")
        bb_mid = latest.get("BB_mid")
        
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
        
        # Volume Score
        avg_volume = df["volume"].tail(20).mean()
        current_volume = latest["volume"]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(100, volume_ratio * 50)  # Cap at 100
        
        # Weighted average
        technical_score = (
            rsi_score * 0.3
            + macd_score * 0.3
            + bb_score * 0.25
            + volume_score * 0.15
        )
        
        details = {
            "rsi": float(rsi_value) if not pd.isna(rsi_value) else None,
            "rsi_score": rsi_score,
            "macd": float(macd_value) if not pd.isna(macd_value) else None,
            "macd_signal": float(macd_signal) if not pd.isna(macd_signal) else None,
            "macd_score": macd_score,
            "bb_score": bb_score,
            "volume_score": volume_score,
            "total": technical_score,
        }
        
        return technical_score, details

    def calculate_fundamental_score(self, symbol: str) -> Tuple[float, Dict]:
        """Calculate fundamental analysis score (neutral - not using external fundamental data)."""
        # Simplified: no external fundamental data, always neutral
        return 50.0, {"note": "Using technical + sentiment only"}

    def calculate_sentiment_score(self, symbol: str) -> Tuple[float, Dict]:
        """Calculate news sentiment score (0-100)."""
        # Sentiment analysis (using NewsAPI)
        try:
            sentiment_score = get_news_sentiment(symbol, max_articles=20)
            logger.info("sentiment_score_calculated", symbol=symbol, score=sentiment_score)
            
            details = {
                "total": sentiment_score,
            }
            
            return sentiment_score, details
            
        except Exception as e:
            logger.error("sentiment_analysis_failed", symbol=symbol, error=str(e))
            return 50.0, {"error": str(e)}

    def score_symbol(self, symbol: str) -> Tuple[float, Dict]:
        """Calculate composite score for a symbol."""
        logger.info("scoring_symbol", symbol=symbol)
        
        # Get price data
        df = alpaca_provider.get_bars(symbol, days=60)
        
        # Calculate all scores
        technical_score, technical_details = self.calculate_technical_score(df)
        fundamental_score, fundamental_details = self.calculate_fundamental_score(symbol)
        sentiment_score, sentiment_details = self.calculate_sentiment_score(symbol)
        
        # Weighted composite score
        composite_score = (
            technical_score * 0.4
            + fundamental_score * 0.3
            + sentiment_score * 0.3
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

    def should_buy(self, symbol: str) -> Tuple[bool, float, Dict]:
        """Determine if symbol should be bought."""
        score, reasoning = self.score_symbol(symbol)
        
        # Check composite score
        if score < self.config.min_composite_score:
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


strategy = TradingStrategy()
