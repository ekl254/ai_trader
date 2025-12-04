"""Data provider for market data from Alpaca."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from config.config import config
from src.logger import logger


class AlpacaProvider:
    """Alpaca API provider for market data and trading."""

    def __init__(self) -> None:
        self.client = StockHistoricalDataClient(
            config.alpaca.api_key, config.alpaca.secret_key
        )

    def get_bars(
        self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day
    ) -> pd.DataFrame:
        """Get historical price bars."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=datetime.now(UTC) - timedelta(days=days),
        )

        try:
            bars = self.client.get_stock_bars(request)
            # bars can be BarSet or dict - ensure we have proper BarSet
            if isinstance(bars, dict):
                raise ValueError("Received dict instead of BarSet")
            df: pd.DataFrame = bars.df
            logger.debug("bars_fetched", symbol=symbol, rows=len(df))
            return df
        except Exception as e:
            logger.error("bars_fetch_failed", symbol=symbol, error=str(e))
            raise

    def get_latest_trade(self, symbol: str) -> dict[str, Any]:
        """Get latest trade price."""
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.client.get_stock_latest_trade(request)
            return {
                "price": float(trade[symbol].price),
                "time": trade[symbol].timestamp,
            }
        except Exception as e:
            logger.error("latest_trade_failed", symbol=symbol, error=str(e))
            raise


# Global instance
alpaca_provider = AlpacaProvider()
