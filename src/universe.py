"""S&P 500 stock universe management."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.logger import logger

CACHE_FILE = Path(__file__).parent.parent / "data" / "sp500_cache.json"
CACHE_DURATION_HOURS = 24  # Refresh daily to catch additions/removals


def get_sp500_symbols() -> List[str]:
    """
    Get list of S&P 500 symbols from Wikipedia with caching.
    Cache is refreshed daily to catch S&P 500 additions/removals.

    Returns:
        List of stock symbols
    """
    # Check cache first
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache["timestamp"])

                # Use cache if less than 24 hours old
                if datetime.now() - cache_time < timedelta(hours=CACHE_DURATION_HOURS):
                    symbols = cache["symbols"]
                    logger.info("sp500_symbols_loaded_from_cache", count=len(symbols))
                    return symbols
        except Exception as e:
            logger.warning("cache_read_failed", error=str(e))

    # Fetch fresh data from Wikipedia
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        # Use requests with proper headers to avoid 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML tables - use lxml and handle the first table
        from io import StringIO

        tables = pd.read_html(StringIO(response.text), flavor="lxml")

        # The first proper table has the S&P 500 constituents
        df = None
        for table in tables:
            if "Symbol" in table.columns or (
                len(table.columns) > 2 and "Ticker" in str(table.columns)
            ):
                df = table
                break

        if df is None:
            raise ValueError("Could not find S&P 500 table in Wikipedia")

        # Try different column names
        symbol_col = None
        for col in ["Symbol", "Ticker", "Ticker symbol"]:
            if col in df.columns:
                symbol_col = col
                break

        if symbol_col is None:
            # Fallback: if first column looks like symbols
            if len(df.columns) > 0:
                symbol_col = df.columns[0]

        symbols = df[symbol_col].tolist()

        # Filter out NaN and clean symbols
        symbols = [
            str(s).strip()
            for s in symbols
            if pd.notna(s) and len(str(s)) <= 5
        ]
        
        # Normalize symbol formats for Alpaca compatibility
        # Wikipedia uses hyphens (BRK-B) but Alpaca uses dots (BRK.B)
        symbols = [s.replace("-", ".") for s in symbols]

        # Save to cache
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(
                {
                    "symbols": symbols,
                    "timestamp": datetime.now().isoformat(),
                    "count": len(symbols),
                },
                f,
            )

        logger.info("sp500_symbols_loaded_from_wikipedia", count=len(symbols))
        return symbols

    except Exception as e:
        logger.error("failed_to_load_sp500_symbols", error=str(e))

        # Try to use stale cache as last resort
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    cache = json.load(f)
                    symbols = cache["symbols"]
                    logger.warning("using_stale_cache", count=len(symbols))
                    return symbols
            except:
                pass

        # Final fallback to hardcoded liquid stocks
        logger.warning("using_fallback_symbols")
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",
            "MA",
            "HD",
            "CVX",
            "MRK",
            "ABBV",
            "KO",
            "PEP",
        ]


def filter_liquid_stocks(symbols: List[str], min_volume: int = 1_000_000) -> List[str]:
    """
    Filter stocks by liquidity using actual volume data from Alpaca.

    Args:
        symbols: List of symbols to filter
        min_volume: Minimum average daily volume (default 1M)

    Returns:
        Filtered list of liquid symbols
    """
    from datetime import datetime, timedelta

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from config.config import config

        client = StockHistoricalDataClient(
            config.alpaca.api_key, config.alpaca.secret_key
        )

        liquid_symbols = []

        # Process in batches to avoid rate limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=20),
                )
                bars = client.get_stock_bars(request)

                for symbol in batch:
                    try:
                        if symbol in bars.data:
                            symbol_bars = bars.data[symbol]
                            if len(symbol_bars) > 0:
                                avg_volume = sum(
                                    bar.volume for bar in symbol_bars
                                ) / len(symbol_bars)
                                if avg_volume >= min_volume:
                                    liquid_symbols.append(symbol)
                    except Exception:
                        # Skip symbols with data issues
                        continue

            except Exception as e:
                logger.warning(
                    "batch_liquidity_check_failed", batch_start=i, error=str(e)
                )
                # On error, include batch symbols (fail-safe)
                liquid_symbols.extend(batch)

        logger.info(
            "liquidity_filter_complete",
            input_count=len(symbols),
            output_count=len(liquid_symbols),
            min_volume=min_volume,
        )

        return liquid_symbols if liquid_symbols else symbols[:100]

    except Exception as e:
        logger.error("liquidity_filter_failed", error=str(e))
        # Fallback to top 100 symbols on error
        return symbols[:100]
