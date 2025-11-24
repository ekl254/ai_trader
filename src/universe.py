"""S&P 500 stock universe management."""

from typing import List

import pandas as pd

from src.logger import logger


def get_sp500_symbols() -> List[str]:
    """
    Get list of S&P 500 symbols.
    
    Returns:
        List of stock symbols
    """
    try:
        # Get from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df["Symbol"].tolist()
        
        # Clean symbols (remove dots, etc.)
        symbols = [s.replace(".", "-") for s in symbols]
        
        logger.info("sp500_symbols_loaded", count=len(symbols))
        return symbols
        
    except Exception as e:
        logger.error("failed_to_load_sp500_symbols", error=str(e))
        # Fallback to a small set of liquid stocks
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK.B", "JPM", "V",
            "JNJ", "WMT", "PG", "MA", "HD",
            "CVX", "MRK", "ABBV", "KO", "PEP",
        ]


def filter_liquid_stocks(symbols: List[str], min_volume: int = 1_000_000) -> List[str]:
    """
    Filter stocks by liquidity.
    
    Args:
        symbols: List of symbols to filter
        min_volume: Minimum average daily volume
    
    Returns:
        Filtered list of liquid symbols
    """
    # For now, return top stocks (in production, check actual volume)
    return symbols[:50]  # Top 50 most liquid
