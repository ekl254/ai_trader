"""News client for fetching financial news articles.

Uses Alpaca News API as primary source (free with trading account, better financial coverage).
Falls back to NewsAPI.org if Alpaca fails.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import requests

from src.logger import logger

# Reputable news domains for NewsAPI fallback
REPUTABLE_SOURCES = {
    "reuters.com",
    "wsj.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "businessinsider.com",
    "nytimes.com",
    "benzinga.com",
    "marketwatch.com",
}


class AlpacaNewsClient:
    """Client for Alpaca's built-in News API - free with trading account."""

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 1800  # 30 minutes (Alpaca news updates frequently)
        self._client = None

    def _get_client(self):
        """Lazy-load Alpaca NewsClient."""
        if self._client is None:
            try:
                from alpaca.data.historical import NewsClient

                self._client = NewsClient(self.api_key, self.secret_key)
            except Exception as e:
                logger.error("alpaca_news_client_init_failed", error=str(e))
                return None
        return self._client

    def get_stock_news(
        self, symbol: str, days_back: int = 7, max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol from Alpaca.

        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of article dictionaries with title, description, url, publishedAt
        """
        # Check cache first
        cache_key = f"alpaca_{symbol}_{days_back}"
        now = datetime.now().timestamp()
        if cache_key in self._cache:
            timestamp, cached_articles = self._cache[cache_key]
            if (now - timestamp) < self._cache_ttl:
                logger.info("alpaca_news_cache_hit", symbol=symbol)
                return cached_articles

        client = self._get_client()
        if client is None:
            return []

        try:
            from alpaca.data.requests import NewsRequest

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)

            request = NewsRequest(
                symbols=symbol,
                start=start_date,
                end=end_date,
                limit=max_articles,
            )

            news_response = client.get_news(request)

            # Convert to standard format
            articles = []
            for article in news_response.data.get("news", []):
                articles.append(
                    {
                        "title": article.headline,
                        "description": article.summary
                        if hasattr(article, "summary")
                        else "",
                        "url": article.url,
                        "publishedAt": str(article.created_at),
                        "source": article.source,
                        "symbols": article.symbols,  # Bonus: related symbols
                    }
                )

            logger.info(
                "alpaca_news_fetched",
                symbol=symbol,
                count=len(articles),
            )

            # Cache the result
            self._cache[cache_key] = (now, articles)
            return articles

        except Exception as e:
            logger.error("alpaca_news_error", symbol=symbol, error=str(e))
            return []


class NewsAPIClient:
    """Client for NewsAPI.org - fallback source."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        self._cache = {}
        self._cache_ttl = 14400  # 4 hours

    def get_stock_news(
        self, symbol: str, days_back: int = 7, max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol from NewsAPI.

        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of article dictionaries with title, description, content, publishedAt
        """
        cache_key = (symbol, days_back)
        if cache_key in self._cache:
            timestamp, cached_articles = self._cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self._cache_ttl:
                logger.info("newsapi_cache_hit", symbol=symbol)
                return cached_articles

        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)

            company_name = self._company_name(symbol)
            if company_name != symbol:
                query = f'("{company_name}" OR "{symbol}") AND (earnings OR stock OR shares OR revenue OR profit OR CEO OR quarterly)'
            else:
                query = f'"{symbol}" AND (stock OR shares OR earnings OR trading)'

            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": max_articles,
                "apiKey": self.api_key,
            }

            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])

                # Filter to reputable sources
                filtered = []
                for article in articles:
                    source_url = article.get("url", "")
                    domain = source_url.split("//")[-1].split("/")[0]
                    if any(rep in domain for rep in REPUTABLE_SOURCES):
                        filtered.append(article)

                # Deduplicate
                seen_urls = set()
                unique_articles = []
                for article in filtered:
                    url = article.get("url")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_articles.append(article)

                logger.info(
                    "newsapi_articles_fetched",
                    symbol=symbol,
                    count=len(unique_articles),
                )

                self._cache[cache_key] = (datetime.now().timestamp(), unique_articles)
                return unique_articles

            elif response.status_code == 426:
                logger.warning(
                    "newsapi_upgrade_required",
                    symbol=symbol,
                    message="NewsAPI free tier has limitations",
                )
                return []
            else:
                logger.error(
                    "newsapi_request_failed",
                    symbol=symbol,
                    status_code=response.status_code,
                    error=response.text,
                )
                return []

        except Exception as e:
            logger.error("newsapi_error", symbol=symbol, error=str(e))
            return []

    def _company_name(self, symbol: str) -> str:
        """Return a friendly company name if we have a mapping."""
        TICKER_COMPANY_MAP = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet Google",
            "GOOG": "Alphabet Google",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta Facebook",
            "NVDA": "Nvidia",
            "BRK.B": "Berkshire Hathaway",
            "JPM": "JPMorgan",
            "V": "Visa",
            "JNJ": "Johnson Johnson",
            "WMT": "Walmart",
            "PG": "Procter Gamble",
            "MA": "Mastercard",
            "HD": "Home Depot",
            "CVX": "Chevron",
            "MRK": "Merck",
            "ABBV": "AbbVie",
            "KO": "Coca-Cola",
            "PEP": "PepsiCo",
            "COST": "Costco",
            "BAC": "Bank of America",
            "AVGO": "Broadcom",
            "LLY": "Eli Lilly",
            "XOM": "ExxonMobil",
            "TMO": "Thermo Fisher",
            "CSCO": "Cisco",
            "MCD": "McDonald's",
            "ACN": "Accenture",
            "ABT": "Abbott",
            "DHR": "Danaher",
            "NKE": "Nike",
            "VZ": "Verizon",
            "ADBE": "Adobe",
            "CRM": "Salesforce",
            "INTC": "Intel",
            "AMD": "AMD",
            "NFLX": "Netflix",
            "DIS": "Disney",
            "UNH": "UnitedHealth",
            "PFE": "Pfizer",
            "ORCL": "Oracle",
            "IBM": "IBM",
            "GE": "General Electric",
            "CAT": "Caterpillar",
            "BA": "Boeing",
            "GS": "Goldman Sachs",
            "RTX": "Raytheon",
            "UPS": "UPS",
            "SBUX": "Starbucks",
            "T": "AT&T",
            "QCOM": "Qualcomm",
            "SPGI": "S&P Global",
            "MS": "Morgan Stanley",
            "BLK": "BlackRock",
            "AXP": "American Express",
            "DE": "John Deere",
            "LOW": "Lowe's",
            "NOW": "ServiceNow",
            "BKNG": "Booking Holdings",
            "ISRG": "Intuitive Surgical",
            "MDLZ": "Mondelez",
            "GILD": "Gilead Sciences",
            "ADI": "Analog Devices",
            "MMM": "3M",
            "TJX": "TJX Companies",
            "CVS": "CVS Health",
            "CI": "Cigna",
            "ZTS": "Zoetis",
            "REGN": "Regeneron",
            "SO": "Southern Company",
            "DUK": "Duke Energy",
            "CL": "Colgate-Palmolive",
            "SYK": "Stryker",
            "FDX": "FedEx",
            "CME": "CME Group",
            "PLD": "Prologis",
            "ITW": "Illinois Tool Works",
            "AON": "Aon",
            "SHW": "Sherwin-Williams",
            "ETN": "Eaton",
            "NSC": "Norfolk Southern",
            "USB": "US Bancorp",
            "PNC": "PNC Financial",
            "TFC": "Truist Financial",
        }
        return TICKER_COMPANY_MAP.get(symbol.upper(), symbol)


class HybridNewsClient:
    """
    Hybrid news client that uses Alpaca News as primary source
    and falls back to NewsAPI.org if needed.

    Benefits of Alpaca News:
    - Free with trading account (no separate API key needed)
    - News is tagged with stock symbols (more accurate)
    - Sources are financial-focused (Benzinga, etc.)
    - No rate limits beyond reasonable usage
    """

    def __init__(self):
        self.alpaca_client = AlpacaNewsClient()
        self.newsapi_client = NewsAPIClient()
        self._use_alpaca = True  # Try Alpaca first

    def get_stock_news(
        self, symbol: str, days_back: int = 7, max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol.
        Uses Alpaca News first, falls back to NewsAPI.

        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of article dictionaries
        """
        articles = []

        # Try Alpaca first (better for stocks)
        if self._use_alpaca:
            articles = self.alpaca_client.get_stock_news(
                symbol, days_back, max_articles
            )
            if articles:
                return articles

        # Fall back to NewsAPI
        articles = self.newsapi_client.get_stock_news(symbol, days_back, max_articles)

        return articles


# Global instance - use hybrid client by default
newsapi_client = HybridNewsClient()
