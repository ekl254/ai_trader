"""NewsAPI client for fetching news articles."""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests

from src.logger import logger

# Reputable news domains (add more as needed)
REPUTABLE_SOURCES = {
    "reuters.com",
    "wsj.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "businessinsider.com",
    "nytimes.com",
}


class NewsAPIClient:
    """Client for NewsAPI.org to fetch news articles."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        self._cache = {}  # Simple in-memory cache: {(symbol, days_back): (timestamp, articles)}
        self._cache_ttl = 14400  # 4 hours in seconds

    def get_stock_news(
        self, symbol: str, days_back: int = 7, max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol.

        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of article dictionaries with title, description, content, publishedAt
        """
        # Check cache first
        cache_key = (symbol, days_back)
        if cache_key in self._cache:
            timestamp, cached_articles = self._cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self._cache_ttl:
                logger.info("newsapi_cache_hit", symbol=symbol)
                return cached_articles

        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)

            # Build targeted query - search for company name AND financial context
            # Avoid generic terms that match unrelated articles
            company_name = self._company_name(symbol)
            if company_name != symbol:
                # We have a known company name - use it with financial context
                query = f'("{company_name}" OR "{symbol}") AND (earnings OR stock OR shares OR revenue OR profit OR CEO OR quarterly)'
            else:
                # Unknown company - use symbol with financial terms
                query = f'"{symbol}" AND (stock OR shares OR earnings OR trading)'

            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "relevancy",  # Changed from publishedAt to get more relevant results
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
                    # Extract domain
                    domain = source_url.split("//")[-1].split("/")[0]
                    if any(rep in domain for rep in REPUTABLE_SOURCES):
                        filtered.append(article)

                # Deduplicate articles by URL to avoid repeats
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

                # Update cache
                self._cache[cache_key] = (datetime.now().timestamp(), unique_articles)

                return unique_articles
            elif response.status_code == 426:
                # Upgrade required - free tier limitation
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
        """Return a friendly company name if we have a mapping, else the symbol itself."""
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

    def get_business_news(
        self, symbol: str, max_articles: int = 20
    ) -> List[Dict[str, Any]]:
        """Fetch business news from US top headlines, filtered by reputable sources and ticker/company name."""
        try:
            query = f"{symbol} OR {self._company_name(symbol)}"
            params = {
                "q": query,
                "category": "business",
                "country": "us",
                "pageSize": max_articles,
                "apiKey": self.api_key,
            }
            response = requests.get(
                f"{self.base_url}/top-headlines",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                # Filter reputable sources
                filtered = []
                for article in articles:
                    source_url = article.get("url", "")
                    domain = source_url.split("//")[-1].split("/")[0]
                    if any(rep in domain for rep in REPUTABLE_SOURCES):
                        filtered.append(article)
                # Deduplicate
                seen = set()
                unique = []
                for a in filtered:
                    u = a.get("url")
                    if u and u not in seen:
                        seen.add(u)
                        unique.append(a)
                logger.info(
                    "newsapi_business_fetched",
                    symbol=symbol,
                    count=len(unique),
                )
                return unique
            else:
                logger.error(
                    "newsapi_business_failed",
                    status_code=response.status_code,
                )
                return []
        except Exception as e:
            logger.error("newsapi_business_error", error=str(e))
            return []

    def get_top_headlines(
        self, category: str = "business", country: str = "us", max_articles: int = 10
    ) -> List[Dict[str, Any]]:
        """Legacy method kept for compatibility – fetch generic top headlines."""
        try:
            params = {
                "category": category,
                "country": country,
                "pageSize": max_articles,
                "apiKey": self.api_key,
            }
            response = requests.get(
                f"{self.base_url}/top-headlines",
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
            else:
                logger.error(
                    "newsapi_headlines_failed",
                    status_code=response.status_code,
                )
                return []
        except Exception as e:
            logger.error("newsapi_headlines_error", error=str(e))
            return []


# Global instance
newsapi_client = NewsAPIClient()
