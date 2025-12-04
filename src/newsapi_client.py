"""News client for fetching financial news from Alpaca.

Uses Alpaca News API exclusively - free with trading account,
better financial coverage, and no rate limits.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from src.logger import logger


class AlpacaNewsClient:
    """Client for Alpaca's News API - free with trading account."""

    def __init__(self) -> None:
        self._client: Any = None
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._cache_ttl = 1800  # 30 minutes

    def _get_client(self) -> Any:
        """Lazy-load Alpaca NewsClient."""
        if self._client is None:
            try:
                from alpaca.data.historical.news import NewsClient

                from config.config import config

                self._client = NewsClient(
                    config.alpaca.api_key, config.alpaca.secret_key
                )
            except Exception as e:
                logger.error("alpaca_news_client_init_failed", error=str(e))
                return None
        return self._client

    def get_stock_news(
        self, symbol: str, days_back: int = 7, max_articles: int = 20
    ) -> list[dict[str, Any]]:
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
        cache_key = f"{symbol}_{days_back}"
        now = datetime.now().timestamp()
        if cache_key in self._cache:
            timestamp, cached_articles = self._cache[cache_key]
            if (now - timestamp) < self._cache_ttl:
                logger.info(
                    "alpaca_news_cache_hit", symbol=symbol, count=len(cached_articles)
                )
                articles_list: list[dict[str, Any]] = cached_articles
                return articles_list

        client = self._get_client()
        if client is None:
            logger.warning("alpaca_news_client_unavailable", symbol=symbol)
            return []

        try:
            from alpaca.data.requests import NewsRequest

            # Calculate date range
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days_back)

            request = NewsRequest(
                symbols=symbol,
                start=start_date,
                end=end_date,
                limit=max_articles,
            )

            response = client.get_news(request)

            # Parse the NewsSet response correctly
            # response.data is a dict with 'news' key containing list of News objects
            news_items = response.data.get("news", [])

            # Convert to standard format
            articles = []
            for article in news_items:
                # article is a News object with attributes
                articles.append(
                    {
                        "title": article.headline,
                        "description": article.summary or "",
                        "content": article.content or "",
                        "url": article.url,
                        "publishedAt": str(article.created_at),
                        "source": article.source,
                        "author": article.author,
                        "symbols": article.symbols,  # Related symbols
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

    def clear_cache(self, symbol: str | None = None) -> None:
        """Clear news cache for a specific symbol or all symbols."""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info("alpaca_news_cache_cleared", symbol=symbol)
        else:
            self._cache.clear()
            logger.info("alpaca_news_cache_cleared_all")


# Global instance
newsapi_client = AlpacaNewsClient()
