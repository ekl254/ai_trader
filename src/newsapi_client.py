"""NewsAPI client for fetching news articles."""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests

from src.logger import logger


class NewsAPIClient:
    """Client for NewsAPI.org to fetch news articles."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        
    def get_stock_news(self, symbol: str, days_back: int = 7, max_articles: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return
            
        Returns:
            List of article dictionaries with title, description, content, publishedAt
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Build query - search for company name or ticker
            query = f"{symbol} OR stock OR shares"
            
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": self.api_key,
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10,
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                logger.info(
                    "newsapi_articles_fetched",
                    symbol=symbol,
                    count=len(articles),
                )
                
                return articles
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
    
    def get_top_headlines(self, category: str = "business", country: str = "us", max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Get top headlines from news sources.
        
        Args:
            category: News category (business, technology, etc.)
            country: Country code (us, gb, etc.)
            max_articles: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
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
                timeout=10,
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
