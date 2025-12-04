"""Sentiment analysis for news articles."""

from datetime import UTC, datetime
from typing import Any

from src.logger import logger
from src.newsapi_client import newsapi_client

# Load sentiment model (cached after first use)
_sentiment_analyzer: Any = None

# Sentiment score cache: {symbol: (timestamp, score, article_count)}
_sentiment_cache: dict[str, tuple[datetime, float, int]] = {}
_SENTIMENT_CACHE_TTL = 1800  # 30 minutes


def get_sentiment_analyzer() -> Any:
    """Get or create sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from transformers import pipeline

        # Using FinBERT - BERT fine-tuned on financial news
        # Better at understanding financial terminology and context
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
        )
    return _sentiment_analyzer


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of text.

    Returns:
        float: Sentiment score from -1 (very negative) to 1 (very positive)
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    analyzer = get_sentiment_analyzer()

    try:
        result = analyzer(text[:512])[0]  # Limit text length
        label: str = result["label"]
        score: float = result["score"]

        # Convert to -1 to 1 scale
        if label.lower() == "positive":
            return float(score)
        elif label.lower() == "negative":
            return float(-score)
        else:
            return 0.0  # Neutral

    except Exception as e:
        logger.warning("sentiment_analysis_failed", error=str(e))
        return 0.0  # Neutral on error


def get_news_sentiment(symbol: str, max_articles: int = 20) -> float:
    """
    Get overall sentiment for a stock based on recent news.
    Results are cached for 30 minutes to reduce API calls and processing time.

    Uses NewsAPI to fetch articles and analyzes sentiment with FinBERT.

    Args:
        symbol: Stock ticker symbol
        max_articles: Maximum number of articles to analyze

    Returns:
        float: Average sentiment score from 0-100 (scaled from -1 to 1)
    """
    global _sentiment_cache

    # Check cache first
    now = datetime.now(UTC)
    if symbol in _sentiment_cache:
        cached_time, cached_score, _cached_count = _sentiment_cache[symbol]
        cache_age = (now - cached_time).total_seconds()
        if cache_age < _SENTIMENT_CACHE_TTL:
            logger.info(
                "sentiment_cache_hit",
                symbol=symbol,
                score=cached_score,
                cache_age_seconds=int(cache_age),
            )
            return cached_score

    try:
        # Fetch news articles using NewsAPI
        articles = newsapi_client.get_stock_news(
            symbol, days_back=7, max_articles=max_articles
        )

        if not articles:
            logger.warning("no_news_articles_found", symbol=symbol)
            # Cache the neutral result too
            _sentiment_cache[symbol] = (now, 50.0, 0)
            return 50.0  # Neutral

        sentiments = []

        for article in articles:
            # Combine title and description for better context
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title}. {description}" if description else title

            if text and len(text.strip()) > 10:  # Skip very short texts
                sentiment = analyze_sentiment(text)
                sentiments.append(sentiment)

        if not sentiments:
            _sentiment_cache[symbol] = (now, 50.0, len(articles))
            return 50.0  # Neutral

        # Average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Convert from -1/+1 scale to 0-100 scale
        score = (avg_sentiment + 1) * 50

        # Cache the result
        _sentiment_cache[symbol] = (now, score, len(articles))

        logger.info(
            "news_sentiment_analyzed",
            symbol=symbol,
            articles_count=len(articles),
            sentiments_count=len(sentiments),
            avg_sentiment=round(avg_sentiment, 3),
            score=round(score, 2),
        )

        return score

    except Exception as e:
        logger.error("news_sentiment_error", symbol=symbol, error=str(e))
        return 50.0  # Neutral on error


def clear_sentiment_cache(symbol: str | None = None) -> None:
    """
    Clear sentiment cache for a specific symbol or all symbols.

    Args:
        symbol: If provided, clear only this symbol's cache. Otherwise clear all.
    """
    global _sentiment_cache
    if symbol:
        _sentiment_cache.pop(symbol, None)
        logger.info("sentiment_cache_cleared", symbol=symbol)
    else:
        _sentiment_cache.clear()
        logger.info("sentiment_cache_cleared_all")


def get_sentiment_cache_stats() -> dict[str, Any]:
    """Get statistics about the sentiment cache."""
    datetime.now(UTC)
    return {
        "cached_symbols": len(_sentiment_cache),
        "symbols": list(_sentiment_cache.keys()),
        "cache_ttl_seconds": _SENTIMENT_CACHE_TTL,
    }
