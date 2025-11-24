"""Sentiment analysis for news articles."""

from typing import List

from src.newsapi_client import newsapi_client
from src.logger import logger


# Load sentiment model (cached after first use)
_sentiment_analyzer = None


def get_sentiment_analyzer():
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
        label = result["label"]
        score = result["score"]
        
        # Convert to -1 to 1 scale
        if label == "POSITIVE":
            return score
        else:
            return -score
            
    except Exception:
        return 0.0  # Neutral on error


def get_news_sentiment(symbol: str, max_articles: int = 20) -> float:
    """
    Get overall sentiment for a stock based on recent news.
    
    Uses NewsAPI to fetch articles and analyzes sentiment.
    
    Args:
        symbol: Stock ticker symbol
        max_articles: Maximum number of articles to analyze
        
    Returns:
        float: Average sentiment score from 0-100 (scaled from -1 to 1)
    """
    try:
        # Fetch news articles using NewsAPI
        articles = newsapi_client.get_stock_news(symbol, days_back=7, max_articles=max_articles)
        
        if not articles:
            logger.warning("no_news_articles_found", symbol=symbol)
            return 50.0  # Neutral
        
        sentiments = []
        
        for article in articles:
            # Combine title and description for better context
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title}. {description}" if description else title
            
            if text:
                sentiment = analyze_sentiment(text)
                sentiments.append(sentiment)
        
        if not sentiments:
            return 50.0  # Neutral
        
        # Average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Convert from -1/+1 scale to 0-100 scale
        score = (avg_sentiment + 1) * 50
        
        logger.info(
            "news_sentiment_analyzed",
            symbol=symbol,
            articles_count=len(articles),
            sentiments_count=len(sentiments),
            avg_sentiment=avg_sentiment,
            score=score,
        )
        
        return score
        
    except Exception as e:
        logger.error("news_sentiment_error", symbol=symbol, error=str(e))
        return 50.0  # Neutral on error
