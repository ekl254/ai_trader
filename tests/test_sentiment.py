"""Tests for sentiment analysis."""

import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def mock_newsapi():
    """Mock NewsAPI client."""
    with patch("src.sentiment.newsapi_client") as mock:
        yield mock


def test_analyze_sentiment_empty_text() -> None:
    """Test sentiment analysis with empty text."""
    from src.sentiment import analyze_sentiment

    assert analyze_sentiment("") == 0.0
    assert analyze_sentiment("   ") == 0.0


@patch("src.sentiment.get_sentiment_analyzer")
def test_analyze_sentiment_positive(mock_analyzer: Mock) -> None:
    """Test positive sentiment detection."""
    from src.sentiment import analyze_sentiment

    mock_pipeline = Mock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.95}]
    mock_analyzer.return_value = mock_pipeline

    result = analyze_sentiment("Great earnings report, stock surges!")

    assert result == 0.95


@patch("src.sentiment.get_sentiment_analyzer")
def test_analyze_sentiment_negative(mock_analyzer: Mock) -> None:
    """Test negative sentiment detection."""
    from src.sentiment import analyze_sentiment

    mock_pipeline = Mock()
    mock_pipeline.return_value = [{"label": "NEGATIVE", "score": 0.85}]
    mock_analyzer.return_value = mock_pipeline

    result = analyze_sentiment("Company faces massive losses")

    assert result == -0.85


@patch("src.sentiment.get_sentiment_analyzer")
def test_analyze_sentiment_error_returns_neutral(mock_analyzer: Mock) -> None:
    """Test that errors return neutral sentiment."""
    from src.sentiment import analyze_sentiment

    mock_analyzer.side_effect = Exception("Model error")

    result = analyze_sentiment("Some text")

    assert result == 0.0


def test_get_news_sentiment_no_articles(mock_newsapi: Mock) -> None:
    """Test sentiment with no articles returns neutral."""
    from src.sentiment import get_news_sentiment

    mock_newsapi.get_stock_news.return_value = []

    result = get_news_sentiment("AAPL")

    assert result == 50.0  # Neutral


@patch("src.sentiment.analyze_sentiment")
def test_get_news_sentiment_with_articles(
    mock_analyze: Mock, mock_newsapi: Mock
) -> None:
    """Test sentiment calculation with multiple articles."""
    from src.sentiment import get_news_sentiment

    mock_newsapi.get_stock_news.return_value = [
        {"title": "Apple hits new high", "description": "Stock surges"},
        {"title": "Strong iPhone sales", "description": "Beat expectations"},
        {"title": "Supply chain issues", "description": "Some concerns"},
    ]

    # Return varying sentiments: +0.8, +0.6, -0.2 → avg = 0.4
    mock_analyze.side_effect = [0.8, 0.6, -0.2]

    result = get_news_sentiment("AAPL", max_articles=10)

    # avg sentiment = 0.4, scaled to 0-100: (0.4 + 1) * 50 = 70
    assert result == pytest.approx(70.0, rel=0.01)
    assert mock_analyze.call_count == 3


def test_get_news_sentiment_api_error(mock_newsapi: Mock) -> None:
    """Test sentiment returns neutral on API error."""
    from src.sentiment import get_news_sentiment

    mock_newsapi.get_stock_news.side_effect = Exception("API unavailable")

    result = get_news_sentiment("AAPL")

    assert result == 50.0  # Neutral on error


@patch("src.sentiment.analyze_sentiment")
def test_get_news_sentiment_empty_content(
    mock_analyze: Mock, mock_newsapi: Mock
) -> None:
    """Test handling of articles with empty content."""
    from src.sentiment import get_news_sentiment

    mock_newsapi.get_stock_news.return_value = [
        {"title": "", "description": ""},  # Empty
        {"title": "Valid headline", "description": "Valid description"},
    ]

    mock_analyze.return_value = 0.5

    result = get_news_sentiment("AAPL")

    # Only one valid article should be analyzed
    assert mock_analyze.call_count == 1


@patch("src.sentiment._sentiment_analyzer", None)
def test_get_sentiment_analyzer_lazy_loading() -> None:
    """Test that sentiment analyzer is lazily loaded."""
    from src.sentiment import get_sentiment_analyzer

    with patch("src.sentiment.pipeline") as mock_pipeline:
        mock_pipeline.return_value = Mock()

        # First call should create analyzer
        analyzer1 = get_sentiment_analyzer()

        # Verify pipeline was called with FinBERT model
        mock_pipeline.assert_called_once_with(
            "sentiment-analysis",
            model="ProsusAI/finbert",
        )


def test_sentiment_text_truncation() -> None:
    """Test that long text is truncated to 512 characters."""
    from src.sentiment import analyze_sentiment

    with patch("src.sentiment.get_sentiment_analyzer") as mock_analyzer:
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_analyzer.return_value = mock_pipeline

        long_text = "x" * 1000
        analyze_sentiment(long_text)

        # Verify the text was truncated to 512 chars
        call_args = mock_pipeline.call_args[0][0]
        assert len(call_args) == 512
