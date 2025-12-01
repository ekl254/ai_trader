"""Tests for trading strategy."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.strategy import TradingStrategy


@pytest.fixture
def strategy() -> TradingStrategy:
    """Create a TradingStrategy instance for testing."""
    with patch("src.strategy.config"):
        return TradingStrategy()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    prices = 100 + np.cumsum(np.random.randn(60) * 2)
    volumes = np.random.randint(1_000_000, 10_000_000, 60)

    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


def test_calculate_technical_score_insufficient_data(strategy: TradingStrategy) -> None:
    """Test technical score returns 0 for insufficient data."""
    df = pd.DataFrame({"close": [100, 101, 102]})
    score, details = strategy.calculate_technical_score(df)

    assert score == 0.0
    assert "error" in details


def test_calculate_technical_score_valid_data(
    strategy: TradingStrategy, sample_df: pd.DataFrame
) -> None:
    """Test technical score calculation with valid data."""
    strategy.config = Mock()
    strategy.config.rsi_period = 14
    strategy.config.rsi_oversold = 30.0
    strategy.config.rsi_overbought = 70.0

    score, details = strategy.calculate_technical_score(sample_df)

    assert 0 <= score <= 100
    assert "rsi_score" in details
    assert "macd_score" in details
    assert "bb_score" in details
    assert "volume_score" in details
    assert "total" in details


def test_calculate_fundamental_score(strategy: TradingStrategy) -> None:
    """Test fundamental score returns neutral (not implemented)."""
    score, details = strategy.calculate_fundamental_score("AAPL")

    assert score == 50.0
    assert "note" in details


@patch("src.strategy.get_news_sentiment")
def test_calculate_sentiment_score_success(
    mock_sentiment: Mock, strategy: TradingStrategy
) -> None:
    """Test sentiment score calculation."""
    mock_sentiment.return_value = 75.0

    score, details = strategy.calculate_sentiment_score("AAPL")

    assert score == 75.0
    assert "total" in details
    mock_sentiment.assert_called_once_with("AAPL", max_articles=20)


@patch("src.strategy.get_news_sentiment")
def test_calculate_sentiment_score_error(
    mock_sentiment: Mock, strategy: TradingStrategy
) -> None:
    """Test sentiment score returns neutral on error."""
    mock_sentiment.side_effect = Exception("API error")

    score, details = strategy.calculate_sentiment_score("AAPL")

    assert score == 50.0
    assert "error" in details


@patch("src.strategy.alpaca_provider")
@patch("src.strategy.get_news_sentiment")
def test_score_symbol(
    mock_sentiment: Mock,
    mock_provider: Mock,
    strategy: TradingStrategy,
    sample_df: pd.DataFrame,
) -> None:
    """Test composite score calculation."""
    mock_provider.get_bars.return_value = sample_df
    mock_sentiment.return_value = 60.0

    strategy.config = Mock()
    strategy.config.rsi_period = 14
    strategy.config.rsi_oversold = 30.0
    strategy.config.rsi_overbought = 70.0

    score, reasoning = strategy.score_symbol("AAPL")

    assert 0 <= score <= 100
    assert "technical" in reasoning
    assert "fundamental" in reasoning
    assert "sentiment" in reasoning
    assert "composite" in reasoning


@patch("src.strategy.alpaca_provider")
@patch("src.strategy.get_news_sentiment")
def test_should_buy_below_threshold(
    mock_sentiment: Mock,
    mock_provider: Mock,
    strategy: TradingStrategy,
    sample_df: pd.DataFrame,
) -> None:
    """Test should_buy returns False when score below threshold."""
    mock_provider.get_bars.return_value = sample_df
    mock_sentiment.return_value = 30.0  # Low sentiment

    strategy.config = Mock()
    strategy.config.rsi_period = 14
    strategy.config.rsi_oversold = 30.0
    strategy.config.rsi_overbought = 70.0
    strategy.config.min_composite_score = 90.0  # High threshold
    strategy.config.min_factor_score = 40.0

    should_buy, score, reasoning = strategy.should_buy("AAPL")

    assert should_buy is False


def test_rescore_positions(strategy: TradingStrategy, sample_df: pd.DataFrame) -> None:
    """Test rescoring multiple positions."""
    with patch.object(strategy, "score_symbol") as mock_score:
        mock_score.return_value = (65.0, {"technical": {"total": 70}})

        result = strategy.rescore_positions(["AAPL", "GOOGL"])

        assert "AAPL" in result
        assert "GOOGL" in result
        assert result["AAPL"]["score"] == 65.0
        assert mock_score.call_count == 2
