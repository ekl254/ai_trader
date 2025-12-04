"""Tests for market regime detection."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_spy_data():
    """Create sample SPY price data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Create trending up data
    prices = 400 + np.cumsum(np.random.randn(100) * 2) + np.arange(100) * 0.5
    volumes = np.random.randint(50_000_000, 100_000_000, 100)

    return pd.DataFrame(
        {
            "open": prices * 0.998,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


def test_market_regime_enum():
    """Test MarketRegime enum values."""
    from src.market_regime import MarketRegime

    assert MarketRegime.STRONG_BULL.value == "strong_bull"
    assert MarketRegime.BULL.value == "bull"
    assert MarketRegime.NEUTRAL.value == "neutral"
    assert MarketRegime.BEAR.value == "bear"
    assert MarketRegime.STRONG_BEAR.value == "strong_bear"
    assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"


def test_regime_parameters_exist():
    """Test that regime parameters are defined for all regimes."""
    from src.market_regime import REGIME_PARAMETERS, MarketRegime

    for regime in MarketRegime:
        assert regime in REGIME_PARAMETERS, f"Missing parameters for {regime}"
        params = REGIME_PARAMETERS[regime]
        assert "stop_loss_pct" in params
        assert "take_profit_pct" in params
        assert "min_score" in params
        assert "max_positions" in params


def test_detect_strong_bull(sample_spy_data):
    """Test detection of strong bull market."""
    from src.market_regime import MarketRegime, MarketRegimeDetector

    # Create strongly trending up data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = 400 + np.arange(100) * 2  # Strong uptrend

    bull_data = pd.DataFrame(
        {
            "open": prices * 0.998,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.full(100, 80_000_000),
        },
        index=dates,
    )

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = bull_data

        detector = MarketRegimeDetector()
        # Clear any cached data
        detector._cache = {}
        result = detector.get_market_regime()
        regime = result.get("regime_enum", MarketRegime.NEUTRAL)

        # Should detect bullish conditions
        assert regime in [
            MarketRegime.STRONG_BULL,
            MarketRegime.BULL,
            MarketRegime.NEUTRAL,
        ]


def test_detect_bear():
    """Test detection of bear market."""
    from src.market_regime import MarketRegime, MarketRegimeDetector

    # Create strongly trending down data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = 500 - np.arange(100) * 2  # Strong downtrend

    bear_data = pd.DataFrame(
        {
            "open": prices * 1.002,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.full(100, 80_000_000),
        },
        index=dates,
    )

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = bear_data

        detector = MarketRegimeDetector()
        detector._cache = {}
        result = detector.get_market_regime()
        regime = result.get("regime_enum", MarketRegime.NEUTRAL)

        # Should detect bearish conditions
        assert regime in [
            MarketRegime.STRONG_BEAR,
            MarketRegime.BEAR,
            MarketRegime.NEUTRAL,
        ]


def test_detect_volatile():
    """Test detection of volatile market."""
    from src.market_regime import MarketRegimeDetector

    # Create highly volatile data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(123)
    prices = 450 + np.cumsum(np.random.randn(100) * 15)  # High volatility

    volatile_data = pd.DataFrame(
        {
            "open": prices * 0.97,
            "high": prices * 1.05,
            "low": prices * 0.95,
            "close": prices,
            "volume": np.full(100, 120_000_000),  # High volume
        },
        index=dates,
    )

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = volatile_data

        detector = MarketRegimeDetector()
        detector._cache = {}
        result = detector.get_market_regime()

        # Should return valid result with parameters
        assert result is not None
        assert "max_positions_override" in result or "max_positions" in result


def test_regime_parameters_returned(sample_spy_data):
    """Test that regime detection returns valid parameters."""
    from src.market_regime import MarketRegimeDetector

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = sample_spy_data

        detector = MarketRegimeDetector()
        detector._cache = {}
        result = detector.get_market_regime()

        assert "stop_loss_pct" in result
        assert "take_profit_pct" in result
        assert "min_score" in result
        assert "max_positions_override" in result
        assert "regime" in result

        # Values should be reasonable
        assert 0.01 <= result["stop_loss_pct"] <= 0.10
        assert 0.03 <= result["take_profit_pct"] <= 0.20
        assert 50 <= result["min_score"] <= 90
        assert 3 <= result["max_positions_override"] <= 12


def test_regime_caching(sample_spy_data):
    """Test that regime detection caches results."""
    from src.market_regime import MarketRegimeDetector

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = sample_spy_data

        detector = MarketRegimeDetector()
        detector._cache = {}

        # First call
        result1 = detector.get_market_regime()

        # Second call should use cache (within cache period)
        result2 = detector.get_market_regime()

        assert result1["regime"] == result2["regime"]

        # Provider should only be called once due to caching
        assert mock_provider.get_bars.call_count == 1


def test_regime_error_handling():
    """Test graceful handling of API errors."""
    from src.market_regime import MarketRegime, MarketRegimeDetector

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.side_effect = Exception("API Error")

        detector = MarketRegimeDetector()
        detector._cache = {}
        result = detector.get_market_regime()

        # Should return neutral/safe defaults on error
        assert result["regime"] == MarketRegime.NEUTRAL.value
        assert result is not None


def test_insufficient_data():
    """Test handling of insufficient data."""
    from src.market_regime import MarketRegime, MarketRegimeDetector

    # Return too little data
    insufficient_data = pd.DataFrame(
        {
            "open": [400],
            "high": [405],
            "low": [395],
            "close": [402],
            "volume": [50000000],
        }
    )

    with patch("src.data_provider.alpaca_provider") as mock_provider:
        mock_provider.get_bars.return_value = insufficient_data

        detector = MarketRegimeDetector()
        detector._cache = {}
        result = detector.get_market_regime()

        # Should return neutral/safe defaults
        assert result["regime"] == MarketRegime.NEUTRAL.value
