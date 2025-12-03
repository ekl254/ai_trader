"""Tests for dynamic position sizer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def mock_trading_client():
    """Mock the trading client."""
    with patch("src.position_sizer.get_trading_client") as mock:
        mock_client = MagicMock()
        mock_account = Mock()
        mock_account.cash = "50000.0"
        mock_account.portfolio_value = "100000.0"
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = []
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_data_client():
    """Mock the data client."""
    with patch("src.position_sizer.get_data_client") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


def test_conviction_multiplier_high_score(mock_trading_client, mock_data_client):
    """Test conviction multiplier for high scores."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Score 90+ should get 1.5x
    mult, reason = sizer.get_conviction_multiplier(92.0)
    assert mult == 1.5
    assert "Exceptional" in reason


def test_conviction_multiplier_standard_score(mock_trading_client, mock_data_client):
    """Test conviction multiplier for standard scores."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Score 75-80 should get 1.0x
    mult, reason = sizer.get_conviction_multiplier(77.0)
    assert mult == 1.0
    assert "Standard" in reason


def test_conviction_multiplier_low_score(mock_trading_client, mock_data_client):
    """Test conviction multiplier for low scores."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Score below 70 should get 0.7x
    mult, reason = sizer.get_conviction_multiplier(65.0)
    assert mult == 0.7
    assert "Minimum" in reason or "Low" in reason


def test_volatility_multiplier_low_vol(mock_trading_client, mock_data_client):
    """Test volatility multiplier for low volatility stocks."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Mock get_volatility to return low vol
    with patch.object(sizer, 'get_volatility', return_value=0.01):
        mult, reason = sizer.get_volatility_multiplier("AAPL")
    
    assert mult >= 1.0
    assert "Low" in reason


def test_volatility_multiplier_high_vol(mock_trading_client, mock_data_client):
    """Test volatility multiplier for high volatility stocks."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Mock get_volatility to return high vol
    with patch.object(sizer, 'get_volatility', return_value=0.05):
        mult, reason = sizer.get_volatility_multiplier("AAPL")
    
    assert mult <= 1.0
    assert "High" in reason


def test_calculate_position_size_basic(mock_trading_client, mock_data_client):
    """Test basic position size calculation."""
    from src.position_sizer import DynamicPositionSizer
    
    # Set up account with plenty of cash
    mock_account = Mock()
    mock_account.cash = "50000.0"
    mock_account.portfolio_value = "100000.0"
    mock_trading_client.get_account.return_value = mock_account
    mock_trading_client.get_all_positions.return_value = []
    
    sizer = DynamicPositionSizer()
    
    # Mock volatility to return normal value
    with patch.object(sizer, 'get_volatility', return_value=0.02):
        result = sizer.calculate_position_size(
            symbol="AAPL",
            current_price=150.0,
            composite_score=75.0,
        )
    
    assert result.symbol == "AAPL"
    assert result.recommended_shares > 0
    assert result.recommended_size > 0
    assert result.conviction_multiplier == 1.0


def test_calculate_position_size_no_cash(mock_trading_client, mock_data_client):
    """Test position size when no cash available."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Create portfolio info with negative cash to pass directly
    # This bypasses the API call entirely
    portfolio_info = {
        "portfolio_value": 100000.0,
        "cash": -5000.0,
        "buying_power": 0.0,
        "position_count": 10,
        "invested_value": 105000.0,
    }
    
    result = sizer.calculate_position_size(
        symbol="AAPL",
        current_price=150.0,
        composite_score=75.0,
        portfolio_info=portfolio_info,
    )
    
    assert result.recommended_shares == 0
    assert result.capped is True
    assert "cash" in result.cap_reason.lower()


def test_calculate_position_size_respects_max(mock_trading_client, mock_data_client):
    """Test that position size respects maximum limit."""
    from src.position_sizer import DynamicPositionSizer
    
    # Set up account with plenty of cash
    mock_account = Mock()
    mock_account.cash = "100000.0"
    mock_account.portfolio_value = "100000.0"
    mock_trading_client.get_account.return_value = mock_account
    mock_trading_client.get_all_positions.return_value = []
    
    sizer = DynamicPositionSizer()
    
    with patch.object(sizer, 'get_volatility', return_value=0.02):
        result = sizer.calculate_position_size(
            symbol="AAPL",
            current_price=10.0,  # Low price to try to get many shares
            composite_score=95.0,  # High score for 1.5x multiplier
        )
    
    # Position should be capped at MAX_POSITION_PCT (10%)
    max_position = 100000.0 * sizer.MAX_POSITION_PCT
    assert result.recommended_size <= max_position


def test_should_deploy_cash(mock_trading_client, mock_data_client):
    """Test cash deployment logic."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # High cash should trigger deployment
    high_cash_info = {"cash": 25000, "portfolio_value": 100000}
    should_deploy, reason = sizer.should_deploy_cash(high_cash_info)
    assert should_deploy is True
    
    # Low cash should not trigger deployment  
    low_cash_info = {"cash": 5000, "portfolio_value": 100000}
    should_deploy, reason = sizer.should_deploy_cash(low_cash_info)
    assert should_deploy is False


def test_record_entry(mock_trading_client, mock_data_client):
    """Test entry recording for daily limits."""
    from src.position_sizer import DynamicPositionSizer
    
    sizer = DynamicPositionSizer()
    
    # Record some entries
    sizer.record_entry("AAPL")
    sizer.record_entry("GOOGL")
    
    today = datetime.now().strftime("%Y-%m-%d")
    assert sizer._daily_entries.get(today, 0) == 2
