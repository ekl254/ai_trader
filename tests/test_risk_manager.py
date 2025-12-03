"""Tests for risk manager."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.risk_manager import RiskManager


@pytest.fixture
def mock_risk_mgr():
    """Create a risk manager with mocked Alpaca client."""
    with patch('src.risk_manager.get_trading_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Default healthy account
        mock_account = Mock()
        mock_account.cash = "50000.0"
        mock_account.portfolio_value = "100000.0"
        mock_account.buying_power = "100000.0"
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = []
        
        risk_mgr = RiskManager()
        yield risk_mgr, mock_client, mock_account


def test_position_size_calculation(mock_risk_mgr) -> None:
    """Test position sizing with 2% risk rule."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    # Set up account with enough cash
    mock_account.cash = "100000.0"
    mock_account.portfolio_value = "100000.0"
    mock_account.buying_power = "100000.0"
    
    result = risk_mgr.calculate_position_size(
        symbol="AAPL",
        entry_price=150.0,
        stop_loss_price=147.0,  # $3 risk per share
    )
    
    # Account: $100k, Risk: 2% = $2000
    # Risk per share: $3
    # Max shares from risk: $2000 / $3 = 666 shares
    # But max position is 10-12% = $10-12k = 66-80 shares (depends on config)
    # So should be capped at max_position_size
    max_position_value = 100000.0 * risk_mgr.config.max_position_size
    max_shares = int(max_position_value / 150.0)
    assert result["shares"] <= max_shares
    assert result["position_value"] <= max_position_value


def test_max_position_size_constraint(mock_risk_mgr) -> None:
    """Test that position size doesn't exceed max_position_size of portfolio."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    mock_account.cash = "100000.0"
    mock_account.portfolio_value = "100000.0"
    mock_account.buying_power = "100000.0"
    
    result = risk_mgr.calculate_position_size(
        symbol="AAPL",
        entry_price=150.0,
        stop_loss_price=149.5,  # Very tight stop = many shares
    )
    
    # Max position should be capped by config.max_position_size
    max_position_value = 100000.0 * risk_mgr.config.max_position_size
    max_shares = int(max_position_value / 150.0)
    assert result["shares"] <= max_shares
    assert result["position_value"] <= max_position_value


def test_validate_trade(mock_risk_mgr) -> None:
    """Test trade validation checks cash (not buying power) to prevent margin."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    # Account with $50k cash
    mock_account.cash = "50000.0"
    mock_account.portfolio_value = "100000.0"
    mock_account.buying_power = "100000.0"
    
    # Valid trade: 50 shares * $150 = $7,500 < available cash
    is_valid, reason = risk_mgr.validate_trade("AAPL", 50, 150.0)
    assert is_valid, f"Should be valid: {reason}"
    
    # Invalid: too many shares - 1000 * $150 = $150k > $50k cash
    is_valid, reason = risk_mgr.validate_trade("AAPL", 1000, 150.0)
    assert not is_valid
    # Could fail for multiple reasons: position size %, or cash
    assert "position" in reason.lower() or "cash" in reason.lower()


def test_validate_trade_blocks_when_low_cash(mock_risk_mgr) -> None:
    """Test that validate_trade blocks trades when cash is below 5% buffer."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    # Cash at only 3% of portfolio - below 5% buffer
    mock_account.cash = "3000.0"
    mock_account.portfolio_value = "100000.0"
    mock_account.buying_power = "100000.0"
    
    is_valid, reason = risk_mgr.validate_trade("AAPL", 10, 150.0)
    assert not is_valid
    assert "cash" in reason.lower() or "buffer" in reason.lower()


def test_check_account_health(mock_risk_mgr) -> None:
    """Test account health check with healthy account."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    mock_account.cash = "50000.0"
    mock_account.portfolio_value = "100000.0"
    mock_client.get_all_positions.return_value = [Mock() for _ in range(5)]
    
    health = risk_mgr.check_account_health()
    assert health["healthy"]
    assert health["cash"] == 50000.0
    assert health["cash_pct"] == 0.5


def test_check_account_health_unhealthy(mock_risk_mgr) -> None:
    """Test account health check with negative cash (margin usage)."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    # Negative cash = margin usage
    mock_account.cash = "-5000.0"
    mock_account.portfolio_value = "100000.0"
    mock_client.get_all_positions.return_value = [Mock() for _ in range(10)]
    
    health = risk_mgr.check_account_health()
    assert not health["healthy"]
    assert health["cash"] == -5000.0
    assert len(health["warnings"]) > 0


def test_should_block_new_buys(mock_risk_mgr) -> None:
    """Test that new buys are blocked when account is unhealthy."""
    risk_mgr, mock_client, mock_account = mock_risk_mgr
    
    # Negative cash should block new buys
    mock_account.cash = "-5000.0"
    mock_account.portfolio_value = "100000.0"
    
    should_block, reason = risk_mgr.should_block_new_buys()
    assert should_block
    assert "cash" in reason.lower() or "negative" in reason.lower()
