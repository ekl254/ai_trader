"""Tests for risk manager."""

import pytest
from unittest.mock import Mock, patch

from src.risk_manager import RiskManager


def test_position_size_calculation() -> None:
    """Test position sizing with 2% risk rule."""
    risk_mgr = RiskManager()
    
    with patch.object(risk_mgr, 'get_account_value', return_value=100000.0):
        with patch.object(risk_mgr, 'get_buying_power', return_value=100000.0):
            result = risk_mgr.calculate_position_size(
                symbol="AAPL",
                entry_price=150.0,
                stop_loss_price=147.0,  # $3 risk per share
            )
            
            # Account: $100k, Risk: 2% = $2000
            # Risk per share: $3
            # Shares: $2000 / $3 = 666 shares
            assert result["shares"] == 666
            assert result["risk_amount"] == pytest.approx(1998.0, rel=0.01)


def test_max_position_size_constraint() -> None:
    """Test that position size doesn't exceed 10% of portfolio."""
    risk_mgr = RiskManager()
    
    with patch.object(risk_mgr, 'get_account_value', return_value=100000.0):
        with patch.object(risk_mgr, 'get_buying_power', return_value=100000.0):
            result = risk_mgr.calculate_position_size(
                symbol="AAPL",
                entry_price=150.0,
                stop_loss_price=149.5,  # Very tight stop = many shares
            )
            
            # Max position: $10k (10% of $100k)
            # Max shares: $10k / $150 = 66 shares
            assert result["shares"] <= 66
            assert result["position_value"] <= 10000.0


def test_validate_trade() -> None:
    """Test trade validation."""
    risk_mgr = RiskManager()
    
    with patch.object(risk_mgr, 'get_account_value', return_value=100000.0):
        with patch.object(risk_mgr, 'get_buying_power', return_value=50000.0):
            # Valid trade
            is_valid, reason = risk_mgr.validate_trade("AAPL", 50, 150.0)
            assert is_valid
            
            # Invalid: too many shares
            is_valid, reason = risk_mgr.validate_trade("AAPL", 1000, 150.0)
            assert not is_valid
            assert "buying power" in reason.lower()
