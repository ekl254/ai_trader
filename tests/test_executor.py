"""Tests for trade executor."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from alpaca.trading.enums import OrderSide


@pytest.fixture
def mock_trading_client():
    """Mock Alpaca trading client via shared factory."""
    with patch("src.executor.get_trading_client") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager."""
    with patch("src.executor.risk_manager") as mock:
        mock.can_open_position.return_value = True
        mock.calculate_position_size.return_value = {
            "shares": 10,
            "position_value": 1500.0,
            "stop_loss_price": 147.0,
            "risk_amount": 30.0,
        }
        mock.validate_trade.return_value = (True, "Trade validated")
        mock.get_current_positions.return_value = {"AAPL": 10}
        # Add return values for get_stop_loss_pct and get_take_profit_pct
        mock.get_stop_loss_pct.return_value = 0.02
        mock.get_take_profit_pct.return_value = 0.06
        yield mock


@pytest.fixture
def mock_position_tracker():
    """Mock position tracker."""
    with patch("src.executor.position_tracker") as mock:
        mock.data = {"positions": {}}
        yield mock


@pytest.fixture
def mock_performance_tracker():
    """Mock performance tracker."""
    with patch("src.executor.PerformanceTracker") as mock:
        yield mock


@pytest.fixture
def mock_position_sizer():
    """Mock position sizer."""
    with patch("src.executor.position_sizer") as mock:
        # Create a mock PositionSizeResult
        mock_result = Mock()
        mock_result.recommended_shares = 10
        mock_result.recommended_size = 1500.0
        mock_result.conviction_multiplier = 1.0
        mock_result.volatility_multiplier = 1.0
        mock_result.rationale = ["Test rationale"]
        mock.calculate_position_size.return_value = mock_result
        yield mock


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker."""
    with patch("src.executor.trading_circuit_breaker") as mock:
        mock.can_proceed.return_value = True
        mock.record_success.return_value = None
        mock.record_failure.return_value = None
        mock.get_status.return_value = {"state": "closed", "failures": 0}
        yield mock


def test_place_market_order_success(
    mock_trading_client: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test successful market order placement."""
    from src.executor import TradeExecutor

    mock_order = Mock()
    mock_order.id = "order123"
    mock_trading_client.submit_order.return_value = mock_order

    executor = TradeExecutor()
    order_id = executor.place_market_order("AAPL", 10, OrderSide.BUY)

    assert order_id == "order123"
    mock_trading_client.submit_order.assert_called_once()


def test_place_market_order_failure(
    mock_trading_client: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test market order failure handling."""
    from src.executor import TradeExecutor

    mock_trading_client.submit_order.side_effect = Exception("API Error")

    executor = TradeExecutor()
    order_id = executor.place_market_order("AAPL", 10, OrderSide.BUY)

    assert order_id is None


def test_buy_stock_success(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_position_sizer: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test successful stock purchase."""
    from alpaca.trading.enums import OrderStatus

    from src.executor import TradeExecutor

    mock_order = Mock()
    mock_order.id = "order123"
    mock_order.symbol = "AAPL"
    mock_order.filled_avg_price = "150.0"
    mock_order.filled_qty = "10"
    mock_order.status = OrderStatus.FILLED
    mock_trading_client.submit_order.return_value = mock_order
    mock_trading_client.get_order_by_id.return_value = mock_order

    executor = TradeExecutor()
    result = executor.buy_stock(
        symbol="AAPL",
        score=75.0,
        reasoning={"technical": {"total": 80}},
        current_price=150.0,
    )

    assert result is True
    mock_risk_manager.can_open_position.assert_called_once_with("AAPL", score=75.0)
    mock_position_sizer.calculate_position_size.assert_called_once()
    mock_position_tracker.track_entry.assert_called_once()


def test_buy_stock_cannot_open_position(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test buy fails when position cannot be opened."""
    from src.executor import TradeExecutor

    mock_risk_manager.can_open_position.return_value = False

    executor = TradeExecutor()
    result = executor.buy_stock(
        symbol="AAPL",
        score=75.0,
        reasoning={},
        current_price=150.0,
    )

    assert result is False


def test_sell_stock_success(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test successful stock sale."""
    from src.executor import TradeExecutor

    mock_order = Mock()
    mock_order.id = "order456"
    mock_trading_client.submit_order.return_value = mock_order

    executor = TradeExecutor()
    result = executor.sell_stock("AAPL", "stop_loss")

    assert result is True
    mock_position_tracker.track_exit.assert_called_once_with(
        symbol="AAPL", exit_reason="stop_loss"
    )


def test_sell_stock_no_position(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test sell fails when no position exists."""
    from src.executor import TradeExecutor

    mock_risk_manager.get_current_positions.return_value = {}  # No positions

    executor = TradeExecutor()
    result = executor.sell_stock("GOOGL", "stop_loss")

    assert result is False


def test_sell_stock_partial(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test partial stock sale."""
    from src.executor import TradeExecutor

    mock_order = Mock()
    mock_order.id = "order789"
    mock_trading_client.submit_order.return_value = mock_order

    executor = TradeExecutor()
    result = executor.sell_stock("AAPL", "rebalancing", sell_pct=0.5)

    assert result is True
    # Partial sell should NOT call track_exit
    mock_position_tracker.track_exit.assert_not_called()


def test_manage_stop_losses_triggers_stop(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test stop loss triggers when loss exceeds threshold."""
    from src.executor import TradeExecutor

    mock_position = Mock()
    mock_position.symbol = "AAPL"
    mock_position.current_price = 145.0
    mock_position.avg_entry_price = 150.0
    mock_position.unrealized_plpc = -0.035  # 3.5% loss, exceeds 2% stop

    mock_trading_client.get_all_positions.return_value = [mock_position]

    mock_order = Mock()
    mock_order.id = "stop_order"
    mock_trading_client.submit_order.return_value = mock_order

    executor = TradeExecutor()

    with patch.object(executor, "sell_stock") as mock_sell:
        executor.manage_stop_losses()
        mock_sell.assert_called_once_with("AAPL", "stop_loss")


def test_manage_stop_losses_triggers_take_profit(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test take profit triggers when gain exceeds threshold."""
    from src.executor import TradeExecutor

    mock_position = Mock()
    mock_position.symbol = "AAPL"
    mock_position.current_price = 160.0
    mock_position.avg_entry_price = 150.0
    mock_position.unrealized_plpc = 0.07  # 7% gain, exceeds 6% take profit

    mock_trading_client.get_all_positions.return_value = [mock_position]

    executor = TradeExecutor()

    with patch.object(executor, "sell_stock") as mock_sell:
        executor.manage_stop_losses()
        mock_sell.assert_called_once_with("AAPL", "take_profit")


def test_close_all_positions(
    mock_trading_client: Mock,
    mock_risk_manager: Mock,
    mock_position_tracker: Mock,
    mock_performance_tracker: Mock,
    mock_circuit_breaker: Mock,
) -> None:
    """Test closing all positions."""
    from src.executor import TradeExecutor

    mock_risk_manager.get_current_positions.return_value = {
        "AAPL": 10,
        "GOOGL": 5,
    }

    executor = TradeExecutor()

    with patch.object(executor, "sell_stock") as mock_sell:
        executor.close_all_positions()

        assert mock_sell.call_count == 2
        mock_sell.assert_any_call("AAPL", "end_of_day_close")
        mock_sell.assert_any_call("GOOGL", "end_of_day_close")
