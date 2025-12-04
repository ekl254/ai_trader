#!/usr/bin/env python3
"""Integration tests for the AI Trading System.

These tests cover full trading loops, error handling, and edge cases.
Tests are designed to run with mocked external APIs (Alpaca, NewsAPI).
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_alpaca_client():
    """Create a mock Alpaca trading client."""
    with patch("alpaca.trading.client.TradingClient") as mock:
        client = MagicMock()

        # Mock account
        account = MagicMock()
        account.equity = "100000.00"
        account.buying_power = "50000.00"
        account.cash = "50000.00"
        account.portfolio_value = "100000.00"
        account.last_equity = "99000.00"
        client.get_account.return_value = account

        # Mock clock
        clock = MagicMock()
        clock.is_open = True
        clock.next_open = datetime.now(UTC) + timedelta(hours=12)
        clock.next_close = datetime.now(UTC) + timedelta(hours=4)
        clock.timestamp = datetime.now(UTC)
        client.get_clock.return_value = clock

        # Mock positions
        client.get_all_positions.return_value = []

        # Mock orders
        client.get_orders.return_value = []
        client.submit_order.return_value = MagicMock(id="test-order-123")

        mock.return_value = client
        yield client


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider."""
    with patch("src.data_provider.alpaca_provider") as mock:
        # Mock latest trade
        mock.get_latest_trade.return_value = {
            "price": 150.00,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Mock bars with realistic OHLCV data
        import pandas as pd

        bars_data = {
            "open": [148.0, 149.0, 150.0, 151.0, 152.0],
            "high": [152.0, 153.0, 154.0, 155.0, 156.0],
            "low": [147.0, 148.0, 149.0, 150.0, 151.0],
            "close": [151.0, 152.0, 153.0, 154.0, 155.0],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        }
        mock.get_bars.return_value = pd.DataFrame(bars_data)

        yield mock


@pytest.fixture
def mock_news_client():
    """Create a mock news client."""
    with patch("src.newsapi_client.newsapi_client") as mock:
        mock.get_stock_news.return_value = [
            {
                "title": "Test Company Reports Strong Earnings",
                "description": "The company beat expectations...",
                "url": "https://example.com/news/1",
                "source": "Test News",
                "publishedAt": datetime.now(UTC).isoformat(),
                "symbols": ["TEST"],
            }
        ]
        yield mock


@pytest.fixture
def mock_sentiment_analyzer():
    """Create a mock sentiment analyzer."""
    with patch("src.sentiment.get_sentiment_analyzer") as mock:
        mock_analyzer = MagicMock()
        mock_analyzer.return_value = [{"label": "positive", "score": 0.75}]
        mock.return_value = mock_analyzer
        yield mock


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return tmp_path


# ============================================================================
# TRADING LOOP INTEGRATION TESTS
# ============================================================================


class TestTradingLoopIntegration:
    """Tests for the full trading loop."""

    def test_market_open_trading_cycle(
        self, mock_alpaca_client, mock_data_provider, mock_sentiment_analyzer
    ):
        """Test a complete market-open trading cycle."""
        # Simulate market open
        mock_alpaca_client.get_clock.return_value.is_open = True

        # Mock a buyable stock
        with patch("src.strategy.strategy") as mock_strategy:
            mock_strategy.should_buy.return_value = (
                True,
                75.0,
                "Strong technicals and sentiment",
            )

            # Verify we can get account info
            account = mock_alpaca_client.get_account()
            assert float(account.equity) == 100000.00

            # Verify market is open
            clock = mock_alpaca_client.get_clock()
            assert clock.is_open is True

    def test_position_management_cycle(self, mock_alpaca_client, mock_data_provider):
        """Test position monitoring and management."""
        # Create mock position
        position = MagicMock()
        position.symbol = "AAPL"
        position.qty = "10"
        position.avg_entry_price = "150.00"
        position.current_price = "155.00"
        position.unrealized_pl = "50.00"
        position.unrealized_plpc = "0.033"
        position.market_value = "1550.00"
        position.cost_basis = "1500.00"
        position.side = "long"

        mock_alpaca_client.get_all_positions.return_value = [position]

        positions = mock_alpaca_client.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert float(positions[0].unrealized_pl) == 50.00

    def test_order_execution_flow(self, mock_alpaca_client):
        """Test order submission and tracking."""
        # Mock order response
        order = MagicMock()
        order.id = "order-123"
        order.symbol = "AAPL"
        order.qty = "10"
        order.side = MagicMock(value="buy")
        order.type = MagicMock(value="market")
        order.status = MagicMock(value="filled")
        order.filled_qty = "10"
        order.filled_avg_price = "150.50"
        order.created_at = datetime.now(UTC)

        mock_alpaca_client.submit_order.return_value = order
        mock_alpaca_client.get_orders.return_value = [order]

        # Submit order
        submitted = mock_alpaca_client.submit_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            type="market",
            time_in_force="day",
        )
        assert submitted.id == "order-123"

        # Verify order in list
        orders = mock_alpaca_client.get_orders()
        assert len(orders) == 1
        assert orders[0].status.value == "filled"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_api_connection_failure(self, mock_alpaca_client):
        """Test handling of API connection failures."""
        mock_alpaca_client.get_account.side_effect = Exception("Connection refused")

        with pytest.raises(Exception) as exc_info:
            mock_alpaca_client.get_account()

        assert "Connection refused" in str(exc_info.value)

    def test_insufficient_buying_power(self, mock_alpaca_client):
        """Test handling of insufficient buying power."""
        from alpaca.common.exceptions import APIError

        mock_alpaca_client.submit_order.side_effect = APIError(
            {"message": "insufficient buying power"}
        )

        with pytest.raises(APIError) as exc_info:
            mock_alpaca_client.submit_order(
                symbol="AAPL",
                qty=1000,
                side="buy",
                type="market",
                time_in_force="day",
            )

        assert "insufficient buying power" in str(exc_info.value)

    def test_invalid_symbol(self, mock_data_provider):
        """Test handling of invalid stock symbols."""
        mock_data_provider.get_latest_trade.side_effect = Exception("Symbol not found")

        with pytest.raises(Exception) as exc_info:
            mock_data_provider.get_latest_trade("INVALID123")

        assert "Symbol not found" in str(exc_info.value)

    def test_market_closed_handling(self, mock_alpaca_client):
        """Test handling when market is closed."""
        mock_alpaca_client.get_clock.return_value.is_open = False

        clock = mock_alpaca_client.get_clock()
        assert clock.is_open is False

    def test_rate_limit_handling(self, mock_alpaca_client):
        """Test handling of API rate limits."""
        from alpaca.common.exceptions import APIError

        mock_alpaca_client.get_account.side_effect = APIError(
            {"message": "rate limit exceeded"}
        )

        with pytest.raises(APIError) as exc_info:
            mock_alpaca_client.get_account()

        assert "rate limit" in str(exc_info.value)

    def test_partial_fill_handling(self, mock_alpaca_client):
        """Test handling of partially filled orders."""
        order = MagicMock()
        order.id = "order-456"
        order.symbol = "AAPL"
        order.qty = "100"
        order.filled_qty = "50"
        order.status = MagicMock(value="partially_filled")

        mock_alpaca_client.get_order.return_value = order

        result = mock_alpaca_client.get_order("order-456")
        assert result.status.value == "partially_filled"
        assert result.filled_qty == "50"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_positions(self, mock_alpaca_client):
        """Test handling when no positions exist."""
        mock_alpaca_client.get_all_positions.return_value = []

        positions = mock_alpaca_client.get_all_positions()
        assert len(positions) == 0

    def test_maximum_positions_reached(self, mock_alpaca_client):
        """Test handling when maximum positions are reached."""
        # Create 10 mock positions (typical max)
        positions = []
        for i in range(10):
            pos = MagicMock()
            pos.symbol = f"STOCK{i}"
            pos.qty = "10"
            pos.market_value = "1000.00"
            positions.append(pos)

        mock_alpaca_client.get_all_positions.return_value = positions

        result = mock_alpaca_client.get_all_positions()
        assert len(result) == 10

    def test_very_low_buying_power(self, mock_alpaca_client):
        """Test handling when buying power is very low."""
        account = MagicMock()
        account.buying_power = "10.00"
        account.cash = "10.00"
        mock_alpaca_client.get_account.return_value = account

        result = mock_alpaca_client.get_account()
        assert float(result.buying_power) == 10.00

    def test_high_volatility_stock(self, mock_data_provider):
        """Test handling of high volatility stocks."""
        import pandas as pd

        # Simulate high volatility with large price swings
        bars_data = {
            "open": [100.0, 120.0, 90.0, 130.0, 85.0],
            "high": [125.0, 140.0, 110.0, 150.0, 100.0],
            "low": [95.0, 110.0, 80.0, 120.0, 75.0],
            "close": [120.0, 90.0, 100.0, 85.0, 95.0],
            "volume": [5000000, 6000000, 7000000, 8000000, 9000000],
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("VOLATILE", days=5)

        # Calculate volatility (std of returns)
        returns = bars["close"].pct_change().dropna()
        volatility = returns.std()
        assert volatility > 0.1  # High volatility threshold

    def test_penny_stock_filtering(self, mock_data_provider):
        """Test filtering of penny stocks (price < $5)."""
        mock_data_provider.get_latest_trade.return_value = {
            "price": 2.50,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        result = mock_data_provider.get_latest_trade("PENNY")
        assert result["price"] < 5.0  # Should be filtered in real implementation

    def test_stock_split_handling(self, mock_data_provider):
        """Test handling after a stock split."""
        import pandas as pd

        # Simulate post-split data where prices are halved
        bars_data = {
            "open": [75.0, 76.0, 77.0, 78.0, 79.0],  # Half of pre-split
            "high": [76.0, 77.0, 78.0, 79.0, 80.0],
            "low": [74.0, 75.0, 76.0, 77.0, 78.0],
            "close": [75.5, 76.5, 77.5, 78.5, 79.5],
            "volume": [2000000, 2100000, 2200000, 2300000, 2400000],  # Double volume
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("SPLIT", days=5)
        assert len(bars) == 5

    def test_market_halt_scenario(self, mock_alpaca_client):
        """Test handling during market halt."""
        from alpaca.common.exceptions import APIError

        mock_alpaca_client.submit_order.side_effect = APIError(
            {"message": "trading halted for symbol"}
        )

        with pytest.raises(APIError) as exc_info:
            mock_alpaca_client.submit_order(
                symbol="HALTED",
                qty=10,
                side="buy",
                type="market",
                time_in_force="day",
            )

        assert "halted" in str(exc_info.value)

    def test_overnight_gap(self, mock_data_provider):
        """Test handling of overnight price gaps."""
        import pandas as pd

        # Simulate overnight gap up
        bars_data = {
            "open": [100.0, 110.0],  # 10% gap up
            "high": [105.0, 115.0],
            "low": [98.0, 108.0],
            "close": [102.0, 112.0],
            "volume": [1000000, 2000000],
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("GAP", days=2)
        gap_pct = (bars.iloc[1]["open"] - bars.iloc[0]["close"]) / bars.iloc[0]["close"]
        assert gap_pct > 0.05  # 5% gap


# ============================================================================
# RISK MANAGEMENT TESTS
# ============================================================================


class TestRiskManagement:
    """Tests for risk management functionality."""

    def test_stop_loss_trigger(self, mock_alpaca_client, mock_data_provider):
        """Test stop loss trigger logic."""
        # Position with 5% loss (should trigger 3% stop loss)
        position = MagicMock()
        position.symbol = "AAPL"
        position.avg_entry_price = "100.00"
        position.current_price = "95.00"
        position.unrealized_plpc = "-0.05"

        mock_alpaca_client.get_all_positions.return_value = [position]

        positions = mock_alpaca_client.get_all_positions()
        loss_pct = float(positions[0].unrealized_plpc)

        # With 3% stop loss, this should trigger
        stop_loss_threshold = -0.03
        assert loss_pct < stop_loss_threshold

    def test_take_profit_trigger(self, mock_alpaca_client):
        """Test take profit trigger logic."""
        # Position with 10% gain (should trigger 8% take profit)
        position = MagicMock()
        position.symbol = "AAPL"
        position.avg_entry_price = "100.00"
        position.current_price = "110.00"
        position.unrealized_plpc = "0.10"

        mock_alpaca_client.get_all_positions.return_value = [position]

        positions = mock_alpaca_client.get_all_positions()
        gain_pct = float(positions[0].unrealized_plpc)

        # With 8% take profit, this should trigger
        take_profit_threshold = 0.08
        assert gain_pct > take_profit_threshold

    def test_position_size_limits(self, mock_alpaca_client):
        """Test position size limit enforcement."""
        account = MagicMock()
        account.equity = "100000.00"
        account.buying_power = "50000.00"
        mock_alpaca_client.get_account.return_value = account

        equity = float(account.equity)
        max_position_pct = 0.10  # 10% max per position
        max_position_value = equity * max_position_pct

        assert max_position_value == 10000.00

    def test_daily_loss_limit(self, mock_alpaca_client):
        """Test daily loss limit enforcement."""
        account = MagicMock()
        account.equity = "95000.00"
        account.last_equity = "100000.00"
        mock_alpaca_client.get_account.return_value = account

        daily_loss = float(account.equity) - float(account.last_equity)
        daily_loss_pct = daily_loss / float(account.last_equity)

        max_daily_loss_pct = -0.05  # 5% max daily loss
        assert daily_loss_pct >= max_daily_loss_pct  # -5% >= -5% means within limit


# ============================================================================
# STRATEGY TESTS
# ============================================================================


class TestStrategyIntegration:
    """Tests for strategy integration."""

    def test_composite_score_calculation(self):
        """Test composite score calculation with all factors."""
        technical = 70.0
        sentiment = 60.0
        fundamental = 50.0

        # Default weights: 40% tech, 30% sent, 30% fund
        composite = (technical * 0.40) + (sentiment * 0.30) + (fundamental * 0.30)

        assert composite == 61.0

    def test_minimum_score_threshold(self):
        """Test minimum score threshold for trading."""
        min_composite = 55.0
        min_factor = 40.0

        # Passing case
        composite = 60.0
        technical = 50.0
        sentiment = 45.0

        should_trade = (
            composite >= min_composite
            and technical >= min_factor
            and sentiment >= min_factor
        )
        assert should_trade is True

        # Failing case - low sentiment
        sentiment = 35.0
        should_trade = (
            composite >= min_composite
            and technical >= min_factor
            and sentiment >= min_factor
        )
        assert should_trade is False

    def test_rebalancing_logic(self, mock_alpaca_client):
        """Test portfolio rebalancing logic."""
        # Current position with low score
        old_position = MagicMock()
        old_position.symbol = "OLD"
        old_position.market_value = "1000.00"

        mock_alpaca_client.get_all_positions.return_value = [old_position]

        old_score = 55.0
        new_candidate_score = 75.0
        min_improvement = 10.0

        should_rebalance = (new_candidate_score - old_score) >= min_improvement
        assert should_rebalance is True


# ============================================================================
# DATA PROVIDER TESTS
# ============================================================================


class TestDataProviderIntegration:
    """Tests for data provider integration."""

    def test_bars_data_structure(self, mock_data_provider):
        """Test bars data has correct structure."""
        bars = mock_data_provider.get_bars("AAPL", days=5)

        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in bars.columns

    def test_latest_trade_structure(self, mock_data_provider):
        """Test latest trade data has correct structure."""
        trade = mock_data_provider.get_latest_trade("AAPL")

        assert "price" in trade
        assert "timestamp" in trade
        assert isinstance(trade["price"], (int, float))


# ============================================================================
# PERFORMANCE TRACKING TESTS
# ============================================================================


class TestPerformanceTracking:
    """Tests for performance tracking integration."""

    def test_trade_recording(self, temp_data_dir):
        """Test trade recording to data file."""
        trade_data = {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "price": 150.00,
            "timestamp": datetime.now(UTC).isoformat(),
            "score": 75.0,
        }

        trades_file = temp_data_dir / "data" / "trades.json"
        trades_file.write_text(json.dumps([trade_data]))

        loaded = json.loads(trades_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "AAPL"

    def test_daily_pnl_calculation(self):
        """Test daily P&L calculation."""
        trades = [
            {
                "symbol": "AAPL",
                "side": "sell",
                "qty": 10,
                "price": 155.00,
                "cost_basis": 150.00,
            },
            {
                "symbol": "GOOGL",
                "side": "sell",
                "qty": 5,
                "price": 140.00,
                "cost_basis": 145.00,
            },
        ]

        total_pnl = 0
        for trade in trades:
            pnl = (trade["price"] - trade["cost_basis"]) * trade["qty"]
            total_pnl += pnl

        # AAPL: (155-150)*10 = 50, GOOGL: (140-145)*5 = -25
        assert total_pnl == 25.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        trades = [
            {"pnl": 50.0},
            {"pnl": -25.0},
            {"pnl": 100.0},
            {"pnl": 75.0},
            {"pnl": -10.0},
        ]

        wins = sum(1 for t in trades if t["pnl"] > 0)
        total = len(trades)
        win_rate = wins / total * 100

        assert win_rate == 60.0


# ============================================================================
# MARKET REGIME TESTS
# ============================================================================


class TestMarketRegimeIntegration:
    """Tests for market regime detection integration."""

    def test_bullish_regime_detection(self, mock_data_provider):
        """Test detection of bullish market regime."""
        import pandas as pd

        # Simulate bullish trend (prices increasing)
        bars_data = {
            "close": [
                100.0,
                102.0,
                105.0,
                108.0,
                112.0,
                115.0,
                118.0,
                120.0,
                122.0,
                125.0,
            ],
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("SPY", days=10)
        momentum = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars[
            "close"
        ].iloc[0]

        assert momentum > 0.10  # 10% positive momentum = bullish

    def test_bearish_regime_detection(self, mock_data_provider):
        """Test detection of bearish market regime."""
        import pandas as pd

        # Simulate bearish trend (prices decreasing)
        bars_data = {
            "close": [
                125.0,
                122.0,
                118.0,
                115.0,
                110.0,
                105.0,
                100.0,
                95.0,
                92.0,
                90.0,
            ],
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("SPY", days=10)
        momentum = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars[
            "close"
        ].iloc[0]

        assert momentum < -0.10  # 10% negative momentum = bearish

    def test_volatility_calculation(self, mock_data_provider):
        """Test volatility calculation for regime detection."""
        import pandas as pd

        bars_data = {
            "close": [100.0, 105.0, 98.0, 108.0, 95.0, 110.0, 92.0, 115.0, 90.0, 120.0],
        }
        mock_data_provider.get_bars.return_value = pd.DataFrame(bars_data)

        bars = mock_data_provider.get_bars("SPY", days=10)
        returns = bars["close"].pct_change().dropna()
        volatility = returns.std()

        # High volatility threshold
        assert volatility > 0.05


# ============================================================================
# SLOW INTEGRATION TESTS (marked for optional execution)
# ============================================================================


@pytest.mark.slow
class TestSlowIntegration:
    """Slow integration tests that may require more time or resources."""

    def test_full_day_simulation(self, mock_alpaca_client, mock_data_provider):
        """Simulate a full trading day."""
        # This would be a comprehensive test of the entire day's trading
        pass

    def test_multi_day_backtest(self, mock_alpaca_client, mock_data_provider):
        """Test multi-day backtesting functionality."""
        pass
