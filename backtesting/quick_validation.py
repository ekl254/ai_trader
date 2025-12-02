#!/usr/bin/env python3
"""
Quick Validation Backtest

Tests if the current technical strategy has any edge over buy-and-hold SPY.
Uses 2022-2024 data (includes bear and bull markets).
No optimization - just validates the current logic works.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

import pandas as pd
import numpy as np
import pandas_ta as ta

# Add parent to path
sys.path.insert(0, '/root/ai_trader')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ''


@dataclass 
class Portfolio:
    cash: float = 100000.0
    positions: Dict[str, dict] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            pos['shares'] * prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value


class TechnicalScorer:
    """Calculates technical score - mirrors live strategy but without API calls."""
    
    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate technical score from OHLCV dataframe."""
        if len(df) < 50:
            return 0.0, {'error': 'insufficient_data'}
        
        df = df.copy()
        close = df['close']
        
        # RSI
        df['RSI'] = ta.rsi(close, length=self.rsi_period)
        
        # MACD
        macd = ta.macd(close)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
        
        # Bollinger Bands
        bbands = ta.bbands(close)
        if bbands is not None:
            bb_cols = bbands.columns.tolist()
            for col in bb_cols:
                if col.startswith('BBU_'):
                    df['BB_upper'] = bbands[col]
                elif col.startswith('BBL_'):
                    df['BB_lower'] = bbands[col]
        
        # EMAs
        df['EMA_20'] = ta.ema(close, length=20)
        df['EMA_50'] = ta.ema(close, length=50)
        
        latest = df.iloc[-1]
        
        # RSI Score
        rsi_value = latest.get('RSI', 50)
        if pd.isna(rsi_value):
            rsi_score = 50.0
        elif rsi_value < self.rsi_oversold:
            rsi_score = 100.0
        elif rsi_value > self.rsi_overbought:
            rsi_score = 0.0
        else:
            rsi_score = 100 - abs(rsi_value - 50) * 2
        
        # MACD Score
        macd_val = latest.get('MACD')
        macd_sig = latest.get('MACD_signal')
        if pd.isna(macd_val) or pd.isna(macd_sig):
            macd_score = 50.0
        else:
            histogram = macd_val - macd_sig
            if histogram > 0:
                macd_score = min(100, 50 + histogram * 25)
            else:
                macd_score = max(0, 50 + histogram * 25)
        
        # Bollinger Score
        bb_upper = latest.get('BB_upper')
        bb_lower = latest.get('BB_lower')
        current_close = latest['close']
        
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            bb_score = 50.0
        elif current_close < bb_lower:
            bb_score = 100.0
        elif current_close > bb_upper:
            bb_score = 0.0
        else:
            bb_range = bb_upper - bb_lower
            position = (current_close - bb_lower) / bb_range if bb_range > 0 else 0.5
            bb_score = (1 - abs(position - 0.5) * 2) * 100
        
        # Volume Score
        avg_volume = df['volume'].tail(20).mean()
        current_volume = latest['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        prev_close = df['close'].iloc[-2]
        price_change = (current_close - prev_close) / prev_close if prev_close > 0 else 0
        
        if price_change > 0:
            volume_score = min(100, 50 + (volume_ratio - 1) * 30)
        elif price_change < -0.005:
            volume_score = max(0, 50 - (volume_ratio - 1) * 30)
        else:
            volume_score = 50.0
        
        # Trend Score
        ema_20 = latest.get('EMA_20')
        ema_50 = latest.get('EMA_50')
        
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            if current_close > ema_20 > ema_50:
                trend_score = 80.0
            elif current_close > ema_20 and current_close > ema_50:
                trend_score = 65.0
            elif current_close < ema_20 < ema_50:
                trend_score = 20.0
            elif current_close < ema_20 and current_close < ema_50:
                trend_score = 35.0
            else:
                trend_score = 50.0
        else:
            trend_score = 50.0
        
        # Weighted composite
        technical_score = (
            rsi_score * 0.25 +
            macd_score * 0.25 +
            bb_score * 0.20 +
            volume_score * 0.15 +
            trend_score * 0.15
        )
        
        return technical_score, {
            'rsi': rsi_value,
            'rsi_score': rsi_score,
            'macd_score': macd_score,
            'bb_score': bb_score,
            'volume_score': volume_score,
            'trend_score': trend_score,
        }


class QuickBacktest:
    """Simple backtest engine for validation."""
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        min_score: float = 65.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.06,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.min_score = min_score
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.client = StockHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY')
        )
        self.scorer = TechnicalScorer()
        self.portfolio = Portfolio(cash=initial_capital)
        
        # Cache for historical data
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, symbols: List[str]) -> None:
        """Load historical data for all symbols."""
        print(f"Loading data for {len(symbols)} symbols...")
        
        # Load in batches to avoid API limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            print(f"  Loading batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}...")
            
            # Need extra days for indicator calculation
            lookback_start = self.start_date - timedelta(days=100)
            
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=lookback_start,
                end=self.end_date,
            )
            
            try:
                bars = self.client.get_stock_bars(request)
                df = bars.df
                
                for symbol in batch:
                    if symbol in df.index.get_level_values(0):
                        symbol_df = df.loc[symbol].copy()
                        symbol_df = symbol_df.reset_index()
                        symbol_df = symbol_df.rename(columns={'timestamp': 'date'})
                        symbol_df = symbol_df.set_index('date')
                        self.data_cache[symbol] = symbol_df
            except Exception as e:
                print(f"  Error loading batch: {e}")
        
        print(f"Loaded data for {len(self.data_cache)} symbols")
    
    def get_data_for_date(self, symbol: str, date: datetime, lookback: int = 60) -> Optional[pd.DataFrame]:
        """Get historical data up to (not including) a specific date."""
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        # Get data up to this date
        mask = df.index < date
        subset = df[mask].tail(lookback)
        
        if len(subset) < 50:
            return None
        
        return subset
    
    def get_price_on_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get opening price on a specific date (for order execution)."""
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        # Find the exact date or next available
        mask = df.index >= date
        if not mask.any():
            return None
        
        return df[mask].iloc[0]['open']
    
    def get_close_on_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get closing price on a specific date."""
        if symbol not in self.data_cache:
            return None
        
        df = self.data_cache[symbol]
        mask = df.index.date == date.date()
        if not mask.any():
            return None
        
        return df[mask].iloc[0]['close']
    
    def run(self, symbols: List[str]) -> Dict:
        """Run the backtest."""
        self.load_data(symbols)
        
        # Get trading days
        spy_data = self.data_cache.get('SPY')
        if spy_data is None:
            raise ValueError("SPY data required for trading calendar")
        
        trading_days = spy_data[
            (spy_data.index >= self.start_date) & 
            (spy_data.index <= self.end_date)
        ].index.tolist()
        
        print(f"\nRunning backtest over {len(trading_days)} trading days...")
        print(f"Strategy: min_score={self.min_score}, stop={self.stop_loss_pct:.1%}, tp={self.take_profit_pct:.1%}")
        
        for i, date in enumerate(trading_days):
            if i % 50 == 0:
                print(f"  Day {i}/{len(trading_days)} - {date.strftime('%Y-%m-%d')} - Positions: {len(self.portfolio.positions)}")
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol in symbols:
                price = self.get_close_on_date(symbol, date)
                if price:
                    current_prices[symbol] = price
            
            # 1. Check exits for existing positions
            self._check_exits(date, current_prices)
            
            # 2. Score all symbols and find candidates
            if len(self.portfolio.positions) < self.max_positions:
                candidates = []
                for symbol in symbols:
                    if symbol in self.portfolio.positions:
                        continue
                    
                    df = self.get_data_for_date(symbol, date)
                    if df is None:
                        continue
                    
                    score, details = self.scorer.calculate_score(df)
                    
                    if score >= self.min_score:
                        candidates.append({
                            'symbol': symbol,
                            'score': score,
                            'price': current_prices.get(symbol),
                        })
                
                # Sort by score and buy top candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                slots_available = self.max_positions - len(self.portfolio.positions)
                for candidate in candidates[:slots_available]:
                    if candidate['price']:
                        self._enter_position(candidate['symbol'], date, candidate['price'], candidate['score'])
            
            # 3. Record equity
            equity = self.portfolio.get_equity(current_prices)
            self.portfolio.equity_curve.append({
                'date': date.isoformat(),
                'equity': equity,
                'positions': len(self.portfolio.positions),
                'cash': self.portfolio.cash,
            })
        
        return self._calculate_results(trading_days)
    
    def _enter_position(self, symbol: str, date: datetime, price: float, score: float) -> None:
        """Enter a new position."""
        # Position sizing: equal weight
        position_value = self.portfolio.cash / (self.max_positions - len(self.portfolio.positions) + 1)
        position_value = min(position_value, self.portfolio.cash * 0.95)  # Keep some cash buffer
        
        shares = int(position_value / price)
        if shares <= 0:
            return
        
        cost = shares * price * 1.0005  # 0.05% slippage
        
        if cost > self.portfolio.cash:
            return
        
        self.portfolio.cash -= cost
        self.portfolio.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'score': score,
            'stop_loss': price * (1 - self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct),
        }
    
    def _check_exits(self, date: datetime, prices: Dict[str, float]) -> None:
        """Check and execute exits."""
        to_exit = []
        
        for symbol, pos in self.portfolio.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            
            exit_reason = None
            
            if price <= pos['stop_loss']:
                exit_reason = 'stop_loss'
            elif price >= pos['take_profit']:
                exit_reason = 'take_profit'
            
            if exit_reason:
                to_exit.append((symbol, price, exit_reason))
        
        for symbol, price, reason in to_exit:
            self._exit_position(symbol, date, price, reason)
    
    def _exit_position(self, symbol: str, date: datetime, price: float, reason: str) -> None:
        """Exit a position."""
        pos = self.portfolio.positions.pop(symbol)
        
        proceeds = pos['shares'] * price * 0.9995  # 0.05% slippage
        self.portfolio.cash += proceeds
        
        pnl = proceeds - (pos['shares'] * pos['entry_price'])
        pnl_pct = (price / pos['entry_price'] - 1) * 100
        
        trade = Trade(
            symbol=symbol,
            entry_date=pos['entry_date'],
            entry_price=pos['entry_price'],
            exit_date=date,
            exit_price=price,
            shares=pos['shares'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.portfolio.trades.append(trade)
    
    def _calculate_results(self, trading_days: List) -> Dict:
        """Calculate backtest metrics."""
        # Close any remaining positions at last price
        if self.portfolio.positions:
            last_date = trading_days[-1]
            for symbol in list(self.portfolio.positions.keys()):
                price = self.get_close_on_date(symbol, last_date)
                if price:
                    self._exit_position(symbol, last_date, price, 'end_of_backtest')
        
        trades = self.portfolio.trades
        equity_curve = pd.DataFrame(self.portfolio.equity_curve)
        
        if len(trades) == 0:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(winners) / len(trades) * 100
        
        avg_winner = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl_pct for t in losers]) if losers else 0
        
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Equity curve metrics
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak'] * 100
        max_drawdown = equity_curve['drawdown'].min()
        
        # Annualized metrics
        days = (trading_days[-1] - trading_days[0]).days
        years = days / 365.25
        cagr = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()
            sharpe = equity_curve['daily_return'].mean() / equity_curve['daily_return'].std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        return {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': round(win_rate, 1),
            'avg_winner_pct': round(avg_winner, 2),
            'avg_loser_pct': round(avg_loser, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return, 2),
            'cagr_pct': round(cagr, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'final_equity': round(final_equity, 2),
            'exit_reasons': exit_reasons,
        }


def run_validation():
    """Run quick validation backtest."""
    
    # Use a representative sample of S&P 500 stocks
    # Mix of sectors and market caps
    symbols = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'CRM', 'ADBE', 'INTC',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'C',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
        # Industrial
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'UNP',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP',
        # SPY for benchmark
        'SPY',
    ]
    
    # Test period: 2022-2024 (includes bear market and recovery)
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 11, 30, tzinfo=timezone.utc)
    
    print("="*60)
    print("QUICK VALIDATION BACKTEST")
    print("="*60)
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Universe: {len(symbols)} stocks")
    print(f"Initial Capital: $100,000")
    print("="*60)
    
    # Run strategy backtest
    print("\n[1/3] Running STRATEGY backtest...")
    bt = QuickBacktest(
        start_date=start,
        end_date=end,
        initial_capital=100000,
        max_positions=10,
        min_score=65.0,  # Current setting
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
    )
    strategy_results = bt.run(symbols)
    
    # Run buy-and-hold SPY benchmark
    print("\n[2/3] Calculating SPY buy-and-hold benchmark...")
    spy_data = bt.data_cache.get('SPY')
    spy_start_price = spy_data[spy_data.index >= start].iloc[0]['open']
    spy_end_price = spy_data[spy_data.index <= end].iloc[-1]['close']
    spy_return = (spy_end_price / spy_start_price - 1) * 100
    
    # Calculate SPY drawdown
    spy_period = spy_data[(spy_data.index >= start) & (spy_data.index <= end)].copy()
    spy_period['peak'] = spy_period['close'].cummax()
    spy_period['dd'] = (spy_period['close'] - spy_period['peak']) / spy_period['peak'] * 100
    spy_max_dd = spy_period['dd'].min()
    
    days = (end - start).days
    years = days / 365.25
    spy_cagr = ((spy_end_price / spy_start_price) ** (1/years) - 1) * 100
    
    # Print results
    print("\n[3/3] Results")
    print("="*60)
    print("\nSTRATEGY PERFORMANCE:")
    print("-"*40)
    print(f"  Total Trades:      {strategy_results.get('total_trades', 0)}")
    print(f"  Win Rate:          {strategy_results.get('win_rate', 0):.1f}%")
    print(f"  Avg Winner:        {strategy_results.get('avg_winner_pct', 0):+.2f}%")
    print(f"  Avg Loser:         {strategy_results.get('avg_loser_pct', 0):.2f}%")
    print(f"  Profit Factor:     {strategy_results.get('profit_factor', 0):.2f}")
    print(f"  Total Return:      {strategy_results.get('total_return_pct', 0):+.2f}%")
    print(f"  CAGR:              {strategy_results.get('cagr_pct', 0):+.2f}%")
    print(f"  Max Drawdown:      {strategy_results.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio:      {strategy_results.get('sharpe_ratio', 0):.2f}")
    print(f"  Final Equity:      ${strategy_results.get('final_equity', 0):,.2f}")
    
    print(f"\n  Exit Reasons:")
    for reason, count in strategy_results.get('exit_reasons', {}).items():
        print(f"    {reason}: {count}")
    
    print("\nSPY BUY-AND-HOLD BENCHMARK:")
    print("-"*40)
    print(f"  Total Return:      {spy_return:+.2f}%")
    print(f"  CAGR:              {spy_cagr:+.2f}%")
    print(f"  Max Drawdown:      {spy_max_dd:.2f}%")
    print(f"  Final Equity:      ${100000 * (1 + spy_return/100):,.2f}")
    
    print("\nSTRATEGY vs BENCHMARK:")
    print("-"*40)
    alpha = strategy_results.get('total_return_pct', 0) - spy_return
    print(f"  Alpha (excess return): {alpha:+.2f}%")
    
    if alpha > 0:
        print("  >>> Strategy OUTPERFORMED benchmark")
    else:
        print("  >>> Strategy UNDERPERFORMED benchmark")
    
    print("="*60)
    
    # Save results
    output = {
        'strategy': strategy_results,
        'benchmark': {
            'total_return_pct': round(spy_return, 2),
            'cagr_pct': round(spy_cagr, 2),
            'max_drawdown_pct': round(spy_max_dd, 2),
        },
        'alpha': round(alpha, 2),
    }
    
    with open('/app/backtesting/validation_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to /app/backtesting/validation_results.json")
    
    return output


if __name__ == '__main__':
    results = run_validation()
