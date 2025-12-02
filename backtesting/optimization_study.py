#!/usr/bin/env python3
"""
Comprehensive Optimization Study: Grid Search vs SGD vs Bayesian vs Regime-Adaptive

This study compares different optimization approaches for the AI Trader strategy.
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/ai_trader')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ML imports
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print('Warning: Optuna not available')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('Warning: XGBoost not available')

import torch
import torch.nn as nn
import torch.optim as optim


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
    score: float = 0.0


class TechnicalScorer:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'rsi': 0.30, 'macd': 0.25, 'bb': 0.25, 'trend': 0.20}
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < 50:
            return None
        
        df = df.copy()
        close = df['close']
        
        df['RSI'] = ta.rsi(close, length=14)
        macd = ta.macd(close)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
        
        bbands = ta.bbands(close)
        if bbands is not None:
            for col in bbands.columns:
                if col.startswith('BBU_'):
                    df['BB_upper'] = bbands[col]
                elif col.startswith('BBL_'):
                    df['BB_lower'] = bbands[col]
        
        df['EMA_20'] = ta.ema(close, length=20)
        df['EMA_50'] = ta.ema(close, length=50)
        df['ATR'] = ta.atr(df['high'], df['low'], close, length=14)
        
        latest = df.iloc[-1]
        return {
            'rsi': latest.get('RSI', 50),
            'macd': latest.get('MACD', 0),
            'macd_signal': latest.get('MACD_signal', 0),
            'bb_upper': latest.get('BB_upper', close.iloc[-1] * 1.02),
            'bb_lower': latest.get('BB_lower', close.iloc[-1] * 0.98),
            'ema_20': latest.get('EMA_20', close.iloc[-1]),
            'ema_50': latest.get('EMA_50', close.iloc[-1]),
            'close': latest['close'],
            'atr': latest.get('ATR', 0),
            'volume': latest['volume'],
        }
    
    def indicators_to_scores(self, ind: Dict) -> Dict[str, float]:
        scores = {}
        
        rsi = ind['rsi']
        if pd.isna(rsi):
            scores['rsi'] = 50.0
        elif rsi < 30:
            scores['rsi'] = 100.0
        elif rsi > 70:
            scores['rsi'] = 0.0
        else:
            scores['rsi'] = 100 - abs(rsi - 50) * 2
        
        macd_val, macd_sig = ind['macd'], ind['macd_signal']
        if pd.isna(macd_val) or pd.isna(macd_sig):
            scores['macd'] = 50.0
        else:
            histogram = macd_val - macd_sig
            scores['macd'] = min(100, max(0, 50 + histogram * 25))
        
        bb_upper, bb_lower, close = ind['bb_upper'], ind['bb_lower'], ind['close']
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            scores['bb'] = 50.0
        elif close < bb_lower:
            scores['bb'] = 100.0
        elif close > bb_upper:
            scores['bb'] = 0.0
        else:
            bb_range = bb_upper - bb_lower
            position = (close - bb_lower) / bb_range if bb_range > 0 else 0.5
            scores['bb'] = (1 - abs(position - 0.5) * 2) * 100
        
        ema_20, ema_50 = ind['ema_20'], ind['ema_50']
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            if close > ema_20 > ema_50:
                scores['trend'] = 80.0
            elif close > ema_20:
                scores['trend'] = 65.0
            elif close < ema_20 < ema_50:
                scores['trend'] = 20.0
            else:
                scores['trend'] = 50.0
        else:
            scores['trend'] = 50.0
        
        return scores
    
    def calculate_score(self, df: pd.DataFrame) -> float:
        ind = self.calculate_indicators(df)
        if ind is None:
            return 0.0
        scores = self.indicators_to_scores(ind)
        return sum(scores[k] * self.weights[k] for k in self.weights)


class SGDWeightOptimizer:
    def __init__(self):
        self.model = None
        self.best_weights = None
        self.training_history = []
    
    def prepare_training_data(self, data_cache, symbols, trading_days, stop_loss, take_profit):
        scorer = TechnicalScorer()
        X, y = [], []
        
        for i, date in enumerate(trading_days[:-20]):
            for symbol in symbols:
                if symbol not in data_cache or symbol == 'SPY':
                    continue
                
                df = data_cache[symbol]
                mask = df.index < date
                subset = df[mask].tail(60)
                
                if len(subset) < 50:
                    continue
                
                ind = scorer.calculate_indicators(subset)
                if ind is None:
                    continue
                
                scores = scorer.indicators_to_scores(ind)
                entry_price = ind['close']
                
                future_mask = (df.index > date) & (df.index <= trading_days[min(i+20, len(trading_days)-1)])
                future_df = df[future_mask]
                
                if len(future_df) < 5:
                    continue
                
                hit_stop = hit_target = False
                for _, row in future_df.iterrows():
                    if row['low'] <= entry_price * (1 - stop_loss):
                        hit_stop = True
                        break
                    if row['high'] >= entry_price * (1 + take_profit):
                        hit_target = True
                        break
                
                label = 1 if hit_target and not hit_stop else 0
                X.append([scores['rsi'], scores['macd'], scores['bb'], scores['trend']])
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=100, lr=0.01):
        class WeightModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.tensor([0.30, 0.25, 0.25, 0.20]))
                self.threshold = nn.Parameter(torch.tensor([70.0]))
            
            def forward(self, x):
                weighted = torch.sum(x * torch.softmax(self.weights, dim=0), dim=1)
                prob = torch.sigmoid((weighted - self.threshold) / 10)
                return prob
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        model = WeightModel()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                with torch.no_grad():
                    pred_binary = (pred > 0.5).float()
                    acc = (pred_binary == y_tensor).float().mean()
                    self.training_history.append({'epoch': epoch, 'loss': loss.item(), 'accuracy': acc.item()})
        
        with torch.no_grad():
            weights = torch.softmax(model.weights, dim=0).numpy()
            threshold = model.threshold.item()
        
        self.best_weights = {'rsi': float(weights[0]), 'macd': float(weights[1]), 
                            'bb': float(weights[2]), 'trend': float(weights[3])}
        self.optimal_threshold = threshold
        self.model = model
        
        return self.best_weights, threshold


class RegimeDetector:
    def __init__(self):
        self.regime_params = {
            'bull': {'stop_loss': 0.03, 'take_profit': 0.10, 'min_score': 65},
            'bear': {'stop_loss': 0.02, 'take_profit': 0.06, 'min_score': 75},
            'sideways': {'stop_loss': 0.025, 'take_profit': 0.05, 'min_score': 70}
        }
    
    def detect_regime(self, spy_df, lookback=60):
        if len(spy_df) < lookback:
            return 'sideways'
        
        recent = spy_df.tail(lookback).copy()
        close = recent['close']
        
        ema_20 = ta.ema(close, length=20).iloc[-1]
        ema_50 = ta.ema(close, length=50).iloc[-1]
        current = close.iloc[-1]
        momentum = (current / close.iloc[0] - 1) * 100
        
        if current > ema_50 and momentum > 5:
            return 'bull'
        elif current < ema_50 and momentum < -5:
            return 'bear'
        return 'sideways'
    
    def get_params_for_regime(self, regime):
        return self.regime_params.get(regime, self.regime_params['sideways'])


class MLEntryModel:
    def __init__(self):
        self.model = None
        self.feature_names = ['rsi', 'macd_hist', 'bb_position', 'trend_strength',
                              'volume_ratio', 'atr_pct', 'momentum_5d', 'momentum_20d']
    
    def extract_features(self, df):
        if len(df) < 50:
            return None
        
        df = df.copy()
        close = df['close']
        
        rsi = ta.rsi(close, length=14).iloc[-1]
        
        macd = ta.macd(close)
        macd_hist = (macd['MACD_12_26_9'] - macd['MACDs_12_26_9']).iloc[-1] if macd is not None else 0
        
        bbands = ta.bbands(close)
        bb_position = 0.5
        if bbands is not None:
            bb_upper = bbands[[c for c in bbands.columns if 'BBU' in c][0]].iloc[-1]
            bb_lower = bbands[[c for c in bbands.columns if 'BBL' in c][0]].iloc[-1]
            if bb_upper != bb_lower:
                bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
        
        ema_20 = ta.ema(close, length=20).iloc[-1]
        ema_50 = ta.ema(close, length=50).iloc[-1]
        trend_strength = (ema_20 - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        
        volume_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
        
        atr = ta.atr(df['high'], df['low'], close, length=14).iloc[-1]
        atr_pct = atr / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        
        momentum_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        momentum_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        
        features = [rsi if not pd.isna(rsi) else 50, macd_hist if not pd.isna(macd_hist) else 0,
                    bb_position if not pd.isna(bb_position) else 0.5, trend_strength if not pd.isna(trend_strength) else 0,
                    volume_ratio if not pd.isna(volume_ratio) else 1, atr_pct if not pd.isna(atr_pct) else 2,
                    momentum_5d if not pd.isna(momentum_5d) else 0, momentum_20d if not pd.isna(momentum_20d) else 0]
        
        return np.array(features)
    
    def prepare_training_data(self, data_cache, symbols, trading_days, stop_loss, take_profit):
        X, y = [], []
        
        for i, date in enumerate(trading_days[:-20]):
            for symbol in symbols:
                if symbol not in data_cache or symbol == 'SPY':
                    continue
                
                df = data_cache[symbol]
                mask = df.index < date
                subset = df[mask].tail(60)
                
                features = self.extract_features(subset)
                if features is None:
                    continue
                
                entry_price = subset['close'].iloc[-1]
                
                future_mask = (df.index > date) & (df.index <= trading_days[min(i+20, len(trading_days)-1)])
                future_df = df[future_mask]
                
                if len(future_df) < 5:
                    continue
                
                hit_stop = hit_target = False
                for _, row in future_df.iterrows():
                    if row['low'] <= entry_price * (1 - stop_loss):
                        hit_stop = True
                        break
                    if row['high'] >= entry_price * (1 + take_profit):
                        hit_target = True
                        break
                
                label = 1 if hit_target and not hit_stop else 0
                X.append(features)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        if not XGBOOST_AVAILABLE:
            return None
        
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                       objective='binary:logistic', random_state=42,
                                       use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)
        return self.model


def run_backtest_with_weights(data_cache, symbols, trading_days, stop_loss, take_profit, min_score, weights=None, initial_capital=100000):
    scorer = TechnicalScorer(weights)
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    max_positions = 10
    
    for date in trading_days:
        current_prices = {}
        for symbol in symbols:
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == date.date()
                if mask.any():
                    current_prices[symbol] = df[mask].iloc[0]['close']
        
        to_exit = []
        for symbol, pos in positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos['stop_loss']:
                to_exit.append((symbol, price, 'stop_loss'))
            elif price >= pos['take_profit']:
                to_exit.append((symbol, price, 'take_profit'))
        
        for symbol, price, reason in to_exit:
            pos = positions.pop(symbol)
            proceeds = pos['shares'] * price * 0.9995
            cash += proceeds
            trades.append(Trade(symbol=symbol, entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                               exit_date=date, exit_price=price, shares=pos['shares'],
                               pnl=proceeds - pos['shares'] * pos['entry_price'],
                               pnl_pct=(price / pos['entry_price'] - 1) * 100, exit_reason=reason))
        
        if len(positions) < max_positions:
            candidates = []
            for symbol in symbols:
                if symbol in positions or symbol not in data_cache or symbol == 'SPY':
                    continue
                df = data_cache[symbol]
                mask = df.index < date
                subset = df[mask].tail(60)
                if len(subset) < 50:
                    continue
                score = scorer.calculate_score(subset)
                if score >= min_score and symbol in current_prices:
                    candidates.append({'symbol': symbol, 'score': score, 'price': current_prices[symbol]})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            slots = max_positions - len(positions)
            
            for c in candidates[:slots]:
                pos_value = min(cash * 0.95, cash / max(slots, 1))
                shares = int(pos_value / c['price'])
                if shares > 0:
                    cost = shares * c['price'] * 1.0005
                    if cost <= cash:
                        cash -= cost
                        positions[c['symbol']] = {'shares': shares, 'entry_price': c['price'], 'entry_date': date,
                                                  'stop_loss': c['price'] * (1 - stop_loss),
                                                  'take_profit': c['price'] * (1 + take_profit)}
        
        position_value = sum(pos['shares'] * current_prices.get(sym, pos['entry_price']) for sym, pos in positions.items())
        equity_curve.append({'date': date, 'equity': cash + position_value})
    
    if positions and trading_days:
        last_date = trading_days[-1]
        for symbol in list(positions.keys()):
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == last_date.date()
                if mask.any():
                    price = df[mask].iloc[0]['close']
                    pos = positions.pop(symbol)
                    proceeds = pos['shares'] * price * 0.9995
                    cash += proceeds
                    trades.append(Trade(symbol=symbol, entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                                       exit_date=last_date, exit_price=price, shares=pos['shares'],
                                       pnl=proceeds - pos['shares'] * pos['entry_price'],
                                       pnl_pct=(price / pos['entry_price'] - 1) * 100, exit_reason='end'))
    
    if not equity_curve:
        return None
    
    final_equity = equity_curve[-1]['equity']
    total_return = (final_equity / initial_capital - 1) * 100
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['dd'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_dd = equity_df['dd'].min()
    
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
    
    return {'total_return': total_return, 'max_drawdown': max_dd, 'total_trades': len(trades),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0}


def run_regime_adaptive_backtest(data_cache, symbols, trading_days, spy_data, initial_capital=100000):
    regime_detector = RegimeDetector()
    scorer = TechnicalScorer()
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    max_positions = 10
    regime_log = []
    
    for date in trading_days:
        spy_mask = spy_data.index < date
        spy_subset = spy_data[spy_mask].tail(60)
        regime = regime_detector.detect_regime(spy_subset)
        params = regime_detector.get_params_for_regime(regime)
        regime_log.append({'date': date, 'regime': regime})
        
        stop_loss, take_profit, min_score = params['stop_loss'], params['take_profit'], params['min_score']
        
        current_prices = {}
        for symbol in symbols:
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == date.date()
                if mask.any():
                    current_prices[symbol] = df[mask].iloc[0]['close']
        
        to_exit = []
        for symbol, pos in positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos['stop_loss']:
                to_exit.append((symbol, price, 'stop_loss'))
            elif price >= pos['take_profit']:
                to_exit.append((symbol, price, 'take_profit'))
        
        for symbol, price, reason in to_exit:
            pos = positions.pop(symbol)
            proceeds = pos['shares'] * price * 0.9995
            cash += proceeds
            trades.append(Trade(symbol=symbol, entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                               exit_date=date, exit_price=price, shares=pos['shares'],
                               pnl=proceeds - pos['shares'] * pos['entry_price'],
                               pnl_pct=(price / pos['entry_price'] - 1) * 100, exit_reason=reason))
        
        if len(positions) < max_positions:
            candidates = []
            for symbol in symbols:
                if symbol in positions or symbol not in data_cache or symbol == 'SPY':
                    continue
                df = data_cache[symbol]
                mask = df.index < date
                subset = df[mask].tail(60)
                if len(subset) < 50:
                    continue
                score = scorer.calculate_score(subset)
                if score >= min_score and symbol in current_prices:
                    candidates.append({'symbol': symbol, 'score': score, 'price': current_prices[symbol]})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            slots = max_positions - len(positions)
            
            for c in candidates[:slots]:
                pos_value = min(cash * 0.95, cash / max(slots, 1))
                shares = int(pos_value / c['price'])
                if shares > 0:
                    cost = shares * c['price'] * 1.0005
                    if cost <= cash:
                        cash -= cost
                        positions[c['symbol']] = {'shares': shares, 'entry_price': c['price'], 'entry_date': date,
                                                  'stop_loss': c['price'] * (1 - stop_loss),
                                                  'take_profit': c['price'] * (1 + take_profit)}
        
        position_value = sum(pos['shares'] * current_prices.get(sym, pos['entry_price']) for sym, pos in positions.items())
        equity_curve.append({'date': date, 'equity': cash + position_value})
    
    if positions and trading_days:
        last_date = trading_days[-1]
        for symbol in list(positions.keys()):
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == last_date.date()
                if mask.any():
                    price = df[mask].iloc[0]['close']
                    pos = positions.pop(symbol)
                    proceeds = pos['shares'] * price * 0.9995
                    cash += proceeds
                    trades.append(Trade(symbol=symbol, entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                                       exit_date=last_date, exit_price=price, shares=pos['shares'],
                                       pnl=proceeds - pos['shares'] * pos['entry_price'],
                                       pnl_pct=(price / pos['entry_price'] - 1) * 100, exit_reason='end'))
    
    if not equity_curve:
        return None
    
    final_equity = equity_curve[-1]['equity']
    total_return = (final_equity / initial_capital - 1) * 100
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['dd'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_dd = equity_df['dd'].min()
    
    winners = [t for t in trades if t.pnl > 0]
    regime_df = pd.DataFrame(regime_log)
    regime_counts = regime_df['regime'].value_counts().to_dict()
    
    return {'total_return': total_return, 'max_drawdown': max_dd, 'total_trades': len(trades),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0, 'regime_counts': regime_counts}


def main():
    print('=' * 80)
    print('COMPREHENSIVE OPTIMIZATION STUDY')
    print('Grid Search vs SGD vs Bayesian vs Regime-Adaptive vs ML')
    print('=' * 80)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'AMD', 'JPM', 'BAC', 'GS', 'V', 'MA',
               'JNJ', 'UNH', 'PFE', 'LLY', 'PG', 'KO', 'WMT', 'HD', 'CAT', 'BA', 'XOM', 'CVX', 'SPY']
    
    start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 11, 30, tzinfo=timezone.utc)
    lookback_start = start_date - timedelta(days=100)
    
    print(f'\nLoading data from {start_date.date()} to {end_date.date()}...')
    
    client = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'))
    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=lookback_start, end=end_date)
    bars = client.get_stock_bars(request)
    df = bars.df
    
    data_cache = {}
    for symbol in symbols:
        if symbol in df.index.get_level_values(0):
            symbol_df = df.loc[symbol].copy().reset_index()
            symbol_df = symbol_df.rename(columns={'timestamp': 'date'}).set_index('date')
            data_cache[symbol] = symbol_df
    
    print(f'Loaded {len(data_cache)} symbols')
    
    spy_data = data_cache['SPY']
    trading_days = [d for d in spy_data.index if start_date <= d <= end_date]
    
    spy_period = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
    spy_return = (spy_period.iloc[-1]['close'] / spy_period.iloc[0]['open'] - 1) * 100
    print(f'SPY Return: {spy_return:+.2f}%')
    
    results = {}
    
    # 1. GRID SEARCH
    print('\n' + '=' * 80)
    print('1. GRID SEARCH (Current Approach)')
    print('=' * 80)
    
    grid_params = [
        {'name': 'Tight Entry (Current)', 'sl': 0.03, 'tp': 0.08, 'ms': 70},
        {'name': 'Quick Sweep Winner', 'sl': 0.03, 'tp': 0.08, 'ms': 60},
        {'name': 'Conservative', 'sl': 0.02, 'tp': 0.06, 'ms': 75},
        {'name': 'Aggressive', 'sl': 0.05, 'tp': 0.12, 'ms': 60},
    ]
    
    for p in grid_params:
        result = run_backtest_with_weights(data_cache, symbols, trading_days, p['sl'], p['tp'], p['ms'])
        if result:
            alpha = result['total_return'] - spy_return
            print(f"  {p['name']}: Return={result['total_return']:+.1f}%, Alpha={alpha:+.1f}%, MaxDD={result['max_drawdown']:.1f}%")
            results[f"Grid: {p['name']}"] = {'return': result['total_return'], 'alpha': alpha, 
                                             'max_dd': result['max_drawdown'], 'win_rate': result['win_rate'], 'method': 'grid_search'}
    
    # 2. SGD
    print('\n' + '=' * 80)
    print('2. SGD WEIGHT OPTIMIZATION')
    print('=' * 80)
    
    split_idx = int(len(trading_days) * 0.6)
    train_days, test_days = trading_days[:split_idx], trading_days[split_idx:]
    
    print(f'  Training: {train_days[0].date()} to {train_days[-1].date()}')
    print(f'  Testing: {test_days[0].date()} to {test_days[-1].date()}')
    
    sgd_optimizer = SGDWeightOptimizer()
    print('  Preparing training data...')
    X_train, y_train = sgd_optimizer.prepare_training_data(data_cache, symbols, train_days, 0.03, 0.08)
    print(f'  Training samples: {len(X_train)}, Positive rate: {y_train.mean()*100:.1f}%')
    
    print('  Training SGD model...')
    best_weights, optimal_threshold = sgd_optimizer.train(X_train, y_train, epochs=200, lr=0.05)
    
    print(f'  Optimized Weights: RSI={best_weights["rsi"]:.3f}, MACD={best_weights["macd"]:.3f}, BB={best_weights["bb"]:.3f}, Trend={best_weights["trend"]:.3f}')
    print(f'  Optimal Threshold: {optimal_threshold:.1f}')
    
    spy_test = spy_data[(spy_data.index >= test_days[0]) & (spy_data.index <= test_days[-1])]
    spy_test_return = (spy_test.iloc[-1]['close'] / spy_test.iloc[0]['open'] - 1) * 100
    
    sgd_result = run_backtest_with_weights(data_cache, symbols, test_days, 0.03, 0.08, optimal_threshold, best_weights)
    if sgd_result:
        sgd_alpha = sgd_result['total_return'] - spy_test_return
        print(f'  SGD Result (Test): Return={sgd_result["total_return"]:+.1f}%, Alpha={sgd_alpha:+.1f}%')
        results['SGD Optimized'] = {'return': sgd_result['total_return'], 'alpha': sgd_alpha,
                                    'max_dd': sgd_result['max_drawdown'], 'win_rate': sgd_result['win_rate'],
                                    'method': 'sgd', 'weights': best_weights, 'threshold': optimal_threshold}
    
    default_result = run_backtest_with_weights(data_cache, symbols, test_days, 0.03, 0.08, 70, None)
    if default_result:
        default_alpha = default_result['total_return'] - spy_test_return
        print(f'  Default Weights (Test): Return={default_result["total_return"]:+.1f}%, Alpha={default_alpha:+.1f}%')
        results['Default Weights (Test)'] = {'return': default_result['total_return'], 'alpha': default_alpha,
                                             'max_dd': default_result['max_drawdown'], 'win_rate': default_result['win_rate'], 'method': 'grid_search'}
    
    # 3. BAYESIAN
    print('\n' + '=' * 80)
    print('3. BAYESIAN OPTIMIZATION (Optuna)')
    print('=' * 80)
    
    if OPTUNA_AVAILABLE:
        train_spy = spy_data[(spy_data.index >= train_days[0]) & (spy_data.index <= train_days[-1])]
        train_spy_return = (train_spy.iloc[-1]['close'] / train_spy.iloc[0]['open'] - 1) * 100
        
        def objective(trial):
            sl = trial.suggest_float('stop_loss', 0.02, 0.08)
            tp = trial.suggest_float('take_profit', 0.04, 0.15)
            ms = trial.suggest_float('min_score', 55, 80)
            w_rsi = trial.suggest_float('w_rsi', 0.1, 0.5)
            w_macd = trial.suggest_float('w_macd', 0.1, 0.4)
            w_bb = trial.suggest_float('w_bb', 0.1, 0.4)
            w_trend = trial.suggest_float('w_trend', 0.1, 0.4)
            
            total = w_rsi + w_macd + w_bb + w_trend
            weights = {'rsi': w_rsi/total, 'macd': w_macd/total, 'bb': w_bb/total, 'trend': w_trend/total}
            
            result = run_backtest_with_weights(data_cache, symbols, train_days, sl, tp, ms, weights)
            if result is None:
                return -100.0
            return result['total_return'] - train_spy_return - max(0, -result['max_drawdown'] - 20) * 0.5
        
        print('  Running Optuna (50 trials)...')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        bp = study.best_params
        print(f'  Best: SL={bp["stop_loss"]*100:.1f}%, TP={bp["take_profit"]*100:.1f}%, Score={bp["min_score"]:.1f}')
        
        total_w = bp['w_rsi'] + bp['w_macd'] + bp['w_bb'] + bp['w_trend']
        bay_weights = {'rsi': bp['w_rsi']/total_w, 'macd': bp['w_macd']/total_w, 'bb': bp['w_bb']/total_w, 'trend': bp['w_trend']/total_w}
        
        bay_result = run_backtest_with_weights(data_cache, symbols, test_days, bp['stop_loss'], bp['take_profit'], bp['min_score'], bay_weights)
        if bay_result:
            bay_alpha = bay_result['total_return'] - spy_test_return
            print(f'  Bayesian (Test): Return={bay_result["total_return"]:+.1f}%, Alpha={bay_alpha:+.1f}%')
            results['Bayesian Optimized'] = {'return': bay_result['total_return'], 'alpha': bay_alpha,
                                             'max_dd': bay_result['max_drawdown'], 'win_rate': bay_result['win_rate'],
                                             'method': 'bayesian', 'params': bp}
    
    # 4. REGIME-ADAPTIVE
    print('\n' + '=' * 80)
    print('4. REGIME-ADAPTIVE STRATEGY')
    print('=' * 80)
    
    regime_result = run_regime_adaptive_backtest(data_cache, symbols, trading_days, spy_data)
    if regime_result:
        regime_alpha = regime_result['total_return'] - spy_return
        print(f'  Regime-Adaptive: Return={regime_result["total_return"]:+.1f}%, Alpha={regime_alpha:+.1f}%')
        print(f'  Regime Distribution: {regime_result["regime_counts"]}')
        results['Regime-Adaptive'] = {'return': regime_result['total_return'], 'alpha': regime_alpha,
                                      'max_dd': regime_result['max_drawdown'], 'win_rate': regime_result['win_rate'],
                                      'method': 'regime_adaptive', 'regime_counts': regime_result['regime_counts']}
    
    # 5. ML (XGBoost)
    print('\n' + '=' * 80)
    print('5. ML ENTRY MODEL (XGBoost)')
    print('=' * 80)
    
    if XGBOOST_AVAILABLE:
        ml_model = MLEntryModel()
        print('  Preparing ML data...')
        X_ml, y_ml = ml_model.prepare_training_data(data_cache, symbols, train_days, 0.03, 0.08)
        print(f'  ML samples: {len(X_ml)}, Positive rate: {y_ml.mean()*100:.1f}%')
        
        print('  Training XGBoost...')
        ml_model.train(X_ml, y_ml)
        
        if ml_model.model:
            importance = ml_model.model.feature_importances_
            print('  Feature Importance:')
            for name, imp in sorted(zip(ml_model.feature_names, importance), key=lambda x: x[1], reverse=True):
                print(f'    {name}: {imp:.3f}')
            results['ML Model (XGBoost)'] = {'method': 'ml', 
                                             'feature_importance': dict(zip(ml_model.feature_names, [float(x) for x in importance]))}
    
    # SUMMARY
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    
    print(f'\n{"Method":<30} {"Return":>10} {"Alpha":>10} {"MaxDD":>10} {"WinRate":>10}')
    print('-' * 72)
    
    for name, r in sorted(results.items(), key=lambda x: x[1].get('alpha', -999), reverse=True):
        if 'return' in r:
            print(f'{name:<30} {r["return"]:>+9.1f}% {r["alpha"]:>+9.1f}% {r["max_dd"]:>9.1f}% {r["win_rate"]:>9.1f}%')
    
    # Save
    with open('/root/ai_trader/backtesting/optimization_study_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print('\nResults saved to optimization_study_results.json')
    return results


if __name__ == '__main__':
    main()
