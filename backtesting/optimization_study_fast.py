#!/usr/bin/env python3
"""
Fast Optimization Study - Reduced scope for quicker results
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/ai_trader')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

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


class TechnicalScorer:
    def __init__(self, weights=None):
        self.weights = weights or {'rsi': 0.30, 'macd': 0.25, 'bb': 0.25, 'trend': 0.20}
    
    def calculate_score(self, df):
        if len(df) < 50:
            return 0.0
        
        close = df['close']
        rsi = ta.rsi(close, length=14).iloc[-1]
        
        macd = ta.macd(close)
        macd_hist = 0
        if macd is not None:
            macd_hist = (macd['MACD_12_26_9'] - macd['MACDs_12_26_9']).iloc[-1]
        
        bbands = ta.bbands(close)
        bb_score = 50.0
        if bbands is not None:
            bb_u = bbands[[c for c in bbands.columns if 'BBU' in c][0]].iloc[-1]
            bb_l = bbands[[c for c in bbands.columns if 'BBL' in c][0]].iloc[-1]
            if bb_u != bb_l:
                pos = (close.iloc[-1] - bb_l) / (bb_u - bb_l)
                bb_score = (1 - abs(pos - 0.5) * 2) * 100
        
        ema_20 = ta.ema(close, length=20).iloc[-1]
        ema_50 = ta.ema(close, length=50).iloc[-1]
        
        rsi_score = 50.0 if pd.isna(rsi) else (100.0 if rsi < 30 else (0.0 if rsi > 70 else 100 - abs(rsi - 50) * 2))
        macd_score = 50.0 if pd.isna(macd_hist) else min(100, max(0, 50 + macd_hist * 25))
        
        trend_score = 50.0
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            c = close.iloc[-1]
            if c > ema_20 > ema_50: trend_score = 80.0
            elif c > ema_20: trend_score = 65.0
            elif c < ema_20 < ema_50: trend_score = 20.0
        
        return (rsi_score * self.weights['rsi'] + macd_score * self.weights['macd'] + 
                bb_score * self.weights['bb'] + trend_score * self.weights['trend'])
    
    def get_component_scores(self, df):
        if len(df) < 50:
            return None
        
        close = df['close']
        rsi = ta.rsi(close, length=14).iloc[-1]
        
        macd = ta.macd(close)
        macd_hist = 0
        if macd is not None:
            macd_hist = (macd['MACD_12_26_9'] - macd['MACDs_12_26_9']).iloc[-1]
        
        bbands = ta.bbands(close)
        bb_score = 50.0
        if bbands is not None:
            bb_u = bbands[[c for c in bbands.columns if 'BBU' in c][0]].iloc[-1]
            bb_l = bbands[[c for c in bbands.columns if 'BBL' in c][0]].iloc[-1]
            if bb_u != bb_l:
                pos = (close.iloc[-1] - bb_l) / (bb_u - bb_l)
                bb_score = (1 - abs(pos - 0.5) * 2) * 100
        
        ema_20 = ta.ema(close, length=20).iloc[-1]
        ema_50 = ta.ema(close, length=50).iloc[-1]
        
        rsi_score = 50.0 if pd.isna(rsi) else (100.0 if rsi < 30 else (0.0 if rsi > 70 else 100 - abs(rsi - 50) * 2))
        macd_score = 50.0 if pd.isna(macd_hist) else min(100, max(0, 50 + macd_hist * 25))
        
        trend_score = 50.0
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            c = close.iloc[-1]
            if c > ema_20 > ema_50: trend_score = 80.0
            elif c > ema_20: trend_score = 65.0
            elif c < ema_20 < ema_50: trend_score = 20.0
        
        return {'rsi': rsi_score, 'macd': macd_score, 'bb': bb_score, 'trend': trend_score}


def run_backtest(data_cache, symbols, trading_days, stop_loss, take_profit, min_score, weights=None, initial_capital=100000):
    scorer = TechnicalScorer(weights)
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    max_positions = 10
    
    for date in trading_days:
        prices = {}
        for sym in symbols:
            if sym in data_cache:
                df = data_cache[sym]
                mask = df.index.date == date.date()
                if mask.any():
                    prices[sym] = df[mask].iloc[0]['close']
        
        # Exits
        exits = []
        for sym, pos in positions.items():
            p = prices.get(sym)
            if p and p <= pos['sl']:
                exits.append((sym, p, 'stop'))
            elif p and p >= pos['tp']:
                exits.append((sym, p, 'target'))
        
        for sym, p, reason in exits:
            pos = positions.pop(sym)
            cash += pos['shares'] * p * 0.9995
            trades.append(Trade(sym, pos['entry_date'], pos['entry_price'], date, p, pos['shares'],
                               pos['shares'] * p * 0.9995 - pos['shares'] * pos['entry_price'],
                               (p / pos['entry_price'] - 1) * 100, reason))
        
        # Entries
        if len(positions) < max_positions:
            cands = []
            for sym in symbols:
                if sym in positions or sym not in data_cache or sym == 'SPY':
                    continue
                df = data_cache[sym]
                subset = df[df.index < date].tail(60)
                if len(subset) < 50:
                    continue
                score = scorer.calculate_score(subset)
                if score >= min_score and sym in prices:
                    cands.append((sym, score, prices[sym]))
            
            cands.sort(key=lambda x: x[1], reverse=True)
            slots = max_positions - len(positions)
            
            for sym, score, price in cands[:slots]:
                shares = int(min(cash * 0.95, cash / max(slots, 1)) / price)
                if shares > 0 and shares * price * 1.0005 <= cash:
                    cash -= shares * price * 1.0005
                    positions[sym] = {'shares': shares, 'entry_price': price, 'entry_date': date,
                                     'sl': price * (1 - stop_loss), 'tp': price * (1 + take_profit)}
        
        pos_val = sum(pos['shares'] * prices.get(sym, pos['entry_price']) for sym, pos in positions.items())
        equity_curve.append({'date': date, 'equity': cash + pos_val})
    
    # Close remaining
    if positions and trading_days:
        last = trading_days[-1]
        for sym in list(positions.keys()):
            if sym in data_cache:
                df = data_cache[sym]
                mask = df.index.date == last.date()
                if mask.any():
                    p = df[mask].iloc[0]['close']
                    pos = positions.pop(sym)
                    cash += pos['shares'] * p * 0.9995
                    trades.append(Trade(sym, pos['entry_date'], pos['entry_price'], last, p, pos['shares'],
                                       pos['shares'] * p * 0.9995 - pos['shares'] * pos['entry_price'],
                                       (p / pos['entry_price'] - 1) * 100, 'end'))
    
    if not equity_curve:
        return None
    
    ret = (equity_curve[-1]['equity'] / initial_capital - 1) * 100
    eq_df = pd.DataFrame(equity_curve)
    eq_df['peak'] = eq_df['equity'].cummax()
    eq_df['dd'] = (eq_df['equity'] - eq_df['peak']) / eq_df['peak'] * 100
    
    winners = [t for t in trades if t.pnl > 0]
    
    return {'return': ret, 'max_dd': eq_df['dd'].min(), 'trades': len(trades),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0}


def run_regime_backtest(data_cache, symbols, trading_days, spy_data, initial_capital=100000):
    scorer = TechnicalScorer()
    
    regime_params = {
        'bull': {'sl': 0.03, 'tp': 0.10, 'ms': 65},
        'bear': {'sl': 0.02, 'tp': 0.06, 'ms': 75},
        'sideways': {'sl': 0.025, 'tp': 0.05, 'ms': 70}
    }
    
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
    
    for date in trading_days:
        # Detect regime
        spy_sub = spy_data[spy_data.index < date].tail(60)
        regime = 'sideways'
        if len(spy_sub) >= 60:
            close = spy_sub['close']
            ema_50 = ta.ema(close, length=50).iloc[-1]
            mom = (close.iloc[-1] / close.iloc[0] - 1) * 100
            if close.iloc[-1] > ema_50 and mom > 5:
                regime = 'bull'
            elif close.iloc[-1] < ema_50 and mom < -5:
                regime = 'bear'
        
        regime_counts[regime] += 1
        params = regime_params[regime]
        sl, tp, ms = params['sl'], params['tp'], params['ms']
        
        prices = {}
        for sym in symbols:
            if sym in data_cache:
                df = data_cache[sym]
                mask = df.index.date == date.date()
                if mask.any():
                    prices[sym] = df[mask].iloc[0]['close']
        
        # Exits
        exits = []
        for sym, pos in positions.items():
            p = prices.get(sym)
            if p and p <= pos['sl']:
                exits.append((sym, p, 'stop'))
            elif p and p >= pos['tp']:
                exits.append((sym, p, 'target'))
        
        for sym, p, reason in exits:
            pos = positions.pop(sym)
            cash += pos['shares'] * p * 0.9995
            trades.append(Trade(sym, pos['entry_date'], pos['entry_price'], date, p, pos['shares'],
                               pos['shares'] * p * 0.9995 - pos['shares'] * pos['entry_price'],
                               (p / pos['entry_price'] - 1) * 100, reason))
        
        # Entries
        if len(positions) < 10:
            cands = []
            for sym in symbols:
                if sym in positions or sym not in data_cache or sym == 'SPY':
                    continue
                df = data_cache[sym]
                subset = df[df.index < date].tail(60)
                if len(subset) < 50:
                    continue
                score = scorer.calculate_score(subset)
                if score >= ms and sym in prices:
                    cands.append((sym, score, prices[sym]))
            
            cands.sort(key=lambda x: x[1], reverse=True)
            slots = 10 - len(positions)
            
            for sym, score, price in cands[:slots]:
                shares = int(min(cash * 0.95, cash / max(slots, 1)) / price)
                if shares > 0 and shares * price * 1.0005 <= cash:
                    cash -= shares * price * 1.0005
                    positions[sym] = {'shares': shares, 'entry_price': price, 'entry_date': date,
                                     'sl': price * (1 - sl), 'tp': price * (1 + tp)}
        
        pos_val = sum(pos['shares'] * prices.get(sym, pos['entry_price']) for sym, pos in positions.items())
        equity_curve.append({'date': date, 'equity': cash + pos_val})
    
    if not equity_curve:
        return None
    
    ret = (equity_curve[-1]['equity'] / initial_capital - 1) * 100
    eq_df = pd.DataFrame(equity_curve)
    eq_df['peak'] = eq_df['equity'].cummax()
    eq_df['dd'] = (eq_df['equity'] - eq_df['peak']) / eq_df['peak'] * 100
    
    winners = [t for t in trades if t.pnl > 0]
    
    return {'return': ret, 'max_dd': eq_df['dd'].min(), 'trades': len(trades),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0, 'regimes': regime_counts}


def prepare_sgd_data(data_cache, symbols, trading_days, stop_loss, take_profit):
    scorer = TechnicalScorer()
    X, y = [], []
    
    sample_days = trading_days[::3]  # Every 3rd day for speed
    
    for i, date in enumerate(sample_days[:-10]):
        for symbol in symbols[:15]:  # Subset of symbols
            if symbol not in data_cache or symbol == 'SPY':
                continue
            
            df = data_cache[symbol]
            subset = df[df.index < date].tail(60)
            if len(subset) < 50:
                continue
            
            scores = scorer.get_component_scores(subset)
            if scores is None:
                continue
            
            entry_price = subset['close'].iloc[-1]
            
            future = df[(df.index > date) & (df.index <= sample_days[min(i+10, len(sample_days)-1)])]
            if len(future) < 3:
                continue
            
            hit_stop = hit_target = False
            for _, row in future.iterrows():
                if row['low'] <= entry_price * (1 - stop_loss):
                    hit_stop = True
                    break
                if row['high'] >= entry_price * (1 + take_profit):
                    hit_target = True
                    break
            
            X.append([scores['rsi'], scores['macd'], scores['bb'], scores['trend']])
            y.append(1 if hit_target and not hit_stop else 0)
    
    return np.array(X), np.array(y)


def train_sgd(X, y, epochs=100, lr=0.05):
    class WeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.tensor([0.30, 0.25, 0.25, 0.20]))
            self.threshold = nn.Parameter(torch.tensor([70.0]))
        
        def forward(self, x):
            w = torch.softmax(self.weights, dim=0)
            score = torch.sum(x * w, dim=1)
            return torch.sigmoid((score - self.threshold) / 10)
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    model = WeightModel()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        w = torch.softmax(model.weights, dim=0).numpy()
        t = model.threshold.item()
    
    return {'rsi': float(w[0]), 'macd': float(w[1]), 'bb': float(w[2]), 'trend': float(w[3])}, t


def main():
    print('=' * 80)
    print('FAST OPTIMIZATION STUDY')
    print('=' * 80)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'AMD', 'JPM', 'BAC', 'V',
               'JNJ', 'UNH', 'PG', 'WMT', 'XOM', 'SPY']
    
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 11, 30, tzinfo=timezone.utc)
    
    print(f'\nPeriod: {start.date()} to {end.date()}')
    print('Loading data...')
    
    client = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'))
    bars = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                                                   start=start - timedelta(days=100), end=end))
    df = bars.df
    
    data_cache = {}
    for sym in symbols:
        if sym in df.index.get_level_values(0):
            sdf = df.loc[sym].copy().reset_index().rename(columns={'timestamp': 'date'}).set_index('date')
            data_cache[sym] = sdf
    
    print(f'Loaded {len(data_cache)} symbols')
    
    spy = data_cache['SPY']
    days = [d for d in spy.index if start <= d <= end]
    
    spy_ret = (spy[spy.index <= end].iloc[-1]['close'] / spy[spy.index >= start].iloc[0]['open'] - 1) * 100
    print(f'SPY Return: {spy_ret:+.1f}%')
    
    results = {}
    
    # 1. Grid Search
    print('\n--- 1. GRID SEARCH ---')
    grids = [
        ('Tight Entry', 0.03, 0.08, 70),
        ('Conservative', 0.02, 0.06, 75),
        ('Aggressive', 0.05, 0.12, 60),
    ]
    
    for name, sl, tp, ms in grids:
        r = run_backtest(data_cache, symbols, days, sl, tp, ms)
        if r:
            alpha = r['return'] - spy_ret
            print(f'  {name}: {r["return"]:+.1f}% (alpha: {alpha:+.1f}%), DD: {r["max_dd"]:.1f}%')
            results[f'Grid_{name}'] = {'return': r['return'], 'alpha': alpha, 'max_dd': r['max_dd'], 
                                       'win_rate': r['win_rate'], 'method': 'grid'}
    
    # 2. SGD
    print('\n--- 2. SGD OPTIMIZATION ---')
    split = int(len(days) * 0.6)
    train_days, test_days = days[:split], days[split:]
    
    print(f'  Train: {train_days[0].date()} to {train_days[-1].date()}')
    print(f'  Test: {test_days[0].date()} to {test_days[-1].date()}')
    
    X, y = prepare_sgd_data(data_cache, symbols, train_days, 0.03, 0.08)
    print(f'  Samples: {len(X)}, Positive: {y.mean()*100:.1f}%')
    
    sgd_weights, sgd_thresh = train_sgd(X, y)
    print(f'  SGD Weights: RSI={sgd_weights["rsi"]:.2f}, MACD={sgd_weights["macd"]:.2f}, BB={sgd_weights["bb"]:.2f}, Trend={sgd_weights["trend"]:.2f}')
    print(f'  SGD Threshold: {sgd_thresh:.1f}')
    
    spy_test_ret = (spy[(spy.index >= test_days[0]) & (spy.index <= test_days[-1])].iloc[-1]['close'] / 
                   spy[(spy.index >= test_days[0]) & (spy.index <= test_days[-1])].iloc[0]['open'] - 1) * 100
    
    sgd_r = run_backtest(data_cache, symbols, test_days, 0.03, 0.08, sgd_thresh, sgd_weights)
    default_r = run_backtest(data_cache, symbols, test_days, 0.03, 0.08, 70, None)
    
    if sgd_r:
        alpha = sgd_r['return'] - spy_test_ret
        print(f'  SGD (test): {sgd_r["return"]:+.1f}% (alpha: {alpha:+.1f}%)')
        results['SGD'] = {'return': sgd_r['return'], 'alpha': alpha, 'max_dd': sgd_r['max_dd'],
                         'win_rate': sgd_r['win_rate'], 'method': 'sgd', 'weights': sgd_weights, 'threshold': sgd_thresh}
    
    if default_r:
        alpha = default_r['return'] - spy_test_ret
        print(f'  Default (test): {default_r["return"]:+.1f}% (alpha: {alpha:+.1f}%)')
        results['Default_test'] = {'return': default_r['return'], 'alpha': alpha, 'max_dd': default_r['max_dd'],
                                   'win_rate': default_r['win_rate'], 'method': 'grid'}
    
    # 3. Bayesian (Optuna)
    if OPTUNA_AVAILABLE:
        print('\n--- 3. BAYESIAN (Optuna) ---')
        
        train_spy_ret = (spy[(spy.index >= train_days[0]) & (spy.index <= train_days[-1])].iloc[-1]['close'] /
                        spy[(spy.index >= train_days[0]) & (spy.index <= train_days[-1])].iloc[0]['open'] - 1) * 100
        
        def objective(trial):
            sl = trial.suggest_float('sl', 0.02, 0.06)
            tp = trial.suggest_float('tp', 0.05, 0.12)
            ms = trial.suggest_float('ms', 60, 80)
            r = run_backtest(data_cache, symbols, train_days, sl, tp, ms)
            if r is None:
                return -100
            return r['return'] - train_spy_ret
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        bp = study.best_params
        print(f'  Best: SL={bp["sl"]*100:.1f}%, TP={bp["tp"]*100:.1f}%, Score={bp["ms"]:.1f}')
        
        bay_r = run_backtest(data_cache, symbols, test_days, bp['sl'], bp['tp'], bp['ms'])
        if bay_r:
            alpha = bay_r['return'] - spy_test_ret
            print(f'  Bayesian (test): {bay_r["return"]:+.1f}% (alpha: {alpha:+.1f}%)')
            results['Bayesian'] = {'return': bay_r['return'], 'alpha': alpha, 'max_dd': bay_r['max_dd'],
                                  'win_rate': bay_r['win_rate'], 'method': 'bayesian', 'params': bp}
    
    # 4. Regime-Adaptive
    print('\n--- 4. REGIME-ADAPTIVE ---')
    regime_r = run_regime_backtest(data_cache, symbols, days, spy)
    if regime_r:
        alpha = regime_r['return'] - spy_ret
        print(f'  Regime-Adaptive: {regime_r["return"]:+.1f}% (alpha: {alpha:+.1f}%)')
        print(f'  Regimes: {regime_r["regimes"]}')
        results['Regime'] = {'return': regime_r['return'], 'alpha': alpha, 'max_dd': regime_r['max_dd'],
                            'win_rate': regime_r['win_rate'], 'method': 'regime', 'regimes': regime_r['regimes']}
    
    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'\n{"Method":<20} {"Return":>10} {"Alpha":>10} {"MaxDD":>10} {"WinRate":>10}')
    print('-' * 62)
    
    for name, r in sorted(results.items(), key=lambda x: x[1].get('alpha', -999), reverse=True):
        print(f'{name:<20} {r["return"]:>+9.1f}% {r["alpha"]:>+9.1f}% {r["max_dd"]:>9.1f}% {r["win_rate"]:>9.1f}%')
    
    # Analysis
    print('\n' + '=' * 80)
    print('ANALYSIS & RECOMMENDATIONS')
    print('=' * 80)
    
    # Find best method
    best = max(results.items(), key=lambda x: x[1].get('alpha', -999))
    print(f'\nBest Method: {best[0]} with {best[1]["alpha"]:+.1f}% alpha')
    
    # Compare SGD vs Default
    if 'SGD' in results and 'Default_test' in results:
        diff = results['SGD']['alpha'] - results['Default_test']['alpha']
        print(f'\nSGD vs Grid Search: {"+" if diff > 0 else ""}{diff:.1f}% alpha difference')
        if diff > 2:
            print('  -> SGD shows meaningful improvement, consider implementing')
        elif diff > 0:
            print('  -> SGD shows marginal improvement, may not be worth complexity')
        else:
            print('  -> SGD underperforms grid search, stick with current approach')
    
    # Regime analysis
    if 'Regime' in results:
        regime_diff = results['Regime']['alpha'] - results.get('Grid_Tight Entry', {}).get('alpha', 0)
        print(f'\nRegime-Adaptive vs Fixed: {"+" if regime_diff > 0 else ""}{regime_diff:.1f}% alpha difference')
        if regime_diff > 3:
            print('  -> Regime adaptation shows significant benefit')
        else:
            print('  -> Regime adaptation shows limited benefit in this period')
    
    # Save
    with open('/root/ai_trader/backtesting/optimization_study_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print('\nResults saved.')
    return results


if __name__ == '__main__':
    main()
