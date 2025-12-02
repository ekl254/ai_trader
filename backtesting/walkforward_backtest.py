#!/usr/bin/env python3
"""
Walk-Forward Optimization Backtest

5-year backtest (2020-2024) with proper train/test split to avoid overfitting.
- Train on 12 months, test on 6 months, roll forward
- Reports only out-of-sample (test) performance
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import itertools

import pandas as pd
import numpy as np
import pandas_ta as ta

sys.path.insert(0, "/root/ai_trader")

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
    exit_reason: str = ""


class TechnicalScorer:
    def calculate_score(self, df: pd.DataFrame) -> float:
        if len(df) < 50:
            return 0.0
        
        df = df.copy()
        close = df["close"]
        
        df["RSI"] = ta.rsi(close, length=14)
        macd = ta.macd(close)
        if macd is not None:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
        
        bbands = ta.bbands(close)
        if bbands is not None:
            for col in bbands.columns:
                if col.startswith("BBU_"):
                    df["BB_upper"] = bbands[col]
                elif col.startswith("BBL_"):
                    df["BB_lower"] = bbands[col]
        
        df["EMA_20"] = ta.ema(close, length=20)
        df["EMA_50"] = ta.ema(close, length=50)
        
        latest = df.iloc[-1]
        
        rsi = latest.get("RSI", 50)
        if pd.isna(rsi):
            rsi_score = 50.0
        elif rsi < 30:
            rsi_score = 100.0
        elif rsi > 70:
            rsi_score = 0.0
        else:
            rsi_score = 100 - abs(rsi - 50) * 2
        
        macd_val = latest.get("MACD")
        macd_sig = latest.get("MACD_signal")
        if pd.isna(macd_val) or pd.isna(macd_sig):
            macd_score = 50.0
        else:
            histogram = macd_val - macd_sig
            macd_score = min(100, max(0, 50 + histogram * 25))
        
        bb_upper = latest.get("BB_upper")
        bb_lower = latest.get("BB_lower")
        current_close = latest["close"]
        
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
        
        ema_20 = latest.get("EMA_20")
        ema_50 = latest.get("EMA_50")
        
        if not pd.isna(ema_20) and not pd.isna(ema_50):
            if current_close > ema_20 > ema_50:
                trend_score = 80.0
            elif current_close > ema_20:
                trend_score = 65.0
            elif current_close < ema_20 < ema_50:
                trend_score = 20.0
            else:
                trend_score = 50.0
        else:
            trend_score = 50.0
        
        return rsi_score * 0.3 + macd_score * 0.25 + bb_score * 0.25 + trend_score * 0.20


def run_backtest_period(data_cache, symbols, trading_days, stop_loss, take_profit, min_score, initial_capital=100000):
    """Run backtest for a specific period."""
    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    scorer = TechnicalScorer()
    max_positions = 10
    
    for date in trading_days:
        current_prices = {}
        for symbol in symbols:
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == date.date()
                if mask.any():
                    current_prices[symbol] = df[mask].iloc[0]["close"]
        
        # Check exits
        to_exit = []
        for symbol, pos in positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos["stop_loss"]:
                to_exit.append((symbol, price, "stop_loss"))
            elif price >= pos["take_profit"]:
                to_exit.append((symbol, price, "take_profit"))
        
        for symbol, price, reason in to_exit:
            pos = positions.pop(symbol)
            proceeds = pos["shares"] * price * 0.9995
            cash += proceeds
            trades.append(Trade(
                symbol=symbol, entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                exit_date=date, exit_price=price, shares=pos["shares"],
                pnl=proceeds - pos["shares"]*pos["entry_price"],
                pnl_pct=(price/pos["entry_price"]-1)*100, exit_reason=reason
            ))
        
        # Find new entries
        if len(positions) < max_positions:
            candidates = []
            for symbol in symbols:
                if symbol in positions or symbol not in data_cache:
                    continue
                df = data_cache[symbol]
                mask = df.index < date
                subset = df[mask].tail(60)
                if len(subset) < 50:
                    continue
                score = scorer.calculate_score(subset)
                if score >= min_score and symbol in current_prices:
                    candidates.append({"symbol": symbol, "score": score, "price": current_prices[symbol]})
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            slots = max_positions - len(positions)
            
            for c in candidates[:slots]:
                pos_value = min(cash * 0.95, cash / max(slots, 1))
                shares = int(pos_value / c["price"])
                if shares > 0:
                    cost = shares * c["price"] * 1.0005
                    if cost <= cash:
                        cash -= cost
                        positions[c["symbol"]] = {
                            "shares": shares, "entry_price": c["price"], "entry_date": date,
                            "stop_loss": c["price"] * (1 - stop_loss),
                            "take_profit": c["price"] * (1 + take_profit),
                        }
        
        # Record equity
        position_value = sum(pos["shares"] * current_prices.get(sym, pos["entry_price"]) 
                           for sym, pos in positions.items())
        equity_curve.append({"date": date, "equity": cash + position_value})
    
    # Close remaining positions
    if positions and trading_days:
        last_date = trading_days[-1]
        for symbol in list(positions.keys()):
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == last_date.date()
                if mask.any():
                    price = df[mask].iloc[0]["close"]
                    pos = positions.pop(symbol)
                    proceeds = pos["shares"] * price * 0.9995
                    cash += proceeds
                    trades.append(Trade(
                        symbol=symbol, entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                        exit_date=last_date, exit_price=price, shares=pos["shares"],
                        pnl=proceeds - pos["shares"]*pos["entry_price"],
                        pnl_pct=(price/pos["entry_price"]-1)*100, exit_reason="end"
                    ))
    
    if not equity_curve:
        return None
    
    final_equity = equity_curve[-1]["equity"]
    total_return = (final_equity / initial_capital - 1) * 100
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
    max_dd = equity_df["dd"].min()
    
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    
    return {
        "total_return": total_return,
        "final_equity": final_equity,
        "max_drawdown": max_dd,
        "total_trades": len(trades),
        "win_rate": len(winners)/len(trades)*100 if trades else 0,
        "trades": trades,
    }


def run_walkforward():
    """Run walk-forward optimization over 5 years."""
    
    # Diversified symbol list
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "CRM", "ADBE", "INTC",
        "JPM", "BAC", "GS", "V", "MA", "WFC",
        "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK",
        "PG", "KO", "WMT", "HD", "MCD", "NKE",
        "CAT", "BA", "HON", "UPS", "GE",
        "XOM", "CVX", "NEE", "SPY"
    ]
    
    # Full 5-year period
    full_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    full_end = datetime(2024, 11, 30, tzinfo=timezone.utc)
    
    print("="*70)
    print("5-YEAR WALK-FORWARD OPTIMIZATION BACKTEST")
    print("="*70)
    print(f"Full Period: {full_start.strftime('%Y-%m-%d')} to {full_end.strftime('%Y-%m-%d')}")
    print(f"Symbols: {len(symbols)}")
    print("Walk-forward: Train 12mo, Test 6mo, Roll forward")
    print("="*70)
    
    # Load all data
    print("\nLoading 5 years of historical data...")
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    )
    
    data_cache = {}
    lookback_start = full_start - timedelta(days=100)
    
    # Load in batches
    batch_size = 40
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"  Loading batch {i//batch_size + 1}...")
        
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=lookback_start,
            end=full_end,
        )
        
        try:
            bars = client.get_stock_bars(request)
            df = bars.df
            for symbol in batch:
                if symbol in df.index.get_level_values(0):
                    symbol_df = df.loc[symbol].copy().reset_index()
                    symbol_df = symbol_df.rename(columns={"timestamp": "date"}).set_index("date")
                    data_cache[symbol] = symbol_df
        except Exception as e:
            print(f"  Error loading batch: {e}")
    
    print(f"Loaded {len(data_cache)} symbols")
    
    # Get all trading days from SPY
    spy_data = data_cache["SPY"]
    all_trading_days = spy_data[
        (spy_data.index >= full_start) & (spy_data.index <= full_end)
    ].index.tolist()
    
    print(f"Total trading days: {len(all_trading_days)}")
    
    # Define walk-forward windows
    # Train: 12 months, Test: 6 months
    windows = [
        # Window 1: Train 2020, Test 2020-H2
        {"train_start": "2020-01-01", "train_end": "2020-06-30", "test_start": "2020-07-01", "test_end": "2020-12-31"},
        # Window 2: Train 2020-H2 to 2021-H1, Test 2021-H2
        {"train_start": "2020-07-01", "train_end": "2021-06-30", "test_start": "2021-07-01", "test_end": "2021-12-31"},
        # Window 3: Train 2021, Test 2022-H1
        {"train_start": "2021-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-06-30"},
        # Window 4: Train 2021-H2 to 2022-H1, Test 2022-H2
        {"train_start": "2021-07-01", "train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2022-12-31"},
        # Window 5: Train 2022, Test 2023-H1
        {"train_start": "2022-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-06-30"},
        # Window 6: Train 2022-H2 to 2023-H1, Test 2023-H2
        {"train_start": "2022-07-01", "train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-12-31"},
        # Window 7: Train 2023, Test 2024-H1
        {"train_start": "2023-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-06-30"},
        # Window 8: Train 2023-H2 to 2024-H1, Test 2024-H2
        {"train_start": "2023-07-01", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-11-30"},
    ]
    
    # Parameter grid for optimization
    stop_losses = [0.02, 0.03, 0.05, 0.07]
    take_profits = [0.06, 0.08, 0.10, 0.15]
    min_scores = [60.0, 65.0, 70.0]
    
    print(f"\nParameter grid: {len(stop_losses)}x{len(take_profits)}x{len(min_scores)} = {len(stop_losses)*len(take_profits)*len(min_scores)} combinations")
    
    # Track out-of-sample results
    oos_results = []
    window_best_params = []
    
    for w_idx, window in enumerate(windows):
        train_start = datetime.strptime(window["train_start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        train_end = datetime.strptime(window["train_end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        test_start = datetime.strptime(window["test_start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        test_end = datetime.strptime(window["test_end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        print(f"\n{'='*70}")
        print(f"Window {w_idx+1}: Train {window['train_start']} to {window['train_end']}")
        print(f"          Test  {window['test_start']} to {window['test_end']}")
        print("="*70)
        
        # Get trading days for train and test periods
        train_days = [d for d in all_trading_days if train_start <= d <= train_end]
        test_days = [d for d in all_trading_days if test_start <= d <= test_end]
        
        if not train_days or not test_days:
            print("  Skipping - insufficient data")
            continue
        
        # Find best params on training data
        print(f"  Training on {len(train_days)} days...")
        best_train_return = -999
        best_params = None
        
        for sl, tp, ms in itertools.product(stop_losses, take_profits, min_scores):
            if tp <= sl:
                continue
            
            result = run_backtest_period(data_cache, symbols, train_days, sl, tp, ms)
            if result and result["total_return"] > best_train_return:
                best_train_return = result["total_return"]
                best_params = {"stop_loss": sl, "take_profit": tp, "min_score": ms}
        
        if not best_params:
            print("  No valid params found")
            continue
        
        print(f"  Best train params: SL={best_params['stop_loss']:.0%}, TP={best_params['take_profit']:.0%}, Score={best_params['min_score']}")
        print(f"  Train return: {best_train_return:+.1f}%")
        
        # Test on out-of-sample data
        print(f"  Testing on {len(test_days)} days (OUT-OF-SAMPLE)...")
        test_result = run_backtest_period(
            data_cache, symbols, test_days,
            best_params["stop_loss"], best_params["take_profit"], best_params["min_score"]
        )
        
        if test_result:
            # Calculate SPY return for same test period
            spy_test = spy_data[(spy_data.index >= test_start) & (spy_data.index <= test_end)]
            if len(spy_test) > 1:
                spy_test_return = (spy_test.iloc[-1]["close"] / spy_test.iloc[0]["open"] - 1) * 100
            else:
                spy_test_return = 0
            
            oos_result = {
                "window": w_idx + 1,
                "test_period": f"{window['test_start']} to {window['test_end']}",
                "params": best_params,
                "strategy_return": round(test_result["total_return"], 2),
                "spy_return": round(spy_test_return, 2),
                "alpha": round(test_result["total_return"] - spy_test_return, 2),
                "max_drawdown": round(test_result["max_drawdown"], 2),
                "trades": test_result["total_trades"],
                "win_rate": round(test_result["win_rate"], 1),
            }
            oos_results.append(oos_result)
            window_best_params.append(best_params)
            
            print(f"  TEST RESULT: {test_result['total_return']:+.1f}% (SPY: {spy_test_return:+.1f}%, Alpha: {oos_result['alpha']:+.1f}%)")
    
    # Aggregate results
    print("\n" + "="*70)
    print("WALK-FORWARD OUT-OF-SAMPLE RESULTS")
    print("="*70)
    print(f"{'Window':>8} {'Period':>25} {'Strategy':>10} {'SPY':>10} {'Alpha':>10} {'MaxDD':>8} {'Trades':>7}")
    print("-"*70)
    
    total_strategy = 100000
    total_spy = 100000
    
    for r in oos_results:
        print(f"{r['window']:>8} {r['test_period']:>25} {r['strategy_return']:>+9.1f}% {r['spy_return']:>+9.1f}% "
              f"{r['alpha']:>+9.1f}% {r['max_drawdown']:>7.1f}% {r['trades']:>7}")
        total_strategy *= (1 + r['strategy_return']/100)
        total_spy *= (1 + r['spy_return']/100)
    
    cumulative_strategy = (total_strategy / 100000 - 1) * 100
    cumulative_spy = (total_spy / 100000 - 1) * 100
    cumulative_alpha = cumulative_strategy - cumulative_spy
    
    print("-"*70)
    print(f"{'TOTAL':>8} {'Cumulative (compounded)':>25} {cumulative_strategy:>+9.1f}% {cumulative_spy:>+9.1f}% {cumulative_alpha:>+9.1f}%")
    
    # Parameter frequency analysis
    print("\n" + "="*70)
    print("OPTIMAL PARAMETER FREQUENCY ACROSS WINDOWS")
    print("="*70)
    
    sl_counts = {}
    tp_counts = {}
    ms_counts = {}
    
    for params in window_best_params:
        sl = params["stop_loss"]
        tp = params["take_profit"]
        ms = params["min_score"]
        sl_counts[sl] = sl_counts.get(sl, 0) + 1
        tp_counts[tp] = tp_counts.get(tp, 0) + 1
        ms_counts[ms] = ms_counts.get(ms, 0) + 1
    
    print(f"\nStop Loss: {dict(sorted(sl_counts.items(), key=lambda x: -x[1]))}")
    print(f"Take Profit: {dict(sorted(tp_counts.items(), key=lambda x: -x[1]))}")
    print(f"Min Score: {dict(sorted(ms_counts.items(), key=lambda x: -x[1]))}")
    
    # Most robust parameters (most frequently selected)
    best_sl = max(sl_counts, key=sl_counts.get) if sl_counts else 0.03
    best_tp = max(tp_counts, key=tp_counts.get) if tp_counts else 0.08
    best_ms = max(ms_counts, key=ms_counts.get) if ms_counts else 60.0
    
    print(f"\nMost Robust Parameters (most frequently optimal):")
    print(f"  Stop Loss: {best_sl:.0%}")
    print(f"  Take Profit: {best_tp:.0%}")
    print(f"  Min Score: {best_ms}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    winning_windows = len([r for r in oos_results if r["alpha"] > 0])
    print(f"Windows with positive alpha: {winning_windows}/{len(oos_results)}")
    print(f"Cumulative Strategy Return: {cumulative_strategy:+.1f}%")
    print(f"Cumulative SPY Return: {cumulative_spy:+.1f}%")
    print(f"Cumulative Alpha: {cumulative_alpha:+.1f}%")
    
    if cumulative_alpha > 0:
        print("\n>>> STRATEGY SHOWS EDGE OVER SPY IN OUT-OF-SAMPLE TESTING <<<")
    else:
        print("\n>>> STRATEGY DOES NOT BEAT SPY IN OUT-OF-SAMPLE TESTING <<<")
    
    # Save results
    output = {
        "cumulative_strategy_return": round(cumulative_strategy, 2),
        "cumulative_spy_return": round(cumulative_spy, 2),
        "cumulative_alpha": round(cumulative_alpha, 2),
        "windows_with_alpha": winning_windows,
        "total_windows": len(oos_results),
        "recommended_params": {
            "stop_loss": best_sl,
            "take_profit": best_tp,
            "min_score": best_ms,
        },
        "window_results": oos_results,
    }
    
    with open("/app/backtesting/walkforward_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\nResults saved to walkforward_results.json")
    return output


if __name__ == "__main__":
    run_walkforward()
