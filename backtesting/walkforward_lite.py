#!/usr/bin/env python3
"""
Walk-Forward Lite - Fewer parameters for faster testing
Tests the promising params from quick sweep across different market conditions.
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
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
    exit_date: datetime | None = None
    exit_price: float | None = None
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

        return (
            rsi_score * 0.3 + macd_score * 0.25 + bb_score * 0.25 + trend_score * 0.20
        )


def run_backtest(
    data_cache,
    symbols,
    trading_days,
    stop_loss,
    take_profit,
    min_score,
    initial_capital=100000,
):
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
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    exit_date=date,
                    exit_price=price,
                    shares=pos["shares"],
                    pnl=proceeds - pos["shares"] * pos["entry_price"],
                    pnl_pct=(price / pos["entry_price"] - 1) * 100,
                    exit_reason=reason,
                )
            )

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
                    candidates.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "price": current_prices[symbol],
                        }
                    )

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
                            "shares": shares,
                            "entry_price": c["price"],
                            "entry_date": date,
                            "stop_loss": c["price"] * (1 - stop_loss),
                            "take_profit": c["price"] * (1 + take_profit),
                        }

        position_value = sum(
            pos["shares"] * current_prices.get(sym, pos["entry_price"])
            for sym, pos in positions.items()
        )
        equity_curve.append({"date": date, "equity": cash + position_value})

    # Close remaining
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
                    trades.append(
                        Trade(
                            symbol=symbol,
                            entry_date=pos["entry_date"],
                            entry_price=pos["entry_price"],
                            exit_date=last_date,
                            exit_price=price,
                            shares=pos["shares"],
                            pnl=proceeds - pos["shares"] * pos["entry_price"],
                            pnl_pct=(price / pos["entry_price"] - 1) * 100,
                            exit_reason="end",
                        )
                    )

    if not equity_curve:
        return None

    final_equity = equity_curve[-1]["equity"]
    total_return = (final_equity / initial_capital - 1) * 100

    equity_df = pd.DataFrame(equity_curve)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = (
        (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
    )
    max_dd = equity_df["dd"].min()

    winners = [t for t in trades if t.pnl > 0]

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "total_trades": len(trades),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "exit_reasons": exit_reasons,
    }


def run_test():
    """Test best params from quick sweep across different market periods."""

    symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "AMD",
        "JPM",
        "BAC",
        "GS",
        "V",
        "MA",
        "JNJ",
        "UNH",
        "PFE",
        "LLY",
        "PG",
        "KO",
        "WMT",
        "HD",
        "CAT",
        "BA",
        "XOM",
        "CVX",
        "SPY",
    ]

    # Test periods covering different market conditions
    periods = [
        {"name": "COVID Crash & Recovery", "start": "2020-02-01", "end": "2020-12-31"},
        {"name": "2021 Bull Market", "start": "2021-01-01", "end": "2021-12-31"},
        {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-12-31"},
        {"name": "2023 Recovery", "start": "2023-01-01", "end": "2023-12-31"},
        {"name": "2024 YTD", "start": "2024-01-01", "end": "2024-11-30"},
    ]

    # Parameters to test - best from quick sweep + alternatives
    param_sets = [
        {"name": "Quick Sweep Winner", "sl": 0.03, "tp": 0.08, "ms": 60},
        {"name": "Current Live", "sl": 0.02, "tp": 0.06, "ms": 65},
        {"name": "Wide Stops", "sl": 0.05, "tp": 0.10, "ms": 60},
        {"name": "Tight Entry", "sl": 0.03, "tp": 0.08, "ms": 70},
    ]

    print("=" * 80)
    print("MULTI-PERIOD STRATEGY VALIDATION")
    print("=" * 80)
    print(f"Symbols: {len(symbols)}")
    print(
        f"Testing {len(param_sets)} parameter sets across {len(periods)} market periods"
    )
    print("=" * 80)

    # Load all data
    print("\nLoading 5 years of data...")
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )

    full_start = datetime(2020, 1, 1, tzinfo=UTC)
    full_end = datetime(2024, 11, 30, tzinfo=UTC)
    lookback_start = full_start - timedelta(days=100)

    data_cache = {}
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=lookback_start,
        end=full_end,
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    for symbol in symbols:
        if symbol in df.index.get_level_values(0):
            symbol_df = df.loc[symbol].copy().reset_index()
            symbol_df = symbol_df.rename(columns={"timestamp": "date"}).set_index(
                "date"
            )
            data_cache[symbol] = symbol_df

    print(f"Loaded {len(data_cache)} symbols")

    spy_data = data_cache["SPY"]

    # Test each parameter set across all periods
    results = {p["name"]: [] for p in param_sets}
    spy_returns = []

    for period in periods:
        p_start = datetime.strptime(period["start"], "%Y-%m-%d").replace(tzinfo=UTC)
        p_end = datetime.strptime(period["end"], "%Y-%m-%d").replace(tzinfo=UTC)

        trading_days = [d for d in spy_data.index if p_start <= d <= p_end]

        # SPY return for this period
        spy_period = spy_data[(spy_data.index >= p_start) & (spy_data.index <= p_end)]
        spy_ret = (
            (spy_period.iloc[-1]["close"] / spy_period.iloc[0]["open"] - 1) * 100
            if len(spy_period) > 1
            else 0
        )
        spy_returns.append({"period": period["name"], "return": spy_ret})

        print(f"\n{period['name']} ({period['start']} to {period['end']})")
        print(f"  SPY: {spy_ret:+.1f}%")

        for params in param_sets:
            result = run_backtest(
                data_cache,
                symbols,
                trading_days,
                params["sl"],
                params["tp"],
                params["ms"],
            )
            if result:
                results[params["name"]].append(
                    {
                        "period": period["name"],
                        "return": result["total_return"],
                        "alpha": result["total_return"] - spy_ret,
                        "max_dd": result["max_drawdown"],
                        "trades": result["total_trades"],
                        "win_rate": result["win_rate"],
                    }
                )
                print(
                    f"  {params['name']}: {result['total_return']:+.1f}% (alpha: {result['total_return']-spy_ret:+.1f}%)"
                )

    # Summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY BY STRATEGY")
    print("=" * 80)

    for params in param_sets:
        print(
            f"\n{params['name']} (SL={params['sl']:.0%}, TP={params['tp']:.0%}, Score={params['ms']})"
        )
        print("-" * 70)
        print(
            f"{'Period':<25} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'WinRate':>10}"
        )
        print("-" * 70)

        total_ret = 100000
        total_alpha = 0

        for r in results[params["name"]]:
            print(
                f"{r['period']:<25} {r['return']:>+9.1f}% {r['alpha']:>+9.1f}% {r['max_dd']:>9.1f}% {r['win_rate']:>9.1f}%"
            )
            total_ret *= 1 + r["return"] / 100
            total_alpha += r["alpha"]

        cumulative = (total_ret / 100000 - 1) * 100
        print("-" * 70)
        print(f"{'CUMULATIVE':<25} {cumulative:>+9.1f}% {total_alpha:>+9.1f}%")

    # SPY benchmark
    print("\n" + "=" * 80)
    print("SPY BENCHMARK")
    print("=" * 80)
    spy_cumulative = 100000
    for s in spy_returns:
        print(f"{s['period']:<25} {s['return']:>+9.1f}%")
        spy_cumulative *= 1 + s["return"] / 100
    print("-" * 70)
    print(f"{'CUMULATIVE':<25} {(spy_cumulative/100000-1)*100:>+9.1f}%")

    # Best strategy determination
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_name = None
    best_alpha = -999

    for params in param_sets:
        total_alpha = sum(r["alpha"] for r in results[params["name"]])
        positive_periods = sum(1 for r in results[params["name"]] if r["alpha"] > 0)

        print(
            f"{params['name']}: Total Alpha = {total_alpha:+.1f}%, Positive periods = {positive_periods}/{len(periods)}"
        )

        if total_alpha > best_alpha:
            best_alpha = total_alpha
            best_name = params["name"]

    print(f"\nBest Strategy: {best_name}")

    # Save results
    output = {
        "results": results,
        "spy_returns": spy_returns,
        "best_strategy": best_name,
    }

    with open("/app/backtesting/validation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nResults saved to validation_results.json")


if __name__ == "__main__":
    run_test()
