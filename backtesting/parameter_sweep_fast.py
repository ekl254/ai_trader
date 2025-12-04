#!/usr/bin/env python3
"""
Fast Parameter Sweep - Using 1 year of data and fewer symbols
"""

import itertools
import json
import os
import sys
from dataclasses import dataclass, field
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


@dataclass
class Portfolio:
    cash: float = 100000.0
    positions: dict[str, dict] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def get_equity(self, prices: dict[str, float]) -> float:
        position_value = sum(
            pos["shares"] * prices.get(symbol, pos["entry_price"])
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value


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

        # RSI Score
        rsi = latest.get("RSI", 50)
        if pd.isna(rsi):
            rsi_score = 50.0
        elif rsi < 30:
            rsi_score = 100.0
        elif rsi > 70:
            rsi_score = 0.0
        else:
            rsi_score = 100 - abs(rsi - 50) * 2

        # MACD Score
        macd_val = latest.get("MACD")
        macd_sig = latest.get("MACD_signal")
        if pd.isna(macd_val) or pd.isna(macd_sig):
            macd_score = 50.0
        else:
            histogram = macd_val - macd_sig
            macd_score = min(100, max(0, 50 + histogram * 25))

        # BB Score
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

        # Trend Score
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


def run_single_backtest(
    data_cache,
    symbols,
    trading_days,
    stop_loss,
    take_profit,
    min_score,
    initial_capital=100000,
):
    """Run a single backtest with given parameters."""
    portfolio = Portfolio(cash=initial_capital)
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
        for symbol, pos in portfolio.positions.items():
            price = current_prices.get(symbol)
            if price is None:
                continue
            if price <= pos["stop_loss"]:
                to_exit.append((symbol, price, "stop_loss"))
            elif price >= pos["take_profit"]:
                to_exit.append((symbol, price, "take_profit"))

        for symbol, price, reason in to_exit:
            pos = portfolio.positions.pop(symbol)
            proceeds = pos["shares"] * price * 0.9995
            portfolio.cash += proceeds
            portfolio.trades.append(
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
        if len(portfolio.positions) < max_positions:
            candidates = []
            for symbol in symbols:
                if symbol in portfolio.positions or symbol not in data_cache:
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
            slots = max_positions - len(portfolio.positions)

            for c in candidates[:slots]:
                pos_value = min(portfolio.cash * 0.95, portfolio.cash / max(slots, 1))
                shares = int(pos_value / c["price"])
                if shares > 0:
                    cost = shares * c["price"] * 1.0005
                    if cost <= portfolio.cash:
                        portfolio.cash -= cost
                        portfolio.positions[c["symbol"]] = {
                            "shares": shares,
                            "entry_price": c["price"],
                            "entry_date": date,
                            "stop_loss": c["price"] * (1 - stop_loss),
                            "take_profit": c["price"] * (1 + take_profit),
                        }

        portfolio.equity_curve.append(
            {"date": date, "equity": portfolio.get_equity(current_prices)}
        )

    # Close remaining
    if portfolio.positions and trading_days:
        last_date = trading_days[-1]
        for symbol in list(portfolio.positions.keys()):
            if symbol in data_cache:
                df = data_cache[symbol]
                mask = df.index.date == last_date.date()
                if mask.any():
                    price = df[mask].iloc[0]["close"]
                    pos = portfolio.positions.pop(symbol)
                    proceeds = pos["shares"] * price * 0.9995
                    portfolio.cash += proceeds
                    portfolio.trades.append(
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

    # Calculate metrics
    trades = portfolio.trades
    if not trades:
        return None

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    equity_df = pd.DataFrame(portfolio.equity_curve)
    final_equity = (
        equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital
    )
    total_return = (final_equity / initial_capital - 1) * 100

    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = (
        (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
    )
    max_dd = equity_df["dd"].min()

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))

    return {
        "total_trades": len(trades),
        "win_rate": round(len(winners) / len(trades) * 100, 1),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_dd, 2),
        "exit_reasons": exit_reasons,
    }


def run_sweep():
    """Run parameter sweep."""
    # Use fewer symbols for faster backtest
    symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "AMD",
        "CRM",
        "JPM",
        "BAC",
        "GS",
        "V",
        "MA",
        "JNJ",
        "UNH",
        "PFE",
        "LLY",
        "ABBV",
        "PG",
        "KO",
        "WMT",
        "HD",
        "MCD",
        "CAT",
        "BA",
        "HON",
        "UPS",
        "XOM",
        "CVX",
        "NEE",
        "SPY",
    ]

    # Use 1.5 years for faster testing
    start = datetime(2023, 6, 1, tzinfo=UTC)
    end = datetime(2024, 11, 30, tzinfo=UTC)

    # Parameter grid - focused set
    stop_losses = [0.02, 0.03, 0.05, 0.07, 0.10]
    take_profits = [0.04, 0.06, 0.08, 0.10, 0.15]
    min_scores = [60.0, 65.0, 70.0]

    print("=" * 70)
    print("FAST PARAMETER SWEEP")
    print("=" * 70)
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Symbols: {len(symbols)}")
    print(f"Combinations: {len(stop_losses) * len(take_profits) * len(min_scores)}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )

    data_cache = {}
    lookback_start = start - timedelta(days=100)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=lookback_start,
        end=end,
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

    # Get trading days
    spy_data = data_cache["SPY"]
    trading_days = spy_data[
        (spy_data.index >= start) & (spy_data.index <= end)
    ].index.tolist()

    # SPY benchmark
    spy_start = spy_data[spy_data.index >= start].iloc[0]["open"]
    spy_end = spy_data[spy_data.index <= end].iloc[-1]["close"]
    spy_return = (spy_end / spy_start - 1) * 100
    print(f"\nSPY Benchmark: {spy_return:+.2f}%")

    # Run sweep
    results = []
    combo_num = 0
    total_combos = len(stop_losses) * len(take_profits) * len(min_scores)

    print("\nRunning sweep...")
    for sl, tp, ms in itertools.product(stop_losses, take_profits, min_scores):
        combo_num += 1

        if tp <= sl:
            continue

        if combo_num % 10 == 0:
            print(f"  {combo_num}/{total_combos}...")

        metrics = run_single_backtest(data_cache, symbols, trading_days, sl, tp, ms)

        if metrics:
            result = {
                "stop_loss": sl,
                "take_profit": tp,
                "min_score": ms,
                **metrics,
                "alpha": round(metrics["total_return"] - spy_return, 2),
            }
            results.append(result)

    # Sort by alpha
    results.sort(key=lambda x: x["alpha"], reverse=True)

    # Print results
    print("\n" + "=" * 70)
    print("TOP 10 COMBINATIONS (by Alpha)")
    print("=" * 70)
    print(
        f"{'SL':>5} {'TP':>5} {'Score':>6} {'Return':>8} {'Alpha':>8} {'WinRate':>8} {'MaxDD':>8} {'PF':>6} {'Trades':>7}"
    )
    print("-" * 70)

    for r in results[:10]:
        sl_str = f"{r['stop_loss']*100:.0f}%"
        tp_str = f"{r['take_profit']*100:.0f}%"
        print(
            f"{sl_str:>5} {tp_str:>5} {r['min_score']:>6.0f} "
            f"{r['total_return']:>+7.1f}% {r['alpha']:>+7.1f}% {r['win_rate']:>7.1f}% "
            f"{r['max_drawdown']:>7.1f}% {r['profit_factor']:>5.2f} {r['total_trades']:>7}"
        )

    print("\n" + "=" * 70)
    print("BOTTOM 5 COMBINATIONS")
    print("=" * 70)
    for r in results[-5:]:
        sl_str = f"{r['stop_loss']*100:.0f}%"
        tp_str = f"{r['take_profit']*100:.0f}%"
        print(
            f"{sl_str:>5} {tp_str:>5} {r['min_score']:>6.0f} "
            f"{r['total_return']:>+7.1f}% {r['alpha']:>+7.1f}% {r['win_rate']:>7.1f}% "
            f"{r['max_drawdown']:>7.1f}% {r['profit_factor']:>5.2f} {r['total_trades']:>7}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"SPY Benchmark: {spy_return:+.2f}%")

    profitable = [r for r in results if r["alpha"] > 0]
    print(f"Combinations that beat SPY: {len(profitable)}/{len(results)}")

    if profitable:
        best = results[0]
        print("\nBest combination:")
        print(f"  Stop Loss: {best['stop_loss']*100:.0f}%")
        print(f"  Take Profit: {best['take_profit']*100:.0f}%")
        print(f"  Min Score: {best['min_score']}")
        print(f"  Return: {best['total_return']:+.2f}%")
        print(f"  Alpha: {best['alpha']:+.2f}%")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Exit Reasons: {best['exit_reasons']}")
    else:
        print("\n>>> NO PARAMETER COMBINATION BEATS SPY <<<")

    # Save
    with open("/root/ai_trader/backtesting/sweep_results.json", "w") as f:
        json.dump(
            {
                "spy_return": spy_return,
                "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
                "results": results,
                "best": results[0] if results else None,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nResults saved to sweep_results.json")


if __name__ == "__main__":
    run_sweep()
