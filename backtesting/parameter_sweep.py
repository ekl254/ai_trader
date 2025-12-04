#!/usr/bin/env python3
"""
Parameter Sweep Backtest

Tests multiple combinations of stop-loss and take-profit parameters
to find if any configuration beats buy-and-hold SPY.
"""

import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
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
    """Calculates technical score - mirrors live strategy."""

    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def calculate_score(self, df: pd.DataFrame) -> tuple[float, dict]:
        if len(df) < 50:
            return 0.0, {"error": "insufficient_data"}

        df = df.copy()
        close = df["close"]

        df["RSI"] = ta.rsi(close, length=self.rsi_period)

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

        rsi_value = latest.get("RSI", 50)
        if pd.isna(rsi_value):
            rsi_score = 50.0
        elif rsi_value < self.rsi_oversold:
            rsi_score = 100.0
        elif rsi_value > self.rsi_overbought:
            rsi_score = 0.0
        else:
            rsi_score = 100 - abs(rsi_value - 50) * 2

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

        avg_volume = df["volume"].tail(20).mean()
        current_volume = latest["volume"]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        prev_close = df["close"].iloc[-2]
        price_change = (
            (current_close - prev_close) / prev_close if prev_close > 0 else 0
        )

        if price_change > 0:
            volume_score = min(100, 50 + (volume_ratio - 1) * 30)
        elif price_change < -0.005:
            volume_score = max(0, 50 - (volume_ratio - 1) * 30)
        else:
            volume_score = 50.0

        ema_20 = latest.get("EMA_20")
        ema_50 = latest.get("EMA_50")

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

        technical_score = (
            rsi_score * 0.25
            + macd_score * 0.25
            + bb_score * 0.20
            + volume_score * 0.15
            + trend_score * 0.15
        )

        return technical_score, {}


class ParameterSweepBacktest:
    """Run backtest with specific parameters."""

    def __init__(
        self,
        data_cache: dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        min_score: float = 65.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.06,
    ):
        self.data_cache = data_cache
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.min_score = min_score
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.scorer = TechnicalScorer()
        self.portfolio = Portfolio(cash=initial_capital)

    def get_data_for_date(
        self, symbol: str, date: datetime, lookback: int = 60
    ) -> pd.DataFrame | None:
        if symbol not in self.data_cache:
            return None
        df = self.data_cache[symbol]
        mask = df.index < date
        subset = df[mask].tail(lookback)
        return subset if len(subset) >= 50 else None

    def get_close_on_date(self, symbol: str, date: datetime) -> float | None:
        if symbol not in self.data_cache:
            return None
        df = self.data_cache[symbol]
        mask = df.index.date == date.date()
        return df[mask].iloc[0]["close"] if mask.any() else None

    def run(self, symbols: list[str], trading_days: list) -> dict:
        for date in trading_days:
            current_prices = {}
            for symbol in symbols:
                price = self.get_close_on_date(symbol, date)
                if price:
                    current_prices[symbol] = price

            # Check exits
            to_exit = []
            for symbol, pos in self.portfolio.positions.items():
                price = current_prices.get(symbol)
                if price is None:
                    continue
                if price <= pos["stop_loss"]:
                    to_exit.append((symbol, price, "stop_loss"))
                elif price >= pos["take_profit"]:
                    to_exit.append((symbol, price, "take_profit"))

            for symbol, price, reason in to_exit:
                pos = self.portfolio.positions.pop(symbol)
                proceeds = pos["shares"] * price * 0.9995
                self.portfolio.cash += proceeds
                pnl = proceeds - (pos["shares"] * pos["entry_price"])
                self.portfolio.trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        exit_date=date,
                        exit_price=price,
                        shares=pos["shares"],
                        pnl=pnl,
                        pnl_pct=(price / pos["entry_price"] - 1) * 100,
                        exit_reason=reason,
                    )
                )

            # Find new entries
            if len(self.portfolio.positions) < self.max_positions:
                candidates = []
                for symbol in symbols:
                    if symbol in self.portfolio.positions:
                        continue
                    df = self.get_data_for_date(symbol, date)
                    if df is None:
                        continue
                    score, _ = self.scorer.calculate_score(df)
                    if score >= self.min_score:
                        candidates.append(
                            {
                                "symbol": symbol,
                                "score": score,
                                "price": current_prices.get(symbol),
                            }
                        )

                candidates.sort(key=lambda x: x["score"], reverse=True)
                slots = self.max_positions - len(self.portfolio.positions)

                for c in candidates[:slots]:
                    if c["price"]:
                        pos_value = min(
                            self.portfolio.cash * 0.95,
                            self.portfolio.cash / (slots + 1),
                        )
                        shares = int(pos_value / c["price"])
                        if shares > 0:
                            cost = shares * c["price"] * 1.0005
                            if cost <= self.portfolio.cash:
                                self.portfolio.cash -= cost
                                self.portfolio.positions[c["symbol"]] = {
                                    "shares": shares,
                                    "entry_price": c["price"],
                                    "entry_date": date,
                                    "stop_loss": c["price"] * (1 - self.stop_loss_pct),
                                    "take_profit": c["price"]
                                    * (1 + self.take_profit_pct),
                                }

            equity = self.portfolio.get_equity(current_prices)
            self.portfolio.equity_curve.append({"date": date, "equity": equity})

        # Close remaining positions
        last_date = trading_days[-1]
        for symbol in list(self.portfolio.positions.keys()):
            price = self.get_close_on_date(symbol, last_date)
            if price:
                pos = self.portfolio.positions.pop(symbol)
                proceeds = pos["shares"] * price * 0.9995
                self.portfolio.cash += proceeds
                self.portfolio.trades.append(
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

        return self._calc_metrics()

    def _calc_metrics(self) -> dict:
        trades = self.portfolio.trades
        if not trades:
            return {"error": "no_trades"}

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        final_equity = equity_df["equity"].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

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
            "win_rate": round(len(winners) / len(trades) * 100, 1) if trades else 0,
            "avg_winner": (
                round(np.mean([t.pnl_pct for t in winners]), 2) if winners else 0
            ),
            "avg_loser": (
                round(np.mean([t.pnl_pct for t in losers]), 2) if losers else 0
            ),
            "profit_factor": (
                round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
            ),
            "total_return": round(total_return, 2),
            "max_drawdown": round(max_dd, 2),
            "final_equity": round(final_equity, 2),
            "exit_reasons": exit_reasons,
        }


def load_data(
    symbols: list[str], start: datetime, end: datetime
) -> dict[str, pd.DataFrame]:
    """Load historical data for all symbols."""
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )

    data_cache = {}
    lookback_start = start - timedelta(days=100)

    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        print(f"Loading batch {i//batch_size + 1}...")

        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=lookback_start,
            end=end,
        )

        try:
            bars = client.get_stock_bars(request)
            df = bars.df
            for symbol in batch:
                if symbol in df.index.get_level_values(0):
                    symbol_df = df.loc[symbol].copy().reset_index()
                    symbol_df = symbol_df.rename(
                        columns={"timestamp": "date"}
                    ).set_index("date")
                    data_cache[symbol] = symbol_df
        except Exception as e:
            print(f"Error: {e}")

    return data_cache


def run_sweep():
    """Run parameter sweep."""
    symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "AMD",
        "CRM",
        "ADBE",
        "INTC",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "V",
        "MA",
        "AXP",
        "BLK",
        "C",
        "JNJ",
        "UNH",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "PG",
        "KO",
        "PEP",
        "WMT",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "TGT",
        "CAT",
        "BA",
        "GE",
        "MMM",
        "HON",
        "UPS",
        "RTX",
        "LMT",
        "DE",
        "UNP",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
        "SPY",
    ]

    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2024, 11, 30, tzinfo=UTC)

    # Parameter grid
    stop_losses = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]  # 1%, 2%, 3%, 5%, 7%, 10%
    take_profits = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]  # 3% to 15%
    min_scores = [60.0, 65.0, 70.0]  # Entry threshold

    print("=" * 70)
    print("PARAMETER SWEEP BACKTEST")
    print("=" * 70)
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Stop losses: {stop_losses}")
    print(f"Take profits: {take_profits}")
    print(f"Min scores: {min_scores}")
    print(
        f"Total combinations: {len(stop_losses) * len(take_profits) * len(min_scores)}"
    )
    print("=" * 70)

    # Load data once
    print("\nLoading historical data...")
    data_cache = load_data(symbols, start, end)

    # Get trading days from SPY
    spy_data = data_cache["SPY"]
    trading_days = spy_data[
        (spy_data.index >= start) & (spy_data.index <= end)
    ].index.tolist()

    # Calculate SPY benchmark
    spy_start = spy_data[spy_data.index >= start].iloc[0]["open"]
    spy_end = spy_data[spy_data.index <= end].iloc[-1]["close"]
    spy_return = (spy_end / spy_start - 1) * 100
    print(f"\nSPY Benchmark Return: {spy_return:+.2f}%")

    # Run sweep
    results = []
    total_combos = len(stop_losses) * len(take_profits) * len(min_scores)
    combo_num = 0

    print("\nRunning parameter sweep...")
    for sl, tp, ms in itertools.product(stop_losses, take_profits, min_scores):
        combo_num += 1

        # Skip invalid combos (take profit must be > stop loss)
        if tp <= sl:
            continue

        bt = ParameterSweepBacktest(
            data_cache=data_cache,
            start_date=start,
            end_date=end,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            min_score=ms,
        )

        metrics = bt.run(symbols, trading_days)

        if "error" not in metrics:
            result = {
                "stop_loss": sl,
                "take_profit": tp,
                "min_score": ms,
                **metrics,
                "alpha": round(metrics["total_return"] - spy_return, 2),
            }
            results.append(result)

            if combo_num % 20 == 0:
                print(f"  Completed {combo_num}/{total_combos} combinations...")

    # Sort by alpha
    results.sort(key=lambda x: x["alpha"], reverse=True)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (by Alpha)")
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
    print("BOTTOM 5 PARAMETER COMBINATIONS")
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
    print(
        f"Combinations that beat SPY: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)"
    )

    if profitable:
        best = results[0]
        print("\nBest combination:")
        print(f"  Stop Loss: {best['stop_loss']*100:.0f}%")
        print(f"  Take Profit: {best['take_profit']*100:.0f}%")
        print(f"  Min Score: {best['min_score']}")
        print(f"  Return: {best['total_return']:+.2f}%")
        print(f"  Alpha: {best['alpha']:+.2f}%")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
    else:
        print("\n>>> NO PARAMETER COMBINATION BEATS SPY <<<")
        print("The technical scoring strategy may not have edge in this market.")

    # Save results
    with open("/root/ai_trader/backtesting/sweep_results.json", "w") as f:
        json.dump(
            {
                "spy_return": spy_return,
                "results": results,
                "best": results[0] if results else None,
                "profitable_count": len(profitable),
                "total_count": len(results),
            },
            f,
            indent=2,
            default=str,
        )

    print("\nResults saved to /root/ai_trader/backtesting/sweep_results.json")
    return results


if __name__ == "__main__":
    run_sweep()
