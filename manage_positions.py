#!/usr/bin/env python3
"""CLI tool for managing position locks and viewing rebalancing history."""

import argparse
import sys
from datetime import datetime

from alpaca.trading.client import TradingClient

from config.config import config
from src.position_tracker import position_tracker


def list_positions() -> None:
    """List all current positions."""
    client = TradingClient(config.alpaca.api_key, config.alpaca.secret_key, paper=True)
    positions = client.get_all_positions()

    if not positions:
        print("No open positions")
        return

    print(
        f"\n{'Symbol':<8} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P/L %':<10} {'Locked':<8}"
    )
    print("-" * 70)

    for pos in positions:
        is_locked = "🔒 YES" if position_tracker.is_locked(pos.symbol) else "   NO"
        pl_pct = float(pos.unrealized_plpc or 0) * 100

        print(
            f"{pos.symbol:<8} "
            f"{pos.qty:<8} "
            f"${float(pos.avg_entry_price):<9.2f} "
            f"${float(pos.current_price or 0):<9.2f} "
            f"{pl_pct:<9.2f}% "
            f"{is_locked:<8}"
        )

    print()


def lock_position(symbol: str) -> None:
    """Lock a position to prevent rebalancing."""
    # Verify position exists
    client = TradingClient(config.alpaca.api_key, config.alpaca.secret_key, paper=True)
    positions = {pos.symbol: pos for pos in client.get_all_positions()}

    if symbol not in positions:
        print(f"❌ Error: No open position for {symbol}")
        print(f"\nAvailable positions: {', '.join(positions.keys())}")
        return

    if position_tracker.is_locked(symbol):
        print(f"ℹ️  {symbol} is already locked")
        return

    position_tracker.lock_position(symbol)
    print(f"✅ Locked {symbol} - position will NOT be rebalanced")


def unlock_position(symbol: str) -> None:
    """Unlock a position to allow rebalancing."""
    if not position_tracker.is_locked(symbol):
        print(f"ℹ️  {symbol} is not locked")
        return

    position_tracker.unlock_position(symbol)
    print(f"✅ Unlocked {symbol} - position can now be rebalanced")


def list_locked() -> None:
    """List all locked positions."""
    locked = position_tracker.get_locked_positions()

    if not locked:
        print("No locked positions")
        return

    print(f"\n🔒 Locked Positions ({len(locked)}):")
    for symbol in locked:
        print(f"  - {symbol}")
    print()


def show_rebalancing_history(limit: int = 20) -> None:
    """Show rebalancing history."""
    history = position_tracker.get_rebalancing_history(limit=limit)

    if not history:
        print("No rebalancing history")
        return

    print(f"\n📊 Rebalancing History (last {len(history)} events):")
    print(
        f"{'Timestamp':<20} {'Old→New':<20} {'Old Score':<12} {'New Score':<12} {'Diff':<8}"
    )
    print("-" * 80)

    for event in history:
        timestamp = datetime.fromisoformat(event["timestamp"]).strftime(
            "%Y-%m-%d %H:%M"
        )
        swap = f"{event['old_symbol']}→{event['new_symbol']}"

        print(
            f"{timestamp:<20} "
            f"{swap:<20} "
            f"{event['old_score']:<12.2f} "
            f"{event['new_score']:<12.2f} "
            f"+{event['score_diff']:<7.2f}"
        )

    print()


def show_rebalancing_stats() -> None:
    """Show rebalancing statistics."""
    stats = position_tracker.get_rebalancing_stats()

    print("\n📈 Rebalancing Statistics:")
    print(f"  Total Rebalances: {stats['total_rebalances']}")

    if stats["total_rebalances"] > 0:
        print(f"  Avg Score Improvement: +{stats['avg_score_improvement']:.2f}")
        print(f"  Min Score Improvement: +{stats['min_score_improvement']:.2f}")
        print(f"  Max Score Improvement: +{stats['max_score_improvement']:.2f}")

    last_rebalance = position_tracker.get_last_rebalance_time()
    if last_rebalance:
        print(f"  Last Rebalance: {last_rebalance.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    cooldown_min = config.trading.rebalance_cooldown_minutes
    can_rebalance = position_tracker.can_rebalance_now(cooldown_min)
    print(f"  Cooldown Period: {cooldown_min} minutes")
    print(
        f"  Can Rebalance Now: {'✅ YES' if can_rebalance else '❌ NO (in cooldown)'}"
    )

    print()


def unlock_all() -> None:
    """Unlock all positions."""
    locked = position_tracker.get_locked_positions()

    if not locked:
        print("No locked positions to unlock")
        return

    for symbol in locked:
        position_tracker.unlock_position(symbol)

    print(f"✅ Unlocked {len(locked)} positions: {', '.join(locked)}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage position locks and view rebalancing history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all positions and their lock status
  python manage_positions.py list

  # Lock a position to prevent rebalancing
  python manage_positions.py lock AAPL

  # Unlock a position
  python manage_positions.py unlock AAPL

  # Show locked positions
  python manage_positions.py locked

  # Show rebalancing history
  python manage_positions.py history

  # Show rebalancing statistics
  python manage_positions.py stats

  # Unlock all positions
  python manage_positions.py unlock-all
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "lock", "unlock", "locked", "history", "stats", "unlock-all"],
        help="Command to execute",
    )

    parser.add_argument(
        "symbol", nargs="?", help="Stock symbol (required for lock/unlock commands)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of history events to show (default: 20)",
    )

    args = parser.parse_args()

    try:
        if args.command == "list":
            list_positions()

        elif args.command == "lock":
            if not args.symbol:
                print("❌ Error: symbol required for lock command")
                print("Usage: python manage_positions.py lock SYMBOL")
                sys.exit(1)
            lock_position(args.symbol.upper())

        elif args.command == "unlock":
            if not args.symbol:
                print("❌ Error: symbol required for unlock command")
                print("Usage: python manage_positions.py unlock SYMBOL")
                sys.exit(1)
            unlock_position(args.symbol.upper())

        elif args.command == "locked":
            list_locked()

        elif args.command == "history":
            show_rebalancing_history(limit=args.limit)

        elif args.command == "stats":
            show_rebalancing_stats()

        elif args.command == "unlock-all":
            unlock_all()

    except KeyboardInterrupt:
        print("\n\nOperation cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
