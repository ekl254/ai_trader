#!/usr/bin/env python3
"""Check market and account status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient
from config.config import config

try:
    client = TradingClient(config.alpaca.api_key, config.alpaca.secret_key, paper=True)
    clock = client.get_clock()
    account = client.get_account()

    print("🕐 Current Market Status")
    print("=" * 50)
    print(f"Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}")
    print(f"Next Open:  {clock.next_open}")
    print(f"Next Close: {clock.next_close}")
    print()
    print("💰 Account Status")
    print("=" * 50)
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Buying Power:    ${float(account.buying_power):,.2f}")
    print(f"Cash:            ${float(account.cash):,.2f}")
    
    positions = client.get_all_positions()
    print(f"\nOpen Positions:  {len(positions)}")
    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        symbol = f"{pnl:+.2f} ({pnl_pct:+.1f}%)"
        print(f"  • {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} → {symbol}")
    
    print()
    if clock.is_open:
        print("✅ MARKET IS OPEN! Run: ./run.sh scan")
    else:
        print("⏳ Market opens at 9:30 AM EST (in a few minutes)")
        print("   Run: ./run.sh scan")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
