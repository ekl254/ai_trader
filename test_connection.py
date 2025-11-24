#!/usr/bin/env python3
"""Test API connections and system readiness."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from alpaca.trading.client import TradingClient
from src.logger import logger

def test_alpaca_connection():
    """Test Alpaca connection."""
    try:
        client = TradingClient(
            config.alpaca.api_key,
            config.alpaca.secret_key,
            paper=True,
        )
        
        # Get account info
        account = client.get_account()
        clock = client.get_clock()
        
        print("✅ Alpaca Connection SUCCESS!")
        print(f"   Account: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Market Status: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")
        
        return True
    except Exception as e:
        print(f"❌ Alpaca Connection FAILED: {e}")
        return False


def test_eodhd_connection():
    """Test EODHD connection."""
    try:
        from src.data_provider import eodhd_provider
        
        # Try to get fundamentals for AAPL
        data = eodhd_provider.get_fundamentals("AAPL")
        
        if data and isinstance(data, dict):
            print("✅ EODHD Connection SUCCESS!")
            print(f"   Test Symbol: AAPL")
            print(f"   API Requests Used: {eodhd_provider.request_count}/20")
            return True
        else:
            print("❌ EODHD returned invalid data")
            return False
            
    except Exception as e:
        print(f"❌ EODHD Connection FAILED: {e}")
        return False


def main():
    """Run all connection tests."""
    print("🔍 Testing AI Trading System Connections\n")
    
    print("=" * 50)
    alpaca_ok = test_alpaca_connection()
    
    print("\n" + "=" * 50)
    eodhd_ok = test_eodhd_connection()
    
    print("\n" + "=" * 50)
    if alpaca_ok and eodhd_ok:
        print("\n✅ ALL SYSTEMS READY!")
        print("\n💡 Next Steps:")
        print("   1. Wait for market open (9:30 AM EST)")
        print("   2. Run: ./run.sh scan")
        print("   3. Watch: tail -f logs/trading.log | jq .")
    else:
        print("\n❌ SOME SYSTEMS FAILED - Check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
