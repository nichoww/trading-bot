"""
Phase 1 Integration Test
========================
Tests all Phase 1 components working together:
- Config loading
- Data streaming from Alpaca
- ATR calculation from real data
- Position sizing for live trades
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
from rt_config import RTConfig
from data_streamer import DataStreamer
from risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_phase1_integration():
    """Run full Phase 1 integration test"""
    
    print("\n" + "="*70)
    print("PHASE 1 INTEGRATION TEST")
    print("="*70)
    
    # Step 1: Load and validate config
    print("\n[1/5] Loading Configuration...")
    config = RTConfig()
    if not config.validate():
        print("[FAIL] Configuration invalid")
        return False
    print("[PASS] Configuration loaded")
    
    # Step 2: Initialize components
    print("\n[2/5] Initializing Components...")
    try:
        streamer = DataStreamer(config)
        risk_mgr = RiskManager(config)
        print("[PASS] Components initialized")
    except Exception as e:
        print(f"[FAIL] Component initialization: {e}")
        return False
    
    # Step 3: Fetch live data for test symbols
    print("\n[3/5] Fetching Live Data...")
    test_symbols = ['AAPL', 'MSFT', 'NVDA']
    all_data = {}
    
    for symbol in test_symbols:
        bars = streamer.get_historical_bars(symbol, lookback_days=5)
        if bars is not None and len(bars) > 0:
            all_data[symbol] = bars
            print(f"  [OK] {symbol}: {len(bars)} bars")
        else:
            print(f"  [WARN] {symbol}: No data")
    
    if len(all_data) == 0:
        print("[FAIL] No data fetched")
        return False
    print(f"[PASS] Fetched data for {len(all_data)} symbols")
    
    # Step 4: Calculate ATR and sizing for each symbol
    print("\n[4/5] Calculating Risk Parameters...")
    portfolio_value = 100000  # $100k paper account
    
    print(f"\n{'Symbol':<8} {'Price':<10} {'ATR':<10} {'Shares':<8} {'Value':<12} {'Stop':<10} {'Risk $':<10}")
    print("-"*70)
    
    for symbol, df in all_data.items():
        current_price = float(df['close'].iloc[-1])
        atr = risk_mgr.calculate_atr(df)
        shares, value, stop = risk_mgr.calculate_position_size(
            portfolio_value, current_price, atr
        )
        risk = shares * (current_price - stop)
        
        print(f"{symbol:<8} ${current_price:<9.2f} ${atr:<9.2f} {shares:<8} ${value:<11,.2f} ${stop:<9.2f} ${risk:<9.2f}")
    
    print("-"*70)
    print("[PASS] Risk parameters calculated")
    
    # Step 5: Test trailing stop with real price movement
    print("\n[5/5] Simulating Trailing Stop...")
    
    # Use AAPL for trailing stop simulation
    if 'AAPL' in all_data:
        df = all_data['AAPL']
        entry = float(df['close'].iloc[-10])
        atr = risk_mgr.calculate_atr(df)
        stop = risk_mgr.calculate_stop_price(entry, atr)
        trail_base = entry
        
        print(f"\n  Entry: ${entry:.2f}, Initial Stop: ${stop:.2f}")
        print(f"  Simulating with last 10 bars...")
        
        for i in range(-10, 0):
            price = float(df['close'].iloc[i])
            new_stop, trail_base = risk_mgr.update_trailing_stop(price, stop, trail_base)
            
            # Check exit conditions
            should_exit, reason = risk_mgr.check_exit_conditions(
                price, new_stop, entry * 1.10  # 10% TP
            )
            
            status = "EXIT" if should_exit else ""
            if new_stop > stop:
                print(f"    Bar {i+11}: ${price:.2f} | Stop: ${stop:.2f} -> ${new_stop:.2f} {status}")
            stop = new_stop
        
        print("[PASS] Trailing stop simulation complete")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 INTEGRATION TEST: ALL PASSED ✓")
    print("="*70)
    print("\nComponents ready:")
    print("  ✓ RTConfig - Configuration management")
    print("  ✓ DataStreamer - Live market data from Alpaca")
    print("  ✓ RiskManager - ATR, position sizing, trailing stops")
    print("\nReady for Phase 2: Order Management")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_phase1_integration()
    exit(0 if success else 1)
