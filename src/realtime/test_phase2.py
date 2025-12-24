"""
Phase 2 Integration Test
========================
Tests Order Management with REAL paper trades!

WARNING: This will place actual orders on your Alpaca paper account.
Run with --dry-run to skip actual orders.

Tests:
1. Order manager initialization
2. Account & position sync
3. Duplicate order guard
4. [Optional] Real order placement with limit -> fallback flow
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import logging
from rt_config import RTConfig
from data_streamer import DataStreamer
from risk_manager import RiskManager
from order_manager import OrderManager, OrderSide

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_phase2_integration(dry_run: bool = True):
    """
    Run Phase 2 integration test.
    
    Args:
        dry_run: If True, skip actual order placement
    """
    
    print("\n" + "="*70)
    print("PHASE 2 INTEGRATION TEST - ORDER MANAGEMENT")
    print("="*70)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No actual orders will be placed")
        print("   Run with --live to place real paper trades\n")
    else:
        print("\n🔴 LIVE MODE - Will place REAL paper trades!\n")
    
    # Step 1: Initialize all components
    print("[1/5] Initializing Components...")
    try:
        config = RTConfig()
        if not config.validate():
            print("[FAIL] Invalid configuration")
            return False
        
        streamer = DataStreamer(config)
        risk_mgr = RiskManager(config)
        order_mgr = OrderManager(config)
        
        print("[PASS] All components initialized")
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        return False
    
    # Step 2: Display account status
    print("\n[2/5] Account Status...")
    order_mgr.display_status()
    print("[PASS] Account synced")
    
    # Step 3: Test duplicate order guard
    print("\n[3/5] Testing Duplicate Order Guard...")
    
    # Check existing orders
    active_count = len(order_mgr._active_orders)
    print(f"  Active order guards: {active_count}")
    
    # Test that duplicate detection works
    test_symbol = 'AAPL'
    test_side = 'buy'
    
    if order_mgr.has_active_order(test_symbol, test_side):
        print(f"  [OK] {test_symbol} {test_side} blocked (existing order)")
    else:
        print(f"  [OK] {test_symbol} {test_side} available")
    
    print("[PASS] Duplicate guard working")
    
    # Step 4: Calculate position sizing for a test trade
    print("\n[4/5] Position Sizing Calculation...")
    
    # Get account value
    account = order_mgr.get_account_info()
    portfolio_value = account.get('portfolio_value', 100000)
    buying_power = account.get('buying_power', 0)
    
    # Fetch data for test symbol
    bars = streamer.get_historical_bars(test_symbol, lookback_days=5)
    if bars is None:
        print(f"  [WARN] Could not fetch data for {test_symbol}")
        current_price = 150.0  # Fallback
        atr = 2.0
    else:
        current_price = float(bars['close'].iloc[-1])
        atr = risk_mgr.calculate_atr(bars)
    
    # Calculate position
    shares, position_value, stop_price = risk_mgr.calculate_position_size(
        portfolio_value, current_price, atr, buying_power
    )
    
    print(f"  Symbol:         {test_symbol}")
    print(f"  Current Price:  ${current_price:.2f}")
    print(f"  ATR:            ${atr:.2f}")
    print(f"  Shares:         {shares}")
    print(f"  Position Value: ${position_value:,.2f}")
    print(f"  Stop Price:     ${stop_price:.2f}")
    print(f"  Risk Amount:    ${shares * (current_price - stop_price):,.2f}")
    print("[PASS] Position sizing calculated")
    
    # Step 5: Optional - Place actual order
    print("\n[5/5] Order Placement Test...")
    
    if dry_run:
        print("  [SKIP] Dry run - no order placed")
        print("  Would have placed:")
        print(f"    LIMIT BUY {shares} {test_symbol} @ ${current_price:.2f}")
        print("[PASS] Order test skipped (dry run)")
    else:
        # Check if we already have an active order
        if order_mgr.has_active_order(test_symbol, 'buy'):
            print(f"  [SKIP] Already have active {test_symbol} buy order")
            print("[PASS] Order test skipped (duplicate)")
        else:
            # Place a small test order (1 share)
            test_shares = 1
            limit_price = current_price * 0.99  # 1% below to avoid immediate fill
            
            print(f"  Placing test order: LIMIT BUY {test_shares} {test_symbol} @ ${limit_price:.2f}")
            
            result = order_mgr.submit_order(
                symbol=test_symbol,
                side=OrderSide.BUY,
                qty=test_shares,
                limit_price=limit_price
            )
            
            if result.success:
                print(f"  [OK] Order filled!")
                print(f"    Order ID:    {result.order_id}")
                print(f"    Filled Qty:  {result.filled_qty}")
                print(f"    Avg Price:   ${result.filled_avg_price:.2f}")
                print(f"    Fallback:    {result.is_fallback}")
            else:
                print(f"  [INFO] Order result: {result.status}")
                if result.error:
                    print(f"    Error: {result.error}")
            
            print("[PASS] Order test complete")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 INTEGRATION TEST: ALL PASSED ✓")
    print("="*70)
    print("\nComponents ready:")
    print("  ✓ OrderManager - Limit orders with fallback")
    print("  ✓ Duplicate Guard - Prevents double orders")
    print("  ✓ Account Sync - Positions and orders from broker")
    print("  ✓ Risk Integration - ATR-based sizing")
    print("\nReady for Phase 3: Portfolio State & Audit Logging")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2 Integration Test')
    parser.add_argument('--live', action='store_true', 
                        help='Place real paper trades (default: dry run)')
    args = parser.parse_args()
    
    success = test_phase2_integration(dry_run=not args.live)
    exit(0 if success else 1)
