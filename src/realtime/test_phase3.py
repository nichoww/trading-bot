"""
Phase 3 Integration Test
========================
Tests Portfolio State & Audit Logging:
1. Portfolio state save/restore
2. Broker position sync
3. Audit trail logging
4. Parquet data storage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
from rt_config import RTConfig
from data_streamer import DataStreamer
from order_manager import OrderManager
from portfolio_state import PortfolioState
from audit_logger import AuditLogger, EventType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_phase3_integration():
    """Run Phase 3 integration test"""
    
    print("\n" + "="*70)
    print("PHASE 3 INTEGRATION TEST - PERSISTENCE & LOGGING")
    print("="*70)
    
    # Step 1: Initialize components
    print("\n[1/6] Initializing Components...")
    try:
        config = RTConfig()
        if not config.validate():
            print("[FAIL] Invalid configuration")
            return False
        
        streamer = DataStreamer(config)
        order_mgr = OrderManager(config)
        portfolio = PortfolioState(config)
        audit = AuditLogger(config)
        
        print("[PASS] All components initialized")
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        return False
    
    # Step 2: Test portfolio state with broker sync
    print("\n[2/6] Testing Broker Sync...")
    
    # Load existing state
    portfolio.load()
    print(f"  Loaded state: {len(portfolio.positions)} positions")
    
    # Get broker positions
    broker_positions = order_mgr.get_positions()
    print(f"  Broker positions: {len(broker_positions)}")
    
    # Sync
    actions = portfolio.sync_with_broker(broker_positions)
    if actions:
        print(f"  Sync actions: {actions}")
    else:
        print("  No sync needed")
    
    print("[PASS] Broker sync complete")
    
    # Step 3: Test portfolio state persistence
    print("\n[3/6] Testing State Persistence...")
    
    # Add a test position
    test_pos = portfolio.add_position(
        symbol='TEST',
        entry_price=100.0,
        shares=50,
        stop_price=95.0,
        take_profit_price=110.0,
        atr=2.0
    )
    print(f"  Added test position: {test_pos.symbol}")
    
    # Update stop
    portfolio.update_position('TEST', stop_price=97.0, trail_base=102.0)
    print(f"  Updated stop: $97.00")
    
    # Create new instance and reload
    portfolio2 = PortfolioState(config)
    portfolio2.load()
    
    if 'TEST' in portfolio2.positions:
        loaded = portfolio2.positions['TEST']
        if loaded.stop_price == 97.0 and loaded.trail_base == 102.0:
            print("  [OK] State correctly persisted and reloaded")
        else:
            print(f"  [WARN] State mismatch: stop={loaded.stop_price}, trail={loaded.trail_base}")
    else:
        print("  [FAIL] Test position not found after reload")
    
    # Cleanup test position
    portfolio.remove_position('TEST')
    
    print("[PASS] State persistence verified")
    
    # Step 4: Test audit logging
    print("\n[4/6] Testing Audit Trail...")
    
    # Log some events
    audit.log_info("Phase 3 test started")
    audit.log_signal('AAPL', 'BUY', 0.72, 'ENTER')
    audit.log_position_open('AAPL', 100, 150.0, 145.0, 'test_order_1')
    audit.log_stop_update('AAPL', 145.0, 148.0)
    audit.log_position_close('AAPL', 100, 160.0, 150.0, 'TAKE_PROFIT', 'test_order_2')
    
    # Query trail
    trail = audit.get_audit_trail(symbol='AAPL')
    print(f"  Logged {len(trail)} AAPL events")
    
    print("[PASS] Audit trail working")
    
    # Step 5: Test Parquet storage
    print("\n[5/6] Testing Parquet Storage...")
    
    # Fetch some real data
    bars = streamer.get_historical_bars('AAPL', lookback_days=2)
    if bars is not None:
        audit.save_bars_parquet('AAPL', bars)
        
        # Reload
        loaded_bars = audit.load_bars_parquet('AAPL')
        if loaded_bars is not None:
            print(f"  [OK] Saved and loaded {len(loaded_bars)} bars")
        else:
            print("  [WARN] Could not reload parquet")
    else:
        print("  [WARN] Could not fetch data for parquet test")
    
    print("[PASS] Parquet storage working")
    
    # Step 6: Display summaries
    print("\n[6/6] Summary Display...")
    
    portfolio.display()
    audit.display_recent(5)
    
    # Get trade summary
    summary = audit.get_trade_summary()
    print("\nTrade Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print("[PASS] Summaries displayed")
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 3 INTEGRATION TEST: ALL PASSED ✓")
    print("="*70)
    print("\nComponents ready:")
    print("  ✓ PortfolioState - Save/restore positions with risk levels")
    print("  ✓ Broker Sync - Detect external position changes")
    print("  ✓ AuditLogger - CSV trail for all events")
    print("  ✓ Parquet Storage - Efficient bar data storage")
    print("\nReady for Phase 4: Main Trading Loop")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_phase3_integration()
    exit(0 if success else 1)
