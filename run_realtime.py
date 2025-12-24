#!/usr/bin/env python3
"""
Run Real-Time Trading
=====================
Main entry point for the real-time trading system.

Usage:
    python run_realtime.py              # Run in test mode (default)
    python run_realtime.py --live       # Run in live mode (real orders!)
    python run_realtime.py --once       # Run single iteration
    python run_realtime.py --dashboard  # Show dashboard while running
"""

import argparse
import logging
import sys
import os
import signal
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "realtime"))

from src.realtime.rt_config import RTConfig
from src.realtime.realtime_engine import RealtimeEngine
from src.realtime.dashboard import SimpleDashboard


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path(__file__).parent / "logs" / "realtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )
    
    return log_file


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    # On Windows, SIGINT can be triggered spuriously by various things
    # We'll track the count and only exit after multiple signals
    global _sigint_count
    _sigint_count += 1
    
    if _sigint_count >= 3:
        print(f"\n\nShutdown confirmed after {_sigint_count} signals...")
        raise KeyboardInterrupt
    else:
        print(f"\n[Signal {signum} received - press Ctrl+C {3 - _sigint_count} more time(s) to exit]")

# Count SIGINT signals
_sigint_count = 0


def execute_demo_trade(engine):
    """Execute a small demo trade to show the bot working."""
    import time
    from src.realtime.order_manager import OrderSide
    
    # Use Ford (F) - it's cheap and liquid
    symbol = 'F'
    qty = 5  # Buy 5 shares
    
    try:
        # Get current price
        price = engine.streamer.get_current_price(symbol)
        if price is None:
            print(f"   [!] Could not get price for {symbol}")
            return
        
        # Check if we already have a position
        if symbol in engine.portfolio.positions:
            print(f"   [!] Already have position in {symbol}, selling instead...")
            pos = engine.portfolio.positions[symbol]
            
            # Sell the position
            order = engine.order_manager.submit_order(
                symbol=symbol,
                qty=pos.shares,
                side=OrderSide.SELL,
                limit_price=price * 0.995  # Slightly below market
            )
            
            if order:
                print(f"   [OK] DEMO SELL: {pos.shares} {symbol} @ ~${price:.2f}")
                engine.portfolio.remove_position(symbol)
        else:
            # Buy shares
            print(f"   [*] Buying {qty} shares of {symbol} @ ${price:.2f}...")
            
            order = engine.order_manager.submit_order(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                limit_price=price * 1.005  # Slightly above market
            )
            
            if order:
                print(f"   [OK] DEMO BUY: {qty} {symbol} @ ~${price:.2f}")
                # Add to portfolio
                stop_price = price * 0.97  # 3% stop
                engine.portfolio.add_position(
                    symbol=symbol,
                    entry_price=price,
                    shares=qty,
                    stop_price=stop_price,
                    take_profit_price=price * 1.06,  # 6% profit target
                    atr=price * 0.02
                )
            else:
                print(f"   [!] Order failed for {symbol}")
        
        # Wait a moment for order to process
        time.sleep(2)
        print(f"   [*] Check Alpaca dashboard for the trade!")
        
    except Exception as e:
        print(f"   [!] Demo trade error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_realtime.py                    # Test mode, all symbols
  python run_realtime.py --live             # LIVE MODE - real orders!
  python run_realtime.py --symbols AAPL NVDA
  python run_realtime.py --once --verbose   # Single iteration, debug output
  python run_realtime.py --dashboard        # With simple dashboard
        """
    )
    
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Enable LIVE trading (real orders!). Default is test mode.'
    )
    
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt for live mode (use with caution!)'
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run single iteration then exit'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show simple dashboard while running'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of symbols to trade (default: config symbols)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Loop interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=2.0,
        help='Risk per trade %% (default: 2.0)'
    )
    
    parser.add_argument(
        '--demo-trade',
        action='store_true',
        help='Execute a small demo trade immediately (buy 1 share of F)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print("\n" + "=" * 60)
    print("  REAL-TIME TRADING BOT")
    print("=" * 60)
    
    # Confirm live mode
    test_mode = not args.live
    
    if not test_mode and not args.no_confirm:
        print("\n[!] WARNING: LIVE MODE ENABLED")
        print("    Real orders will be placed!")
        print("\n    Type 'YES' to confirm: ", end='')
        
        confirm = input().strip()
        if confirm != 'YES':
            print("Cancelled.")
            return 1
    elif not test_mode:
        print("\n[!] LIVE MODE ENABLED (auto-confirmed)")
    else:
        print("\n[*] Running in TEST MODE (no real orders)")
    
    # Create config
    config = RTConfig()
    
    if args.symbols:
        config.symbols = args.symbols
    
    config.loop_interval_seconds = args.interval
    # Convert percentage to decimal (2.0% -> 0.02)
    config.risk_per_trade_pct = args.risk / 100.0
    
    print(f"\n   Symbols: {', '.join(config.symbols)}")
    print(f"   Interval: {config.loop_interval_seconds}s")
    print(f"   Risk: {args.risk}%")
    print(f"   Log: {log_file}")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create engine
    engine = RealtimeEngine(config, test_mode=test_mode)
    
    try:
        if args.once:
            # Single iteration
            print("\n   Running single iteration...")
            
            # Manual startup so we can demo trade before completion
            engine.startup()
            
            # Execute demo trade if requested
            if args.demo_trade and not test_mode:
                print("\n   Executing demo trade...")
                execute_demo_trade(engine)
            
            # Run single iteration
            engine._run_single_iteration()
            engine.stop()
            
        elif args.dashboard:
            # Start with dashboard
            engine.startup()
            engine.start_background()
            
            dashboard = SimpleDashboard(engine)
            dashboard.start()  # This blocks until Ctrl+C
            
            engine.stop()
            
        else:
            # Normal run
            print("\n   Starting trading loop...")
            print("   Press Ctrl+C to stop\n")
            
            engine.startup()
            engine.run_loop()  # This blocks until Ctrl+C
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        engine.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[OK] Trading session ended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
