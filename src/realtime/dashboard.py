"""
Real-Time Trading Dashboard
============================
Console-based monitoring display for the trading engine.

Features:
- Live position tracking
- P&L display
- Signal history
- System status
"""

import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, List
import logging

# Handle both package and standalone imports
try:
    from .realtime_engine import RealtimeEngine
    from .rt_config import RTConfig
except ImportError:
    from realtime_engine import RealtimeEngine
    from rt_config import RTConfig

logger = logging.getLogger(__name__)


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_currency(value: float) -> str:
    """Format as currency"""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_pct(value: float) -> str:
    """Format as percentage"""
    return f"{value:+.2%}"


class TradingDashboard:
    """
    Console dashboard for monitoring real-time trading.
    """
    
    def __init__(self, engine: RealtimeEngine):
        """
        Initialize dashboard.
        
        Args:
            engine: RealtimeEngine instance to monitor
        """
        self.engine = engine
        self.running = False
        self.refresh_rate = 5  # seconds
        self.signal_history: List[Dict] = []
        self.max_history = 10
    
    def start(self):
        """Start dashboard refresh loop"""
        self.running = True
        
        while self.running:
            try:
                self._render()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.running = False
                break
    
    def stop(self):
        """Stop dashboard"""
        self.running = False
    
    def _render(self):
        """Render dashboard to console"""
        clear_screen()
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = self.engine.get_status()
        
        print("=" * 70)
        print(f"  REAL-TIME TRADING DASHBOARD                     {now}")
        print("=" * 70)
        
        # Engine Status
        running_str = "🟢 RUNNING" if status['running'] else "🔴 STOPPED"
        mode_str = "[TEST MODE]" if status['test_mode'] else "[LIVE]"
        print(f"\n  Engine: {running_str} {mode_str}")
        print(f"  Loops: {status['loop_count']}")
        if status['last_loop']:
            print(f"  Last: {status['last_loop']}")
        
        # Account Info
        print("\n" + "-" * 70)
        print("  ACCOUNT")
        print("-" * 70)
        print(f"  Equity:      {format_currency(status['equity'])}")
        
        # Positions
        positions = status.get('positions', {})
        print(f"  Positions:   {len(positions)}/{self.engine.config.max_positions}")
        
        print("\n" + "-" * 70)
        print("  OPEN POSITIONS")
        print("-" * 70)
        
        if positions:
            print(f"  {'Symbol':<8} {'Qty':>8} {'Entry':>10} {'Stop':>10} {'Trail':>10}")
            print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
            
            for symbol, pos in positions.items():
                print(f"  {symbol:<8} {pos['quantity']:>8} "
                      f"${pos['entry']:>9.2f} ${pos['stop']:>9.2f} ${pos['trail']:>9.2f}")
        else:
            print("  No open positions")
        
        # Recent Signals
        print("\n" + "-" * 70)
        print("  RECENT SIGNALS")
        print("-" * 70)
        
        if self.signal_history:
            for sig in self.signal_history[-5:]:
                print(f"  {sig['time']} | {sig['symbol']:<6} | {sig['signal']:<4} | {sig['confidence']:.0%}")
        else:
            print("  Waiting for signals...")
        
        # Errors
        if status['errors'] > 0:
            print("\n" + "-" * 70)
            print(f"  ERRORS ({status['errors']})")
            print("-" * 70)
            for err in status['last_errors']:
                print(f"  ⚠️  {err[:60]}...")
        
        # Footer
        print("\n" + "=" * 70)
        print("  Press Ctrl+C to stop")
        print("=" * 70)
    
    def add_signal(self, symbol: str, signal: str, confidence: float):
        """
        Add signal to history.
        
        Args:
            symbol: Stock ticker
            signal: Signal type (BUY/SELL/HOLD)
            confidence: Signal confidence
        """
        self.signal_history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence
        })
        
        # Trim history
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]


class SimpleDashboard:
    """
    Simple text-based dashboard without clearing screen.
    Better for logging/debugging.
    """
    
    def __init__(self, engine: RealtimeEngine):
        self.engine = engine
        self.running = False
        self.refresh_rate = 30  # seconds
    
    def start(self):
        """Start dashboard"""
        self.running = True
        
        print("\n" + "=" * 60)
        print("SIMPLE DASHBOARD - Status updates every 30s")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        while self.running:
            try:
                self._print_status()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.running = False
                break
    
    def _print_status(self):
        """Print status line"""
        status = self.engine.get_status()
        now = datetime.now().strftime("%H:%M:%S")
        
        pos_count = len(status.get('positions', {}))
        
        running = "RUN" if status['running'] else "STOP"
        mode = "TEST" if status['test_mode'] else "LIVE"
        
        print(f"[{now}] {running}/{mode} | "
              f"Loop #{status['loop_count']} | "
              f"Positions: {pos_count}/{self.engine.config.max_positions} | "
              f"Equity: {format_currency(status['equity'])} | "
              f"Errors: {status['errors']}")


if __name__ == "__main__":
    # Demo dashboard with test engine
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for dashboard
        format='%(message)s'
    )
    
    print("\n" + "=" * 60)
    print("DASHBOARD DEMO")
    print("=" * 60)
    
    # Create test engine
    config = RTConfig()
    config.symbols = ['AAPL', 'NVDA', 'TSLA']
    
    engine = RealtimeEngine(config, test_mode=True)
    engine.startup()
    
    # Create dashboard
    dashboard = SimpleDashboard(engine)
    
    print("\nDashboard initialized. Starting status display...")
    print("(Ctrl+C to exit)\n")
    
    # Just show one status update for demo
    dashboard._print_status()
    
    print("\n[OK] Dashboard demo complete!")
