"""
Audit Logger
============
Comprehensive logging and data storage:
- CSV audit trail for all orders and trades
- Parquet storage for bar data
- Event logging with timestamps
"""

import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import logging
import pandas as pd

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
    from .order_manager import OrderResult
except ImportError:
    from rt_config import RTConfig
    from order_manager import OrderResult

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events to log"""
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_UPDATED = "STOP_UPDATED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ERROR = "ERROR"
    INFO = "INFO"


class AuditLogger:
    """
    Handles all audit logging and data storage.
    
    Features:
    - CSV audit trail for orders/trades
    - Parquet storage for market data
    - Structured event logging
    - Query interface for analysis
    """
    
    def __init__(self, config: RTConfig):
        """
        Initialize AuditLogger.
        
        Args:
            config: RTConfig instance
        """
        self.config = config
        
        # Setup directories
        self.logs_dir = Path(config.logs_dir)
        self.parquet_dir = Path(config.parquet_dir)
        self.audit_csv = Path(config.audit_csv)
        
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if needed
        self._init_csv()
        
        logger.info(f"[OK] AuditLogger initialized (csv: {self.audit_csv})")
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.audit_csv.exists():
            headers = [
                'timestamp',
                'event_type',
                'symbol',
                'side',
                'qty',
                'price',
                'order_id',
                'client_order_id',
                'status',
                'filled_qty',
                'filled_price',
                'pnl',
                'message'
            ]
            
            with open(self.audit_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_event(
        self,
        event_type: EventType,
        symbol: str = '',
        side: str = '',
        qty: float = 0,
        price: float = 0,
        order_id: str = '',
        client_order_id: str = '',
        status: str = '',
        filled_qty: float = 0,
        filled_price: float = 0,
        pnl: float = 0,
        message: str = ''
    ):
        """
        Log an event to the audit trail.
        
        Args:
            event_type: Type of event
            symbol: Stock ticker
            side: buy/sell
            qty: Order quantity
            price: Order price
            order_id: Alpaca order ID
            client_order_id: Client order ID
            status: Order status
            filled_qty: Filled quantity
            filled_price: Average fill price
            pnl: Profit/loss (for closes)
            message: Additional message
        """
        timestamp = datetime.now().isoformat()
        
        # Handle both enum and string event types
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        
        row = [
            timestamp,
            event_type_str,
            symbol,
            side,
            qty,
            price,
            order_id,
            client_order_id,
            status,
            filled_qty,
            filled_price,
            pnl,
            message
        ]
        
        try:
            with open(self.audit_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            logger.debug(f"Logged {event_type_str}: {symbol} {message}")
            
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")
    
    def log_order(self, result: OrderResult, event_type: EventType = None):
        """
        Log an order result to the audit trail.
        
        Args:
            result: OrderResult from OrderManager
            event_type: Override event type (auto-detected if None)
        """
        if event_type is None:
            if result.success and result.filled_qty > 0:
                event_type = EventType.ORDER_FILLED
            elif result.success:
                event_type = EventType.ORDER_SUBMITTED
            elif 'cancel' in str(result.error).lower():
                event_type = EventType.ORDER_CANCELLED
            else:
                event_type = EventType.ORDER_REJECTED
        
        self.log_event(
            event_type=event_type,
            symbol=result.symbol,
            side=result.side,
            qty=result.qty,
            price=result.filled_avg_price or 0,
            order_id=result.order_id or '',
            client_order_id=result.client_order_id or '',
            status=result.status,
            filled_qty=result.filled_qty,
            filled_price=result.filled_avg_price or 0,
            message=result.error or ('fallback' if result.is_fallback else '')
        )
    
    def log_position_open(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_price: float,
        order_id: str = ''
    ):
        """Log position opening"""
        self.log_event(
            event_type=EventType.POSITION_OPENED,
            symbol=symbol,
            side='buy',
            qty=shares,
            price=entry_price,
            order_id=order_id,
            message=f"Stop: ${stop_price:.2f}"
        )
    
    def log_position_close(
        self,
        symbol: str,
        shares: int,
        exit_price: float,
        entry_price: float,
        reason: str,
        order_id: str = ''
    ):
        """Log position closing"""
        pnl = shares * (exit_price - entry_price)
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        self.log_event(
            event_type=EventType.POSITION_CLOSED,
            symbol=symbol,
            side='sell',
            qty=shares,
            price=exit_price,
            order_id=order_id,
            pnl=pnl,
            message=f"{reason} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
        )
    
    def log_stop_update(
        self,
        symbol: str,
        old_stop: float,
        new_stop: float
    ):
        """Log stop price update"""
        self.log_event(
            event_type=EventType.STOP_UPDATED,
            symbol=symbol,
            price=new_stop,
            message=f"${old_stop:.2f} -> ${new_stop:.2f}"
        )
    
    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        action: str
    ):
        """Log signal generation"""
        self.log_event(
            event_type=EventType.SIGNAL_GENERATED,
            symbol=symbol,
            message=f"Signal: {signal} | Confidence: {confidence:.2%} | Action: {action}"
        )
    
    def log_error(self, message: str, symbol: str = ''):
        """Log an error"""
        self.log_event(
            event_type=EventType.ERROR,
            symbol=symbol,
            message=message
        )
    
    def log_info(self, message: str, symbol: str = ''):
        """Log info message"""
        self.log_event(
            event_type=EventType.INFO,
            symbol=symbol,
            message=message
        )
    
    # =========================================================================
    # PARQUET STORAGE
    # =========================================================================
    
    def save_bars_parquet(self, symbol: str, df: pd.DataFrame):
        """
        Save bar data to Parquet file.
        
        Args:
            symbol: Stock ticker
            df: DataFrame with OHLCV data
        """
        try:
            filepath = self.parquet_dir / f"{symbol}_bars.parquet"
            
            # If file exists, append new data
            if filepath.exists():
                existing = pd.read_parquet(filepath)
                df = pd.concat([existing, df]).drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp')
            
            df.to_parquet(filepath, index=False)
            logger.debug(f"Saved {len(df)} bars for {symbol} to parquet")
            
        except Exception as e:
            logger.error(f"Error saving parquet for {symbol}: {e}")
    
    def load_bars_parquet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load bar data from Parquet file.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            DataFrame or None
        """
        try:
            filepath = self.parquet_dir / f"{symbol}_bars.parquet"
            
            if filepath.exists():
                return pd.read_parquet(filepath)
            return None
            
        except Exception as e:
            logger.error(f"Error loading parquet for {symbol}: {e}")
            return None
    
    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================
    
    def get_audit_trail(
        self,
        symbol: str = None,
        event_type: EventType = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Query the audit trail.
        
        Args:
            symbol: Filter by symbol
            event_type: Filter by event type
            start_date: Filter from date (ISO format)
            end_date: Filter to date (ISO format)
            
        Returns:
            DataFrame with matching events
        """
        try:
            df = pd.read_csv(self.audit_csv)
            
            if symbol:
                df = df[df['symbol'] == symbol]
            if event_type:
                df = df[df['event_type'] == event_type.value]
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading audit trail: {e}")
            return pd.DataFrame()
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from audit trail.
        
        Returns:
            Dict with trade statistics
        """
        try:
            df = pd.read_csv(self.audit_csv)
            
            closed = df[df['event_type'] == EventType.POSITION_CLOSED.value]
            
            if len(closed) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            winning = closed[closed['pnl'] > 0]
            losing = closed[closed['pnl'] <= 0]
            
            return {
                'total_trades': len(closed),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': len(winning) / len(closed) if len(closed) > 0 else 0,
                'total_pnl': closed['pnl'].sum(),
                'avg_pnl': closed['pnl'].mean(),
                'best_trade': closed['pnl'].max(),
                'worst_trade': closed['pnl'].min()
            }
            
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            return {}
    
    def display_recent(self, n: int = 10):
        """Display recent audit entries"""
        try:
            df = pd.read_csv(self.audit_csv)
            
            print("\n" + "="*100)
            print(f"RECENT AUDIT ENTRIES (last {n})")
            print("="*100)
            
            if len(df) == 0:
                print("No entries yet")
            else:
                recent = df.tail(n)
                for _, row in recent.iterrows():
                    ts = row['timestamp'][:19]  # Trim microseconds
                    event = row['event_type']
                    symbol = row['symbol'] if pd.notna(row['symbol']) else ''
                    msg = row['message'] if pd.notna(row['message']) else ''
                    
                    print(f"{ts} | {event:<18} | {symbol:<6} | {msg}")
            
            print("="*100 + "\n")
            
        except Exception as e:
            print(f"Error displaying audit: {e}")


if __name__ == "__main__":
    # Test the audit logger
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    logging.basicConfig(level=logging.INFO)
    
    config = RTConfig()
    
    # Use test paths
    config.audit_csv = 'logs/realtime/test_audit.csv'
    config.parquet_dir = 'data/realtime/test_parquet'
    
    audit = AuditLogger(config)
    
    print("\n--- Testing Audit Logger ---")
    
    # Log some events
    audit.log_info("Test started")
    
    audit.log_signal('AAPL', 'BUY', 0.75, 'ENTER')
    
    audit.log_position_open('AAPL', 100, 150.00, 145.00, 'order123')
    
    audit.log_stop_update('AAPL', 145.00, 148.00)
    
    audit.log_position_close('AAPL', 100, 160.00, 150.00, 'TAKE_PROFIT', 'order456')
    
    audit.log_error("Test error message", 'MSFT')
    
    # Display recent
    audit.display_recent()
    
    # Get summary
    print("\n--- Trade Summary ---")
    summary = audit.get_trade_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Test parquet
    print("\n--- Testing Parquet Storage ---")
    test_df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='1min'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100.5 + i for i in range(10)],
        'volume': [1000 * (i+1) for i in range(10)]
    })
    
    audit.save_bars_parquet('TEST', test_df)
    loaded = audit.load_bars_parquet('TEST')
    print(f"[OK] Saved and loaded {len(loaded)} bars")
    
    # Cleanup
    import shutil
    if os.path.exists(config.audit_csv):
        os.remove(config.audit_csv)
    if os.path.exists(config.parquet_dir):
        shutil.rmtree(config.parquet_dir)
    print("[OK] Cleaned up test files")
    
    print("\n[OK] AuditLogger tests passed!")
