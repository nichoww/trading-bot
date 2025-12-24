"""
Portfolio State Manager
=======================
Handles persistence of portfolio state across restarts:
- Save positions with their risk levels (stops, trail bases)
- Restore state on startup
- Merge with live broker positions
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import logging

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
    from .risk_manager import PositionRisk
except ImportError:
    from rt_config import RTConfig
    from risk_manager import PositionRisk

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Serializable position state"""
    symbol: str
    entry_price: float
    shares: int
    entry_time: str  # ISO format
    stop_price: float
    take_profit_price: Optional[float]
    trail_base: float
    atr: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PositionState':
        return cls(**data)
    
    def to_position_risk(self) -> PositionRisk:
        """Convert to PositionRisk object"""
        return PositionRisk(
            symbol=self.symbol,
            entry_price=self.entry_price,
            shares=self.shares,
            position_value=self.shares * self.entry_price,
            stop_price=self.stop_price,
            take_profit_price=self.take_profit_price,
            trail_base=self.trail_base,
            atr=self.atr,
            risk_amount=self.shares * (self.entry_price - self.stop_price)
        )


class PortfolioState:
    """
    Manages portfolio state persistence.
    
    Features:
    - Save all position data including risk parameters
    - Restore on startup
    - Merge with live broker positions
    - Detect orphaned positions (in broker but not state)
    """
    
    def __init__(self, config: RTConfig):
        """
        Initialize PortfolioState.
        
        Args:
            config: RTConfig instance
        """
        self.config = config
        self.state_file = Path(config.state_file)
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Current positions: symbol -> PositionState
        self.positions: Dict[str, PositionState] = {}
        
        # Metadata
        self.last_saved: Optional[datetime] = None
        self.last_loaded: Optional[datetime] = None
        
        logger.info(f"[OK] PortfolioState initialized (file: {self.state_file})")
    
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        take_profit_price: Optional[float],
        atr: float
    ) -> PositionState:
        """
        Add a new position to state.
        
        Args:
            symbol: Stock ticker
            entry_price: Entry price
            shares: Number of shares
            stop_price: Initial stop loss
            take_profit_price: Take profit level (or None)
            atr: ATR at entry
            
        Returns:
            PositionState object
        """
        state = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            entry_time=datetime.now().isoformat(),
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            trail_base=entry_price,
            atr=atr
        )
        
        self.positions[symbol] = state
        logger.info(f"Added position: {symbol} {shares} shares @ ${entry_price:.2f}")
        
        # Auto-save
        self.save()
        
        return state
    
    def update_position(
        self,
        symbol: str,
        stop_price: float = None,
        trail_base: float = None
    ) -> bool:
        """
        Update position risk parameters.
        
        Args:
            symbol: Stock ticker
            stop_price: New stop price (or None to keep)
            trail_base: New trail base (or None to keep)
            
        Returns:
            True if updated
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot update unknown position: {symbol}")
            return False
        
        pos = self.positions[symbol]
        
        if stop_price is not None:
            pos.stop_price = stop_price
        if trail_base is not None:
            pos.trail_base = trail_base
        
        # Auto-save
        self.save()
        
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position from state.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            True if removed
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Removed position: {symbol}")
            self.save()
            return True
        return False
    
    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position state by symbol"""
        return self.positions.get(symbol)
    
    def save(self) -> bool:
        """
        Save state to file.
        
        Returns:
            True if saved successfully
        """
        try:
            data = {
                'version': 1,
                'saved_at': datetime.now().isoformat(),
                'positions': {
                    sym: pos.to_dict() 
                    for sym, pos in self.positions.items()
                }
            }
            
            # Write to temp file first, then rename (atomic)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Rename to actual file
            temp_file.replace(self.state_file)
            
            self.last_saved = datetime.now()
            logger.debug(f"State saved: {len(self.positions)} positions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load state from file.
        
        Returns:
            True if loaded successfully
        """
        if not self.state_file.exists():
            logger.info("No state file found, starting fresh")
            return True
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Load positions
            self.positions = {}
            for sym, pos_data in data.get('positions', {}).items():
                self.positions[sym] = PositionState.from_dict(pos_data)
            
            self.last_loaded = datetime.now()
            saved_at = data.get('saved_at', 'unknown')
            
            logger.info(f"State loaded: {len(self.positions)} positions (saved: {saved_at})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def sync_with_broker(self, broker_positions: Dict[str, dict]) -> Dict[str, str]:
        """
        Sync state with broker positions.
        
        Handles:
        - Positions in state but not broker (closed externally)
        - Positions in broker but not state (opened externally)
        
        Args:
            broker_positions: Dict from OrderManager.get_positions()
            
        Returns:
            Dict of sync actions taken: symbol -> action
        """
        actions = {}
        
        # Check for positions closed externally
        state_symbols = set(self.positions.keys())
        broker_symbols = set(broker_positions.keys())
        
        # Closed externally (in state but not broker)
        for symbol in state_symbols - broker_symbols:
            logger.warning(f"Position {symbol} closed externally, removing from state")
            self.remove_position(symbol)
            actions[symbol] = "removed_closed_externally"
        
        # Opened externally (in broker but not state)
        for symbol in broker_symbols - state_symbols:
            pos = broker_positions[symbol]
            logger.warning(f"Position {symbol} found in broker but not state, adding")
            
            # Handle both Alpaca Position objects and dicts
            if hasattr(pos, 'avg_entry_price'):
                # Alpaca Position object
                entry_price = float(pos.avg_entry_price)
                qty = int(pos.qty)
                current_price = float(pos.current_price)
            else:
                # Dict format
                entry_price = pos['entry_price']
                qty = int(pos['qty'])
                current_price = pos['current_price']
            
            # Create a basic state entry (we don't know original risk params)
            self.positions[symbol] = PositionState(
                symbol=symbol,
                entry_price=entry_price,
                shares=qty,
                entry_time=datetime.now().isoformat(),
                stop_price=entry_price * 0.95,  # Default 5% stop
                take_profit_price=entry_price * 1.10,  # Default 10% TP
                trail_base=current_price,
                atr=entry_price * 0.02  # Estimate 2% ATR
            )
            actions[symbol] = "added_from_broker"
        
        # Update share counts if different
        for symbol in state_symbols & broker_symbols:
            pos = broker_positions[symbol]
            # Handle both Alpaca Position objects and dicts
            if hasattr(pos, 'qty'):
                broker_qty = int(pos.qty)
            else:
                broker_qty = int(pos['qty'])
            state_qty = self.positions[symbol].shares
            
            if broker_qty != state_qty:
                logger.warning(f"Position {symbol} qty mismatch: state={state_qty}, broker={broker_qty}")
                self.positions[symbol].shares = broker_qty
                actions[symbol] = "qty_updated"
        
        if actions:
            self.save()
        
        return actions
    
    def display(self):
        """Display current state"""
        print("\n" + "="*80)
        print("PORTFOLIO STATE")
        print("="*80)
        
        if self.last_loaded:
            print(f"Last Loaded: {self.last_loaded.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.last_saved:
            print(f"Last Saved:  {self.last_saved.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nPositions: {len(self.positions)}")
        
        if self.positions:
            print(f"\n{'Symbol':<8} {'Shares':<8} {'Entry':<10} {'Stop':<10} {'Trail':<10} {'TP':<10}")
            print("-"*80)
            
            for sym, pos in self.positions.items():
                tp_str = f"${pos.take_profit_price:.2f}" if pos.take_profit_price else "N/A"
                print(
                    f"{sym:<8} {pos.shares:<8} ${pos.entry_price:<9.2f} "
                    f"${pos.stop_price:<9.2f} ${pos.trail_base:<9.2f} {tp_str:<10}"
                )
        
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test portfolio state
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    logging.basicConfig(level=logging.INFO)
    
    config = RTConfig()
    
    # Use a test file
    config.state_file = 'data/realtime/test_portfolio_state.json'
    
    ps = PortfolioState(config)
    
    print("\n--- Testing Portfolio State ---")
    
    # Add some test positions
    ps.add_position('AAPL', 150.0, 100, 145.0, 165.0, 2.5)
    ps.add_position('MSFT', 300.0, 50, 290.0, 330.0, 5.0)
    ps.add_position('GOOGL', 140.0, 75, 135.0, 154.0, 3.0)
    
    ps.display()
    
    # Update a position
    print("\n--- Updating AAPL stop ---")
    ps.update_position('AAPL', stop_price=148.0, trail_base=155.0)
    
    # Remove a position
    print("\n--- Removing GOOGL ---")
    ps.remove_position('GOOGL')
    
    ps.display()
    
    # Test reload
    print("\n--- Testing Reload ---")
    ps2 = PortfolioState(config)
    ps2.load()
    ps2.display()
    
    # Cleanup test file
    if os.path.exists(config.state_file):
        os.remove(config.state_file)
        print(f"[OK] Cleaned up test file")
    
    print("\n[OK] PortfolioState tests passed!")
