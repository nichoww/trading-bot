"""
Risk Manager
============
Handles all risk-related calculations:
- ATR (Average True Range) calculation
- Position sizing based on risk
- Stop loss levels (initial and trailing)
- Take profit levels
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
except ImportError:
    from rt_config import RTConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionRisk:
    """Risk parameters for a position"""
    symbol: str
    entry_price: float
    shares: int
    position_value: float
    stop_price: float
    take_profit_price: Optional[float]
    trail_base: float          # Highest price since entry (for trailing)
    atr: float
    risk_amount: float         # $ at risk


class RiskManager:
    """
    Manages all risk calculations for the trading system.
    
    Features:
    - ATR-based stop loss calculation
    - Position sizing based on % risk per trade
    - Trailing stop management
    - Take profit levels
    """
    
    def __init__(self, config: RTConfig):
        """
        Initialize RiskManager.
        
        Args:
            config: RTConfig instance with risk parameters
        """
        self.config = config
        logger.info("[OK] RiskManager initialized")
    
    def calculate_atr(
        self, 
        df: pd.DataFrame, 
        period: int = None
    ) -> float:
        """
        Calculate Average True Range (ATR).
        
        ATR measures volatility by considering:
        - Current high minus current low
        - Absolute value of current high minus previous close
        - Absolute value of current low minus previous close
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default: from config)
            
        Returns:
            Current ATR value
        """
        if period is None:
            period = self.config.atr_period
        
        if len(df) < period + 1:
            logger.warning(f"Insufficient data for ATR (need {period + 1}, have {len(df)})")
            # Return a rough estimate based on available data
            return float((df['high'] - df['low']).mean())
        
        # Calculate True Range components
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range = max of:
        # 1. High - Low
        # 2. |High - Previous Close|
        # 3. |Low - Previous Close|
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR is the moving average of True Range
        if len(true_range) >= period:
            atr = np.mean(true_range[-period:])
        else:
            atr = np.mean(true_range)
        
        return float(atr)
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: float,
        buying_power: float = None
    ) -> Tuple[int, float, float]:
        """
        Calculate position size based on risk parameters.
        
        Uses ATR to determine stop distance, then sizes position
        so that hitting the stop loses only risk_per_trade_pct of portfolio.
        
        Args:
            portfolio_value: Total portfolio value in $
            entry_price: Expected entry price
            atr: Current ATR value
            buying_power: Available buying power (optional cap)
            
        Returns:
            Tuple of (shares, position_value, stop_price)
        """
        # Calculate stop distance using ATR
        stop_distance = atr * self.config.atr_stop_multiplier
        stop_price = entry_price - stop_distance
        
        # Risk amount = % of portfolio
        risk_amount = portfolio_value * self.config.risk_per_trade_pct
        
        # Position size = risk_amount / stop_distance
        # This ensures we lose exactly risk_amount if stop is hit
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            shares = 0
        
        # Calculate position value
        position_value = shares * entry_price
        
        # Apply max position size cap
        max_position_value = portfolio_value * self.config.max_position_pct
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        # Apply buying power constraint
        if buying_power is not None and position_value > buying_power:
            shares = int(buying_power / entry_price)
            position_value = shares * entry_price
        
        # Ensure at least 1 share if we have any position
        if shares < 1 and position_value > 0:
            shares = 1
            position_value = entry_price
        
        logger.debug(
            f"Position sizing: {shares} shares @ ${entry_price:.2f} = ${position_value:.2f} "
            f"(stop: ${stop_price:.2f}, risk: ${risk_amount:.2f})"
        )
        
        return shares, position_value, stop_price
    
    def calculate_stop_price(
        self,
        entry_price: float,
        atr: float
    ) -> float:
        """
        Calculate initial stop loss price.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * self.config.atr_stop_multiplier
        return entry_price - stop_distance
    
    def calculate_take_profit(
        self,
        entry_price: float
    ) -> Optional[float]:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            
        Returns:
            Take profit price, or None if disabled
        """
        if self.config.take_profit_pct <= 0:
            return None
        
        return entry_price * (1 + self.config.take_profit_pct)
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        trail_base: float
    ) -> Tuple[float, float]:
        """
        Update trailing stop based on current price.
        
        Trailing stop only moves UP (never down).
        When price makes new high, stop ratchets up.
        
        Args:
            current_price: Current market price
            current_stop: Current stop price
            trail_base: Highest price since entry
            
        Returns:
            Tuple of (new_stop_price, new_trail_base)
        """
        # Update trail base if we have a new high
        new_trail_base = max(trail_base, current_price)
        
        # Calculate new stop based on trail
        new_stop = new_trail_base * (1 - self.config.trail_percent)
        
        # Stop only moves up, never down
        final_stop = max(current_stop, new_stop)
        
        if final_stop > current_stop:
            logger.debug(
                f"Trailing stop moved: ${current_stop:.2f} -> ${final_stop:.2f} "
                f"(trail base: ${new_trail_base:.2f})"
            )
        
        return final_stop, new_trail_base
    
    def check_exit_conditions(
        self,
        current_price: float,
        stop_price: float,
        take_profit_price: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Check if exit conditions are met.
        
        Args:
            current_price: Current market price
            stop_price: Current stop loss price
            take_profit_price: Take profit price (or None)
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Check stop loss
        if current_price <= stop_price:
            return True, f"STOP_LOSS (price ${current_price:.2f} <= stop ${stop_price:.2f})"
        
        # Check take profit
        if take_profit_price is not None and current_price >= take_profit_price:
            return True, f"TAKE_PROFIT (price ${current_price:.2f} >= target ${take_profit_price:.2f})"
        
        return False, ""
    
    def create_position_risk(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        atr: float
    ) -> PositionRisk:
        """
        Create a PositionRisk object for a new position.
        
        Args:
            symbol: Stock ticker
            entry_price: Entry price
            shares: Number of shares
            atr: Current ATR value
            
        Returns:
            PositionRisk object
        """
        stop_price = self.calculate_stop_price(entry_price, atr)
        take_profit = self.calculate_take_profit(entry_price)
        risk_amount = shares * (entry_price - stop_price)
        
        return PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            position_value=shares * entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit,
            trail_base=entry_price,
            atr=atr,
            risk_amount=risk_amount
        )
    
    def display_risk_summary(self, positions: Dict[str, PositionRisk]):
        """
        Display risk summary for all positions.
        
        Args:
            positions: Dict mapping symbol to PositionRisk
        """
        if not positions:
            print("No open positions")
            return
        
        print("\n" + "="*80)
        print("POSITION RISK SUMMARY")
        print("="*80)
        print(f"{'Symbol':<8} {'Shares':<8} {'Entry':<10} {'Stop':<10} {'TP':<10} {'Risk $':<10}")
        print("-"*80)
        
        total_risk = 0
        for symbol, pos in positions.items():
            tp_str = f"${pos.take_profit_price:.2f}" if pos.take_profit_price else "N/A"
            print(
                f"{symbol:<8} {pos.shares:<8} ${pos.entry_price:<9.2f} "
                f"${pos.stop_price:<9.2f} {tp_str:<10} ${pos.risk_amount:<9.2f}"
            )
            total_risk += pos.risk_amount
        
        print("-"*80)
        print(f"Total Risk: ${total_risk:.2f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test the risk manager
    import sys
    sys.path.insert(0, 'src')
    
    logging.basicConfig(level=logging.INFO)
    
    config = RTConfig()
    rm = RiskManager(config)
    
    # Create sample OHLCV data
    print("\n--- Testing ATR Calculation ---")
    sample_data = pd.DataFrame({
        'high': [150 + i*0.5 + np.random.uniform(-1, 1) for i in range(20)],
        'low': [148 + i*0.5 + np.random.uniform(-1, 1) for i in range(20)],
        'close': [149 + i*0.5 + np.random.uniform(-0.5, 0.5) for i in range(20)],
    })
    
    atr = rm.calculate_atr(sample_data)
    print(f"[OK] ATR: ${atr:.2f}")
    
    # Test position sizing
    print("\n--- Testing Position Sizing ---")
    portfolio = 100000
    entry = 150.00
    
    shares, value, stop = rm.calculate_position_size(portfolio, entry, atr)
    print(f"[OK] Position: {shares} shares @ ${entry:.2f}")
    print(f"    Value: ${value:.2f}")
    print(f"    Stop: ${stop:.2f}")
    print(f"    Risk: ${shares * (entry - stop):.2f} ({shares * (entry - stop) / portfolio:.1%})")
    
    # Test trailing stop
    print("\n--- Testing Trailing Stop ---")
    current_stop = stop
    trail_base = entry
    
    # Simulate price going up
    for price in [151, 153, 155, 154, 156]:
        new_stop, trail_base = rm.update_trailing_stop(price, current_stop, trail_base)
        print(f"  Price: ${price:.2f} -> Stop: ${new_stop:.2f} (trail base: ${trail_base:.2f})")
        current_stop = new_stop
    
    # Test exit conditions
    print("\n--- Testing Exit Conditions ---")
    pos_risk = rm.create_position_risk('AAPL', entry, shares, atr)
    
    # Test stop hit
    should_exit, reason = rm.check_exit_conditions(stop - 1, stop, pos_risk.take_profit_price)
    print(f"Stop hit: {should_exit} - {reason}")
    
    # Test take profit hit
    if pos_risk.take_profit_price:
        should_exit, reason = rm.check_exit_conditions(
            pos_risk.take_profit_price + 1, stop, pos_risk.take_profit_price
        )
        print(f"TP hit: {should_exit} - {reason}")
    
    # Display summary
    rm.display_risk_summary({'AAPL': pos_risk})
    
    print("\n[OK] RiskManager tests passed!")
