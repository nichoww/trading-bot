"""
Real-Time Trading Configuration
===============================
Central configuration for the real-time trading system.
All tunable parameters in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RTConfig:
    """
    Real-Time Trading Configuration
    
    All parameters for the trading system are defined here.
    Modify these values to tune the bot's behavior.
    """
    
    # =========================================================================
    # API CREDENTIALS (from .env)
    # =========================================================================
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'))
    
    # =========================================================================
    # TRADING UNIVERSE
    # =========================================================================
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'AMD', 'AMZN', 'BA', 'COST', 
        'DIS', 'F', 'GE', 'GM', 'GOOGL',
        'INTC', 'JPM', 'META', 'MSFT', 'NFLX', 
        'NVDA', 'PYPL', 'TSLA', 'V', 'WMT'
    ])
    
    # =========================================================================
    # TIMING
    # =========================================================================
    bar_interval: str = '1Min'          # Bar timeframe: '1Min', '5Min', '15Min', '1Hour'
    loop_interval_seconds: int = 60     # Main loop runs every N seconds
    limit_order_timeout: int = 30       # Seconds to wait for limit fill
    cancel_check_interval: int = 2      # Seconds between order status checks
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    risk_per_trade_pct: float = 0.02    # Risk 2% of portfolio per trade
    max_position_pct: float = 0.10      # Max 10% of portfolio in one position
    max_positions: int = 15             # Maximum concurrent positions (increased for demo)
    
    # =========================================================================
    # STOPS & TRAILING
    # =========================================================================
    atr_period: int = 14                # ATR lookback period
    atr_stop_multiplier: float = 2.0    # Initial stop = entry - (ATR * multiplier)
    trail_percent: float = 0.03         # 3% trailing stop
    take_profit_pct: float = 0.10       # 10% take profit (optional, 0 to disable)
    
    # =========================================================================
    # ORDER SETTINGS
    # =========================================================================
    use_limit_orders: bool = True       # Use limit orders (fallback to market)
    limit_offset_pct: float = 0.001     # Limit price offset (0.1% from current)
    extended_hours: bool = True         # Allow extended hours trading (equities)
    
    # =========================================================================
    # SIGNAL SETTINGS
    # =========================================================================
    signal_type: str = 'MA_CROSSOVER'   # 'MA_CROSSOVER' or 'ML_MODEL'
    ma_fast_period: int = 10            # Fast MA period
    ma_slow_period: int = 30            # Slow MA period
    ml_model_path: str = 'models/rf_model_1d.pkl'
    ml_scaler_path: str = 'models/scaler_1d.pkl'
    ml_confidence_threshold: float = 0.60
    
    # =========================================================================
    # DATA STORAGE
    # =========================================================================
    data_dir: str = 'data/realtime'
    parquet_dir: str = 'data/realtime/parquet'
    logs_dir: str = 'logs/realtime'
    audit_csv: str = 'logs/realtime/audit_trail.csv'
    state_file: str = 'data/realtime/portfolio_state.json'
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    dashboard_refresh_seconds: int = 5
    
    # =========================================================================
    # TIMEZONE
    # =========================================================================
    timezone: str = 'US/Eastern'
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        if not self.alpaca_api_key:
            errors.append("ALPACA_API_KEY not set")
        if not self.alpaca_secret_key:
            errors.append("ALPACA_SECRET_KEY not set")
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 0.10:
            errors.append("risk_per_trade_pct should be between 0 and 0.10")
        if self.max_positions < 1:
            errors.append("max_positions must be at least 1")
        if self.atr_period < 5:
            errors.append("atr_period should be at least 5")
        if len(self.symbols) == 0:
            errors.append("symbols list is empty")
            
        if errors:
            for e in errors:
                print(f"[ERROR] Config: {e}")
            return False
        
        return True
    
    def display(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("REAL-TIME TRADING CONFIGURATION")
        print("="*70)
        print(f"API Base URL:      {self.alpaca_base_url}")
        print(f"Symbols:           {len(self.symbols)} tickers")
        print(f"Bar Interval:      {self.bar_interval}")
        print(f"Loop Interval:     {self.loop_interval_seconds}s")
        print("-"*70)
        print("RISK SETTINGS:")
        print(f"  Risk per Trade:  {self.risk_per_trade_pct:.1%}")
        print(f"  Max Position:    {self.max_position_pct:.1%}")
        print(f"  Max Positions:   {self.max_positions}")
        print(f"  ATR Period:      {self.atr_period}")
        print(f"  ATR Stop Mult:   {self.atr_stop_multiplier}x")
        print(f"  Trail %:         {self.trail_percent:.1%}")
        print("-"*70)
        print("ORDER SETTINGS:")
        print(f"  Limit Orders:    {self.use_limit_orders}")
        print(f"  Limit Timeout:   {self.limit_order_timeout}s")
        print(f"  Extended Hours:  {self.extended_hours}")
        print("-"*70)
        print("SIGNAL:")
        print(f"  Type:            {self.signal_type}")
        if self.signal_type == 'MA_CROSSOVER':
            print(f"  Fast MA:         {self.ma_fast_period}")
            print(f"  Slow MA:         {self.ma_slow_period}")
        else:
            print(f"  ML Model:        {self.ml_model_path}")
            print(f"  Confidence:      {self.ml_confidence_threshold:.0%}")
        print("="*70 + "\n")


# Default configuration instance
DEFAULT_CONFIG = RTConfig()


if __name__ == "__main__":
    # Test configuration
    config = RTConfig()
    if config.validate():
        print("[OK] Configuration valid")
        config.display()
    else:
        print("[ERROR] Configuration invalid")
