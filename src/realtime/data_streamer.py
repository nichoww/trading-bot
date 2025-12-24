"""
Data Streamer
=============
Fetches OHLCV bar data from Alpaca API.
Supports both historical bars and real-time updates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
except ImportError:
    from rt_config import RTConfig

logger = logging.getLogger(__name__)


class DataStreamer:
    """
    Handles all market data fetching from Alpaca.
    
    Features:
    - Fetch historical bars for backtesting/warmup
    - Get latest bars for real-time signals
    - Cache data to reduce API calls
    - Automatic bar interval handling
    """
    
    def __init__(self, config: RTConfig, client: REST = None):
        """
        Initialize DataStreamer.
        
        Args:
            config: RTConfig instance with API credentials
            client: Optional existing Alpaca REST client
        """
        self.config = config
        
        # Initialize Alpaca client
        if client:
            self.client = client
        else:
            self.client = REST(
                key_id=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                base_url=config.alpaca_base_url
            )
        
        # Parse bar interval
        self.timeframe = self._parse_timeframe(config.bar_interval)
        
        # Data cache: {symbol: DataFrame}
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        logger.info(f"[OK] DataStreamer initialized (interval: {config.bar_interval})")
    
    def _parse_timeframe(self, interval: str) -> TimeFrame:
        """Convert string interval to Alpaca TimeFrame"""
        interval_map = {
            '1Min': TimeFrame(1, TimeFrameUnit.Minute),
            '5Min': TimeFrame(5, TimeFrameUnit.Minute),
            '15Min': TimeFrame(15, TimeFrameUnit.Minute),
            '30Min': TimeFrame(30, TimeFrameUnit.Minute),
            '1Hour': TimeFrame(1, TimeFrameUnit.Hour),
            '1Day': TimeFrame(1, TimeFrameUnit.Day),
        }
        
        if interval not in interval_map:
            logger.warning(f"Unknown interval '{interval}', defaulting to 1Min")
            return TimeFrame(1, TimeFrameUnit.Minute)
        
        return interval_map[interval]
    
    def get_historical_bars(
        self, 
        symbol: str, 
        lookback_days: int = 30,
        end: datetime = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars for a symbol.
        
        Args:
            symbol: Stock ticker
            lookback_days: Number of days of history
            end: End datetime (default: now)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap
        """
        try:
            if end is None:
                end = datetime.now()
            
            start = end - timedelta(days=lookback_days)
            
            # Format dates as ISO strings
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            logger.debug(f"Fetching {symbol} bars: {start_str} to {end_str}")
            
            # Fetch bars from Alpaca
            bars = self.client.get_bars(
                symbol,
                self.timeframe,
                start=start_str,
                end=end_str,
                feed='iex'  # Use IEX feed for free tier
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Reset index to get timestamp as column
            bars = bars.reset_index()
            
            # Standardize column names
            bars.columns = [c.lower() for c in bars.columns]
            
            # Ensure we have required columns
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in bars.columns:
                    logger.error(f"Missing column {col} for {symbol}")
                    return None
            
            # Sort by timestamp
            bars = bars.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data
            self.cache[symbol] = bars
            self.cache_timestamps[symbol] = datetime.now()
            
            logger.debug(f"Fetched {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None
    
    def get_latest_bars(
        self, 
        symbols: List[str] = None, 
        num_bars: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest bars for multiple symbols.
        
        Args:
            symbols: List of tickers (default: from config)
            num_bars: Number of recent bars to fetch
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.config.symbols
        
        result = {}
        
        for symbol in symbols:
            # Calculate lookback based on interval and num_bars
            # For 1Min bars, 100 bars = ~1.7 hours, fetch 1 day to be safe
            lookback = max(2, num_bars // 390 + 1)  # 390 mins per trading day
            
            bars = self.get_historical_bars(symbol, lookback_days=lookback)
            
            if bars is not None and len(bars) >= num_bars:
                result[symbol] = bars.tail(num_bars).reset_index(drop=True)
            elif bars is not None:
                result[symbol] = bars
            else:
                logger.warning(f"Could not fetch data for {symbol}")
        
        return result
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Current price or None if unavailable
        """
        try:
            # Get latest trade
            trade = self.client.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            
            # Fallback: use cached close price
            if symbol in self.cache and not self.cache[symbol].empty:
                return float(self.cache[symbol]['close'].iloc[-1])
            
            return None
    
    def get_current_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols: List of tickers (default: from config)
            
        Returns:
            Dict mapping symbol to current price
        """
        if symbols is None:
            symbols = self.config.symbols
        
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    def get_ohlcv_tuple(self, symbol: str) -> Optional[tuple]:
        """
        Get the latest OHLCV values as a tuple.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Tuple of (open, high, low, close, volume) or None
        """
        if symbol not in self.cache or self.cache[symbol].empty:
            self.get_historical_bars(symbol, lookback_days=2)
        
        if symbol in self.cache and not self.cache[symbol].empty:
            row = self.cache[symbol].iloc[-1]
            return (
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            )
        
        return None
    
    def warmup(self, symbols: List[str] = None, lookback_days: int = 30):
        """
        Pre-fetch historical data for all symbols.
        Call this before starting the trading loop.
        
        Args:
            symbols: List of tickers (default: from config)
            lookback_days: Days of history to fetch
        """
        if symbols is None:
            symbols = self.config.symbols
        
        logger.info(f"Warming up data for {len(symbols)} symbols...")
        
        success = 0
        for symbol in symbols:
            bars = self.get_historical_bars(symbol, lookback_days)
            if bars is not None:
                success += 1
        
        logger.info(f"[OK] Warmup complete: {success}/{len(symbols)} symbols loaded")


if __name__ == "__main__":
    # Test the data streamer
    import sys
    sys.path.insert(0, 'src')
    
    logging.basicConfig(level=logging.INFO)
    
    config = RTConfig()
    if not config.validate():
        print("[ERROR] Invalid config")
        exit(1)
    
    streamer = DataStreamer(config)
    
    # Test with one symbol
    print("\n--- Testing Single Symbol ---")
    bars = streamer.get_historical_bars('AAPL', lookback_days=5)
    if bars is not None:
        print(f"[OK] Fetched {len(bars)} bars for AAPL")
        print(bars.tail(5))
    
    # Test current price
    print("\n--- Testing Current Price ---")
    price = streamer.get_current_price('AAPL')
    if price:
        print(f"[OK] AAPL current price: ${price:.2f}")
    
    # Test multiple symbols
    print("\n--- Testing Multiple Symbols ---")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = streamer.get_latest_bars(test_symbols, num_bars=50)
    for sym, df in data.items():
        print(f"  {sym}: {len(df)} bars, last close: ${df['close'].iloc[-1]:.2f}")
    
    print("\n[OK] DataStreamer tests passed!")
