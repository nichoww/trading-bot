"""
Signal Generator
================
Generates trading signals using:
- MA Crossover strategy
- ML Model predictions (optional)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from enum import Enum
import logging
import joblib
from pathlib import Path

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
except ImportError:
    from rt_config import RTConfig

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalGenerator:
    """
    Generates trading signals based on configured strategy.
    
    Strategies:
    - MA_CROSSOVER: Fast MA crosses above/below slow MA
    - ML_MODEL: Machine learning model predictions
    """
    
    def __init__(self, config: RTConfig):
        """
        Initialize SignalGenerator.
        
        Args:
            config: RTConfig instance
        """
        self.config = config
        self.signal_type = config.signal_type
        
        # Load ML model if needed
        self.model = None
        self.scaler = None
        if self.signal_type == 'ML_MODEL':
            self._load_ml_model()
        
        logger.info(f"[OK] SignalGenerator initialized (strategy: {self.signal_type})")
    
    def _load_ml_model(self):
        """Load ML model and scaler"""
        try:
            model_path = Path(self.config.ml_model_path)
            scaler_path = Path(self.config.ml_scaler_path)
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Disable parallel processing
                if hasattr(self.model, 'n_jobs'):
                    self.model.n_jobs = 1
                
                logger.info(f"[OK] ML model loaded: {model_path}")
            else:
                logger.warning(f"ML model not found, falling back to MA_CROSSOVER")
                self.signal_type = 'MA_CROSSOVER'
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.signal_type = 'MA_CROSSOVER'
    
    def generate_signal(
        self, 
        df: pd.DataFrame,
        symbol: str = ''
    ) -> Tuple[Signal, float]:
        """
        Generate trading signal from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock ticker (for logging)
            
        Returns:
            Tuple of (Signal, confidence)
        """
        if self.signal_type == 'MA_CROSSOVER':
            return self._ma_crossover_signal(df, symbol)
        elif self.signal_type == 'ML_MODEL':
            return self._ml_model_signal(df, symbol)
        else:
            return Signal.HOLD, 0.0
    
    def _ma_crossover_signal(
        self, 
        df: pd.DataFrame,
        symbol: str = ''
    ) -> Tuple[Signal, float]:
        """
        Generate MA crossover signal.
        
        BUY: Fast MA crosses above Slow MA
        SELL: Fast MA crosses below Slow MA
        HOLD: No crossover
        
        Args:
            df: DataFrame with 'close' column
            symbol: Stock ticker
            
        Returns:
            Tuple of (Signal, confidence)
        """
        fast_period = self.config.ma_fast_period
        slow_period = self.config.ma_slow_period
        
        if len(df) < slow_period + 2:
            logger.debug(f"{symbol}: Insufficient data for MA ({len(df)} bars)")
            return Signal.HOLD, 0.0
        
        close = df['close'].values
        
        # Calculate MAs
        fast_ma = pd.Series(close).rolling(fast_period).mean().values
        slow_ma = pd.Series(close).rolling(slow_period).mean().values
        
        # Current and previous values
        fast_now = fast_ma[-1]
        fast_prev = fast_ma[-2]
        slow_now = slow_ma[-1]
        slow_prev = slow_ma[-2]
        
        # Check for crossover
        if np.isnan(fast_now) or np.isnan(slow_now):
            return Signal.HOLD, 0.0
        
        # Calculate signal strength (distance between MAs as % of price)
        ma_diff_pct = abs(fast_now - slow_now) / slow_now
        confidence = min(0.5 + ma_diff_pct * 10, 1.0)  # Scale to 0.5-1.0
        
        # Bullish crossover (fast crosses above slow)
        if fast_prev <= slow_prev and fast_now > slow_now:
            logger.debug(f"{symbol}: BUY signal (MA crossover up)")
            return Signal.BUY, confidence
        
        # Bearish crossover (fast crosses below slow)
        if fast_prev >= slow_prev and fast_now < slow_now:
            logger.debug(f"{symbol}: SELL signal (MA crossover down)")
            return Signal.SELL, confidence
        
        # No crossover - check trend direction
        if fast_now > slow_now:
            # Uptrend but no new crossover
            return Signal.HOLD, confidence * 0.5
        else:
            # Downtrend
            return Signal.HOLD, confidence * 0.5
    
    def _ml_model_signal(
        self, 
        df: pd.DataFrame,
        symbol: str = ''
    ) -> Tuple[Signal, float]:
        """
        Generate ML model signal.
        
        Args:
            df: DataFrame with features
            symbol: Stock ticker
            
        Returns:
            Tuple of (Signal, confidence)
        """
        if self.model is None or self.scaler is None:
            logger.warning("ML model not loaded, returning HOLD")
            return Signal.HOLD, 0.0
        
        try:
            # Calculate features from OHLCV
            features = self._calculate_features(df)
            if features is None:
                return Signal.HOLD, 0.0
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Assuming binary classification: [prob_down, prob_up]
            prob_up = proba[1] if len(proba) > 1 else proba[0]
            prob_down = 1 - prob_up
            
            # Generate signal based on confidence threshold
            threshold = self.config.ml_confidence_threshold
            
            if prob_up >= threshold:
                return Signal.BUY, prob_up
            elif prob_down >= threshold:
                return Signal.SELL, prob_down
            else:
                return Signal.HOLD, max(prob_up, prob_down)
                
        except Exception as e:
            logger.error(f"Error in ML prediction for {symbol}: {e}")
            return Signal.HOLD, 0.0
    
    def _calculate_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate features for ML model from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature array or None
        """
        try:
            if len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Calculate basic features (match training features)
            features = {}
            
            # Price features
            features['Daily_Return'] = (close[-1] - close[-2]) / close[-2]
            features['HL_Range'] = (high[-1] - low[-1]) / close[-1]
            
            # Moving averages
            features['SMA_10'] = np.mean(close[-10:])
            features['SMA_20'] = np.mean(close[-20:])
            features['SMA_50'] = np.mean(close[-50:])
            features['Distance_SMA50'] = (close[-1] - features['SMA_50']) / features['SMA_50']
            
            # RSI
            delta = np.diff(close[-15:])
            gain = np.mean([d for d in delta if d > 0]) if any(d > 0 for d in delta) else 0
            loss = np.mean([-d for d in delta if d < 0]) if any(d < 0 for d in delta) else 0.0001
            rs = gain / loss
            features['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            features['MACD'] = ema12 - ema26
            features['MACD_Signal'] = pd.Series(close).ewm(span=9).mean().iloc[-1]
            features['MACD_Histogram'] = features['MACD'] - features['MACD_Signal']
            
            # Bollinger Bands
            bb_ma = np.mean(close[-20:])
            bb_std = np.std(close[-20:])
            features['BB_Upper'] = bb_ma + 2 * bb_std
            features['BB_Lower'] = bb_ma - 2 * bb_std
            features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / bb_ma
            features['BB_Position'] = (close[-1] - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'])
            
            # Volume
            features['Volume_Ratio'] = volume[-1] / np.mean(volume[-20:])
            features['Volume_MA_20'] = np.mean(volume[-20:])
            features['volume'] = volume[-1]
            
            # Convert to array in expected order
            feature_names = [
                'Daily_Return', 'HL_Range', 'SMA_10', 'SMA_20', 'SMA_50',
                'Distance_SMA50', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
                'Volume_Ratio', 'Volume_MA_20', 'volume'
            ]
            
            feature_array = np.array([[features.get(f, 0) for f in feature_names]])
            return feature_array
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
    
    def get_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength (0-1).
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Trend strength from 0 (no trend) to 1 (strong trend)
        """
        if len(df) < self.config.ma_slow_period:
            return 0.0
        
        close = df['close'].values
        
        fast_ma = np.mean(close[-self.config.ma_fast_period:])
        slow_ma = np.mean(close[-self.config.ma_slow_period:])
        
        # Trend strength as % difference
        diff_pct = abs(fast_ma - slow_ma) / slow_ma
        
        return min(diff_pct * 20, 1.0)  # Scale to 0-1


if __name__ == "__main__":
    # Test signal generator
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    logging.basicConfig(level=logging.INFO)
    
    from data_streamer import DataStreamer
    
    config = RTConfig()
    streamer = DataStreamer(config)
    sig_gen = SignalGenerator(config)
    
    print("\n--- Testing Signal Generator ---")
    
    # Fetch data
    test_symbols = ['AAPL', 'NVDA', 'TSLA']
    
    for symbol in test_symbols:
        bars = streamer.get_historical_bars(symbol, lookback_days=5)
        
        if bars is not None:
            signal, confidence = sig_gen.generate_signal(bars, symbol)
            trend = sig_gen.get_trend_strength(bars)
            
            print(f"  {symbol}: {signal.value:<4} | Confidence: {confidence:.2%} | Trend: {trend:.2%}")
        else:
            print(f"  {symbol}: No data")
    
    print("\n[OK] SignalGenerator tests passed!")
