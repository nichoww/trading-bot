"""
Feature Engineering Module
Calculates technical indicators and creates machine learning features
Reads from SQLite database and saves processed features
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from pathlib import Path
from datetime import datetime

# Configure logging
log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data paths
DB_PATH = 'data/market_data.db'
PROCESSED_DIR = 'data/processed'
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """
    Technical indicator calculator and feature creator
    """
    
    def __init__(self, db_path=DB_PATH):
        """
        Initialize feature engineer
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        logger.info("FeatureEngineer initialized")
    
    
    def load_ticker_data(self, ticker):
        """
        Load OHLCV data for a specific ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: OHLCV data sorted by date
        """
        try:
            query = """
                SELECT ticker, date, open, high, low, close, volume
                FROM ohlcv
                WHERE ticker = ?
                ORDER BY date ASC
            """
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise
    
    
    def calculate_sma(self, df, column='close', periods=[20, 50]):
        """
        Calculate Simple Moving Average
        SMA = sum of closing prices over N periods / N
        
        Args:
            df (pd.DataFrame): OHLCV data
            column (str): Column to calculate SMA on
            periods (list): SMA periods
            
        Returns:
            pd.DataFrame: Data with SMA columns added
        """
        try:
            for period in periods:
                df[f'SMA_{period}'] = df[column].rolling(window=period).mean()
            logger.debug(f"Calculated SMA for periods: {periods}")
            return df
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            raise
    
    
    def calculate_ema(self, df, column='close', periods=[12]):
        """
        Calculate Exponential Moving Average
        EMA gives more weight to recent prices
        EMA = (Close - EMA_prev) * multiplier + EMA_prev
        
        Args:
            df (pd.DataFrame): OHLCV data
            column (str): Column to calculate EMA on
            periods (list): EMA periods
            
        Returns:
            pd.DataFrame: Data with EMA columns added
        """
        try:
            for period in periods:
                df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
            logger.debug(f"Calculated EMA for periods: {periods}")
            return df
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            raise
    
    
    def calculate_rsi(self, df, column='close', period=14):
        """
        Calculate Relative Strength Index
        RSI measures momentum and magnitude of directional price changes
        RSI = 100 - (100 / (1 + RS)) where RS = avg gain / avg loss
        Range: 0-100 (Below 30 = oversold, Above 70 = overbought)
        
        Args:
            df (pd.DataFrame): OHLCV data
            column (str): Column to calculate RSI on
            period (int): RSI period (typically 14)
            
        Returns:
            pd.DataFrame: Data with RSI column added
        """
        try:
            delta = df[column].diff()
            
            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            df[f'RSI_{period}'] = rsi
            logger.debug(f"Calculated RSI_{period}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise
    
    
    def calculate_macd(self, df, column='close', fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        MACD is momentum indicator showing relationship between two moving averages
        MACD = EMA_12 - EMA_26
        Signal = EMA_9 of MACD
        Histogram = MACD - Signal
        
        Args:
            df (pd.DataFrame): OHLCV data
            column (str): Column to calculate MACD on
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            pd.DataFrame: Data with MACD columns added
        """
        try:
            ema_fast = df[column].ewm(span=fast, adjust=False).mean()
            ema_slow = df[column].ewm(span=slow, adjust=False).mean()
            
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            df['MACD'] = macd
            df['MACD_Signal'] = signal_line
            df['MACD_Histogram'] = histogram
            
            logger.debug(f"Calculated MACD({fast},{slow},{signal})")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise
    
    
    def calculate_bollinger_bands(self, df, column='close', period=20, num_std=2):
        """
        Calculate Bollinger Bands
        BB identifies overbought/oversold conditions and volatility
        Upper Band = SMA + (std * 2)
        Middle Band = SMA
        Lower Band = SMA - (std * 2)
        
        Args:
            df (pd.DataFrame): OHLCV data
            column (str): Column to calculate BB on
            period (int): Moving average period
            num_std (float): Number of standard deviations
            
        Returns:
            pd.DataFrame: Data with Bollinger Band columns added
        """
        try:
            sma = df[column].rolling(window=period).mean()
            std = df[column].rolling(window=period).std()
            
            df['BB_Upper'] = sma + (std * num_std)
            df['BB_Middle'] = sma
            df['BB_Lower'] = sma - (std * num_std)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            
            logger.debug(f"Calculated Bollinger Bands({period},{num_std})")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise
    
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range
        ATR measures market volatility
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA of TR
        
        Args:
            df (pd.DataFrame): OHLCV data with high, low, close
            period (int): ATR period (typically 14)
            
        Returns:
            pd.DataFrame: Data with ATR column added
        """
        try:
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as EMA of TR
            atr = tr.ewm(span=period, adjust=False).mean()
            
            df[f'ATR_{period}'] = atr
            logger.debug(f"Calculated ATR_{period}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise
    
    
    def calculate_volume_features(self, df, period=20):
        """
        Calculate volume-based features
        Volume MA: Moving average of trading volume
        Volume Ratio: Current volume / Volume MA
        
        Args:
            df (pd.DataFrame): OHLCV data
            period (int): Volume MA period
            
        Returns:
            pd.DataFrame: Data with volume features
        """
        try:
            df[f'Volume_MA_{period}'] = df['volume'].rolling(window=period).mean()
            df['Volume_Ratio'] = df['volume'] / df[f'Volume_MA_{period}']
            
            logger.debug(f"Calculated volume features (period={period})")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume features: {str(e)}")
            raise
    
    
    def calculate_derived_features(self, df):
        """
        Create derived features from base indicators
        
        Args:
            df (pd.DataFrame): Data with technical indicators
            
        Returns:
            pd.DataFrame: Data with derived features
        """
        try:
            # 1. Daily return (%)
            df['Daily_Return'] = df['close'].pct_change() * 100
            
            # 2. Distance from SMA_50 (%)
            df['Distance_SMA50'] = ((df['close'] - df['SMA_50']) / df['SMA_50']) * 100
            
            # 3. RSI Category (oversold, neutral, overbought)
            df['RSI_Category'] = pd.cut(
                df['RSI_14'],
                bins=[0, 30, 70, 100],
                labels=['oversold', 'neutral', 'overbought']
            )
            
            # 4. MACD Signal (bullish/bearish crossover)
            # Bullish: MACD > Signal and MACD_prev < Signal_prev
            # Bearish: MACD < Signal and MACD_prev > Signal_prev
            macd_above_current = df['MACD'] > df['MACD_Signal']
            macd_above_prev = df['MACD'].shift(1) > df['MACD_Signal'].shift(1)
            
            # Crossover conditions
            bullish = macd_above_current & ~macd_above_prev
            bearish = ~macd_above_current & macd_above_prev
            
            df['MACD_Crossover'] = 'no_cross'
            df.loc[bullish, 'MACD_Crossover'] = 'bullish'
            df.loc[bearish, 'MACD_Crossover'] = 'bearish'
            
            # 5. High-Low range (%)
            df['HL_Range'] = ((df['high'] - df['low']) / df['close']) * 100
            
            # 6. Price position in Bollinger Bands (0-1, 0=lower, 1=upper)
            bb_range = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['close'] - df['BB_Lower']) / bb_range
            df['BB_Position'] = df['BB_Position'].clip(0, 1)  # Clip to 0-1 range
            
            logger.debug("Calculated derived features")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating derived features: {str(e)}")
            raise
    
    
    def calculate_price_momentum(self, df):
        """
        Calculate price momentum features
        Measures rate of price change over multiple timeframes
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with momentum features added
        """
        try:
            # 5-day price momentum
            df['Momentum_5d'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            
            # 10-day price momentum
            df['Momentum_10d'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # 20-day price momentum
            df['Momentum_20d'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
            
            logger.debug("Calculated price momentum features")
            return df
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            raise
    
    
    def calculate_volatility_features(self, df):
        """
        Calculate volatility features
        Measures price variability and risk
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with volatility features added
        """
        try:
            # Calculate daily returns
            returns = df['close'].pct_change() * 100
            
            # 5-day rolling volatility (std dev of returns)
            df['Volatility_5d'] = returns.rolling(window=5).std()
            
            # 10-day rolling volatility
            df['Volatility_10d'] = returns.rolling(window=10).std()
            
            # 20-day average volatility
            volatility_20d = returns.rolling(window=20).std()
            
            # Volatility ratio: current volatility / 20-day average volatility
            df['Volatility_Ratio'] = df['Volatility_5d'] / (volatility_20d + 1e-10)
            
            logger.debug("Calculated volatility features")
            return df
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            raise
    
    
    def calculate_advanced_volume_features(self, df):
        """
        Calculate advanced volume features including Money Flow Index
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with volume features added
        """
        try:
            # 5-day average volume
            df['Volume_5d_Avg'] = df['volume'].rolling(window=5).mean()
            
            # Volume trend: (current - 5d avg) / 5d avg
            df['Volume_Trend'] = ((df['volume'] - df['Volume_5d_Avg']) / (df['Volume_5d_Avg'] + 1e-10)) * 100
            
            # Money Flow Index (MFI) - like RSI but incorporates volume
            # Typical price = (high + low + close) / 3
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Raw Money Flow = Typical Price × Volume
            raw_money_flow = typical_price * df['volume']
            
            # Positive Money Flow (when price goes up)
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
            
            # Negative Money Flow (when price goes down)
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
            
            # 14-period Money Flow Index
            positive_flow_14 = positive_flow.rolling(window=14).sum()
            negative_flow_14 = negative_flow.rolling(window=14).sum()
            
            # MFI = 100 - (100 / (1 + (Positive MF / Negative MF)))
            mfi_ratio = positive_flow_14 / (negative_flow_14 + 1e-10)
            df['Money_Flow_Index'] = 100 - (100 / (1 + mfi_ratio))
            
            logger.debug("Calculated advanced volume features")
            return df
        except Exception as e:
            logger.error(f"Error calculating volume features: {str(e)}")
            raise
    
    
    def calculate_trend_strength(self, df):
        """
        Calculate trend strength features
        Measures how strong the current trend is
        
        Args:
            df (pd.DataFrame): OHLCV data with ATR already calculated
            
        Returns:
            pd.DataFrame: Data with trend features added
        """
        try:
            # Calculate consecutive up/down days
            price_changes = df['close'].diff()
            is_up = price_changes > 0
            
            # Count consecutive ups (reset to 0 when down)
            consecutive_ups = (is_up).astype(int)
            consecutive_ups[~is_up] = 0
            consecutive_ups = consecutive_ups.groupby((is_up != is_up.shift()).cumsum()).cumsum()
            df['Consecutive_Up_Days'] = consecutive_ups
            
            # Count consecutive downs (reset to 0 when up)
            consecutive_downs = (~is_up).astype(int)
            consecutive_downs[is_up] = 0
            consecutive_downs = consecutive_downs.groupby((is_up == is_up.shift()).cumsum()).cumsum()
            df['Consecutive_Down_Days'] = consecutive_downs
            
            # ADX calculation (Average Directional Index)
            # Measures trend strength (0-100, >25 = strong trend)
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional movements
            up_move = high.diff()
            down_move = -low.diff()
            
            # Determine +DM and -DM
            plus_dm = up_move.copy()
            minus_dm = down_move.copy()
            
            plus_dm[(up_move <= down_move) | (up_move <= 0)] = 0
            minus_dm[(down_move <= up_move) | (down_move <= 0)] = 0
            
            # 14-period ADX
            period = 14
            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
            df['ADX'] = dx.rolling(window=period).mean()
            df['Plus_DI'] = plus_di
            df['Minus_DI'] = minus_di
            
            logger.debug("Calculated trend strength features")
            return df
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            raise
    
    
    def calculate_market_regime(self, df):
        """
        Calculate market regime features
        Identifies current market conditions (trend, volatility regime)
        
        Args:
            df (pd.DataFrame): OHLCV data with SMA and ATR already calculated
            
        Returns:
            pd.DataFrame: Data with regime features added
        """
        try:
            # 1. Is price above/below 50-day SMA?
            df['Above_SMA50'] = (df['close'] > df['SMA_50']).astype(int)
            
            # 2. Is short-term SMA above long-term SMA?
            df['SMA20_Above_SMA50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
            
            # 3. Volatility regime (based on ATR percentile)
            # Simpler approach: calculate ATR percentile rank over 50 days
            atr_50_min = df['ATR_14'].rolling(window=50).min()
            atr_50_max = df['ATR_14'].rolling(window=50).max()
            atr_percentile = ((df['ATR_14'] - atr_50_min) / (atr_50_max - atr_50_min + 1e-10)) * 100
            
            # Classify as high (>75th), medium (25-75th), low (<25th)
            df['Volatility_Regime'] = 0  # Default to medium
            df.loc[atr_percentile > 75, 'Volatility_Regime'] = 2  # High
            df.loc[atr_percentile < 25, 'Volatility_Regime'] = 1  # Low
            
            logger.debug("Calculated market regime features")
            return df
        except Exception as e:
            logger.error(f"Error calculating market regime: {str(e)}")
            raise

    
    def create_multi_day_targets(self, df):
        """
        Create multi-day binary target variables for prediction
        
        Creates 8 target columns:
        - Target_1d, Target_3d, Target_5d, Target_7d: 1 if return > 0%, else 0
        - Target_1d_1pct, Target_3d_1pct, Target_5d_1pct, Target_7d_1pct: 1 if return > 1%, else 0
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Data with all target columns added
        """
        try:
            # Calculate returns for different time horizons
            windows = [1, 3, 5, 7]
            
            for window in windows:
                # Get close price N days in the future
                future_close = df['close'].shift(-window)
                
                # Calculate return percentage
                returns = ((future_close - df['close']) / df['close'] * 100)
                
                # Target 1: 1 if return > 0% (any positive gain)
                df[f'Target_{window}d'] = (returns > 0).astype(int)
                
                # Target 2: 1 if return > 1% (meaningful gain)
                df[f'Target_{window}d_1pct'] = (returns > 1).astype(int)
                
                # Store return percentage for analysis
                df[f'Return_{window}d'] = returns
            
            logger.info("Created multi-day target variables")
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {str(e)}")
            raise
    
    
    def handle_missing_values(self, df):
        """
        Handle missing values created by indicator calculations and targets
        
        - Forward fill first N days where indicators can't be calculated
        - Drop the last 7 rows (incomplete targets - not enough future data)
        - Drop rows where any target is NaN
        
        Args:
            df (pd.DataFrame): Data with indicators
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        try:
            # Count initial NaN rows (from moving averages)
            initial_nans = df.isna().sum().max()
            
            # Forward fill for first N days (where indicators need history)
            df = df.bfill()
            
            # Drop last 7 rows (has NaN targets - no enough future data for 7-day target)
            df = df.iloc[:-7]
            
            # Drop any remaining NaN rows in targets
            target_cols = [col for col in df.columns if col.startswith('Target_')]
            if target_cols:
                df = df.dropna(subset=target_cols)
            
            logger.debug(f"Handled missing values (filled first {initial_nans} days, dropped last 7 rows for targets)")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    
    def engineer_features_for_ticker(self, ticker):
        """
        Run complete feature engineering pipeline for one ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: Complete feature set
        """
        try:
            logger.info(f"Engineering features for {ticker}...")
            
            # Load raw data
            df = self.load_ticker_data(ticker)
            
            # Calculate technical indicators
            df = self.calculate_sma(df, periods=[20, 50])
            df = self.calculate_ema(df, periods=[12])
            df = self.calculate_rsi(df, period=14)
            df = self.calculate_macd(df)
            df = self.calculate_bollinger_bands(df, period=20)
            df = self.calculate_atr(df, period=14)
            df = self.calculate_volume_features(df, period=20)
            
            # NEW FEATURES: Advanced momentum and volatility
            df = self.calculate_price_momentum(df)
            df = self.calculate_volatility_features(df)
            df = self.calculate_advanced_volume_features(df)
            df = self.calculate_trend_strength(df)
            df = self.calculate_market_regime(df)
            
            # Create derived features
            df = self.calculate_derived_features(df)
            
            # Create multi-day target variables
            df = self.create_multi_day_targets(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            logger.info(f"Feature engineering complete for {ticker}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features for {ticker}: {str(e)}")
            raise
    
    
    def save_features_to_csv(self, df, ticker):
        """
        Save feature dataframe to CSV
        
        Args:
            df (pd.DataFrame): Features
            ticker (str): Stock ticker
            
        Returns:
            str: Path to saved file
        """
        try:
            filepath = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved features to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            raise
    
    
    def save_combined_features(self, all_features_dict):
        """
        Combine all ticker features and save to single multiday file
        
        Args:
            all_features_dict (dict): Features for each ticker
            
        Returns:
            str: Path to combined file
        """
        try:
            combined_df = pd.concat(
                [df.assign(ticker=ticker) for ticker, df in all_features_dict.items()],
                ignore_index=True
            )
            
            filepath = os.path.join(PROCESSED_DIR, 'features_multiday.csv')
            combined_df.to_csv(filepath, index=False)
            logger.info(f"Saved combined features to {filepath} ({len(combined_df)} rows)")
            return filepath
        except Exception as e:
            logger.error(f"Error saving combined features: {str(e)}")
            raise
    
    
    def generate_target_statistics(self, all_features_dict):
        """
        Generate statistics comparing different target variables
        
        Args:
            all_features_dict (dict): Features for each ticker
            
        Returns:
            dict: Statistics for each target
        """
        try:
            # Combine all data
            combined_df = pd.concat(all_features_dict.values(), ignore_index=True)
            
            stats = {}
            
            # Calculate statistics for each target
            targets = ['Target_1d', 'Target_3d', 'Target_5d', 'Target_7d']
            targets_1pct = ['Target_1d_1pct', 'Target_3d_1pct', 'Target_5d_1pct', 'Target_7d_1pct']
            returns = ['Return_1d', 'Return_3d', 'Return_5d', 'Return_7d']
            
            print("\n" + "="*80)
            print("MULTI-DAY TARGET STATISTICS")
            print("="*80)
            
            for target, target_1pct, ret in zip(targets, targets_1pct, returns):
                pos_count = combined_df[target].sum()
                pos_pct = (pos_count / len(combined_df)) * 100
                
                pos_count_1pct = combined_df[target_1pct].sum()
                pos_pct_1pct = (pos_count_1pct / len(combined_df)) * 100
                
                avg_return = combined_df[ret].mean()
                std_return = combined_df[ret].std()
                
                stats[target] = {
                    'positive_count': int(pos_count),
                    'positive_pct': round(pos_pct, 2),
                    'positive_count_1pct': int(pos_count_1pct),
                    'positive_pct_1pct': round(pos_pct_1pct, 2),
                    'avg_return': round(avg_return, 4),
                    'std_return': round(std_return, 4),
                    'total_samples': len(combined_df)
                }
                
                window = target.split('_')[1]
                print(f"\n{window.upper()} TARGETS:")
                print(f"  Threshold 0%:     {int(pos_count):,} positive ({pos_pct:.1f}%)")
                print(f"  Threshold 1%:     {int(pos_count_1pct):,} positive ({pos_pct_1pct:.1f}%)")
                print(f"  Avg Return:       {avg_return:+.2f}%")
                print(f"  Std Dev Return:   {std_return:.2f}%")
            
            # Correlations between targets
            print("\n" + "="*80)
            print("CORRELATION BETWEEN TARGETS")
            print("="*80)
            
            corr_matrix = combined_df[targets].corr()
            print("\nTarget Correlation Matrix:")
            print(corr_matrix.round(3))
            
            # Compare short vs long term
            print("\n" + "="*80)
            print("SHORT-TERM vs LONG-TERM COMPARISON")
            print("="*80)
            
            corr_1d_5d = combined_df['Target_1d'].corr(combined_df['Target_5d'])
            print(f"\n1-day target vs 5-day target correlation: {corr_1d_5d:.3f}")
            print("  → Lower correlation = different signals")
            print("  → 5-day smoother, less noise than 1-day")
            
            logger.info("Generated target statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {str(e)}")
            raise
    
    
    def get_all_tickers(self):
        """
        Get list of all tickers in database
        
        Returns:
            list: Ticker symbols
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
            tickers = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return tickers
        except Exception as e:
            logger.error(f"Error getting tickers: {str(e)}")
            raise
    
    
    def engineer_all_tickers(self):
        """
        Run feature engineering for all tickers
        
        Returns:
            dict: Features for each ticker
        """
        logger.info("Starting feature engineering for all tickers...")
        
        tickers = self.get_all_tickers()
        all_features = {}
        
        for ticker in tickers:
            try:
                features = self.engineer_features_for_ticker(ticker)
                all_features[ticker] = features
                self.save_features_to_csv(features, ticker)
                
            except Exception as e:
                logger.error(f"Failed to engineer features for {ticker}: {str(e)}")
                continue
        
        logger.info(f"Feature engineering complete for {len(all_features)}/{len(tickers)} tickers")
        return all_features


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features for all tickers
    print("\nStarting feature engineering pipeline with MULTI-DAY TARGETS...\n")
    all_features = engineer.engineer_all_tickers()
    
    # Save combined features
    multiday_file = engineer.save_combined_features(all_features)
    
    # Generate comparison statistics
    stats = engineer.generate_target_statistics(all_features)
    
    # Display summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(all_features)} tickers")
    print(f"Individual features saved to: {PROCESSED_DIR}/*.csv")
    print(f"Combined features saved to:  {multiday_file}")
    
    # Show sample of first ticker
    first_ticker = list(all_features.keys())[0]
    sample_data = all_features[first_ticker]
    
    print(f"\nSample: {first_ticker} ({len(sample_data)} rows)")
    
    # Show target columns
    target_cols = [col for col in sample_data.columns if col.startswith('Target_') or col.startswith('Return_')]
    print(f"\nTarget Columns ({len(target_cols)}):")
    for col in sorted(target_cols):
        print(f"  • {col}")
    
    print(f"\nTotal Features: {len(sample_data.columns)}")
    print(f"\nFirst row sample:")
    for col in target_cols:
        if col.startswith('Return_'):
            print(f"  {col}: {sample_data[col].iloc[0]:+.2f}%")
        else:
            print(f"  {col}: {sample_data[col].iloc[0]}")
    
    print("\n" + "="*80)
    print("Multi-day features ready for model training!")
    print("="*80)
