"""
Data Collection Module - OHLCV Data Download
Handles fetching 2 years of market data from Alpaca API and yfinance
Stores data in SQLite database and CSV backups
"""

import pandas as pd
import yfinance as yf
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time

# Configure logging
log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'data_collection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = 'data/market_data.db'
DATA_DIR = 'data/raw'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


class MarketDataCollector:
    """
    Main class for downloading and storing market data
    """
    
    def __init__(self, db_path=DB_PATH):
        """
        Initialize the data collector
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.init_database()
        logger.info("MarketDataCollector initialized")
    
    
    def init_database(self):
        """
        Initialize SQLite database with OHLCV schema
        Creates table if it doesn't exist
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table with proper schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ticker_date 
                ON ohlcv(ticker, date)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    
    def data_exists(self, ticker, start_date, end_date):
        """
        Check if data already exists in database for a ticker in the date range
        Prevents unnecessary re-downloading
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            tuple: (exists: bool, row_count: int)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM ohlcv 
                WHERE ticker = ? AND date >= ? AND date <= ?
            ''', (ticker, start_date, end_date))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0, count
            
        except Exception as e:
            logger.error(f"Error checking data existence: {str(e)}")
            return False, 0
    
    
    def fetch_data_yfinance(self, ticker, start_date, end_date):
        """
        Fetch historical OHLCV data from yfinance
        Used as primary source with retry logic
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: OHLCV data with columns [date, open, high, low, close, volume]
        """
        try:
            logger.info(f"Fetching {ticker} from yfinance ({start_date} to {end_date})")
            
            # Download data from yfinance
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1d',
                progress=False
            )
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            
            # Handle MultiIndex columns (flatten if necessary)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex - take first level
                data.columns = data.columns.get_level_values(0)
            
            # Rename Date column to date
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
            
            # Standardize column names to lowercase
            data.columns = data.columns.str.lower()
            
            # Select only needed columns
            data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert date to string format
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            
            # Remove any NaN rows
            data = data.dropna()
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    
    def save_to_database(self, ticker, data):
        """
        Save OHLCV data to SQLite database
        Handles duplicates by skipping them (UNIQUE constraint)
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): OHLCV data to save
            
        Returns:
            int: Number of rows inserted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            inserted_count = 0
            
            # Insert data row by row (handles duplicates gracefully)
            for idx, row in data.iterrows():
                try:
                    cursor.execute('''
                        INSERT INTO ohlcv 
                        (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        row['date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume'])
                    ))
                    inserted_count += 1
                except sqlite3.IntegrityError:
                    # Skip duplicate records
                    pass
            
            conn.commit()
            conn.close()
            
            logger.info(f"Inserted {inserted_count} rows for {ticker} to database")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error saving data to database: {str(e)}")
            raise
    
    
    def save_to_csv(self, ticker, data):
        """
        Save OHLCV data to CSV file as backup
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): OHLCV data to save
            
        Returns:
            str: Path to saved CSV file
        """
        try:
            csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            data.to_csv(csv_path, index=False)
            logger.info(f"Saved {ticker} to {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error saving CSV for {ticker}: {str(e)}")
            raise
    
    
    def download_ticker_data(self, ticker, years=2):
        """
        Download data for a single ticker (2 years by default)
        Checks if data already exists before downloading
        
        Args:
            ticker (str): Stock ticker symbol
            years (int): Number of years of data to download
            
        Returns:
            dict: {ticker: data_rows_added, status: 'success'/'skipped'/'error'}
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            
            # Check if data already exists
            exists, count = self.data_exists(ticker, start_date, end_date)
            if exists and count > 400:  # 2 years should have ~500 trading days
                logger.info(f"{ticker}: Data already exists ({count} rows). Skipping download.")
                return {'ticker': ticker, 'rows': count, 'status': 'skipped'}
            
            # Fetch data from yfinance
            data = self.fetch_data_yfinance(ticker, start_date, end_date)
            
            if len(data) == 0:
                logger.warning(f"{ticker}: No data returned")
                return {'ticker': ticker, 'rows': 0, 'status': 'no_data'}
            
            # Save to database
            rows_added = self.save_to_database(ticker, data)
            
            # Save CSV backup
            self.save_to_csv(ticker, data)
            
            # Rate limiting to avoid API throttling
            time.sleep(0.5)
            
            return {'ticker': ticker, 'rows': rows_added, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            return {'ticker': ticker, 'rows': 0, 'status': 'error', 'error': str(e)}
    
    
    def download_multiple_tickers(self, tickers, years=2):
        """
        Download data for multiple tickers with progress bar
        Includes error handling for rate limits and connection issues
        
        Args:
            tickers (list): List of stock ticker symbols
            years (int): Number of years of data to download
            
        Returns:
            pd.DataFrame: Summary of download results
        """
        logger.info(f"Starting bulk download of {len(tickers)} tickers ({years} years each)")
        
        results = []
        
        # Progress bar for tracking download status
        with tqdm(total=len(tickers), desc="Downloading data", unit="ticker") as pbar:
            for ticker in tickers:
                try:
                    result = self.download_ticker_data(ticker, years)
                    results.append(result)
                    
                    # Update progress bar with result
                    status = result.get('status', 'unknown')
                    pbar.update(1)
                    pbar.set_postfix({'ticker': ticker, 'status': status})
                    
                except Exception as e:
                    logger.error(f"Exception downloading {ticker}: {str(e)}")
                    results.append({
                        'ticker': ticker,
                        'rows': 0,
                        'status': 'exception',
                        'error': str(e)
                    })
                    pbar.update(1)
        
        # Convert results to DataFrame for summary
        results_df = pd.DataFrame(results)
        logger.info(f"\nDownload Summary:\n{results_df.to_string()}")
        
        return results_df
    
    
    def load_from_database(self, ticker, start_date=None, end_date=None):
        """
        Load OHLCV data from database for a specific ticker
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in format 'YYYY-MM-DD' (optional)
            end_date (str): End date in format 'YYYY-MM-DD' (optional)
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            if start_date is None:
                start_date = '1990-01-01'
            if end_date is None:
                end_date = '2099-12-31'
            
            query = '''
                SELECT ticker, date, open, high, low, close, volume
                FROM ohlcv
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            '''
            
            df = pd.read_sql_query(
                query,
                sqlite3.connect(self.db_path),
                params=(ticker, start_date, end_date)
            )
            
            logger.info(f"Loaded {len(df)} rows for {ticker} from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise
    
    
    def get_database_stats(self):
        """
        Get statistics about stored data
        
        Returns:
            pd.DataFrame: Stats by ticker (count of records)
        """
        try:
            query = '''
                SELECT ticker, COUNT(*) as record_count, 
                       MIN(date) as earliest_date, MAX(date) as latest_date
                FROM ohlcv
                GROUP BY ticker
                ORDER BY ticker
            '''
            
            df = pd.read_sql_query(query, sqlite3.connect(self.db_path))
            logger.info(f"\nDatabase Statistics:\n{df.to_string()}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # List of 20 major stocks to download
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'JPM', 'V', 'WMT',
        'DIS', 'NFLX', 'PYPL', 'AMD', 'INTC',
        'BA', 'GE', 'F', 'GM', 'COST'
    ]
    
    # Initialize collector
    collector = MarketDataCollector()
    
    # Download 2 years of data for all tickers
    logger.info(f"Downloading data for {len(TICKERS)} tickers")
    results = collector.download_multiple_tickers(TICKERS, years=2)
    
    # Display statistics
    logger.info("\n" + "="*60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("="*60)
    stats = collector.get_database_stats()
    print("\n" + stats.to_string())
    
    logger.info(f"\nData stored in: {collector.db_path}")
    logger.info(f"CSV backups in: {DATA_DIR}")
    logger.info(f"Logs available in: logs/data_collection.log")
