"""
Data Validation Module
Validates OHLCV data quality and generates comprehensive reports
Checks for missing dates, null values, duplicates, and price anomalies
"""

import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'data_validation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation class for OHLCV data
    """
    
    def __init__(self, db_path='data/market_data.db'):
        """
        Initialize validator
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.report_path = 'logs/data_quality_report.txt'
        self.validation_results = {}
        self.issues_found = []
        logger.info("DataValidator initialized")
    
    
    def load_ticker_data(self, ticker):
        """
        Load data for a specific ticker from database
        
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
    
    
    def get_all_tickers(self):
        """
        Get list of all tickers in database
        
        Returns:
            list: List of ticker symbols
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
    
    
    def check_null_values(self, df, ticker):
        """
        Check for null/NaN values in OHLCV columns
        
        Args:
            df (pd.DataFrame): OHLCV data
            ticker (str): Ticker symbol
            
        Returns:
            dict: Null value statistics
        """
        null_stats = {}
        
        # Check each column for nulls
        for col in ['open', 'high', 'low', 'close', 'volume']:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_stats[col] = null_count
                issue = f"{ticker}: Found {null_count} null values in {col}"
                self.issues_found.append(issue)
                logger.warning(issue)
        
        return null_stats if null_stats else {'status': 'OK - No nulls found'}
    
    
    def check_duplicates(self, df, ticker):
        """
        Check for duplicate rows (same date entries)
        
        Args:
            df (pd.DataFrame): OHLCV data
            ticker (str): Ticker symbol
            
        Returns:
            dict: Duplicate statistics
        """
        # Check for duplicate dates
        duplicate_dates = df[df.duplicated(subset=['date'], keep=False)]
        
        if len(duplicate_dates) > 0:
            dup_count = len(duplicate_dates) // 2  # Each duplicate appears twice
            issue = f"{ticker}: Found {dup_count} duplicate date entries"
            self.issues_found.append(issue)
            logger.warning(issue)
            return {'duplicate_count': dup_count, 'dates': duplicate_dates['date'].unique().tolist()}
        
        return {'status': 'OK - No duplicates found'}
    
    
    def check_price_anomalies(self, df, ticker, threshold=0.50):
        """
        Check for extreme price movements (potential data errors)
        Daily returns > 50% are flagged as anomalies
        
        Args:
            df (pd.DataFrame): OHLCV data
            ticker (str): Ticker symbol
            threshold (float): Return threshold (default 50%)
            
        Returns:
            dict: Anomaly statistics
        """
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # Find anomalies (absolute return > threshold)
        anomalies = df[df['daily_return'].abs() > threshold].copy()
        
        anomaly_stats = {
            'threshold': f"{threshold*100}%",
            'anomaly_count': len(anomalies)
        }
        
        if len(anomalies) > 0:
            issue = f"{ticker}: Found {len(anomalies)} days with price movements > {threshold*100}%"
            self.issues_found.append(issue)
            logger.warning(issue)
            
            # Show details of anomalies
            anomaly_stats['details'] = []
            for idx, row in anomalies.iterrows():
                detail = {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'return': f"{row['daily_return']*100:.2f}%",
                    'close': f"${row['close']:.2f}"
                }
                anomaly_stats['details'].append(detail)
                logger.warning(f"  {detail['date']}: {detail['return']} (Close: {detail['close']})")
        else:
            anomaly_stats['status'] = 'OK - No anomalies found'
        
        return anomaly_stats
    
    
    def check_missing_dates(self, df, ticker):
        """
        Check for missing trading dates (gaps in data)
        Note: Market holidays and weekends are expected
        Gaps up to 5% are considered normal
        
        Args:
            df (pd.DataFrame): OHLCV data
            ticker (str): Ticker symbol
            
        Returns:
            dict: Missing date statistics
        """
        # Generate expected trading day range (Business days only)
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Create business day range
        expected_dates = pd.bdate_range(start=min_date, end=max_date)
        actual_dates = set(df['date'].dt.date)
        expected_dates_set = set(expected_dates.date)
        
        # Find missing dates
        missing_dates = expected_dates_set - actual_dates
        
        expected_count = len(expected_dates)
        actual_count = len(actual_dates)
        missing_count = len(missing_dates)
        gap_percentage = (missing_count / expected_count * 100) if expected_count > 0 else 0
        
        missing_stats = {
            'expected_trading_days': expected_count,
            'actual_days': actual_count,
            'missing_count': missing_count,
            'gap_percentage': f"{gap_percentage:.1f}%"
        }
        
        # Only flag if gap is > 5% (indicates real issue, not just holidays)
        if gap_percentage > 5.0:
            issue = f"{ticker}: Found {missing_count} missing trading dates ({gap_percentage:.1f}% gap)"
            self.issues_found.append(issue)
            logger.warning(issue)
            missing_stats['missing_dates'] = sorted(list(missing_dates))[:10]  # Show first 10
        else:
            missing_stats['note'] = f'Small gap ({gap_percentage:.1f}%) - normal due to market holidays'
        
        return missing_stats
    
    
    def validate_ticker(self, ticker):
        """
        Run all validation checks for a single ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Comprehensive validation results
        """
        logger.info(f"\nValidating {ticker}...")
        
        try:
            # Load data
            df = self.load_ticker_data(ticker)
            
            # Initialize results dictionary
            results = {
                'ticker': ticker,
                'status': 'VALID',
                'row_count': len(df),
                'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            }
            
            # Run validation checks
            results['null_values'] = self.check_null_values(df, ticker)
            results['duplicates'] = self.check_duplicates(df, ticker)
            results['missing_dates'] = self.check_missing_dates(df, ticker)
            results['price_anomalies'] = self.check_price_anomalies(df, ticker)
            
            # Determine overall status
            if len([issue for issue in self.issues_found if issue.startswith(ticker)]) > 0:
                results['status'] = 'WARNING'
            
            self.validation_results[ticker] = results
            logger.info(f"{ticker} validation completed - Status: {results['status']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'error': str(e)
            }
    
    
    def validate_all_tickers(self):
        """
        Run validation for all tickers in database
        
        Returns:
            dict: Results for all tickers
        """
        logger.info("Starting validation of all tickers...")
        
        tickers = self.get_all_tickers()
        logger.info(f"Found {len(tickers)} tickers to validate")
        
        for ticker in tickers:
            self.validate_ticker(ticker)
        
        return self.validation_results
    
    
    def generate_report(self):
        """
        Generate comprehensive validation report
        
        Returns:
            str: Report content
        """
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("DATA QUALITY VALIDATION REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # Summary statistics
        report_lines.append("\nSUMMARY STATISTICS")
        report_lines.append("-"*80)
        
        total_tickers = len(self.validation_results)
        valid_tickers = sum(1 for r in self.validation_results.values() if r.get('status') == 'VALID')
        warning_tickers = sum(1 for r in self.validation_results.values() if r.get('status') == 'WARNING')
        error_tickers = sum(1 for r in self.validation_results.values() if r.get('status') == 'ERROR')
        
        report_lines.append(f"Total Tickers Validated: {total_tickers}")
        report_lines.append(f"  - VALID:   {valid_tickers}")
        report_lines.append(f"  - WARNING: {warning_tickers}")
        report_lines.append(f"  - ERROR:   {error_tickers}")
        
        total_rows = sum(r.get('row_count', 0) for r in self.validation_results.values())
        report_lines.append(f"Total Data Rows: {total_rows:,}")
        
        # Issues found
        if self.issues_found:
            report_lines.append("\n" + "="*80)
            report_lines.append("DATA QUALITY ISSUES FOUND")
            report_lines.append("="*80)
            for i, issue in enumerate(self.issues_found, 1):
                report_lines.append(f"{i}. {issue}")
        else:
            report_lines.append("\n" + "="*80)
            report_lines.append("NO ISSUES FOUND - DATA QUALITY IS EXCELLENT!")
            report_lines.append("="*80)
        
        # Detailed results by ticker
        report_lines.append("\n" + "="*80)
        report_lines.append("DETAILED VALIDATION RESULTS BY TICKER")
        report_lines.append("="*80)
        
        for ticker in sorted(self.validation_results.keys()):
            result = self.validation_results[ticker]
            
            report_lines.append(f"\n{ticker}")
            report_lines.append("-"*80)
            report_lines.append(f"Status:       {result.get('status', 'N/A')}")
            report_lines.append(f"Row Count:    {result.get('row_count', 'N/A'):,}")
            report_lines.append(f"Date Range:   {result.get('date_range', 'N/A')}")
            
            # Null values
            if 'null_values' in result:
                null_info = result['null_values']
                if null_info.get('status') == 'OK - No nulls found':
                    report_lines.append("Null Values:  PASS - No null values")
                else:
                    report_lines.append("Null Values:  FAIL - Found nulls:")
                    for col, count in null_info.items():
                        report_lines.append(f"  - {col}: {count}")
            
            # Duplicates
            if 'duplicates' in result:
                dup_info = result['duplicates']
                if dup_info.get('status') == 'OK - No duplicates found':
                    report_lines.append("Duplicates:   PASS - No duplicates")
                else:
                    report_lines.append(f"Duplicates:   FAIL - {dup_info.get('duplicate_count')} duplicate entries")
            
            # Missing dates
            if 'missing_dates' in result:
                miss_info = result['missing_dates']
                expected = miss_info.get('expected_trading_days', 0)
                actual = miss_info.get('actual_days', 0)
                missing = miss_info.get('missing_count', 0)
                report_lines.append(f"Trading Days: Expected {expected}, Found {actual} (Gap: {missing})")
                if miss_info.get('note'):
                    report_lines.append(f"              {miss_info['note']}")
            
            # Price anomalies
            if 'price_anomalies' in result:
                anom_info = result['price_anomalies']
                if anom_info.get('status') == 'OK - No anomalies found':
                    report_lines.append(f"Price Anomalies: PASS - No anomalies (threshold: {anom_info.get('threshold', 'N/A')})")
                else:
                    count = anom_info.get('anomaly_count', 0)
                    threshold = anom_info.get('threshold', 'N/A')
                    report_lines.append(f"Price Anomalies: WARNING - {count} anomalies found (threshold: {threshold})")
                    if anom_info.get('details'):
                        for detail in anom_info['details'][:3]:  # Show first 3
                            report_lines.append(f"  - {detail['date']}: {detail['return']} (Close: {detail['close']})")
        
        # Footer
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)
    
    
    def save_report(self, report_content):
        """
        Save report to file
        
        Args:
            report_content (str): Report text
        """
        try:
            with open(self.report_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to {self.report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise
    
    
    def print_summary(self):
        """
        Print summary to console
        """
        print("\n" + "="*80)
        print("DATA VALIDATION SUMMARY")
        print("="*80)
        
        for ticker in sorted(self.validation_results.keys()):
            result = self.validation_results[ticker]
            status = result.get('status', 'N/A')
            row_count = result.get('row_count', 'N/A')
            
            # Color coding for status (text-based)
            status_str = f"{ticker:8} | Rows: {row_count:>4,} | Status: {status}"
            print(status_str)
        
        print("="*80)
        
        # Summary counts
        total = len(self.validation_results)
        valid = sum(1 for r in self.validation_results.values() if r.get('status') == 'VALID')
        warning = sum(1 for r in self.validation_results.values() if r.get('status') == 'WARNING')
        
        print(f"\nTotal: {total} | Valid: {valid} | Warnings: {warning}")
        
        if self.issues_found:
            print(f"\nIssues Found: {len(self.issues_found)}")
            for issue in self.issues_found[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(self.issues_found) > 5:
                print(f"  ... and {len(self.issues_found) - 5} more")
        else:
            print("\nNo issues found - Data quality is excellent!")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Initialize validator
    validator = DataValidator()
    
    # Validate all tickers
    print("Starting data validation...\n")
    validator.validate_all_tickers()
    
    # Generate and save report
    report = validator.generate_report()
    validator.save_report(report)
    
    # Print summary to console
    validator.print_summary()
    
    # Print full report
    print("\n" + report)
    
    logger.info("Data validation complete!")
