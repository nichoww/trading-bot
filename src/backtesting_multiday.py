"""
Comprehensive Backtesting System for Multi-Day Trading Model
Simulates realistic trading with 5-day (or 7-day) holding periods
"""

import logging
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = Path('data/processed')
MODELS_PATH = Path('models')
LOGS_PATH = Path('logs')
LOGS_PATH.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    'holding_period': 5,  # days (change to 7 for 7-day model)
    'model_threshold': 0.60,  # probability threshold for BUY signal
    'position_size': 10000,  # per position
    'max_positions': 5,  # max concurrent positions
    'stop_loss': 0.05,  # 5% stop loss
    'take_profit': 0.10,  # 10% take profit
    'slippage': 0.001,  # 0.1% slippage
    'initial_capital': 100000,
    'tickers': ['AAPL', 'AMD', 'AMZN', 'BA', 'COST', 'DIS', 'F', 'GE', 'GM',
                'GOOGL', 'INTC', 'JPM', 'META', 'MSFT', 'NFLX', 'NVDA', 'PYPL',
                'TSLA', 'V', 'WMT']
}


class Position:
    """Represents a single position"""
    def __init__(self, ticker, entry_date, entry_price, shares):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0
        self.pnl_pct = 0
    
    def close(self, exit_price, exit_date, exit_reason):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_reason = exit_reason
        self.pnl = (self.exit_price - self.entry_price) * self.shares
        self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'shares': self.shares,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'hold_days': (self.exit_date - self.entry_date).days if self.exit_date else 0
        }


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config):
        self.config = config
        self.initial_capital = config['initial_capital']
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.positions = {}  # {ticker: Position}
        self.closed_positions = []
        self.portfolio_history = []
        self.trade_log = []
        
        # Load model and data
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load the trained model"""
        model_file = MODELS_PATH / f'rf_model_{self.config["holding_period"]}d_optimized.pkl'
        scaler_file = MODELS_PATH / f'scaler_{self.config["holding_period"]}d_optimized.pkl'
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LOADING MODEL FOR {self.config['holding_period']}-DAY TRADING")
        logger.info(f"{'='*80}")
        
        # Try different model versions if exact one not found
        if not model_file.exists():
            logger.info(f"Model {self.config['holding_period']}d not found, checking available models...")
            available_models = list(MODELS_PATH.glob('*_optimized.pkl'))
            if available_models:
                # Use the first available model
                model_file = available_models[0]
                logger.info(f"Using available model: {model_file.name}")
            else:
                logger.error("No models found")
                raise FileNotFoundError(f"No models found in {MODELS_PATH}")
        
        if not scaler_file.exists():
            logger.info(f"Scaler {self.config['holding_period']}d not found, checking available scalers...")
            available_scalers = list(MODELS_PATH.glob('scaler_*_optimized.pkl'))
            if available_scalers:
                scaler_file = available_scalers[0]
                logger.info(f"Using available scaler: {scaler_file.name}")
            else:
                logger.error("No scalers found")
                raise FileNotFoundError(f"No scalers found in {MODELS_PATH}")
        
        try:
            # Try joblib first (more reliable), then pickle
            try:
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                logger.info(f"✓ Model loaded with joblib: {model_file}")
                logger.info(f"✓ Scaler loaded with joblib: {scaler_file}")
            except Exception as e:
                logger.info(f"Joblib failed, trying pickle...")
                self.model = pickle.load(open(model_file, 'rb'))
                self.scaler = pickle.load(open(scaler_file, 'rb'))
                logger.info(f"✓ Model loaded with pickle: {model_file}")
                logger.info(f"✓ Scaler loaded with pickle: {scaler_file}")
            
            # Disable parallel processing to avoid hanging during backtest
            self.model.n_jobs = 1
            logger.info("✓ Disabled parallel processing (n_jobs=1) for stable predictions")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_data(self):
        """Load historical features and prices"""
        logger.info(f"\nLoading historical data...")
        
        self.all_data = {}
        for ticker in self.config['tickers']:
            feature_file = DATA_PATH / f'{ticker}_features.csv'
            if feature_file.exists():
                df = pd.read_csv(feature_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                self.all_data[ticker] = df
                logger.info(f"✓ Loaded {ticker}: {len(df)} rows")
            else:
                logger.warning(f"✗ Not found: {ticker}")
        
        # Get date range for backtesting (most recent 1 year)
        all_dates = []
        for df in self.all_data.values():
            all_dates.extend(df['date'].values)
        
        all_dates = pd.to_datetime(all_dates)
        self.end_date = all_dates.max()
        self.start_date = self.end_date - timedelta(days=365)
        
        logger.info(f"\nBacktest period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Total trading days in period: {len(pd.bdate_range(self.start_date, self.end_date))}")
    
    def _get_features_for_date(self, ticker, date):
        """Get scaled features for a specific date"""
        df = self.all_data.get(ticker)
        if df is None or len(df) == 0:
            return None
        
        # Find row for this date
        row = df[df['date'] == date]
        if len(row) == 0:
            return None
        
        # Get numeric features (excluding date and target)
        row_data = row.drop(['date', 'ticker'], axis=1, errors='ignore')
        numeric_cols = row_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        target_cols = ['Target_1d', 'Target_3d', 'Target_5d', 'Target_7d']
        numeric_cols = [col for col in numeric_cols if col not in target_cols]
        
        # Only use first N features that match model's expected input
        expected_features = self.model.n_features_in_
        numeric_cols = numeric_cols[:expected_features]
        
        # Scale features
        features = row_data[numeric_cols].values.reshape(1, -1)
        
        # Handle any NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaled_features = self.scaler.transform(features)
        
        return scaled_features
    
    def _get_price_at_date(self, ticker, date, price_type='close'):
        """Get price for a specific date"""
        df = self.all_data.get(ticker)
        if df is None or len(df) == 0:
            return None
        
        row = df[df['date'] == date]
        if len(row) == 0:
            return None
        
        return row[price_type].values[0]
    
    def _get_trading_dates(self):
        """Get all trading dates in the backtest period"""
        all_dates = set()
        for df in self.all_data.values():
            dates = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]['date']
            all_dates.update(dates)
        
        return sorted(list(all_dates))
    
    def _generate_signal(self, ticker, date):
        """Generate BUY/SELL signal for a ticker on a date"""
        features = self._get_features_for_date(ticker, date)
        if features is None:
            return None, None
        
        try:
            prob = self.model.predict_proba(features)[0]
            signal_prob = prob[1]  # probability of UP (class 1)
            
            if signal_prob > self.config['model_threshold']:
                return 'BUY', signal_prob
            else:
                return 'HOLD', signal_prob
        except Exception as e:
            logger.debug(f"Error generating signal for {ticker} on {date}: {e}")
            return None, None
    
    def _apply_slippage(self, price, is_entry=True):
        """Apply slippage to price"""
        if is_entry:
            return price * (1 + self.config['slippage'])
        else:
            return price * (1 - self.config['slippage'])
    
    def run_backtest(self):
        """Run the backtest"""
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING BACKTEST")
        logger.info(f"{'='*80}")
        
        trading_dates = self._get_trading_dates()
        logger.info(f"Total trading dates: {len(trading_dates)}")
        
        for current_date in trading_dates:
            # Check positions for exit conditions
            self._process_exits(current_date)
            
            # Check for new entry signals
            self._process_entries(current_date)
            
            # Record portfolio state
            self._record_portfolio(current_date)
        
        logger.info(f"\n{'='*80}")
        logger.info("BACKTEST COMPLETE")
        logger.info(f"{'='*80}")
    
    def _process_exits(self, current_date):
        """Process position exits"""
        tickers_to_close = []
        
        for ticker, position in self.positions.items():
            exit_price = self._get_price_at_date(ticker, current_date, 'close')
            if exit_price is None:
                continue
            
            hold_days = (current_date - position.entry_date).days
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            
            exit_reason = None
            
            # Check stop loss
            if pnl_pct < -self.config['stop_loss']:
                exit_reason = 'STOP_LOSS'
            
            # Check take profit
            elif pnl_pct > self.config['take_profit']:
                exit_reason = 'TAKE_PROFIT'
            
            # Check holding period expiration
            elif hold_days >= self.config['holding_period']:
                exit_reason = 'EXPIRED'
            
            if exit_reason:
                exit_price_with_slippage = self._apply_slippage(exit_price, is_entry=False)
                position.close(exit_price_with_slippage, current_date, exit_reason)
                
                pnl = (exit_price_with_slippage - position.entry_price) * position.shares
                self.cash += pnl + position.shares * exit_price_with_slippage
                self.closed_positions.append(position)
                tickers_to_close.append(ticker)
                
                logger.debug(f"CLOSED {ticker} on {current_date.date()}: {exit_reason} - PnL: ${pnl:,.2f}")
        
        for ticker in tickers_to_close:
            del self.positions[ticker]
    
    def _process_entries(self, current_date):
        """Process new position entries"""
        if len(self.positions) >= self.config['max_positions']:
            return
        
        for ticker in self.config['tickers']:
            if ticker in self.positions:
                continue
            
            signal, prob = self._generate_signal(ticker, current_date)
            if signal != 'BUY':
                continue
            
            entry_price = self._get_price_at_date(ticker, current_date, 'close')
            if entry_price is None:
                continue
            
            # Check if we have enough cash
            entry_price_with_slippage = self._apply_slippage(entry_price, is_entry=True)
            position_cost = self.config['position_size'] + (self.config['position_size'] * self.config['slippage'])
            
            if self.cash < position_cost:
                logger.debug(f"Insufficient cash for {ticker}")
                continue
            
            # Open position
            shares = self.config['position_size'] / entry_price_with_slippage
            position = Position(ticker, current_date, entry_price_with_slippage, shares)
            self.positions[ticker] = position
            self.cash -= position_cost
            
            self.trade_log.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'BUY',
                'price': entry_price_with_slippage,
                'shares': shares,
                'signal_prob': prob
            })
            
            logger.debug(f"OPENED {ticker} on {current_date.date()}: ${entry_price_with_slippage:.2f} - Prob: {prob:.2%}")
            
            if len(self.positions) >= self.config['max_positions']:
                break
    
    def _record_portfolio(self, current_date):
        """Record portfolio state"""
        current_value = self.cash
        
        for ticker, position in self.positions.items():
            price = self._get_price_at_date(ticker, current_date, 'close')
            if price is not None:
                current_value += position.shares * price
        
        self.portfolio_value = current_value
        
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital
        })
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.closed_positions) == 0:
            logger.warning("No closed positions to analyze")
            return {}
        
        # Basic trade statistics
        total_trades = len(self.closed_positions)
        profitable_trades = sum(1 for p in self.closed_positions if p.pnl > 0)
        losing_trades = sum(1 for p in self.closed_positions if p.pnl < 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = sum(p.pnl for p in self.closed_positions)
        profitable_pnls = [p.pnl for p in self.closed_positions if p.pnl > 0]
        losing_pnls = [p.pnl for p in self.closed_positions if p.pnl < 0]
        
        avg_win = np.mean(profitable_pnls) if profitable_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        profit_factor = sum(profitable_pnls) / abs(sum(losing_pnls)) if losing_pnls else 0
        
        # Holding period statistics
        hold_days = [p.to_dict()['hold_days'] for p in self.closed_positions]
        avg_hold_days = np.mean(hold_days) if hold_days else 0
        
        # Return calculations
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if len(portfolio_df) > 0:
            daily_returns = portfolio_df['total_return'].diff().dropna()
            
            # Annualized return
            trading_days = len(portfolio_df)
            annualized_return = total_return * (252 / trading_days) if trading_days > 0 else 0
            
            # Sharpe ratio
            daily_std = daily_returns.std()
            risk_free_rate = 0.02 / 252  # 2% annual risk-free rate
            sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_std if daily_std > 0 else 0
            
            # Max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_days': avg_hold_days,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return metrics
    
    def _calculate_benchmark(self):
        """Calculate benchmark returns (SPY buy-and-hold)"""
        logger.info("\nCalculating benchmarks...")
        
        # For simplicity, we'll calculate equal-weight returns of the 20 stocks
        # and compare to a buy-and-hold strategy
        
        spy_data = []
        for ticker in self.config['tickers']:
            df = self.all_data.get(ticker)
            if df is not None:
                spy_data.append(df)
        
        if spy_data:
            # Calculate equal-weight returns
            prices_at_start = []
            prices_at_end = []
            
            for ticker in self.config['tickers']:
                df = self.all_data.get(ticker)
                if df is None:
                    continue
                
                start_price = df[df['date'] <= self.start_date]['close'].iloc[-1] if len(df[df['date'] <= self.start_date]) > 0 else None
                end_price = df[df['date'] <= self.end_date]['close'].iloc[-1] if len(df[df['date'] <= self.end_date]) > 0 else None
                
                if start_price and end_price:
                    prices_at_start.append(start_price)
                    prices_at_end.append(end_price)
            
            if prices_at_start and prices_at_end:
                avg_start = np.mean(prices_at_start)
                avg_end = np.mean(prices_at_end)
                benchmark_return = (avg_end - avg_start) / avg_start
            else:
                benchmark_return = 0
        else:
            benchmark_return = 0
        
        return benchmark_return
    
    def save_results(self):
        """Save backtest results"""
        logger.info(f"\n{'='*80}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*80}")
        
        # Save trade log
        trade_df = pd.DataFrame([p.to_dict() for p in self.closed_positions])
        trade_csv = DATA_PATH / f'backtest_trades_{self.config["holding_period"]}d.csv'
        trade_df.to_csv(trade_csv, index=False)
        logger.info(f"✓ Saved trade log: {trade_csv}")
        
        # Save portfolio history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_csv = DATA_PATH / f'portfolio_history_{self.config["holding_period"]}d.csv'
        portfolio_df.to_csv(portfolio_csv, index=False)
        logger.info(f"✓ Saved portfolio history: {portfolio_csv}")
        
        # Calculate and save metrics
        metrics = self._calculate_metrics()
        benchmark_return = self._calculate_benchmark()
        
        results_text = self._format_results(metrics, benchmark_return)
        results_file = LOGS_PATH / f'backtest_results_{self.config["holding_period"]}d.txt'
        with open(results_file, 'w') as f:
            f.write(results_text)
        logger.info(f"✓ Saved results: {results_file}")
        
        return metrics, benchmark_return, trade_df, portfolio_df
    
    def _format_results(self, metrics, benchmark_return):
        """Format results for display and saving"""
        text = f"""
{'='*80}
BACKTEST RESULTS - {self.config['holding_period']}-DAY MODEL
{'='*80}

BACKTEST PERIOD
{'='*80}
Start Date: {self.start_date.date()}
End Date: {self.end_date.date()}
Initial Capital: ${self.initial_capital:,.2f}
Final Portfolio Value: ${self.portfolio_value:,.2f}

TRADE STATISTICS
{'='*80}
Total Trades: {metrics.get('total_trades', 0)}
Profitable Trades: {metrics.get('profitable_trades', 0)}
Losing Trades: {metrics.get('losing_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.2%}
Average Holding Period: {metrics.get('avg_hold_days', 0):.1f} days

PROFIT & LOSS
{'='*80}
Total PnL: ${metrics.get('total_pnl', 0):,.2f}
Average Win: ${metrics.get('avg_win', 0):,.2f}
Average Loss: ${metrics.get('avg_loss', 0):,.2f}
Profit Factor: {metrics.get('profit_factor', 0):.2f}x

RETURNS
{'='*80}
Total Return: {metrics.get('total_return', 0):.2%}
Annualized Return: {metrics.get('annualized_return', 0):.2%}
Benchmark Return (Equal-Weight): {benchmark_return:.2%}
Outperformance: {(metrics.get('total_return', 0) - benchmark_return):.2%}

RISK METRICS
{'='*80}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}

STRATEGY CONFIGURATION
{'='*80}
Holding Period: {self.config['holding_period']} days
Model Threshold: {self.config['model_threshold']:.0%}
Position Size: ${self.config['position_size']:,.2f}
Max Positions: {self.config['max_positions']}
Stop Loss: {self.config['stop_loss']:.0%}
Take Profit: {self.config['take_profit']:.0%}
Slippage: {self.config['slippage']:.2%}

{'='*80}
"""
        return text
    
    def plot_results(self, metrics, trade_df, portfolio_df):
        """Create visualizations"""
        logger.info("\nCreating visualizations...")
        
        # 1. Portfolio value over time
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Strategy', linewidth=2)
        ax.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DATA_PATH / f'chart_portfolio_value_{self.config["holding_period"]}d.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: chart_portfolio_value_{self.config['holding_period']}d.png")
        plt.close()
        
        # 2. Distribution of returns
        if len(trade_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(trade_df['pnl_pct'], bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Trade Returns')
            ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(DATA_PATH / f'chart_returns_distribution_{self.config["holding_period"]}d.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved: chart_returns_distribution_{self.config['holding_period']}d.png")
            plt.close()
        
        # 3. Cumulative returns
        fig, ax = plt.subplots(figsize=(14, 6))
        portfolio_df['cumulative_return'] = portfolio_df['total_return'] + 1
        ax.plot(portfolio_df['date'], portfolio_df['cumulative_return'], linewidth=2, label='Strategy')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (x)')
        ax.set_title('Cumulative Returns Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(DATA_PATH / f'chart_cumulative_returns_{self.config["holding_period"]}d.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: chart_cumulative_returns_{self.config['holding_period']}d.png")
        plt.close()
        
        # 4. Win rate by ticker
        if len(trade_df) > 0 and 'ticker' in trade_df.columns:
            ticker_stats = trade_df.groupby('ticker').apply(
                lambda x: pd.Series({
                    'trades': len(x),
                    'wins': (x['pnl'] > 0).sum(),
                    'win_rate': (x['pnl'] > 0).sum() / len(x) if len(x) > 0 else 0
                })
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ticker_stats['win_rate'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
            ax.set_xlabel('Ticker')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate by Ticker')
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(DATA_PATH / f'chart_win_rate_by_ticker_{self.config["holding_period"]}d.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved: chart_win_rate_by_ticker_{self.config['holding_period']}d.png")
            plt.close()


def main():
    """Main execution"""
    logger.info("\n" + "="*80)
    logger.info("MULTI-DAY BACKTEST ENGINE")
    logger.info("="*80)
    
    try:
        # Create backtesting engine
        engine = BacktestEngine(CONFIG)
        
        # Run backtest
        engine.run_backtest()
        
        # Save results
        metrics, benchmark_return, trade_df, portfolio_df = engine.save_results()
        
        # Print results
        print(engine._format_results(metrics, benchmark_return))
        
        # Create visualizations
        engine.plot_results(metrics, trade_df, portfolio_df)
        
        logger.info("\n" + "="*80)
        logger.info("BACKTEST COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nKey Metrics:")
        logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Benchmark: {benchmark_return:.2%}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
