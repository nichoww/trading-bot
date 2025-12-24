"""
Backtesting Module
Simulates trading strategies on historical data to evaluate performance
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Backtest:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, data, initial_capital=100000, commission=0.001):
        """
        Initialize backtest
        
        Args:
            data (pd.DataFrame): Historical price data with signals
            initial_capital (float): Starting capital
            commission (float): Commission per trade (0.1%)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.results = None
        
        logger.info(f"Backtest initialized with ${initial_capital:,.2f} capital")
    
    
    def apply_signals(self, signal_column='Signal'):
        """
        Apply trading signals from DataFrame
        
        Args:
            signal_column (str): Name of signal column (1=buy, -1=sell, 0=hold)
        """
        try:
            if signal_column not in self.data.columns:
                raise ValueError(f"Signal column '{signal_column}' not found")
            
            portfolio_values = []
            
            for idx, row in self.data.iterrows():
                signal = row[signal_column]
                price = row['Close']
                
                # Execute trades based on signals
                if signal == 1 and self.position == 0:  # Buy signal
                    shares = (self.capital * 0.95) / price  # Use 95% of capital
                    cost = shares * price * (1 + self.commission)
                    self.position = shares
                    self.capital -= cost
                    self.trades.append({
                        'date': row['date'],
                        'signal': 'BUY',
                        'price': price,
                        'shares': shares,
                        'capital_remaining': self.capital
                    })
                    logger.debug(f"BUY: {shares:.2f} shares at ${price:.2f}")
                
                elif signal == -1 and self.position > 0:  # Sell signal
                    proceeds = self.position * price * (1 - self.commission)
                    self.capital += proceeds
                    self.trades.append({
                        'date': row['date'],
                        'signal': 'SELL',
                        'price': price,
                        'shares': self.position,
                        'capital_remaining': self.capital
                    })
                    logger.debug(f"SELL: {self.position:.2f} shares at ${price:.2f}")
                    self.position = 0
                
                # Calculate current portfolio value
                portfolio_value = self.capital + (self.position * price)
                portfolio_values.append(portfolio_value)
            
            self.data['Portfolio_Value'] = portfolio_values
            logger.info(f"Applied {len(self.trades)} signals")
            
        except Exception as e:
            logger.error(f"Error applying signals: {str(e)}")
            raise
    
    
    def calculate_returns(self):
        """
        Calculate strategy returns
        
        Returns:
            dict: Return metrics
        """
        try:
            final_value = self.data['Portfolio_Value'].iloc[-1]
            total_return = (final_value - self.initial_capital) / self.initial_capital
            annualized_return = (1 + total_return) ** (252 / len(self.data)) - 1
            
            # Calculate drawdown
            cummax = self.data['Portfolio_Value'].cummax()
            drawdown = (self.data['Portfolio_Value'] - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            daily_returns = self.data['Portfolio_Value'].pct_change()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(self.trades),
                'buy_trades': sum(1 for t in self.trades if t['signal'] == 'BUY'),
                'sell_trades': sum(1 for t in self.trades if t['signal'] == 'SELL')
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    
    def get_performance_summary(self):
        """
        Get detailed performance summary
        
        Returns:
            pd.DataFrame: Performance summary
        """
        if self.results is None:
            self.calculate_returns()
        
        summary = pd.DataFrame([self.results])
        return summary
    
    
    def plot_equity_curve(self):
        """
        Plot equity curve (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data['Portfolio_Value'], linewidth=2)
            plt.title('Strategy Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            logger.info("Equity curve plotted")
            return plt
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
    
    
    def get_trade_log(self):
        """
        Get log of all trades
        
        Returns:
            pd.DataFrame: Trade log
        """
        return pd.DataFrame(self.trades)


def evaluate_strategy(data, signal_column='Signal', initial_capital=100000):
    """
    Evaluate a trading strategy
    
    Args:
        data (pd.DataFrame): Data with trading signals
        signal_column (str): Column containing signals
        initial_capital (float): Starting capital
        
    Returns:
        dict: Performance metrics
    """
    try:
        backtest = Backtest(data, initial_capital)
        backtest.apply_signals(signal_column)
        results = backtest.calculate_returns()
        
        logger.info(f"Strategy Performance: "
                   f"Return: {results['total_return_pct']:.2f}%, "
                   f"Sharpe: {results['sharpe_ratio']:.2f}, "
                   f"Max DD: {results['max_drawdown']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating strategy: {str(e)}")
        raise


if __name__ == "__main__":
    print("Backtesting Module")
    print("Use Backtest class or evaluate_strategy() to test trading signals")
