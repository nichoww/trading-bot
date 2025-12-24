"""
Real-Time Trading Engine
========================
Main trading loop that orchestrates all components:
- Data streaming
- Signal generation
- Risk management
- Order execution
- State persistence
- Audit logging
"""

import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
import traceback

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
    from .data_streamer import DataStreamer
    from .risk_manager import RiskManager
    from .order_manager import OrderManager
    from .portfolio_state import PortfolioState
    from .audit_logger import AuditLogger
    from .signal_generator import SignalGenerator, Signal
except ImportError:
    from rt_config import RTConfig
    from data_streamer import DataStreamer
    from risk_manager import RiskManager
    from order_manager import OrderManager
    from portfolio_state import PortfolioState
    from audit_logger import AuditLogger
    from signal_generator import SignalGenerator, Signal

logger = logging.getLogger(__name__)


class RealtimeEngine:
    """
    Main real-time trading engine.
    
    Coordinates all components:
    1. Fetches market data on each loop
    2. Generates trading signals
    3. Manages positions and risk
    4. Places orders with fallback logic
    5. Updates trailing stops
    6. Persists state and logs events
    """
    
    def __init__(
        self,
        config: Optional[RTConfig] = None,
        test_mode: bool = False
    ):
        """
        Initialize trading engine.
        
        Args:
            config: RTConfig instance (creates default if None)
            test_mode: If True, don't place real orders
        """
        self.config = config or RTConfig()
        self.test_mode = test_mode
        
        # Validate config
        self.config.validate()
        
        # Initialize components
        self.streamer = DataStreamer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.order_manager = OrderManager(self.config)
        self.portfolio = PortfolioState(self.config)
        self.audit = AuditLogger(self.config)
        self.signal_gen = SignalGenerator(self.config)
        
        # Engine state
        self.running = False
        self.loop_thread: Optional[threading.Thread] = None
        self.last_loop_time: Optional[datetime] = None
        self.loop_count = 0
        self.errors: List[str] = []
        
        # Position tracking
        self.pending_entries: Set[str] = set()  # Symbols with pending buy orders
        
        logger.info("[OK] RealtimeEngine initialized")
        if test_mode:
            logger.info("    Running in TEST MODE - no real orders")
    
    def startup(self):
        """
        Initialize engine and restore state.
        """
        logger.info("=" * 50)
        logger.info("REALTIME ENGINE STARTUP")
        logger.info("=" * 50)
        
        # Log startup event
        self.audit.log_event('ENGINE', 'startup', {
            'symbols': self.config.symbols,
            'test_mode': self.test_mode,
            'risk_per_trade': self.config.risk_per_trade_pct,
            'max_positions': self.config.max_positions
        })
        
        # Load saved portfolio state
        self.portfolio.load()
        logger.info(f"[OK] Loaded {len(self.portfolio.positions)} saved positions")
        
        # Sync with broker - get actual positions from Alpaca
        try:
            broker_positions = self.order_manager.api.list_positions()
            # Convert list to dict for sync
            positions_dict = {p.symbol: p for p in broker_positions}
            self.portfolio.sync_with_broker(positions_dict)
        except Exception as e:
            logger.warning(f"Could not sync broker positions: {e}")
        
        # Get account info
        account = self.order_manager.api.get_account()
        self.config.account_equity = float(account.equity)
        logger.info(f"[OK] Account equity: ${self.config.account_equity:,.2f}")
        
        # Warm up data for all symbols
        logger.info(f"[OK] Warming up data for {len(self.config.symbols)} symbols...")
        self.streamer.warmup(self.config.symbols)
        
        # Initialize risk for existing positions
        for symbol, pos in self.portfolio.positions.items():
            if pos.shares > 0:
                # Fetch current data and update ATR-based stops
                bars = self.streamer.get_historical_bars(symbol)
                if bars is not None:
                    current_price = float(bars['close'].iloc[-1])
                    atr = self.risk_manager.calculate_atr(bars)
                    # Update stop if not set or ATR-based stop is tighter
                    new_stop = self.risk_manager.calculate_stop_price(pos.entry_price, atr)
                    if pos.stop_price is None or new_stop > pos.stop_price:
                        pos.stop_price = new_stop
                    pos.atr = atr
                    pos.trail_base = max(pos.trail_base or current_price, current_price)
                    logger.info(f"    {symbol}: Stop=${pos.stop_price:.2f}, Trail base=${pos.trail_base:.2f}")
        
        # Save state
        self.portfolio.save()
        
        # Display configuration
        self.config.display()
        
        logger.info("-" * 50)
        logger.info("STARTUP COMPLETE - Ready to trade")
        logger.info("-" * 50)
    
    def run_loop(self):
        """
        Main trading loop - runs continuously.
        
        This method is typically run in a background thread.
        """
        self.running = True
        
        while self.running:
            try:
                # Check if market is open
                if not self._is_market_open():
                    logger.info("Market closed (outside regular + extended hours), waiting 60s...")
                    time.sleep(60)
                    continue
                
                # Run single iteration
                self._run_single_iteration()
                
                # Wait for next interval
                interval = self.config.loop_interval_seconds
                logger.debug(f"Sleeping {interval}s until next loop...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt - stopping engine")
                self.running = False
                break
            except Exception as e:
                error_msg = f"Loop error: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.errors.append(error_msg)
                self.audit.log_event('ENGINE', 'error', {'message': str(e)})
                time.sleep(10)  # Brief pause on error
        
        self._shutdown()
    
    def _run_single_iteration(self):
        """
        Execute one trading loop iteration.
        """
        self.loop_count += 1
        self.last_loop_time = datetime.now(timezone.utc)
        
        logger.info(f"\n--- Loop #{self.loop_count} | {self.last_loop_time.strftime('%H:%M:%S')} UTC ---")
        
        # Refresh account equity
        try:
            account = self.order_manager.api.get_account()
            self.config.account_equity = float(account.equity)
        except Exception as e:
            logger.warning(f"Could not refresh equity: {e}")
        
        # Process each symbol
        for symbol in self.config.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.audit.log_event('ENGINE', 'symbol_error', {
                    'symbol': symbol, 
                    'error': str(e)
                })
        
        # Update trailing stops for all positions
        self._update_all_trailing_stops()
        
        # Check for exit conditions
        self._check_exit_conditions()
        
        # Save state periodically
        self.portfolio.save()
        
        # Log loop summary
        pos_count = len([p for p in self.portfolio.positions.values() if p.shares > 0])
        logger.info(f"    Positions: {pos_count}/{self.config.max_positions} | Equity: ${self.config.account_equity:,.0f}")
    
    def _process_symbol(self, symbol: str):
        """
        Process a single symbol - generate signal and act.
        
        Args:
            symbol: Stock ticker
        """
        # Get current bars
        bars = self.streamer.get_historical_bars(symbol)
        if bars is None or len(bars) < self.config.atr_period:
            logger.debug(f"{symbol}: Insufficient data")
            return
        
        current_price = bars['close'].iloc[-1]
        
        # Check if we have a position
        has_position = (
            symbol in self.portfolio.positions and
            self.portfolio.positions[symbol].shares > 0
        )
        
        # Check if we have pending order
        has_pending = symbol in self.pending_entries
        
        # Generate signal
        signal, confidence = self.signal_gen.generate_signal(bars, symbol)
        
        logger.debug(f"{symbol}: ${current_price:.2f} | Signal: {signal.value} ({confidence:.0%})")
        
        # Act on signal
        if signal == Signal.BUY:
            if not has_position and not has_pending:
                self._try_entry(symbol, current_price, bars, confidence)
        
        elif signal == Signal.SELL:
            if has_position:
                self._try_exit(symbol, current_price, reason='Signal SELL')
    
    def _try_entry(
        self, 
        symbol: str, 
        price: float, 
        bars, 
        confidence: float
    ):
        """
        Attempt to enter a position.
        
        Args:
            symbol: Stock ticker
            price: Current price
            bars: Historical bars for ATR
            confidence: Signal confidence
        """
        # Check position limits
        open_positions = len([p for p in self.portfolio.positions.values() if p.shares > 0])
        if open_positions >= self.config.max_positions:
            logger.debug(f"{symbol}: Max positions reached ({open_positions})")
            return
        
        # Check confidence threshold
        min_confidence = 0.5  # Could be configurable
        if confidence < min_confidence:
            logger.debug(f"{symbol}: Confidence too low ({confidence:.0%})")
            return
        
        # Calculate ATR for position sizing
        atr = self.risk_manager.calculate_atr(bars)
        
        # Calculate position size
        quantity, position_value, stop_loss = self.risk_manager.calculate_position_size(
            portfolio_value=self.config.account_equity,
            entry_price=price,
            atr=atr
        )
        
        if quantity <= 0:
            logger.debug(f"{symbol}: Calculated quantity is 0")
            return
        
        # Submit order
        logger.info(f"[BUY] {symbol}: {quantity} shares @ ~${price:.2f} (stop: ${stop_loss:.2f})")
        
        if self.test_mode:
            logger.info(f"    TEST MODE - order not submitted")
            return
        
        # Mark as pending
        self.pending_entries.add(symbol)
        
        try:
            from .order_manager import OrderSide
            order = self.order_manager.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                qty=quantity,
                limit_price=price * 1.001  # Slight buffer for fills
            )
            
            if order and order.success:
                # Log the order
                self.audit.log_order(order)
                
                # Add to portfolio
                self.portfolio.add_position(
                    symbol=symbol,
                    entry_price=price,
                    shares=quantity,
                    stop_price=stop_loss,
                    take_profit_price=price * (1 + self.config.take_profit_pct),
                    atr=atr
                )
                logger.info(f"[OK] {symbol} position opened")
            
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            self.pending_entries.discard(symbol)
    
    def _try_exit(self, symbol: str, price: float, reason: str = 'Signal'):
        """
        Attempt to exit a position.
        
        Args:
            symbol: Stock ticker
            price: Current price
            reason: Reason for exit
        """
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        if pos.shares <= 0:
            return
        
        pnl = (price - pos.entry_price) * pos.shares
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        
        logger.info(f"[SELL] {symbol}: {pos.shares} shares @ ~${price:.2f} ({reason})")
        logger.info(f"       P&L: ${pnl:+,.2f} ({pnl_pct:+.1%})")
        
        if self.test_mode:
            logger.info(f"    TEST MODE - order not submitted")
            return
        
        try:
            from .order_manager import OrderSide
            # Use limit order slightly below current price for extended hours compatibility
            limit_price = price * 0.995  # 0.5% below to ensure fill
            order = self.order_manager.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=pos.shares,
                limit_price=limit_price
            )
            
            if order and order.success:
                # Log the order result
                self.audit.log_order(order)
                
                # Log position close for trade summary
                self.audit.log_position_close(
                    symbol=symbol,
                    shares=pos.shares,
                    exit_price=price,
                    entry_price=pos.entry_price,
                    reason=reason,
                    order_id=order.order_id or ''
                )
                
                # Remove from portfolio
                self.portfolio.remove_position(symbol)
                logger.info(f"[OK] {symbol} position closed")
                
        except Exception as e:
            logger.error(f"Exit order failed for {symbol}: {e}")
    
    def _update_all_trailing_stops(self):
        """
        Update trailing stops for all open positions.
        """
        for symbol, pos in self.portfolio.positions.items():
            if pos.shares <= 0:
                continue
            
            bars = self.streamer.get_historical_bars(symbol)
            if bars is None:
                continue
            
            current_price = bars['close'].iloc[-1]
            
            # Update trailing stop (stored in stop_price)
            new_stop, new_trail_base = self.risk_manager.update_trailing_stop(
                current_price=current_price,
                current_stop=pos.stop_price,
                trail_base=pos.trail_base
            )
            
            if new_stop > pos.stop_price:
                old_stop = pos.stop_price
                pos.stop_price = new_stop
                pos.trail_base = new_trail_base
                logger.debug(f"{symbol}: Trail stop raised ${old_stop:.2f} -> ${new_stop:.2f}")
    
    def _check_exit_conditions(self):
        """
        Check all positions for stop-loss or trailing stop triggers.
        """
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.shares <= 0:
                continue
            
            bars = self.streamer.get_historical_bars(symbol)
            if bars is None:
                continue
            
            current_price = bars['close'].iloc[-1]
            
            # Check exits (stop_price and take_profit_price)
            should_exit, reason = self.risk_manager.check_exit_conditions(
                current_price=current_price,
                stop_price=pos.stop_price,
                take_profit_price=pos.take_profit_price
            )
            
            if should_exit:
                self._try_exit(symbol, current_price, reason=reason)
    
    def _is_market_open(self) -> bool:
        """
        Check if US stock market is currently open (including extended hours if enabled).
        
        Returns:
            True if market is open or in extended hours (when enabled)
        """
        try:
            clock = self.order_manager.api.get_clock()
            
            # If regular market is open, always trade
            if clock.is_open:
                return True
            
            # If extended hours enabled, check if we're in extended hours
            if self.config.extended_hours:
                now = datetime.now(timezone.utc)
                hour = now.hour
                minute = now.minute
                weekday = now.weekday()
                
                # Skip weekends
                if weekday >= 5:
                    return False
                
                # Extended hours (in UTC):
                # Pre-market: 9:00 - 14:30 UTC (4:00 AM - 9:30 AM EST)
                # After-hours: 21:00 - 01:00 UTC (4:00 PM - 8:00 PM EST)
                in_premarket = (hour >= 9 and hour < 14) or (hour == 14 and minute < 30)
                in_afterhours = hour >= 21 or hour < 1
                
                if in_premarket or in_afterhours:
                    logger.debug(f"Extended hours active (UTC hour: {hour})")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not check market hours: {e}")
            # Default to checking time
            now = datetime.now(timezone.utc)
            # Rough EST hours: 14:30 - 21:00 UTC
            hour = now.hour
            return 14 <= hour <= 21
    
    def _shutdown(self):
        """
        Clean shutdown - save state and log.
        """
        logger.info("\n" + "=" * 50)
        logger.info("ENGINE SHUTDOWN")
        logger.info("=" * 50)
        
        # Save final state
        self.portfolio.save()
        
        # Log shutdown
        self.audit.log_event('ENGINE', 'shutdown', {
            'loop_count': self.loop_count,
            'errors': len(self.errors)
        })
        
        # Summary
        summary = self.audit.get_trade_summary()
        if summary:
            logger.info(f"Session summary:")
            logger.info(f"  Total trades: {summary.get('total_trades', 0)}")
            logger.info(f"  Total P&L: ${summary.get('total_pnl', 0):+,.2f}")
        
        logger.info("Shutdown complete")
    
    def start_background(self):
        """
        Start engine in background thread.
        """
        if self.running:
            logger.warning("Engine already running")
            return
        
        self.startup()
        
        self.loop_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.loop_thread.start()
        
        logger.info("Engine started in background thread")
    
    def stop(self):
        """
        Stop the engine gracefully.
        """
        logger.info("Stop requested...")
        self.running = False
        
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=10)
    
    def run_once(self):
        """
        Run a single iteration (for testing).
        """
        self.startup()
        self._run_single_iteration()
        self._shutdown()
    
    def get_status(self) -> Dict:
        """
        Get current engine status.
        
        Returns:
            Status dictionary
        """
        return {
            'running': self.running,
            'test_mode': self.test_mode,
            'loop_count': self.loop_count,
            'last_loop': self.last_loop_time.isoformat() if self.last_loop_time else None,
            'equity': self.config.account_equity,
            'positions': {
                sym: {
                    'quantity': pos.shares,
                    'entry': pos.entry_price,
                    'stop': pos.stop_price,
                    'trail': pos.trail_base
                }
                for sym, pos in self.portfolio.positions.items()
                if pos.shares > 0
            },
            'errors': len(self.errors),
            'last_errors': self.errors[-3:] if self.errors else []
        }


if __name__ == "__main__":
    # Test engine
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("\n" + "=" * 60)
    print("REALTIME ENGINE TEST")
    print("=" * 60)
    
    # Create config with limited symbols for testing
    config = RTConfig()
    config.symbols = ['AAPL', 'NVDA']  # Just test with 2 symbols
    
    # Create engine in test mode
    engine = RealtimeEngine(config, test_mode=True)
    
    # Run one iteration
    engine.run_once()
    
    print("\n[OK] Engine test complete!")
