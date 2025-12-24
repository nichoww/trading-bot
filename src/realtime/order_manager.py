"""
Order Manager
=============
Handles all order placement, monitoring, and execution:
- Limit orders with timeout
- Fallback to market orders
- Order status tracking
- Duplicate order prevention (symbol+side guard)
- Client order ID for idempotency
"""

import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Tuple
from enum import Enum
import logging
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

# Handle both package and standalone imports
try:
    from .rt_config import RTConfig
except ImportError:
    from rt_config import RTConfig

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderResult:
    """Result of an order operation"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    qty: float = 0
    filled_qty: float = 0
    filled_avg_price: Optional[float] = None
    status: str = ""
    error: Optional[str] = None
    is_fallback: bool = False  # True if this was a fallback market order


class OrderManager:
    """
    Manages order placement and execution.
    
    Features:
    - Submit limit orders with configurable timeout
    - Automatic fallback to market orders if limit doesn't fill
    - Duplicate order prevention (one active order per symbol+side)
    - Client order IDs for broker-side idempotency
    - Order status polling and cancellation
    """
    
    def __init__(self, config: RTConfig, client: REST = None):
        """
        Initialize OrderManager.
        
        Args:
            config: RTConfig instance
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
        
        # Alias for external access
        self.api = self.client
        
        # Duplicate order guard: Set of "symbol:side" keys with active orders
        self._active_orders: Dict[str, str] = {}  # "AAPL:buy" -> order_id
        
        # Sync with broker on init
        self._sync_open_orders()
        
        logger.info("[OK] OrderManager initialized")
    
    def _generate_client_order_id(self, symbol: str, side: str) -> str:
        """Generate unique client order ID for idempotency"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"RT_{symbol}_{side}_{timestamp}_{unique}"
    
    def _get_order_key(self, symbol: str, side: str) -> str:
        """Generate key for duplicate order tracking"""
        return f"{symbol}:{side}"
    
    def _sync_open_orders(self):
        """Sync open orders from broker to prevent duplicates on restart"""
        try:
            open_orders = self.client.list_orders(status='open')
            
            self._active_orders.clear()
            for order in open_orders:
                key = self._get_order_key(order.symbol, order.side)
                self._active_orders[key] = order.id
            
            if self._active_orders:
                logger.info(f"Synced {len(self._active_orders)} open orders from broker")
                for key, order_id in self._active_orders.items():
                    logger.debug(f"  {key}: {order_id}")
        except Exception as e:
            logger.warning(f"Could not sync open orders: {e}")
    
    def has_active_order(self, symbol: str, side: str) -> bool:
        """Check if there's already an active order for symbol+side"""
        key = self._get_order_key(symbol, side)
        return key in self._active_orders
    
    def _wait_for_fill(
        self, 
        order_id: str, 
        timeout: int = None
    ) -> Tuple[bool, dict]:
        """
        Wait for an order to fill.
        
        Args:
            order_id: Alpaca order ID
            timeout: Seconds to wait (default: from config)
            
        Returns:
            Tuple of (is_filled, order_dict)
        """
        if timeout is None:
            timeout = self.config.limit_order_timeout
        
        check_interval = self.config.cancel_check_interval
        elapsed = 0
        
        while elapsed < timeout:
            try:
                order = self.client.get_order(order_id)
                status = order.status
                
                if status == 'filled':
                    return True, order._raw
                elif status in ['cancelled', 'rejected', 'expired']:
                    return False, order._raw
                
                time.sleep(check_interval)
                elapsed += check_interval
                
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
                time.sleep(check_interval)
                elapsed += check_interval
        
        # Timeout reached, get final status
        try:
            order = self.client.get_order(order_id)
            return order.status == 'filled', order._raw
        except:
            return False, {}
    
    def _cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order and wait for terminal state.
        
        Args:
            order_id: Alpaca order ID
            
        Returns:
            True if successfully cancelled
        """
        try:
            self.client.cancel_order(order_id)
            logger.info(f"Cancel requested for order {order_id}")
            
            # Wait for cancellation to complete
            for _ in range(5):
                time.sleep(1)
                order = self.client.get_order(order_id)
                if order.status in ['cancelled', 'filled', 'rejected', 'expired']:
                    return True
            
            return True
        except APIError as e:
            if 'already been cancelled' in str(e) or 'already filled' in str(e):
                return True
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        limit_price: float = None
    ) -> OrderResult:
        """
        Submit an order with limit-first strategy.
        
        Flow:
        1. Check for duplicate orders
        2. Submit limit order
        3. Wait for fill (configurable timeout)
        4. If not filled, cancel and submit market fallback
        
        Args:
            symbol: Stock ticker
            side: BUY or SELL
            qty: Number of shares
            limit_price: Limit price (if None, uses market order)
            
        Returns:
            OrderResult with fill details
        """
        side_str = side.value
        key = self._get_order_key(symbol, side_str)
        
        # Check duplicate guard
        if key in self._active_orders:
            existing_id = self._active_orders[key]
            logger.warning(f"Duplicate order blocked: {symbol} {side_str} (existing: {existing_id})")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side_str,
                qty=qty,
                error=f"Active order exists: {existing_id}"
            )
        
        client_order_id = self._generate_client_order_id(symbol, side_str)
        
        # Determine order type
        use_limit = self.config.use_limit_orders and limit_price is not None
        
        try:
            if use_limit:
                # Submit limit order
                logger.info(f"Submitting LIMIT {side_str.upper()} {qty} {symbol} @ ${limit_price:.2f}")
                
                order = self.client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side_str,
                    type='limit',
                    time_in_force='day',
                    limit_price=round(limit_price, 2),
                    client_order_id=client_order_id,
                    extended_hours=self.config.extended_hours
                )
                
                order_id = order.id
                self._active_orders[key] = order_id
                
                # Wait for fill
                is_filled, order_data = self._wait_for_fill(order_id)
                
                if is_filled:
                    # Remove from active orders
                    self._active_orders.pop(key, None)
                    
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        client_order_id=client_order_id,
                        symbol=symbol,
                        side=side_str,
                        qty=qty,
                        filled_qty=float(order_data.get('filled_qty', qty)),
                        filled_avg_price=float(order_data.get('filled_avg_price', limit_price)),
                        status='filled',
                        is_fallback=False
                    )
                
                # Limit order didn't fill - cancel and try market
                logger.info(f"Limit order timeout, cancelling {order_id}")
                self._cancel_order(order_id)
                self._active_orders.pop(key, None)
                
                # Check if partially filled
                try:
                    order = self.client.get_order(order_id)
                    filled_qty = float(order.filled_qty) if order.filled_qty else 0
                    remaining_qty = qty - int(filled_qty)
                    
                    if remaining_qty <= 0:
                        # Actually filled during cancel
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            client_order_id=client_order_id,
                            symbol=symbol,
                            side=side_str,
                            qty=qty,
                            filled_qty=filled_qty,
                            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else limit_price,
                            status='filled',
                            is_fallback=False
                        )
                except:
                    remaining_qty = qty
                
                # Fallback to market order for remaining quantity
                return self._submit_market_fallback(symbol, side, remaining_qty, client_order_id + "_MKT")
            
            else:
                # Direct market order
                return self._submit_market_fallback(symbol, side, qty, client_order_id)
                
        except APIError as e:
            self._active_orders.pop(key, None)
            logger.error(f"API Error submitting order: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side_str,
                qty=qty,
                error=str(e)
            )
        except Exception as e:
            self._active_orders.pop(key, None)
            logger.error(f"Error submitting order: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side_str,
                qty=qty,
                error=str(e)
            )
    
    def _submit_market_fallback(
        self, 
        symbol: str, 
        side: OrderSide, 
        qty: int,
        client_order_id: str
    ) -> OrderResult:
        """
        Submit a market order as fallback.
        
        Args:
            symbol: Stock ticker
            side: BUY or SELL
            qty: Number of shares
            client_order_id: Client order ID
            
        Returns:
            OrderResult
        """
        side_str = side.value
        key = self._get_order_key(symbol, side_str)
        
        if qty <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side_str,
                qty=0,
                error="No quantity to fill"
            )
        
        try:
            logger.info(f"Submitting MARKET {side_str.upper()} {qty} {symbol}")
            
            order = self.client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side_str,
                type='market',
                time_in_force='day',
                client_order_id=client_order_id,
                extended_hours=self.config.extended_hours
            )
            
            order_id = order.id
            self._active_orders[key] = order_id
            
            # Market orders should fill quickly
            is_filled, order_data = self._wait_for_fill(order_id, timeout=10)
            
            self._active_orders.pop(key, None)
            
            if is_filled:
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side_str,
                    qty=qty,
                    filled_qty=float(order_data.get('filled_qty', qty)),
                    filled_avg_price=float(order_data.get('filled_avg_price', 0)),
                    status='filled',
                    is_fallback=True
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side_str,
                    qty=qty,
                    status=order_data.get('status', 'unknown'),
                    error="Market order did not fill",
                    is_fallback=True
                )
                
        except Exception as e:
            self._active_orders.pop(key, None)
            logger.error(f"Error submitting market fallback: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side_str,
                qty=qty,
                error=str(e),
                is_fallback=True
            )
    
    def get_positions(self) -> Dict[str, dict]:
        """
        Get all current positions from broker.
        
        Returns:
            Dict mapping symbol to position info
        """
        try:
            positions = self.client.list_positions()
            result = {}
            
            for pos in positions:
                result[pos.symbol] = {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
            
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account_info(self) -> dict:
        """Get account information from broker"""
        try:
            account = self.client.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'day_trade_count': int(account.daytrade_count) if account.daytrade_count else 0,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def close_position(self, symbol: str) -> OrderResult:
        """
        Close an entire position.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            OrderResult
        """
        positions = self.get_positions()
        
        if symbol not in positions:
            return OrderResult(
                success=False,
                symbol=symbol,
                side='sell',
                error=f"No position in {symbol}"
            )
        
        pos = positions[symbol]
        qty = int(abs(pos['qty']))
        side = OrderSide.SELL if pos['qty'] > 0 else OrderSide.BUY
        
        # Get current price for limit
        current_price = pos['current_price']
        
        # For sells, use slightly below current price to ensure fill
        if side == OrderSide.SELL:
            limit_price = current_price * (1 - self.config.limit_offset_pct)
        else:
            limit_price = current_price * (1 + self.config.limit_offset_pct)
        
        return self.submit_order(symbol, side, qty, limit_price)
    
    def display_status(self):
        """Display order manager status"""
        print("\n" + "="*60)
        print("ORDER MANAGER STATUS")
        print("="*60)
        
        # Account info
        account = self.get_account_info()
        if account:
            print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"Buying Power:    ${account['buying_power']:,.2f}")
            print(f"Cash:            ${account['cash']:,.2f}")
        
        # Active orders
        print(f"\nActive Order Guards: {len(self._active_orders)}")
        for key, order_id in self._active_orders.items():
            print(f"  {key}: {order_id}")
        
        # Positions
        positions = self.get_positions()
        print(f"\nOpen Positions: {len(positions)}")
        if positions:
            print(f"{'Symbol':<8} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L':<12}")
            print("-"*50)
            for sym, pos in positions.items():
                print(f"{sym:<8} {pos['qty']:<8.0f} ${pos['entry_price']:<9.2f} "
                      f"${pos['current_price']:<9.2f} ${pos['unrealized_pl']:<11.2f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test the order manager (READ-ONLY - no actual orders in this test)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    logging.basicConfig(level=logging.INFO)
    
    config = RTConfig()
    if not config.validate():
        print("[ERROR] Invalid config")
        exit(1)
    
    om = OrderManager(config)
    
    print("\n--- Testing Order Manager (Read-Only) ---")
    
    # Display status
    om.display_status()
    
    # Test duplicate guard
    print("\n--- Testing Duplicate Guard ---")
    print(f"Has active AAPL buy? {om.has_active_order('AAPL', 'buy')}")
    print(f"Has active AAPL sell? {om.has_active_order('AAPL', 'sell')}")
    
    # Test client order ID generation
    print("\n--- Testing Client Order ID ---")
    for _ in range(3):
        coid = om._generate_client_order_id('AAPL', 'buy')
        print(f"  {coid}")
    
    print("\n[OK] OrderManager tests passed (read-only)!")
    print("\nTo test actual order placement, use test_phase2.py")
