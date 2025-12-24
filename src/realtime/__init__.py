"""
Real-Time Trading System
========================
A sophisticated real-time paper trading bot with:
- Streaming price data from Alpaca
- ATR-based position sizing and stops
- Trailing stop logic
- Limit order flow with market fallback
- Duplicate order prevention
- Portfolio state persistence
"""

from .rt_config import RTConfig
from .data_streamer import DataStreamer
from .risk_manager import RiskManager, PositionRisk
from .order_manager import OrderManager, OrderResult
from .portfolio_state import PortfolioState, PositionState
from .audit_logger import AuditLogger, EventType
from .signal_generator import SignalGenerator, Signal
from .realtime_engine import RealtimeEngine
from .dashboard import TradingDashboard, SimpleDashboard

__version__ = "1.0.0"
__all__ = [
    "RTConfig",
    "DataStreamer",
    "RiskManager",
    "PositionRisk",
    "OrderManager",
    "OrderResult",
    "PortfolioState",
    "PositionState",
    "AuditLogger",
    "EventType",
    "SignalGenerator",
    "Signal",
    "RealtimeEngine",
    "TradingDashboard",
    "SimpleDashboard",
]
