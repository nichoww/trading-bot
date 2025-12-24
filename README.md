# Real-Time Trading Bot

A sophisticated real-time paper trading system using the Alpaca API with ML-based and technical analysis signals.

## Project Evolution

This project started as a 5-day holding period ML trading bot but was **pivoted to a real-time trading system** with:
- Minute-by-minute price monitoring
- ATR-based dynamic position sizing
- Trailing stops that ratchet up (never down)
- Limit orders with automatic market fallback
- State persistence across restarts

The original ML infrastructure is preserved and can be used for signal generation.

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your Alpaca credentials:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Run Trading Bot

```bash
# Test mode (no real orders) - single iteration
python run_realtime.py --once

# Test mode - continuous
python run_realtime.py

# Live mode (real paper trading orders!)
python run_realtime.py --live

# Live mode - skip confirmation prompt
python run_realtime.py --live --no-confirm

# Demo trade (buy/sell 5 shares of Ford to test)
python run_realtime.py --live --no-confirm --demo-trade --once

# Specific symbols
python run_realtime.py --symbols AAPL NVDA TSLA
```

---

## System Architecture

```
Trading Bot/
|-- run_realtime.py          # Main entry point
|-- src/
|   |-- realtime/            # Real-time trading system
|   |   |-- rt_config.py         # Configuration
|   |   |-- data_streamer.py     # Alpaca data fetching
|   |   |-- risk_manager.py      # ATR, position sizing, stops
|   |   |-- order_manager.py     # Order execution
|   |   |-- portfolio_state.py   # State persistence
|   |   |-- audit_logger.py      # Trade logging
|   |   |-- signal_generator.py  # Trading signals
|   |   |-- realtime_engine.py   # Main loop
|   |   |-- dashboard.py         # Monitoring
|   |
|   |-- model_training.py    # ML model training (legacy)
|   |-- backtesting.py       # Strategy backtesting
|   |-- feature_engineering.py
|   |-- data_collection.py
|
|-- models/                  # Trained ML models
|-- data/                    # Historical data
|-- logs/                    # Trading logs
|-- notebooks/               # Analysis notebooks
```

---

## Features

### Real-Time Trading Engine
- **60-second loop** monitoring all symbols
- **Alpaca API** integration for data and orders
- **MA Crossover** signals (10/30 period)
- Optional **ML Model** signals

### Risk Management
- **ATR-based stops**: Stop distance = ATR x 2.0
- **Position sizing**: Risk 2% of equity per trade
- **Trailing stops**: Ratchet up as price rises, never down
- **Max positions**: 5 concurrent positions

### Order Execution
- **Limit orders first**: Better fills
- **30s timeout**: Then fallback to market
- **Duplicate guard**: One order per symbol
- **Client order IDs**: Idempotency

### State & Logging
- **Portfolio state**: Saved to JSON after each loop
- **Audit trail**: CSV log of all events
- **Parquet storage**: Efficient bar data storage

---

## Configuration

Edit `src/realtime/rt_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbols` | 20 stocks | Tickers to trade |
| `bar_interval` | 1Min | Data timeframe |
| `loop_interval_seconds` | 60 | Loop frequency |
| `risk_per_trade_pct` | 0.02 | 2% risk per trade |
| `max_positions` | 15 | Max open positions |
| `atr_period` | 14 | ATR lookback |
| `atr_stop_multiplier` | 2.0 | Stop = 2x ATR |
| `trail_stop_pct` | 0.03 | 3% trailing stop |
| `use_limit_orders` | True | Limit before market |
| `limit_order_timeout` | 30 | Seconds to wait |

---

## Command Line Options

```
python run_realtime.py [OPTIONS]

Options:
  --live              Enable LIVE trading (real orders!)
  --no-confirm        Skip the confirmation prompt for live mode
  --demo-trade        Execute a demo trade (buy/sell Ford) on startup
  --once              Run single iteration then exit
  --dashboard         Show monitoring dashboard
  --symbols SYM ...   Override symbols to trade
  --interval SEC      Loop interval (default: 60)
  --risk PCT          Risk per trade % (default: 2.0)
  --verbose, -v       Debug logging
```

---

## Legacy Components

The following components from the original 5-day ML system are preserved:

- `train_models.py` - Train ML models on historical data
- `backtest_strategy.py` - Backtest trading strategies
- `models/` - Pre-trained Random Forest, Gradient Boosting models
- `data/` - Historical price data and features
- `notebooks/` - Jupyter analysis notebooks

To use ML signals instead of MA crossover, set `signal_type='ML_MODEL'` in config.

---

## Files Removed in Pivot

The following files were removed as redundant:
- `run_live_trader.py` - Old 5-day entry point
- `demo.py`, `silent_demo.py` - Old demos
- `test_live_trading.py` - Old tests
- `src/live_trading.py` - Old trading logic
- `src/live_trading_5d.py` - Old 5-day system
- `src/live_trading_dashboard.py` - Old dashboard

---

## License

MIT License
