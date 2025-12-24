# Real-Time Trading System

A sophisticated real-time paper trading bot built on top of the ML trading framework.

## Features

- **Real-Time Data**: Minute-by-minute bars from Alpaca API
- **ATR-Based Risk Management**: Position sizing and stop-loss based on Average True Range
- **Trailing Stops**: Dynamic stops that ratchet up as price rises (never down)
- **Smart Order Flow**: Limit orders with timeout and automatic market order fallback
- **Duplicate Prevention**: Guards against duplicate orders for the same symbol
- **State Persistence**: Portfolio state saved/restored across restarts
- **Audit Trail**: Complete CSV logging of all trades and events
- **Parquet Storage**: Efficient bar data storage for analysis

## Quick Start

### 1. Test Mode (Recommended First)

Run in test mode - no real orders placed:

```bash
python run_realtime.py --once
```

This runs a single iteration and exits.

### 2. Continuous Test Mode

Run continuously in test mode:

```bash
python run_realtime.py
```

Press `Ctrl+C` to stop.

### 3. With Dashboard

```bash
python run_realtime.py --dashboard
```

### 4. Live Mode (Real Orders!)

⚠️ **WARNING**: This places real orders on your paper account!

```bash
python run_realtime.py --live
```

You will be prompted to confirm.

## Command Line Options

```
Usage: run_realtime.py [OPTIONS]

Options:
  --live              Enable LIVE trading (real orders!)
  --once              Run single iteration then exit
  --dashboard         Show simple dashboard while running
  --symbols AAPL NVDA Override symbols to trade
  --interval 60       Loop interval in seconds (default: 60)
  --risk 2.0          Risk per trade % (default: 2.0)
  --verbose, -v       Enable debug logging
```

## Architecture

```
src/realtime/
├── rt_config.py         # Configuration dataclass
├── data_streamer.py     # Alpaca data fetching
├── risk_manager.py      # ATR, position sizing, stops
├── order_manager.py     # Order placement & fallback
├── portfolio_state.py   # State persistence
├── audit_logger.py      # CSV/Parquet logging
├── signal_generator.py  # MA crossover signals
├── realtime_engine.py   # Main trading loop
├── dashboard.py         # Console monitoring
└── test_phase*.py       # Integration tests
```

## Configuration

Edit `src/realtime/rt_config.py` to modify:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbols` | 20 stocks | Tickers to trade |
| `bar_interval` | `1Min` | Bar timeframe |
| `loop_interval_seconds` | 60 | Main loop frequency |
| `risk_per_trade_pct` | 0.02 | 2% risk per trade |
| `max_position_pct` | 0.10 | 10% max single position |
| `max_positions` | 5 | Max concurrent positions |
| `atr_period` | 14 | ATR calculation period |
| `atr_stop_multiplier` | 2.0 | Stop distance = 2x ATR |
| `trail_stop_pct` | 0.03 | 3% trailing stop |
| `use_limit_orders` | True | Start with limit orders |
| `limit_order_timeout` | 30 | Seconds before fallback |

## Trading Logic

### Entry Signals

Uses MA Crossover strategy:
- **BUY**: Fast MA (10) crosses above Slow MA (30)
- **SELL**: Fast MA crosses below Slow MA

Alternatively, can use ML model predictions (set `signal_type='ML_MODEL'`).

### Position Sizing

Position size is calculated to risk a fixed percentage of equity:

```
Position Size = (Account Equity × Risk %) / (Entry Price × ATR Multiplier × ATR)
```

### Exit Conditions

Positions are closed when:
1. **Signal SELL**: Opposite signal generated
2. **Stop-Loss Hit**: Price drops below fixed stop (Entry - ATR × Multiplier)
3. **Trailing Stop Hit**: Price drops below trailing stop

### Trailing Stop Logic

The trailing stop "ratchets up" as price rises:
- Initial: Entry Price × (1 - Trail %)
- Updated: Current High × (1 - Trail %)
- **Never moves down**, only up

## Data Files

### Logs
- `logs/realtime/audit_trail.csv` - All events and orders
- `logs/realtime/trading_*.log` - Session logs

### State
- `data/realtime/portfolio_state.json` - Saved positions
- `data/realtime/bars/*.parquet` - Historical bars

## Testing

Run all integration tests:

```bash
python src/realtime/test_phase1.py
python src/realtime/test_phase2.py
python src/realtime/test_phase3.py
python src/realtime/test_phase4.py
```

## Market Hours

The engine checks Alpaca's clock API to determine if the market is open:
- US Stock Market: 9:30 AM - 4:00 PM Eastern
- Extended hours trading is enabled by default

During market closed hours, the loop sleeps and waits.

## Safety Features

1. **Test Mode by Default**: No real orders without `--live` flag
2. **Confirmation Required**: Must type "YES" for live mode
3. **Duplicate Guard**: Prevents duplicate orders per symbol
4. **Position Limits**: Configurable max positions
5. **Risk Limits**: Position sizing capped by risk % and max position %
6. **State Persistence**: Positions saved after each loop
7. **Graceful Shutdown**: Ctrl+C saves state before exit

## Troubleshooting

### "Market closed, waiting..."
The US stock market is closed. Run during market hours or during extended hours (pre-market 4am-9:30am, after-hours 4pm-8pm ET).

### "Insufficient data"
Not enough historical bars to calculate indicators. The engine needs ~50 bars minimum.

### "Max positions reached"
You've hit the `max_positions` limit. Close existing positions or increase the limit.

### Order Not Filling
Limit orders may not fill if price moves. The system will automatically fall back to market orders after the timeout.

## Future Improvements

- [ ] WebSocket streaming for real-time prices
- [ ] Web-based dashboard
- [ ] SMS/Email alerts
- [ ] Multi-strategy support
- [ ] Backtesting integration
- [ ] Options support
