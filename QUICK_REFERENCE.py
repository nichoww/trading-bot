"""
QUICK REFERENCE - Real-Time Trading Bot
========================================
Run this file to see all available commands.

Project Location: c:\\Users\\Administrator\\Desktop\\Projects\\Trading Bot
"""

COMMANDS = """
================================================================================
                        REAL-TIME TRADING BOT
                           Quick Reference
================================================================================

ACTIVATE ENVIRONMENT
--------------------
cd "c:\\Users\\Administrator\\Desktop\\Projects\\Trading Bot"
.\\.venv\\Scripts\\Activate.ps1


RUN TRADING BOT
---------------
# Test mode - single iteration (safe, no orders)
python run_realtime.py --once

# Test mode - continuous monitoring
python run_realtime.py

# Test specific symbols
python run_realtime.py --symbols AAPL NVDA TSLA

# With dashboard
python run_realtime.py --dashboard

# LIVE MODE (places real paper trading orders!)
python run_realtime.py --live


COMMAND LINE OPTIONS
--------------------
--live              Enable live trading (real orders)
--once              Run single iteration then exit
--dashboard         Show monitoring dashboard
--symbols SYM ...   Override symbols to trade
--interval SEC      Loop interval in seconds (default: 60)
--risk PCT          Risk per trade % (default: 2.0)
--verbose, -v       Enable debug logging


RUN INTEGRATION TESTS
---------------------
python src/realtime/test_phase1.py    # Config, Data, Risk
python src/realtime/test_phase2.py    # Order Management
python src/realtime/test_phase3.py    # State Persistence
python src/realtime/test_phase4.py    # Full Engine


LEGACY ML COMMANDS
------------------
# Train new ML models
python train_models.py

# Backtest strategy
python backtest_strategy.py


KEY FILES
---------
run_realtime.py              Main entry point
src/realtime/rt_config.py    Configuration settings
src/realtime/README.md       Detailed documentation
logs/realtime/               Trading logs and audit trail
data/realtime/               Saved state and bar data


CONFIGURATION
-------------
Edit src/realtime/rt_config.py to change:
  - symbols: List of tickers to trade
  - risk_per_trade_pct: Risk % per trade (default 0.02 = 2%)
  - max_positions: Max concurrent positions (default 5)
  - atr_stop_multiplier: Stop distance in ATR units (default 2.0)
  - trail_stop_pct: Trailing stop % (default 0.03 = 3%)
  - signal_type: 'MA_CROSSOVER' or 'ML_MODEL'


================================================================================
"""

if __name__ == "__main__":
    print(COMMANDS)
