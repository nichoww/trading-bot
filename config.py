"""
Configuration module for Alpaca Trading Bot
Handles API credentials and environment variables securely
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Paper trading base URL (set APCA_API_BASE_URL=https://paper-api.alpaca.markets in .env)
ALPACA_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

# Data source for market data (IEX, sip, or crypto_us)
DATA_SOURCE = os.getenv('DATA_SOURCE', 'iex')

# Validate that required environment variables are set
def validate_credentials():
    """
    Validates that all required API credentials are available
    Raises ValueError if any required credentials are missing
    """
    if not ALPACA_API_KEY:
        raise ValueError("ALPACA_API_KEY environment variable is not set")
    if not ALPACA_SECRET_KEY:
        raise ValueError("ALPACA_SECRET_KEY environment variable is not set")
    
    print("✓ API credentials validated successfully")


if __name__ == "__main__":
    try:
        validate_credentials()
        print(f"Base URL: {ALPACA_BASE_URL}")
        print(f"Data Source: {DATA_SOURCE}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
