"""
Alpaca Trading Bot - Main Connection Module
Establishes connection to Alpaca API and retrieves account information
"""

from alpaca_trade_api import REST
import sys
import os

# Add src directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, validate_credentials


def initialize_trading_client():
    """
    Initializes and returns an Alpaca REST API client
    Uses environment variables for API credentials
    
    Returns:
        REST: Authenticated Alpaca REST client
        
    Raises:
        ValueError: If API credentials are missing
        Exception: If connection to Alpaca API fails
    """
    try:
        # Validate credentials before attempting connection
        validate_credentials()
        
        # Initialize the REST client with API credentials
        client = REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL
        )
        
        print("[OK] Successfully connected to Alpaca API")
        return client
        
    except ValueError as e:
        print(f"[ERROR] Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Connection Error: Failed to connect to Alpaca API")
        print(f"  Details: {str(e)}")
        raise


def get_account_info(client):
    """
    Retrieves and displays current account information
    
    Args:
        client (REST): Authenticated Alpaca REST client
        
    Returns:
        dict: Dictionary containing account information
    """
    try:
        # Get account information
        account = client.get_account()
        
        # Extract key account metrics
        account_info = {
            'account_number': account.account_number,
            'status': account.status,
            'currency': 'USD',
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'multiplier': account.multiplier,
        }
        
        return account_info
        
    except Exception as e:
        print(f"[ERROR] Error retrieving account information: {str(e)}")
        raise


def display_account_summary(account_info):
    """
    Displays a formatted summary of account information
    
    Args:
        account_info (dict): Dictionary containing account information
    """
    print("\n" + "="*60)
    print("ALPACA TRADING ACCOUNT SUMMARY")
    print("="*60)
    print(f"Account Number:              {account_info['account_number']}")
    print(f"Status:                      {account_info['status']}")
    print(f"Currency:                    {account_info['currency']}")
    print("-"*60)
    print(f"Portfolio Value:             ${account_info['portfolio_value']:,.2f}")
    print(f"Cash Available:              ${account_info['cash']:,.2f}")
    print(f"Buying Power:                ${account_info['buying_power']:,.2f}")
    print(f"Multiplier:                  {account_info['multiplier']}")
    print("="*60 + "\n")


def test_connection():
    """
    Tests the connection to Alpaca API and displays account information
    This is the main entry point for testing the bot setup
    """
    try:
        print("\nInitializing Alpaca Trading Bot...")
        print("-" * 60)
        
        # Step 1: Initialize trading client
        client = initialize_trading_client()
        
        # Step 2: Retrieve account information
        print("\nRetrieving account information...")
        account_info = get_account_info(client)
        
        # Step 3: Display account summary
        display_account_summary(account_info)
        
        print("[OK] Connection test completed successfully!")
        return True
        
    except ValueError as e:
        print(f"\n[ERROR] Validation Error: {e}")
        print("\nPlease ensure the following environment variables are set:")
        print("  - ALPACA_API_KEY")
        print("  - ALPACA_SECRET_KEY")
        print("  - APCA_API_BASE_URL (optional, defaults to paper trading)")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Connection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Main execution block
    Runs connection test when script is executed directly
    """
    success = test_connection()
    sys.exit(0 if success else 1)
