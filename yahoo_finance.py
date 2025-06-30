import yfinance as yf
import json

def get_multiple_stock_data(trading_symbols, period='1mo'):
    """
    Fetches historical closing prices for a list of stock symbols.

    Args:
        trading_symbols (list): A list of stock ticker symbols.
        period (str): The period for which to fetch data (e.g., '1mo', '1y').

    Returns:
        dict: A dictionary where keys are symbols and values are dictionaries
              containing 'closing_prices' (list of floats/None) or 'error' (str).
              Example:
              {
                  "AAPL": {"closing_prices": [150.0, 151.0, ...], "error": null},
                  "INVALID_SYMBOL": {"closing_prices": null, "error": "Failed to fetch data"}
              }
    """
    all_stock_data = {}
    if not isinstance(trading_symbols, list):
        print("⚠️ Input symbols must be a list.")
        return {"error": "Input symbols must be a list."}

    for symbol_input in trading_symbols:
        symbol = str(symbol_input).upper() # Ensure symbol is string and uppercase
        try:
            stock = yf.Ticker(symbol)
            # Fetch data for the specified period
            historical_data = stock.history(period=period.upper())
            
            if historical_data.empty:
                print(f"ℹ️ No data found for symbol {symbol} for period {period}.")
                all_stock_data[symbol] = {"closing_prices": None, "error": f"No data found for symbol {symbol} for period {period}."}
                continue
            
            # Ensure 'Close' column exists
            if 'Close' not in historical_data.columns:
                print(f"⚠️ 'Close' column not in historical data for {symbol}.")
                all_stock_data[symbol] = {"closing_prices": None, "error": f"'Close' column not found in historical data for {symbol}."}
                continue

            closing_prices = historical_data['Close'].tolist()
            # Convert NaN to None if any, or handle as needed
            closing_prices = [None if price != price else price for price in closing_prices] # price != price checks for NaN
            all_stock_data[symbol] = {"closing_prices": closing_prices, "error": None}
            print(f"✅ Fetched data for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            all_stock_data[symbol] = {"closing_prices": None, "error": str(e)}
            
    return all_stock_data
