import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")

# Ensure API key is available
if not api_key:
    print("Error: FINNHUB_API_KEY not found in .env file or environment variables.")
    exit()

finnhub_client = finnhub.Client(api_key=api_key)

# Define the date range for the news
today = datetime.today().date()
two_weeks_ago = today - timedelta(days=14) # You can adjust the number of days

_from_date_str = two_weeks_ago.strftime('%Y-%m-%d')
_to_date_str = today.strftime('%Y-%m-%d')

def get_company_news_for_symbol(symbol, days_back=14):
    """
    Get company news for a specific stock symbol.
    This function can be imported and used by other modules.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        days_back (int): Number of days to look back for news
    
    Returns:
        str: Formatted news summary or error message
    """
    try:
        # Calculate date range
        today = datetime.today().date()
        start_date = today - timedelta(days=days_back)
        
        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = today.strftime('%Y-%m-%d')
        
        print(f"Fetching news for {symbol} from {from_date_str} to {to_date_str}")
        
        # Fetch company news
        news_list = finnhub_client.company_news(symbol, _from=from_date_str, to=to_date_str)
        
        if not news_list:
            return f"No news found for {symbol} from {from_date_str} to {to_date_str}."
        
        # Format news output
        output = []
        output.append(f"\n--- Key News for {symbol} ({from_date_str} to {to_date_str}) ---\n")
        
        # Limit to top 5 news items for better readability
        for i, news_item in enumerate(news_list[:5]):
            output.append(f"News Item #{i + 1}")
            output.append("-" * 20)
            output.append(f"  Headline: {news_item.get('headline', 'N/A')}")
            
            # Get the full summary
            summary = news_item.get('summary', '')
            if summary:
                # Truncate very long summaries to keep context manageable
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                output.append(f"  Summary:  {summary}")
            else:
                output.append(f"  Summary:  N/A")
            
            output.append(f"  URL:      {news_item.get('url', 'N/A')}")
            output.append("\n" + "=" * 50 + "\n")  # Separator for each news item
        
        return "\n".join(output)
        
    except finnhub.FinnhubAPIException as e:
        error_msg = f"Finnhub API Exception for {symbol}: {e}"
        if hasattr(e, 'args') and e.args:
            error_msg += f" Details: {e.args}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while fetching news for {symbol}: {e}"
        print(error_msg)
        return error_msg

# --- Main Script Execution (for standalone usage) ---
if __name__ == "__main__":
    # You can modify this list to include multiple stock symbols
    stock_symbols = ['AAPL']  # Add more symbols like ['AAPL', 'GOOGL', 'MSFT'] for batch processing
    
    for symbol in stock_symbols:
        print(f"\n{'='*80}")
        print(f"PROCESSING NEWS FOR: {symbol}")
        print(f"{'='*80}")
        
        result = get_company_news_for_symbol(symbol)
        print(result)
