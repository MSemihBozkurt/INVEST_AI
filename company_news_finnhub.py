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
two_weeks_ago = today - timedelta(days=1) # You can adjust the number of days

_from_date_str = two_weeks_ago.strftime('%Y-%m-%d')
_to_date_str = today.strftime('%Y-%m-%d')

# Fetch company news
try:
    company_symbol = 'AAPL' # You can change this to any other stock symbol
    news_list = finnhub_client.company_news(company_symbol, _from=_from_date_str, to=_to_date_str)

    if not news_list:
        print(f"No news found for {company_symbol} from {_from_date_str} to {_to_date_str}.")
    else:
        print(f"\n--- Key News for {company_symbol} ({_from_date_str} to {_to_date_str}) ---\n")
        for i, news_item in enumerate(news_list):
            print(f"News Item #{i + 1}")
            print("-" * 20)

            print(f"  Headline: {news_item.get('headline', 'N/A')}")

            # Get the full summary
            summary = news_item.get('summary', '')
            if summary: # Only print summary if it exists and is not empty
                print(f"  Summary:  {summary}") # Print the entire summary
            else:
                print(f"  Summary:  N/A")

            print(f"  URL:      {news_item.get('url', 'N/A')}")
            print("\n" + "=" * 50 + "\n") # Separator for each news item

except finnhub.FinnhubAPIException as e:
    print(f"Finnhub API Exception: {e}")
    print(f"Details: {e.args}") # FinnhubAPIException might have more details in args
except Exception as e:
    print(f"An unexpected error occurred: {e}")