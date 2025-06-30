import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=api_key)

def format_value(value, unit):
    """
    Formats a financial value based on its unit.
    """
    if value is None:
        return "N/A"
    
    if unit == 'usd':
        # Format as currency with comma separators, large numbers without decimals
        if abs(value) >= 1_000_000_000: # Billions
            return f"${value / 1_000_000_000:,.2f} B"
        elif abs(value) >= 1_000_000: # Millions
            return f"${value / 1_000_000:,.2f} M"
        else: # Smaller values
            return f"${value:,.2f}"
    elif unit == 'shares':
        return f"{value:,.0f} shares"
    elif unit == 'usd/share':
        return f"${value:.2f}/share"
    else:
        return f"{value} {unit}"

def format_financial_report_full(report_data):
    """
    Formats the raw financial report data into a human-readable string,
    displaying all available concepts in each section.
    """
    if not report_data:
        return "No financial report data available."

    output = []
    output.append(f"---")
    output.append(f"## Annual Financial Report for {report_data.get('symbol', 'N/A')}")
    output.append(f"---")
    output.append(f"**Fiscal Year:** {report_data.get('year', 'N/A')}")
    output.append(f"**Form Type:** {report_data.get('form', 'N/A')}")
    output.append(f"**Filed Date:** {report_data.get('filedDate', 'N/A').split(' ')[0]}") # Only date part
    output.append(f"**Fiscal Period:** {report_data.get('startDate', 'N/A').split(' ')[0]} to {report_data.get('endDate', 'N/A').split(' ')[0]}")

    # Balance Sheet
    bs_data = report_data.get('report', {}).get('bs', [])
    if bs_data:
        output.append("\n### Balance Sheet (BS)")
        for item in bs_data:
            label = item.get('label', item.get('concept', 'N/A'))
            value = item.get('value')
            unit = item.get('unit', '')
            formatted_val = format_value(value, unit)
            output.append(f"- {label}: {formatted_val}")
    else:
        output.append("\n### Balance Sheet (BS): No data available.")

    # Income Statement
    ic_data = report_data.get('report', {}).get('ic', [])
    if ic_data:
        output.append("\n### Income Statement (IC)")
        for item in ic_data:
            label = item.get('label', item.get('concept', 'N/A'))
            value = item.get('value')
            unit = item.get('unit', '')
            formatted_val = format_value(value, unit)
            output.append(f"- {label}: {formatted_val}")
    else:
        output.append("\n### Income Statement (IC): No data available.")

    # Cash Flow Statement
    cf_data = report_data.get('report', {}).get('cf', [])
    if cf_data:
        output.append("\n### Cash Flow Statement (CF)")
        for item in cf_data:
            label = item.get('label', item.get('concept', 'N/A'))
            value = item.get('value')
            unit = item.get('unit', '')
            formatted_val = format_value(value, unit)
            output.append(f"- {label}: {formatted_val}")
    else:
        output.append("\n### Cash Flow Statement (CF): No data available.")

    return "\n".join(output)

def get_company_news(symbol, days_back=14):
    """
    Fetch company news for a given stock symbol from Finnhub.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        days_back (int): Number of days to look back for news (default: 14)
    
    Returns:
        str: Formatted news summary or error message
    """
    try:
        # Calculate date range
        today = datetime.today().date()
        start_date = today - timedelta(days=days_back)
        
        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = today.strftime('%Y-%m-%d')
        
        print(f"  Fetching news for {symbol} from {from_date_str} to {to_date_str}")
        
        # Fetch company news
        news_list = finnhub_client.company_news(symbol, _from=from_date_str, to=to_date_str)
        
        if not news_list:
            return f"No recent news found for {symbol} in the last {days_back} days."
        
        # Format news output
        output = []
        output.append(f"Recent News for {symbol} ({from_date_str} to {to_date_str}):")
        output.append("-" * 50)
        
        # Limit to top 5 news items to avoid overwhelming the context
        for i, news_item in enumerate(news_list[:5]):
            output.append(f"\nNews #{i + 1}:")
            output.append(f"Headline: {news_item.get('headline', 'N/A')}")
            
            summary = news_item.get('summary', '')
            if summary:
                # Truncate very long summaries
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                output.append(f"Summary: {summary}")
            else:
                output.append("Summary: N/A")
            
            # Include source URL for reference
            output.append(f"Source: {news_item.get('url', 'N/A')}")
            
        return "\n".join(output)
        
    except finnhub.FinnhubAPIException as e:
        error_msg = f"Finnhub API error for {symbol}: {str(e)}"
        print(f"    ❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching news for {symbol}: {str(e)}"
        print(f"    ❌ {error_msg}")
        return error_msg

def get_financial_reports(symbol):
    """
    Fetch the most recent annual financial report for a given stock symbol from Finnhub.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
    
    Returns:
        str: Formatted financial report or error message
    """
    try:
        # Calculate date range for the last two years
        today = datetime.today().date()
        two_years_ago = today - timedelta(days=5 * 365)
        
        from_date_str = two_years_ago.strftime('%Y-%m-%d')
        to_date_str = today.strftime('%Y-%m-%d')
        
        print(f"  Fetching financial report for {symbol} from {from_date_str} to {to_date_str}")
        
        # Fetch financial reports
        financial_data = finnhub_client.financials_reported(
            symbol=symbol, 
            freq='annual', 
            _from=from_date_str, 
            to=to_date_str
        )
        
        if not financial_data or not financial_data.get('data'):
            return f"No annual financial reports found for {symbol} in the last two years."
        
        # Get the most recent annual report (usually the first one in the list)
        latest_annual_report = financial_data['data'][0]
        
        # Format and return the report
        return format_financial_report_full(latest_annual_report)
        
    except finnhub.FinnhubAPIException as e:
        error_msg = f"Finnhub API error for {symbol}: {str(e)}"
        print(f"    ❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching financial report for {symbol}: {str(e)}"
        print(f"    ❌ {error_msg}")
        return error_msg

# Test functions (for debugging purposes)
if __name__ == "__main__":
    # Test with AAPL
    test_symbol = "AAPL"
    
    print("Testing company news...")
    news_result = get_company_news(test_symbol)
    print(news_result)
    
    print("\n" + "="*80 + "\n")
    
    print("Testing financial reports...")
    report_result = get_financial_reports(test_symbol)
    print(report_result)
    