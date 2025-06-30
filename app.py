import gradio as gr
import json
import os 
import sys
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai

# Gemini API Pricing (per million tokens for gemini-2.0-flash)
GEMINI_INPUT_COST_PER_MILLION = 0.10  # $0.10 per 1M input tokens
GEMINI_OUTPUT_COST_PER_MILLION = 0.40  # $0.40 per 1M output tokens

load_dotenv()
SERPAPI_KEY = os.getenv("SERP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Add Search imports
from google_search.search_api import SearchAPI
from google_search.gemini_api import GeminiAPI
from google_search.summarization_agent import SummarizationAgent

# Start SearchAPI and its dependencies
gemini_processor = GeminiAPI(GEMINI_API_KEY)
summarizer = SummarizationAgent(GEMINI_API_KEY)
search_api = SearchAPI(SERPAPI_KEY, gemini_processor, summarizer)

from insight_agent import insight_chat

# Add RAG imports
from RAG.document_loader import load_documents_from_folder
from RAG.embedder import get_embedding_function
from RAG.vector_store import create_chroma_client, add_document_to_collection, retrieveDocs, summarize_collection, clear_collection, get_document_sources
from RAG.gemini_rag import build_chatBot, generate_LLM_answer
import google.generativeai as genai

# Ensure the project directory is in sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports for main chat
from yahoo_finance import get_multiple_stock_data
from recommendation_agent import generate_recommendations as generate_investment_recommendations

# Import Finnhub utilities
from finnhub_utils import get_company_news, get_financial_reports

# --- Global Variables ---
user_risk_profile_json = {}
chat_initialized = False
guidance_chat_initialized = False
latest_financial_data = {}  # Store latest financial data for download
process_logs = []  # Store specific process logs for UI display
token_usage_logs = []  # Store token usage logs for UI display
total_session_cost = 0.0 # Total session cost for token usage

# RAG Global Variables
RAG_LLM = None
chroma_collection = None
document_folder = None

# Initialize Gemini client for token counting
genai.configure(api_key=GEMINI_API_KEY)
gemini_client = genai.GenerativeModel('gemini-2.0-flash')

# Initialize ChromaDB for RAG (loaded once at startup)
try:
    # Configuration
    collection_name = "Investment_Guidance"
    sentence_transformer_model = "distiluse-base-multilingual-cased-v1"
    chromaDB_path = os.path.join(".", "RAG/chromadb_data")
    document_folder = os.path.join(".", "RAG/sample_data")  # Your documents folder
    
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)
    
    # Initialize Gemini for RAG
    genai.configure(api_key=GEMINI_API_KEY)
    system_prompt = """You are a patient and knowledgeable investment guide. Your role is to:
    1. Teach basic investment concepts in simple, easy-to-understand language
    2. Answer questions about stocks, bonds, ETFs, mutual funds, risk management, and portfolio diversification
    3. Provide step-by-step explanations for complex topics
    4. Use real-world examples and analogies to make concepts clear
    5. Encourage questions and create a supportive learning environment
    6. Focus on education rather than specific investment advice
    
    Always be encouraging, patient, and thorough in your explanations. Break down complex topics into digestible pieces.
    Answer based on the provided context from financial documents and glossaries."""
    
    RAG_LLM = build_chatBot(system_prompt)
    
    # Initialize ChromaDB and embedding function
    embedding_function = get_embedding_function(sentence_transformer_model)
    chroma_client, chroma_collection = create_chroma_client(collection_name, embedding_function, chromaDB_path)
    
    # Load documents if they exist
    document_chunks = load_documents_from_folder(document_folder, sentence_transformer_model)
    if document_chunks:
        for document_name, chunks in document_chunks:
            chroma_collection = add_document_to_collection(chroma_collection, chunks, document_name, category="Investment Guide")
    else:
        pass
        
except Exception as e:
    import traceback
    traceback.print_exc()
    RAG_LLM = None
    chroma_collection = None

# Function to format logs for UI display
def get_formatted_logs():
    return "\n".join(process_logs[-10:])  # Show last 10 logs to keep UI clean


def calculate_token_cost(prompt_tokens, output_tokens):
    """Calculate cost based on Gemini pricing"""
    input_cost = (prompt_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_MILLION
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost


# Function to format token usage logs for UI display
def get_formatted_token_usage():
    if not token_usage_logs:
        return "No token usage recorded yet."
    
    # Show last 10 token usage logs
    recent_logs = token_usage_logs[-10:]
    formatted_logs = "\n".join(recent_logs)
    
    # Add total session cost at the bottom
    formatted_logs += f"\n💰 TOTAL SESSION COST: ${total_session_cost:.6f}\n"
    
    return formatted_logs

# Function to log token usage
def log_token_usage(operation, prompt_tokens, output_tokens, total_tokens):
    global total_session_cost
    
    input_cost, output_cost, operation_cost = calculate_token_cost(prompt_tokens, output_tokens)
    total_session_cost += operation_cost
    
    log_entry = (f"{operation}: "
                f"Input: {prompt_tokens:,} tokens (${input_cost:.6f}), "
                f"Output: {output_tokens:,} tokens (${output_cost:.6f}), "
                f"Total: {total_tokens:,} tokens (${operation_cost:.6f})")
    
    token_usage_logs.append(log_entry)
    print(log_entry)  # For debugging

# Function to generate Excel file from financial data
def generate_excel_file():
    global latest_financial_data
    if not latest_financial_data:
        return None, "No financial data available to download. Please request stock recommendations first."
    
    # Create a DataFrame for each stock and combine into an Excel file
    excel_data = {}
    for symbol, data in latest_financial_data.items():
        if data.get("closing_prices") and not data.get("error"):
            df = pd.DataFrame({
                "Date": data.get("dates", [f"Day {i+1}" for i in range(len(data["closing_prices"]))]),
                "Closing Price": [round(p, 2) if p is not None else None for p in data["closing_prices"]]
            })
            excel_data[symbol] = df
    
    if not excel_data:
        return None, "No valid financial data available to export to Excel."
    
    # Write to Excel file with multiple sheets
    output_file = "stock_data.xlsx"
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        for symbol, df in excel_data.items():
            df.to_excel(writer, sheet_name=symbol, index=False)
    
    return output_file, None

# Chatbot function - Updated to use Finnhub and track tokens
def chatbot_fn(message, history):
    global user_risk_profile_json, latest_financial_data, process_logs, token_usage_logs

    if not message.strip():
        yield "Please enter a message.", None, get_formatted_logs(), get_formatted_token_usage()
        return

    print(f"💬 User message to main chatbot: {message}")

    if not user_risk_profile_json:
        yield "Please complete the risk profile questionnaire first. I need that information to provide tailored stock suggestions.", None, get_formatted_logs(), get_formatted_token_usage()
        return

    profile_json_str = json.dumps(user_risk_profile_json, indent=2)
    current_input_for_insight_agent = f"User Profile (JSON):\n{profile_json_str}\n\nUser's Original Request:\n{message}"

    try:
        # Stage 1: Get stock symbols from Insight Agent
        process_logs.append("🧠 Sending to Insight Agent...")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"🧠 Sending to Insight Agent... Input length: {len(current_input_for_insight_agent)}")

        # Count input tokens for Insight Agent
        input_tokens_insight = gemini_client.count_tokens(current_input_for_insight_agent).total_tokens
        insight_response = insight_chat.send_message(current_input_for_insight_agent)
        
        # Get token usage from response
        prompt_tokens_insight = insight_response.usage_metadata.prompt_token_count
        output_tokens_insight = insight_response.usage_metadata.candidates_token_count
        total_tokens_insight = insight_response.usage_metadata.total_token_count
        log_token_usage("Insight Agent", prompt_tokens_insight, output_tokens_insight, total_tokens_insight)
        
        insight_text = ""
        if insight_response.candidates and insight_response.candidates[0].content.parts:
            insight_text = insight_response.candidates[0].content.parts[0].text.strip()
        elif hasattr(insight_response, 'text'):
            insight_text = insight_response.text.strip()
        else:
            print("⚠️ Insight Agent returned no parseable content.")

        print(f"🔍 Insight Agent Raw Response: '{insight_text}'")

        stock_symbols = []
        try:
            if insight_text.startswith("```json"):
                insight_text = insight_text.replace("```json", "", 1).strip()
            if insight_text.startswith("```"):
                insight_text = insight_text.replace("```", "", 1).strip()
            if insight_text.endswith("```"):
                insight_text = insight_text[:-3].strip()
            
            extracted_symbols = json.loads(insight_text)
            if isinstance(extracted_symbols, list) and all(isinstance(s, str) for s in extracted_symbols):
                stock_symbols = [s.strip().upper() for s in extracted_symbols if s.strip()]
                stock_symbols = [s for s in stock_symbols if 1 <= len(s) <= 10]
                
                if not stock_symbols:
                    print("⚠️ Insight Agent returned an empty list or invalid symbols after cleaning.")
                elif not (1 <= len(stock_symbols) <= 3):
                    print(f"⚠️ Insight Agent returned {len(stock_symbols)} symbols. Original request was 1-3. Proceeding.")
            else:
                raise ValueError("Insight agent response was not a valid JSON list of strings.")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Error parsing stock symbols from Insight Agent: {e}. Response: '{insight_text}'")
            yield f"I had trouble understanding the stock suggestions from my analysis engine. The format was unexpected. (Details: {e})", None, get_formatted_logs(), get_formatted_token_usage()
            return

        if not stock_symbols:
            yield "I couldn't identify any specific stocks matching your criteria right now. You might want to adjust your profile preferences or try a broader request.", None, get_formatted_logs(), get_formatted_token_usage()
            return

        process_logs.append(f"📈 Identified stock symbols: {stock_symbols}")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"📈 Identified stock symbols: {stock_symbols}")

        # Stage 2: Fetch stock data from Yahoo Finance (default 1 month)
        process_logs.append(f"📉 Fetching historical data for {stock_symbols}...")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"📉 Fetching historical data for {stock_symbols} (period: 1mo)...")
        latest_financial_data = get_multiple_stock_data(stock_symbols, period='1mo')
        print(f"📊 Fetched financial data: {json.dumps(latest_financial_data, indent=2, default=str)}")

        # Stage 3: Fetch company news from Finnhub for each stock
        company_news = {}
        process_logs.append(f"📰 Fetching company news from Finnhub for {stock_symbols}...")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"📰 Fetching company news from Finnhub for {stock_symbols}...")
        for symbol in stock_symbols:
            try:
                print(f"  Fetching news for {symbol}...")
                news_data = get_company_news(symbol)
                company_news[symbol] = news_data
                print(f"    ✅ News data for {symbol} retrieved.")
            except Exception as e:
                print(f"    ❌ Error fetching news for {symbol}: {str(e)}")
                company_news[symbol] = f"Error fetching news for {symbol}: {str(e)}"

        # Stage 4: Fetch financial reports from Finnhub for each stock
        financial_reports = {}
        process_logs.append(f"📋 Fetching financial reports from Finnhub for {stock_symbols}...")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"📋 Fetching financial reports from Finnhub for {stock_symbols}...")
        for symbol in stock_symbols:
            try:
                print(f"  Fetching financial report for {symbol}...")
                report_data = get_financial_reports(symbol)
                financial_reports[symbol] = report_data
                print(f"    ✅ Financial report for {symbol} retrieved.")
            except Exception as e:
                print(f"    ❌ Error fetching financial report for {symbol}: {str(e)}")
                financial_reports[symbol] = f"Error fetching financial report for {symbol}: {str(e)}"

        # Stage 5: Prepare context for the Recommendation Agent
        recommendation_context_parts = ["Please analyze the following comprehensive stock information based on their recent performance, news, and financial reports:\n"]
        
        for symbol_key in stock_symbols:
            symbol_data_str = f"\n=== Stock Analysis: {symbol_key} ===\n"
            
            fdata = latest_financial_data.get(symbol_key, {})
            if fdata.get("error"):
                symbol_data_str += f"Historical Price Data: Error - {fdata['error']}\n"
            elif fdata.get("closing_prices") and fdata.get("closing_prices") is not None:
                prices = fdata['closing_prices']
                if not prices:
                    symbol_data_str += "Historical Price Data (1mo): No price data points available.\n"
                else:
                    price_snippet = [round(p, 2) if p is not None else 'N/A' for p in prices[:5]]
                    if len(prices) > 5: price_snippet.append("...")
                    symbol_data_str += f"Historical Price Data (1mo, recent sample): {price_snippet}\n"
            else:
                symbol_data_str += "Historical Price Data (1mo): Data not available or not found.\n"

            news_data = company_news.get(symbol_key, "News data not available for this stock.")
            if isinstance(news_data, str):
                symbol_data_str += f"Recent Company News: {news_data}\n"
            else:
                symbol_data_str += f"Recent Company News:\n{news_data}\n"

            report_data = financial_reports.get(symbol_key, "Financial report not available for this stock.")
            if isinstance(report_data, str):
                symbol_data_str += f"Latest Financial Report: {report_data}\n"
            else:
                symbol_data_str += f"Latest Financial Report:\n{report_data}\n"
            
            recommendation_context_parts.append(symbol_data_str)

        full_context_for_recommendation = "".join(recommendation_context_parts)
        print(f"📜 Context for Recommendation Agent (approx. length {len(full_context_for_recommendation)} chars):\n{full_context_for_recommendation[:1000]}...")

        # Stage 6: Get recommendations from the Recommendation Agent
        process_logs.append("🤖 Generating investment recommendations...")
        yield "", None, get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print("🤖 Generating investment recommendations...")

        # Count input tokens for Recommendation Agent
        input_tokens_recommendation = gemini_client.count_tokens(full_context_for_recommendation).total_tokens
        bot_reply = generate_investment_recommendations(full_context_for_recommendation)

        if isinstance(bot_reply, tuple):
            bot_reply, usage_metadata = bot_reply
            prompt_tokens_recommendation = usage_metadata.prompt_token_count
            output_tokens_recommendation = usage_metadata.candidates_token_count
            total_tokens_recommendation = usage_metadata.total_token_count
        else:
            # Fallback: count output tokens assuming text-only response
            prompt_tokens_recommendation = input_tokens_recommendation
            output_tokens_recommendation = gemini_client.count_tokens(bot_reply).total_tokens
            total_tokens_recommendation = prompt_tokens_recommendation + output_tokens_recommendation
        log_token_usage("Recommendation Agent", prompt_tokens_recommendation, output_tokens_recommendation, total_tokens_recommendation)
        
        print(f"💡 Recommendation Agent Response:\n{bot_reply}")
        yield bot_reply, None, get_formatted_logs(), get_formatted_token_usage()

    except Exception as e:
        print(f"❌❌❌ Major error in chatbot_fn: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"I encountered an unexpected issue while processing your request. Please try again later. (Error: {str(e)})", str(e), get_formatted_logs(), get_formatted_token_usage()

# --- Beginner Guidance Chatbot function
def guidance_chatbot_fn(message, history):
    global guidance_chat_initialized, RAG_LLM, chroma_collection, process_logs, token_usage_logs

    print(f"📩 Received guidance message: {message}")

    message = message.strip()
    if not message:
        yield "Please enter a valid question about investments.", get_formatted_logs(), get_formatted_token_usage()
        return

    if RAG_LLM is None or chroma_collection is None:
        yield "I apologize, but the guidance system is not fully initialized. Please check if the document folder contains PDF files and restart the application.", get_formatted_logs(), get_formatted_token_usage()
        return

    try:
        process_logs.append("🔍 Searching documents for relevant information...")
        yield "", get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print("🔍 Searching documents for relevant information...")
        
        retrieved_results = retrieveDocs(chroma_collection, message, n_results=5, return_only_docs=False)
        
        if not retrieved_results or not retrieved_results.get('documents'):
            print("⚠️ No relevant documents found in knowledge base.")
            context = "\nNo specific documents found in knowledge base. Please provide a general investment education response."
        else:
            retrieved_documents = retrieved_results['documents'][0]
            context = f"\nRelevant information from knowledge base:\n" + "\n".join(retrieved_documents)
            print(f"📚 Retrieved {len(retrieved_documents)} relevant document chunks")

        prompt = f"Educational Investment Question: {message}"
        
        # Count input tokens for Guidance Chatbot
        input_content = prompt + context
        input_tokens_guidance = gemini_client.count_tokens(input_content).total_tokens
        
        process_logs.append("🤖 Generating educational response...")
        yield "", get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print("🤖 Generating educational response...")
        bot_reply = generate_LLM_answer(prompt, context, RAG_LLM)
        
        # Get token usage from response
        response = RAG_LLM.send_message(input_content)
        prompt_tokens_guidance = response.usage_metadata.prompt_token_count
        output_tokens_guidance = response.usage_metadata.candidates_token_count
        total_tokens_guidance = response.usage_metadata.total_token_count
        log_token_usage("Guidance Chatbot", prompt_tokens_guidance, output_tokens_guidance, total_tokens_guidance)
        
        sources = get_document_sources(retrieved_results) if retrieved_results else []
        
        if sources and sources != ['Unknown']:
            bot_reply += f"\n\n📚 **Sources consulted:** {', '.join(sources)}"
        
        bot_reply = bot_reply.strip()
        print(f"✅ Generated guidance response with {len(sources)} sources")
        
        yield bot_reply, get_formatted_logs(), get_formatted_token_usage()
        
    except Exception as e:
        error_reply = f"I apologize, but I'm having trouble responding right now. Please try asking your question again."
        print(f"❌ Error in guidance_chatbot_fn: {str(e)}")
        import traceback
        traceback.print_exc()
        yield error_reply, get_formatted_logs(), get_formatted_token_usage()

# --- Sentiment Analysis function
def analyze_sentiment(topic):
    global process_logs, token_usage_logs
    if not topic.strip():
        yield "Please enter a valid stock symbol or topic.", get_formatted_logs(), get_formatted_token_usage()
        return
    
    try:
        # Log the sentiment analysis search and yield immediately
        process_logs.append(f"🔍 Performing sentiment analysis...")
        yield "", get_formatted_logs(), get_formatted_token_usage()  # Stream log update
        print(f"🔍 Performing sentiment analysis for topic: {topic}")
        
        # Count input tokens for Sentiment Analysis
        input_content = f"USER QUERY: \"{topic}\"\nSummarize the sentiment based on recent news."
        input_tokens_sentiment = gemini_client.count_tokens(input_content).total_tokens
        
        results = search_api.get_enhanced_search_results_with_summary(topic, include_summary=True)
        print(f"Debug: Raw results = {results}")
        if results["error"]:
            yield f"Error analyzing sentiment: {results['error']}", get_formatted_logs(), get_formatted_token_usage()
            return
        
        summary = results["summary"]
        if not summary:
            yield "No summary could be generated for the given topic.", get_formatted_logs(), get_formatted_token_usage()
            return
        
        # Count output tokens
        output_tokens_sentiment = gemini_client.count_tokens(summary).total_tokens
        total_tokens_sentiment = input_tokens_sentiment + output_tokens_sentiment
        log_token_usage("Sentiment Analysis", input_tokens_sentiment, output_tokens_sentiment, total_tokens_sentiment)
        
        yield summary.strip(), get_formatted_logs(), get_formatted_token_usage()
    except Exception as e:
        yield f"Error analyzing sentiment: {str(e)}", get_formatted_logs(), get_formatted_token_usage()

# --- Process questionnaire function ---
def submit_questionnaire(age, occupation, stability, net_income, expenses, debts, emergency_fund,
                        investment_goal, horizon, amount,
                        knowledge, fluctuation_feeling, reaction, max_loss,
                        preferred_sectors, avoided_sectors):
    global user_risk_profile_json, chat_initialized

    risk_profile, score, confidence, details = calculate_risk_profile(
        age, stability, net_income, expenses, debts, emergency_fund,
        investment_goal, horizon, amount,
        knowledge, fluctuation_feeling, reaction, max_loss
    )

    user_risk_profile_json = {
        "personal_financial_information": {
            "age": age,
            "occupation": occupation,
            "income_stability": stability,
            "monthly_net_income": net_income,
            "monthly_expenses": expenses,
            "has_debts": debts,
            "emergency_fund": emergency_fund,
        },
        "investment_goals_and_approach": {
            "goal": investment_goal,
            "horizon": horizon,
            "investment_amount": amount,
        },
        "investment_knowledge_and_risk_tolerance": {
            "knowledge_level": knowledge,
            "feeling_about_fluctuations": fluctuation_feeling,
            "reaction_to_market_drop": reaction,
            "max_tolerable_loss": max_loss,
        },
        "preferences": {
            "preferred_sectors": preferred_sectors,
            "avoided_sectors": avoided_sectors,
        },
        "risk_profile_result": {
            "profile": risk_profile,
        }
    }

    chat_initialized = False
    return (f"✅ Risk profile saved for personalized investment advice.\n\n"
            f"--- Personal & Financial Information ---\n"
            f"- Age: {age}\n- Occupation: {occupation}\n- Income Stability: {stability}\n"
            f"- Monthly Net Income: ${net_income:,.2f}\n- Monthly Average Expenses: ${expenses:,.2f}\n"
            f"- Outstanding Debts: {debts}\n- Emergency Fund: {emergency_fund}\n\n"
            f"--- Investment Goals & Approach ---\n"
            f"- Primary Investment Goal: {investment_goal}\n- Investment Horizon: {horizon}\n"
            f"- Intended Investment Amount: ${amount:,.2f}\n\n"
            f"--- Investment Knowledge & Risk Tolerance ---\n"
            f"- Investment Knowledge: {knowledge}\n"
            f"- Feeling about Short-term Fluctuations: {fluctuation_feeling}\n"
            f"- Reaction to a 20% Market Drop: {reaction}\n"
            f"- Maximum Tolerable Annual Loss: {max_loss}\n\n"
            f"--- Preferences ---\n"
            f"- Preferred Sectors: {preferred_sectors}\n"
            f"- Avoided Sectors: {avoided_sectors}\n\n"
            f"- Calculated Risk Profile: {risk_profile}"), get_formatted_logs(), get_formatted_token_usage()

def format_chat_history(chatbot_state):
    if not chatbot_state:
        return "No conversation history available."
    
    formatted_history = ["--- Conversation History ---"]
    for message in chatbot_state:
        role = message.get("role", "Unknown")
        content = message.get("content", "")
        if role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
        else:
            formatted_history.append(f"{role}: {content}")
    return "\n\n".join(formatted_history)

# --- Risk profile calculation ---
def calculate_risk_profile(age, stability, net_income, expenses, debts, emergency_fund,
                          investment_goal, horizon, amount,
                          knowledge, fluctuation_feeling, reaction, max_loss):
    score = 0
    details = {}

    if age < 30:
        age_score = 4
    elif age < 50:
        age_score = 3
    elif age < 65:
        age_score = 2
    else:
        age_score = 1
    score += age_score
    details['age'] = age_score

    stability_score = {"Very Stable": 3, "Moderately Stable": 2, "Unstable": 0}[stability]
    score += stability_score
    details['stability'] = stability_score

    savings_ratio = (net_income - expenses) / (net_income + 1e-6)
    if savings_ratio >= 0.6:
        savings_score = 4
    elif savings_ratio >= 0.4:
        savings_score = 3
    elif savings_ratio >= 0.2:
        savings_score = 2
    else:
        savings_score = 1
    score += savings_score
    details['savings_ratio'] = savings_score

    debt_score = 0 if debts == "Yes" else 2
    score += debt_score
    details['debts'] = debt_score

    fund_score = {
        "Yes, covers 3-6 months of expenses": 3,
        "Yes, but covers less": 1,
        "No, I don't have one": -1
    }[emergency_fund]
    score += fund_score
    details['emergency_fund'] = fund_score

    horizon_score = {
        "Long-term (More than 10 years)": 4,
        "Medium-term (3-10 years)": 2,
        "Short-term (Less than 3 years)": 0
    }[horizon]
    score += horizon_score
    details['horizon'] = horizon_score

    investment_ratio = amount / (net_income * 12 + 1e-6)
    if investment_ratio >= 1.0:
        amount_score = 4
    elif investment_ratio >= 0.5:
        amount_score = 3
    elif investment_ratio >= 0.2:
        amount_score = 2
    else:
        amount_score = 1
    score += amount_score
    details['investment_amount_ratio'] = amount_score

    knowledge_score = {
        "Advanced (I am knowledgeable about complex investment strategies)": 4,
        "Intermediate (I understand basic investment concepts)": 2,
        "Beginner (I have little to no knowledge about investing)": 0
    }[knowledge]
    score += knowledge_score
    details['knowledge'] = knowledge_score

    emotion_score = {
        "Largely unfazed; I focus on the long term.": 3,
        "Mildly concerned, but it wouldn't significantly impact my life.": 2,
        "Very anxious; it would keep me up at night.": 0
    }[fluctuation_feeling]
    score += emotion_score
    details['emotion'] = emotion_score

    reaction_score = {
        "Buy more investments (see it as an opportunity).": 3,
        "Hold onto my investments.": 2,
        "Sell some of my investments.": 0
    }[reaction]
    score += reaction_score
    details['reaction'] = reaction_score

    loss_score = {
        "More than 20%": 4,
        "Between 10% and 20%": 3,
        "Between 5% and 10%": 1,
        "Less than 5%": 0
    }[max_loss]
    score += loss_score
    details['max_loss'] = loss_score

    max_possible_score = 36
    confidence = round((score / max_possible_score) * 100)

    if score >= 28:
        risk_profile = "Aggressive"
    elif score >= 18:
        risk_profile = "Moderate"
    else:
        risk_profile = "Conservative"

    return risk_profile, score, confidence, details

# --- Updated Gradio Application
with gr.Blocks(title="Investment Assistant") as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### 📋 Menu")
            open_btn = gr.Button("📊 Open Questionnaire", size="sm")
            sentiment_btn = gr.Button("📈 Sentiment Analysis", size="sm")
            beginner_guide_btn = gr.Button("📚 Beginner Guidance", size="sm")
            history_btn = gr.Button("💬 Chat History", size="sm")
            gr.Markdown("### 🖥️ Process Logs")
            log_output = gr.Textbox(label="Activity Log", lines=10, interactive=False, value=get_formatted_logs())
            gr.Markdown("### 📊 Token Usage")
            token_usage_output = gr.Textbox(label="Token Usage Log", lines=5, interactive=False, value=get_formatted_token_usage())

        with gr.Column(scale=5):
            with gr.Column(visible=True) as chat_panel:
                initial_bot_message = (
                    "Hello! I am your financial investment assistant. "
                    "To help you get started, please fill out the risk profile questionnaire "
                    "by clicking the 'Open Questionnaire' button in the menu on the left."
                )
                
                chatbot_display = gr.Chatbot(
                    value=[{"role": "assistant", "content": initial_bot_message}],
                    label="Chat Assistant",
                    type="messages",
                    height=600
                )
                
                chat_interface = gr.ChatInterface(
                    fn=chatbot_fn,
                    chatbot=chatbot_display,
                    type="messages",
                    additional_outputs=[gr.Textbox(visible=False), log_output, token_usage_output]  # Hidden error output, log output, and token usage
                )
                
                download_btn = gr.Button("📥 Download Stock Data (Excel)", size="sm")
                download_output = gr.File(label="Download Stock Data")
                download_error = gr.Textbox(label="Download Status", visible=False)

            with gr.Column(visible=False) as form_panel:
                gr.Markdown("## Risk Profile Questionnaire")
                gr.Markdown("### Personal & General Financial Information")
                age = gr.Number(label="Age", minimum=18, value=18, step=1)
                occupation = gr.Textbox(label="Occupation")
                stability = gr.Radio(["Very Stable", "Moderately Stable", "Unstable"], label="Income Stability")
                net_income = gr.Number(label="Monthly Net Income ($)", minimum=0, value=2000, step=100)
                expenses = gr.Number(label="Average Monthly Expenses ($)", minimum=0, value=1000, step=100)
                debts = gr.Radio(["Yes", "No"], label="Do you have any outstanding debts (e.g., credit cards, loans)?")
                emergency_fund = gr.Radio(
                    ["Yes, covers 3-6 months of expenses", "Yes, but covers less", "No, I don't have one"],
                    label="Do you have an emergency fund?"
                )

                gr.Markdown("### Investment Goals & General Approach")
                investment_goal = gr.Textbox(label="What is your Primary Investment Goal? (e.g., Retirement, Buying a house, Education, etc.)")
                horizon = gr.Radio(["Short-term (Less than 3 years)", "Medium-term (3-10 years)", "Long-term (More than 10 years)"], label="Investment Horizon")
                amount = gr.Number(label="Intended Investment Amount ($)", minimum=0, value=1000, step=100)

                gr.Markdown("### Investment Knowledge & Risk Tolerance")
                knowledge = gr.Radio([
                    "Beginner (I have little to no knowledge about investing)",
                    "Intermediate (I understand basic investment concepts)",
                    "Advanced (I am knowledgeable about complex investment strategies)"
                ], label="Investment Knowledge Level")

                fluctuation_feeling = gr.Radio([
                    "Very anxious; it would keep me up at night.",
                    "Mildly concerned, but it wouldn't significantly impact my life.",
                    "Largely unfazed; I focus on the long term."
                ], label="How do short-term fluctuations in your investment value make you feel?")

                reaction = gr.Radio([
                    "Sell some of my investments.",
                    "Hold onto my investments.",
                    "Buy more investments (see it as an opportunity)."
                ], label="If the market were to drop by 20% overall, what would you most likely do?")

                max_loss = gr.Radio([
                    "Less than 5%", "Between 5% and 10%", "Between 10% and 20%", "More than 20%"
                ], label="What is the maximum annual loss you are comfortable with for your investments in a worst-case scenario?")

                gr.Markdown("### Specific Preferences")
                preferred_sectors = gr.Textbox(label="Sectors you prefer to invest in (e.g., Technology, Healthcare)")
                avoided_sectors = gr.Textbox(label="Sectors you definitely want to avoid (e.g., Tobacco, Gambling)")

                submit_btn = gr.Button("Calculate & Save Risk Profile")
                output = gr.Textbox(label="Result", lines=18, interactive=False)
                back_btn = gr.Button("⬅ Back to Chat")

            with gr.Column(visible=False) as sentiment_panel:
                gr.Markdown("# Market Sentiment Analysis")
                gr.Markdown("Enter a stock symbol or topic (e.g., AAPL, Forex, Oil) to analyze the current market sentiment based on recent news.")
                
                gr.Markdown("### Topic")
                with gr.Row():
                    topic_input = gr.Textbox(
                        placeholder="e.g., TSLA",
                        label="",
                        scale=4
                    )
                    analyze_btn = gr.Button("Analyze Sentiment", variant="primary", scale=1)
                
                sentiment_output = gr.Textbox(
                    label="Sentiment Analysis Result",
                    lines=15,
                    interactive=False,
                    placeholder="Analysis results will appear here..."
                )
                
                back_sentiment_btn = gr.Button("⬅ Back to Chat")

            with gr.Column(visible=False) as guidance_panel:
                gr.Markdown("# 🎓 Investment Learning Guide")
                gr.Markdown(""" 
                The guide is designed to help you understand investment concepts step-by-step. Whether you're completely new to investing or want to refresh your knowledge.
                
                *How to use this guide:*
                Simply ask any investment-related question, and It'll provide clear, easy-to-understand explanations based on curated educational content. Don't hesitate to ask for clarification or more details on any topic!
                """)
                
                initial_guidance_message = (
                    "Hello! I am your personal investment guide, and I'm excited to help you on your learning journey! 🚀\n\n"    
                    "Feel free to ask me anything, such as:\n"
                    "• 'What is a stock and how does it work?'\n"
                    "• 'What's the difference between stocks and bonds?'\n"
                    "• 'Can you explain what an ETF is?'\n\n"
                    "Remember, there are no silly questions here! My goal is to make you feel confident and knowledgeable about investing. What would you like to learn about first?"
                )
                
                guidance_chatbot_display = gr.Chatbot(
                    value=[{"role": "assistant", "content": initial_guidance_message}],
                    label="Investment Learning Guide",
                    type="messages",
                    height=500
                )
                
                guidance_chat_interface = gr.ChatInterface(
                    fn=guidance_chatbot_fn,
                    chatbot=guidance_chatbot_display,
                    type="messages",
                    additional_outputs=[log_output, token_usage_output]
                )
                
                back_guidance_btn = gr.Button("⬅ Back to Chat")

            with gr.Column(visible=False) as history_panel:
                gr.Markdown("## Chat History")
                history_output = gr.Textbox(lines=20, label="Full Conversation", interactive=False)
                close_history_btn = gr.Button("⬅ Back to Chat")

    # --- Event Handlers ---
    open_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    sentiment_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    beginner_guide_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    back_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    back_sentiment_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    back_guidance_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    history_btn.click(
        fn=lambda chatbot_state: (
            gr.update(visible=False),  # chat_panel
            gr.update(visible=False),  # form_panel
            gr.update(visible=False),  # sentiment_panel
            gr.update(visible=False),  # guidance_panel
            gr.update(visible=True),   # history_panel
            format_chat_history(chatbot_state),  # Update history_output
            get_formatted_logs(),  # Update log_output
            get_formatted_token_usage()  # Update token_usage_output
        ),
        inputs=[chatbot_display],
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, history_output, log_output, token_usage_output]
    )

    close_history_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), get_formatted_logs(), get_formatted_token_usage()),
        inputs=None,
        outputs=[chat_panel, form_panel, sentiment_panel, guidance_panel, history_panel, log_output, token_usage_output]
    )

    submit_btn.click(
        submit_questionnaire,
        inputs=[age, occupation, stability, net_income, expenses, debts, emergency_fund,
                investment_goal, horizon, amount,
                knowledge, fluctuation_feeling, reaction, max_loss,
                preferred_sectors, avoided_sectors],
        outputs=[output, log_output, token_usage_output]
    )

    analyze_btn.click(
        analyze_sentiment,
        inputs=[topic_input],
        outputs=[sentiment_output, log_output, token_usage_output],
    )

    download_btn.click(
        generate_excel_file,
        inputs=None,
        outputs=[download_output, download_error]
    )

# --- Run the App ---
if __name__ == "__main__":
    demo.launch()