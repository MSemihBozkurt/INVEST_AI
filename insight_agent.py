from google.generativeai import GenerativeModel
import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# System Prompt - Insight Agent

insight_agent_prompt = """
Your task is to act as a data extraction tool.
Based on the provided User Profile (JSON) and the User's Original Request, identify 1 to 3 relevant stock symbols.
YOUR RESPONSE MUST BE A VALID JSON ARRAY ONLY. Each string should be a stock symbol.
EXAMPLE RESPONSE: ["MSFT", "AAPL", "GOOG"]
DO NOT ADD any other text, explanations, apologies, summaries, or introductory sentences.
DO NOT USE markdown syntax such as ```json or ```.
The entire response should only be the JSON array.
If no relevant symbols can be identified, return an empty JSON array: [].

The User Profile (JSON) and the User's Original Request will be provided below.
"""

insight_model = GenerativeModel(
model_name="gemini-2.0-flash", 
system_instruction=insight_agent_prompt,
generation_config={
"temperature": 0.2,
}
)

insight_chat = insight_model.start_chat(history=[])