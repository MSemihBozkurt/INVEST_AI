import os
import sys
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Fix gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
logging.getLogger('google.api_core.grpc_helpers').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport.grpc').setLevel(logging.ERROR)

SUMMARIZATION_SYSTEM_PROMPT = """You are an expert at summarization and market sentiment analysis. Your task is to take the expanded information coming from multiple search results and create a comprehensive and coherent answer to the user's original query.

Your job:
1. Analyze the user's original query and expanded information from multiple search results
2. Synthesize this information to create a coherent, comprehensive, and useful summary
3. Highlight main topics, important points, and information valuable to the user
4. If there is contradictory information, mention this and clarify if possible
5. Organize the information in a logical order
6. **MARKET SENTIMENT ANALYSIS**: Based on the news, analyst opinions, price movements, and overall information gathered, provide a clear market sentiment assessment

When referencing information in your summary:
- Use direct URL links in parentheses instead of numbered source references
- Format references as: (https://example.com/page)
- Place the URL reference immediately after the relevant information
- Do not use "Source 1", "Source 2" etc. - always use the actual URLs

Market Sentiment Assessment:
- Analyze the tone and content of news articles, analyst ratings, price forecasts, and market data
- Classify the overall market sentiment as one of: Positive, Negative, Neutral, Optimistic, Pessimistic, Cautious, or Bullish/Bearish.
- Provide reasoning for your sentiment assessment based on the gathered information
- Include this sentiment analysis in a dedicated section at the end of your summary
- Format: "**Market Sentiment Analysis:** [Brief explanation of why this sentiment based on the sources]"
- After the explanation, add a final line with just the sentiment in bold and prominent format: "**OVERALL SENTIMENT: [SENTIMENT]**"

Summary traits:
- Should be written in English
- Should answer the user's query directly
- Should be clear, easy to understand, and well-structured
- Avoid unnecessary repetition
- Consider sources and multiple perspectives
- Should be medium-length (not too short, not too long)
- Include direct URL references in parentheses for all cited information
- End with a clear market sentiment assessment

**IMPORTANT: Give your answer entirely in English and create a comprehensive summary focused on the user's query. Always reference information using direct URLs in parentheses, never use numbered source references. Always conclude with a market sentiment analysis based on all gathered information.**
"""
DEFAULT_SUMMARIZATION_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 4096,
    "response_mime_type": "text/plain",
}
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
class SummarizationAgent:
    def __init__(self, api_key: str,
                 model_name: str = "gemini-2.0-flash",
                 generation_config: dict = DEFAULT_SUMMARIZATION_CONFIG,
                 system_instruction: str = SUMMARIZATION_SYSTEM_PROMPT,
                 safety_settings: dict = DEFAULT_SAFETY_SETTINGS):
        if not api_key:
            raise ValueError("Google API key is required for SummarizationAgent.")
        
        genai.configure(api_key=api_key, transport='rest')
        
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Summarization Model: {e}") from e

    def create_comprehensive_summary(self, user_query: str, enhanced_results: list) -> tuple:
        """
        Takes enhanced search results and creates a comprehensive summary.
        Returns a tuple of (summary, usage_metadata).
        """
        if not enhanced_results:
            return "Sorry, no search results were found to summarize.", None
        
        content_for_summary = self._prepare_content_for_summary(user_query, enhanced_results)
        
        try:
            response = self.model.generate_content(content_for_summary)
            
            if response.text:
                return response.text.strip(), response.usage_metadata
            else:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return f"[Summary blocked: {response.prompt_feedback.block_reason.name}]", None
                if response.candidates:
                    return "[The model generated a response but there is no usable text content]", None
                return "[The model didn't return text content for an unknown reason]", None
                
        except Exception as e:
            return f"[Error while generating summary: {str(e)}]", None

    def _prepare_content_for_summary(self, user_query: str, enhanced_results: list) -> str:
        content = f"""
USER QUERY: "{user_query}"
EXPANDED INFORMATION FROM SEARCH RESULTS:
"""
        for i, result in enumerate(enhanced_results, 1):
            title = result.get('title', 'No title')
            expanded_info = result.get('expanded_info_gemini', 'No expanded information')
            link = result.get('link', 'No link')
            content += f"""
--- SOURCE {i} ---
Title: {title}
Link: {link}
Expanded Information: {expanded_info}
"""
        content += """
Using the information above, create a comprehensive and coherent English response to the user's query. Summarize main topics from all sources and provide a valuable general overview for the user.
"""
        return content

    def create_bullet_point_summary(self, user_query: str, enhanced_results: list) -> tuple:
        """
        Creates a bullet-point style summary of the enhanced results.
        Returns a tuple of (summary, usage_metadata).
        """
        if not enhanced_results:
            return "• No search results found to summarize.", None
        
        content = f"""
USER QUERY: "{user_query}"
INFORMATION FROM SOURCES:
"""
        for i, result in enumerate(enhanced_results, 1):
            title = result.get('title', 'No title')
            expanded_info = result.get('expanded_info_gemini', 'No expanded information')
            content += f"""
Source {i} - {title}:
{expanded_info}
"""
        content += """
Summarize the information above in bullet points as an English list. Each bullet should highlight an important point and focus on answering the user's query.
"""
        try:
            response = self.model.generate_content(content)
            return response.text.strip() if response.text else "[Could not generate bullet-point summary]", response.usage_metadata
        except Exception as e:
            return f"[Bullet-point summary error: {str(e)}]", None