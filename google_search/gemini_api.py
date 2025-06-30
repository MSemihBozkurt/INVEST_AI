import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Default configurations
DEFAULT_MODEL_NAME = "gemini-2.0-flash"

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.90,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in content development for a search engine.
Your primary function is to expand short search result snippets to provide users with more comprehensive information.

You will receive the following details for a specific search result:
- `USER_QUERY`: The original query submitted by the user.
- `TITLE`: The title of the search result.
- `ORIGINAL_SNIPPET`: The short text snippet provided by the search API.
- `LINK`: The URL of the search result.

Based on these inputs, your task is to:
1. Generate a detailed and informative paragraph (or several concise paragraphs if necessary) that significantly expands the `ORIGINAL_SNIPPET`.
2. Use the content of the `TITLE` and `ORIGINAL_SNIPPET` as the foundation for expansion.
3. Elaborate on the main points mentioned in the `ORIGINAL_SNIPPET`, providing additional context, explanations, and relevant details that help the user better understand the topic of the linked page.
4. The goal is to "augment the information" presented to the user, making the snippet far more useful than its original brief version.
5. Ensure your response is factual, directly relevant to the provided inputs, and maintains a neutral, informative tone.
6. Do not simply rephrase or repeat the `ORIGINAL_SNIPPET`. Add significant new value and depth.
7. Do not fabricate information that cannot be reasonably inferred from the given title and snippet. If the provided information is too limited for meaningful expansion, acknowledge it, but always try your best.

**IMPORTANT: Provide all your responses in English. Always present the expanded information in English.**
"""

DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

class GeminiAPI:
    def __init__(self, api_key: str,
                 model_name: str = DEFAULT_MODEL_NAME,
                 generation_config: dict = DEFAULT_GENERATION_CONFIG,
                 system_instruction: str = DEFAULT_SYSTEM_PROMPT,
                 safety_settings: dict = DEFAULT_SAFETY_SETTINGS):
        if not api_key:
            raise ValueError("Google API key is required for GeminiAPI.")
        
        genai.configure(api_key=api_key)
        
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name or DEFAULT_MODEL_NAME,
                generation_config=generation_config or DEFAULT_GENERATION_CONFIG,
                system_instruction=system_instruction or DEFAULT_SYSTEM_PROMPT,
                safety_settings=safety_settings or DEFAULT_SAFETY_SETTINGS
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini Model: {e}") from e

    def get_expanded_information(self, user_query: str, title: str, original_snippet: str, link: str) -> tuple:
        """
        Sends the search result details to the Gemini LLM to get an expanded version.
        Returns a tuple of (expanded_text, usage_metadata).
        """
        user_prompt_for_llm = f"""
        USER_QUERY: "{user_query}"
        TITLE: "{title}"
        ORIGINAL_SNIPPET: "{original_snippet}"
        LINK: "{link}"

        Based on the system instructions provided to you, please generate an expanded and more informative version of the original snippet.
"""
        try:
            response = self.model.generate_content(user_prompt_for_llm)
            
            if response.text:
                return response.text.strip(), response.usage_metadata
            else:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return f"[LLM Output Blocked: {response.prompt_feedback.block_reason.name}]", None
                if response.candidates:
                    return "[LLM returned candidates but no usable text content]", None
                return "[LLM did not return text content for an unknown reason]", None
                
        except Exception as e:
            return f"[Error generating expanded information from LLM: {str(e)}]", None
