import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from google_search.gemini_api import GeminiAPI
from google_search.summarization_agent import SummarizationAgent
load_dotenv()
serpapi_key = os.getenv("SERP_API_KEY")

class SearchAPI:
    def __init__(self, serpapi_key: str, gemini_processor: GeminiAPI, summarizer: SummarizationAgent = None):
        if not serpapi_key:
            raise ValueError("SerpAPI key is required for SearchAPI.")
        if not gemini_processor:
            raise ValueError("GeminiAPI instance is required for SearchAPI.")
        self.serpapi_key = serpapi_key
        self.gemini_processor = gemini_processor
        self.summarizer = summarizer
    
    def _search_google_serp(self, query: str, location: str = "United States"):
        search_params = {
            "q": query,
            "location": location,
            "api_key": self.serpapi_key,
            "engine": "google",
        }
        try:
            search = GoogleSearch(search_params)
            results_dict = search.get_dict()
            return results_dict
        except Exception as e:
            return {"error": f"SerpAPI search failed: {str(e)}"}
    
    def _parse_serp_results(self, results_dict: dict):
        entries = []
        if 'error' in results_dict:
            return entries
        
        for result in results_dict.get('organic_results', []):
            title = result.get('title')
            snippet = result.get('snippet')
            link = result.get('link')
            if title and snippet and link:
                entries.append({'title': title, 'snippet': snippet, 'link': link})
        return entries
    
    def get_enhanced_search_results(self, user_query: str):
        """
        Gets search results from SerpAPI and enhances their snippets using GeminiAPI.
        Returns a dictionary with "error", "results", and "token_usage".
        """
        enhanced_results_list = []
        token_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        serp_results_data = self._search_google_serp(user_query)
    
        if 'error' in serp_results_data:
            return {"error": serp_results_data['error'], "results": [], "token_usage": token_usage}
        
        parsed_entries = self._parse_serp_results(serp_results_data)
        if not parsed_entries:
            return {"error": "No organic search results found or parsed.", "results": [], "token_usage": token_usage}
        
        for entry in parsed_entries[:5]:  # Limit to top 5 results
            title = entry['title']
            original_snippet = entry['snippet']
            link = entry['link']
            
            expanded_info, usage_metadata = self.gemini_processor.get_expanded_information(
                user_query, title, original_snippet, link
            )
            
            enhanced_results_list.append({
                "title": title,
                "original_snippet": original_snippet,
                "link": link,
                "expanded_info_gemini": expanded_info
            })
            
            if usage_metadata:
                token_usage["prompt_tokens"] += usage_metadata.prompt_token_count
                token_usage["output_tokens"] += usage_metadata.candidates_token_count
                token_usage["total_tokens"] += usage_metadata.total_token_count
                
        return {"error": None, "results": enhanced_results_list, "token_usage": token_usage}
        
    def get_enhanced_search_results_with_summary(self, user_query: str, include_summary: bool = True):
        """
        Gets search results from SerpAPI, enhances their snippets using GeminiAPI,
        and optionally creates a comprehensive summary.
        Returns a dictionary with "error", "results", "summary", and "token_usage".
        """
        enhanced_data = self.get_enhanced_search_results(user_query)
        token_usage = enhanced_data.get("token_usage", {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        
        if enhanced_data["error"]:
            return {
                "error": enhanced_data["error"], 
                "results": [], 
                "summary": None,
                "bullet_summary": None,
                "token_usage": token_usage
            }
        
        enhanced_results = enhanced_data["results"]
        summary = None
        bullet_summary = None
        
        if include_summary and self.summarizer and enhanced_results:
            try:
                summary, summary_usage = self.summarizer.create_comprehensive_summary(user_query, enhanced_results)
                bullet_summary, bullet_usage = self.summarizer.create_bullet_point_summary(user_query, enhanced_results)
                
                if summary_usage:
                    token_usage["prompt_tokens"] += summary_usage.prompt_token_count
                    token_usage["output_tokens"] += summary_usage.candidates_token_count
                    token_usage["total_tokens"] += summary_usage.total_token_count
                if bullet_usage:
                    token_usage["prompt_tokens"] += bullet_usage.prompt_token_count
                    token_usage["output_tokens"] += bullet_usage.candidates_token_count
                    token_usage["total_tokens"] += bullet_usage.total_token_count
                    
            except Exception as e:
                summary = f"[Error while generating summary: {str(e)}]"
                bullet_summary = f"[Bullet point summary error: {str(e)}]"
        
        return {
            "error": None,
            "results": enhanced_results,
            "summary": summary,
            "bullet_summary": bullet_summary,
            "total_results": len(enhanced_results),
            "token_usage": token_usage
        }

# Example usage and testing
if __name__ == "__main__":
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        gemini_processor = GeminiAPI(google_api_key)
        summarizer = SummarizationAgent(google_api_key)
        
        if not serpapi_key:
            raise ValueError("SERPAPI_KEY not found in environment variables")
        
        search_api = SearchAPI(serpapi_key, gemini_processor, summarizer)
        
        query = input("Enter your search query: ")
        if not query:
            raise ValueError("Search query cannot be empty.")
        print(f"Searching for: {query}")
        print("=" * 50)
        
        results = search_api.get_enhanced_search_results_with_summary(query)
        
        if results["error"]:
            print(f"Error: {results['error']}")
        else:
            print(f"Found {results['total_results']} results")
            print(f"Token Usage: {results['token_usage']}\n")
            
            if results["summary"]:
                print("🔍 COMPREHENSIVE SUMMARY:")
                print("-" * 30)
                print(results["summary"])
                print("\n")
            
            if results["bullet_summary"]:
                print("📋 BULLET POINT SUMMARY:")
                print("-" * 30)
                print(results["bullet_summary"])
                print("\n")
            
            print("📚 DETAILED RESULTS:")
            print("-" * 30)
            for i, result in enumerate(results['results'], 1):
                print(f"\n{i}. {result['title']}")
                print(f"   🔗 Link: {result['link']}")
                print(f"   📝 Original: {result['original_snippet']}")
                print(f"   ✨ Expanded: {result['expanded_info_gemini']}")
                print("-" * 50)
                
    except Exception as e:
        print(f"Error running search API: {e}")