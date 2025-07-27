# import re
# import json
# import google.generativeai as genai
# from app.config import settings

# genai.configure(api_key=settings.GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-2.5-pro")  # Gemini 2.5

# class PromptAnalyzerAgent:
#     """
#     Parses natural-language requirement into a structured search spec.
#     Uses CoT prompting for transparent reasoning[13].
#     """

#     SYS_PROMPT = """You are an expert recruiter assistant. 
# Break the user's requirement into:
# - keywords (array of strings)
# - years_of_experience (array with min and max, e.g. [2, 5])
# - education (array of education keywords like ["IIT", "MIT", "Stanford"])
# - location (array of locations like ["India", "Bangalore"])
# Return ONLY valid JSON with these exact field names."""

#     def run(self, user_prompt: str) -> dict:
#         response = model.generate_content(
#             f"{self.SYS_PROMPT}\n\nUser request: {user_prompt}"
#         )
#         # Gemini returns JSON dictionary if the prompt is clear
#         json_text = response.text.strip("` ")
#         # Remove any markdown json code block markers
#         if json_text.startswith("json\n"):
#             json_text = json_text[5:]
#         try:
#             parsed = json.loads(json_text)
#             # Ensure all required fields exist with proper defaults
#             if "keywords" not in parsed:
#                 parsed["keywords"] = [user_prompt]
#             if "years_of_experience" not in parsed:
#                 parsed["years_of_experience"] = [0, 20]
#             if "education" not in parsed:
#                 parsed["education"] = []
#             if "location" not in parsed:
#                 parsed["location"] = []
#             return parsed
#         except json.JSONDecodeError:
#             # Fallback: return a basic structure if JSON parsing fails
#             return {
#                 "keywords": [user_prompt],
#                 "years_of_experience": [0, 20],
#                 "education": ["IIT"],  # Default based on user's query
#                 "location": ["India"]  # Default based on user's query
#             }


import re
import json
import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")  # Gemini 2.5

class PromptAnalyzerAgent:
    """
    Parses natural-language requirement into a structured search spec.
    Uses CoT prompting for transparent reasoning[13].
    """

    SYS_PROMPT = """You are an expert recruiter assistant. 
Break the user's requirement into:
- keywords (array of strings)
- years_of_experience (array with min and max, e.g. [2, 5]. If not specified, use [0, 20])
- education (array of education keywords like ["IIT", "MIT", "Stanford"])
- location (array of locations like ["India", "Bangalore"])
Return ONLY valid JSON with these exact field names. NEVER use null values - always provide arrays."""

    def run(self, user_prompt: str) -> dict:
        response = model.generate_content(
            f"{self.SYS_PROMPT}\n\nUser request: {user_prompt}"
        )
        # Gemini returns JSON dictionary if the prompt is clear
        json_text = response.text.strip("` ")
        # Remove any markdown json code block markers
        if json_text.startswith("json\n"):
            json_text = json_text[5:]
        try:
            parsed = json.loads(json_text)
            
            # Ensure all required fields exist with proper defaults and no None values
            if "keywords" not in parsed or parsed["keywords"] is None:
                parsed["keywords"] = self._extract_keywords_fallback(user_prompt)
            if "years_of_experience" not in parsed or parsed["years_of_experience"] is None:
                parsed["years_of_experience"] = [0, 20]  # Default range
            if "education" not in parsed or parsed["education"] is None:
                parsed["education"] = []
            if "location" not in parsed or parsed["location"] is None:
                parsed["location"] = []
            
            # Additional validation to ensure years_of_experience is a proper list
            if not isinstance(parsed["years_of_experience"], list):
                parsed["years_of_experience"] = [0, 20]
            elif len(parsed["years_of_experience"]) != 2:
                parsed["years_of_experience"] = [0, 20]
            elif any(x is None for x in parsed["years_of_experience"]):
                parsed["years_of_experience"] = [0, 20]
                
            return parsed
        except json.JSONDecodeError:
            # Fallback: return a basic structure if JSON parsing fails
            return {
                "keywords": self._extract_keywords_fallback(user_prompt),
                "years_of_experience": [0, 20],
                "education": self._extract_education_fallback(user_prompt),
                "location": self._extract_location_fallback(user_prompt)
            }
    
    def _extract_keywords_fallback(self, text: str) -> list:
        """Extract keywords using basic pattern matching as fallback"""
        keywords = []
        text_lower = text.lower()
        
        # Technical terms
        tech_terms = ["ai", "ml", "machine learning", "deep learning", "robotics", "cse", "btech", "masters", "phd"]
        for term in tech_terms:
            if term in text_lower:
                keywords.append(term)
        
        # If no tech terms found, use the full text as keywords
        if not keywords:
            keywords = [word for word in text.split() if len(word) > 2][:5]  # Take first 5 meaningful words
            
        return keywords
    
    def _extract_education_fallback(self, text: str) -> list:
        """Extract education keywords as fallback"""
        education = []
        text_lower = text.lower()
        
        education_terms = ["iit", "mit", "stanford", "harvard", "berkeley", "cmu", "iisc"]
        for term in education_terms:
            if term in text_lower:
                education.append(term.upper())
                
        return education
    
    def _extract_location_fallback(self, text: str) -> list:
        """Extract location keywords as fallback"""
        locations = []
        text_lower = text.lower()
        
        location_terms = ["usa", "india", "bangalore", "mumbai", "delhi", "hyderabad", "silicon valley", "california"]
        for term in location_terms:
            if term in text_lower:
                locations.append(term.title())
                
        return locations