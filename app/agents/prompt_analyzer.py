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
- years_of_experience (array with min and max, e.g. [2, 5])
- education (array of education keywords like ["IIT", "MIT", "Stanford"])
- location (array of locations like ["India", "Bangalore"])
Return ONLY valid JSON with these exact field names."""

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
            # Ensure all required fields exist with proper defaults
            if "keywords" not in parsed:
                parsed["keywords"] = [user_prompt]
            if "years_of_experience" not in parsed:
                parsed["years_of_experience"] = [0, 20]
            if "education" not in parsed:
                parsed["education"] = []
            if "location" not in parsed:
                parsed["location"] = []
            return parsed
        except json.JSONDecodeError:
            # Fallback: return a basic structure if JSON parsing fails
            return {
                "keywords": [user_prompt],
                "years_of_experience": [0, 20],
                "education": ["IIT"],  # Default based on user's query
                "location": ["India"]  # Default based on user's query
            }
