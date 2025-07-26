# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     SERPAPI_KEY: str = os.getenv("SERPAPI_KEY")
#     GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
#     SHEETS_CREDENTIALS: str = os.getenv("SHEETS_SERVICE_ACCOUNT_FILE")
#     GOOGLE_SHEET_ID: str = os.getenv("GOOGLE_SHEET_ID")

# settings = Settings()


import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    SHEETS_CREDENTIALS: str = os.getenv("SHEETS_SERVICE_ACCOUNT_FILE")
    GOOGLE_SHEET_ID: str = os.getenv("GOOGLE_SHEET_ID")
    
    # Conditional Logic Parameters
    MIN_SEARCH_RESULTS: int = int(os.getenv("MIN_SEARCH_RESULTS", "10"))
    MIN_QUALITY_RATIO: float = float(os.getenv("MIN_QUALITY_RATIO", "0.7"))
    MAX_RETRY_ATTEMPTS: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    
    # Scoring Thresholds
    HIGH_SCORE_THRESHOLD: int = int(os.getenv("HIGH_SCORE_THRESHOLD", "7"))
    MEDIUM_SCORE_THRESHOLD: int = int(os.getenv("MEDIUM_SCORE_THRESHOLD", "5"))
    MIN_HIGH_SCORE_CANDIDATES: int = int(os.getenv("MIN_HIGH_SCORE_CANDIDATES", "5"))
    
    # Processing Modes
    ENABLE_SENIOR_SCORING: bool = os.getenv("ENABLE_SENIOR_SCORING", "true").lower() == "true"
    ENABLE_AI_SCORING: bool = os.getenv("ENABLE_AI_SCORING", "true").lower() == "true"
    ENABLE_QUALITY_FILTERING: bool = os.getenv("ENABLE_QUALITY_FILTERING", "true").lower() == "true"

settings = Settings()
