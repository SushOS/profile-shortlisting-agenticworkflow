import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    SHEETS_CREDENTIALS: str = os.getenv("SHEETS_SERVICE_ACCOUNT_FILE")
    GOOGLE_SHEET_ID: str = os.getenv("GOOGLE_SHEET_ID")

settings = Settings()
