import serpapi
from app.config import settings

class LinkedInSearch:
    """
    Uses SerpAPI to fetch public LinkedIn profile snippets ethically.
    Only publicly indexed data is accessed, aligning with recent US rulings that
    permit scraping of public pages[20].
    """

    def __init__(self):
        self.params = {
            "engine": "google",
            "api_key": settings.SERPAPI_KEY,
            "num": 100,
            "q": "",
            "hl": "en"
        }

    def query(self, google_query: str) -> list[dict]:
        self.params["q"] = google_query + " site:linkedin.com/in/"
        search = serpapi.search(self.params)
        return search.get("organic_results", [])
