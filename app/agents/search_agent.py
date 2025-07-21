from app.tools.serpapi_client import LinkedInSearch

class SearchAgent:
    """
    Builds Google-style queries and fetches up to 100 profile snippets per query[2].
    """

    def __init__(self):
        self.client = LinkedInSearch()

    def run(self, spec: dict) -> list[dict]:
        keywords = spec.get("keywords", "")
        loc = " ".join(spec.get("location", []))
        edu = " ".join(spec.get("education", []))
        query = f'{keywords} {edu} "{loc}" linkedin'
        return self.client.query(query)