import pandas as pd
from tqdm import tqdm
from app.tools.email_extractor import extract_emails

class EnrichAgent:
    """
    Normalises raw snippets into a DataFrame & extracts email addresses.
    """

    def run(self, snippets: list[dict]) -> pd.DataFrame:
        rows = []
        for snip in tqdm(snippets, desc="Enriching"):
            title = snip.get("title", "")
            link = snip.get("link", "")
            snippet = snip.get("snippet", "")
            rows.append(
                {
                    "name": title.split(" - ")[0],
                    "headline": title,
                    "profile_url": link,
                    "snippet": snippet,
                    "emails": ", ".join(extract_emails(snippet))
                }
            )
        return pd.DataFrame(rows)
