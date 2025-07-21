import pandas as pd
from app.tools.xai_scoring import explain_score

class ScoringAgent:
    """
    Adds boolean feature columns & applies XAI scoring[7].
    """

    def __init__(self, spec: dict):
        self.spec = spec

    def _feature_flags(self, row: pd.Series) -> pd.Series:
        # Safely get snippet and headline with fallback to empty string
        snippet = str(row.get("snippet", "")).lower()
        headline = str(row.get("headline", "")).lower()
        
        college_match = (len(self.spec.get("education", [])) > 0 and 
                         any(k.lower() in snippet for k in self.spec["education"]))
        industry_match = any(k.lower() in headline for k in ["tech", "software", "it"])
        loc_match = (len(self.spec.get("location", [])) > 0 and 
                     self.spec["location"][0].lower() in snippet)
        exp_match = (len(self.spec.get("years_of_experience", [])) >= 2 and 
                     self.spec["years_of_experience"][0] is not None and 
                     self.spec["years_of_experience"][1] is not None and
                     any(str(y) in snippet for y in range(
                         self.spec["years_of_experience"][0],
                         self.spec["years_of_experience"][1] + 1)))

        row["college_match"] = college_match
        row["industry_match"] = industry_match
        row["location_match"] = loc_match
        row["exp_years_match"] = exp_match
        return row

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(self._feature_flags, axis=1)
        df[["score", "category_wise_score", "strengths", "weaknesses", "recommendation"]] = df.apply(
            lambda r: pd.Series(explain_score(r)), axis=1
        )
        return df.sort_values("score", ascending=False).head(20)
