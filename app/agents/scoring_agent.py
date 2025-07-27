# import pandas as pd
# from app.tools.xai_scoring import explain_score

# class ScoringAgent:
#     """
#     Adds boolean feature columns & applies XAI scoring[7].
#     """

#     def __init__(self, spec: dict):
#         self.spec = spec

#     def _feature_flags(self, row: pd.Series) -> pd.Series:
#         # Safely get snippet and headline with fallback to empty string
#         snippet = str(row.get("snippet", "")).lower()
#         headline = str(row.get("headline", "")).lower()
        
#         college_match = (len(self.spec.get("education", [])) > 0 and 
#                          any(k.lower() in snippet for k in self.spec["education"]))
#         industry_match = any(k.lower() in headline for k in ["tech", "software", "it"])
#         loc_match = (len(self.spec.get("location", [])) > 0 and 
#                      self.spec["location"][0].lower() in snippet)
#         exp_match = (len(self.spec.get("years_of_experience", [])) >= 2 and 
#                      self.spec["years_of_experience"][0] is not None and 
#                      self.spec["years_of_experience"][1] is not None and
#                      any(str(y) in snippet for y in range(
#                          self.spec["years_of_experience"][0],
#                          self.spec["years_of_experience"][1] + 1)))

#         row["college_match"] = college_match
#         row["industry_match"] = industry_match
#         row["location_match"] = loc_match
#         row["exp_years_match"] = exp_match
#         return row

#     def run(self, df: pd.DataFrame) -> pd.DataFrame:
#         df = df.apply(self._feature_flags, axis=1)
#         df[["score", "category_wise_score", "strengths", "weaknesses", "recommendation"]] = df.apply(
#             lambda r: pd.Series(explain_score(r)), axis=1
#         )
#         return df.sort_values("score", ascending=False).head(20)


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
        
        # Education match with safe handling
        education_list = self.spec.get("education", [])
        college_match = (
            len(education_list) > 0 and 
            any(k.lower() in snippet for k in education_list if k is not None)
        )
        
        # Industry match
        industry_match = any(k.lower() in headline for k in ["tech", "software", "it", "ai", "ml", "robotics"])
        
        # Location match with safe handling
        location_list = self.spec.get("location", [])
        loc_match = (
            len(location_list) > 0 and 
            location_list[0] is not None and
            location_list[0].lower() in snippet
        )
        
        # Experience match with robust None handling
        years_list = self.spec.get("years_of_experience", [])
        exp_match = False
        
        if (years_list and 
            len(years_list) >= 2 and 
            years_list[0] is not None and 
            years_list[1] is not None):
            try:
                min_years = int(years_list[0])
                max_years = int(years_list[1])
                exp_match = any(
                    str(y) in snippet 
                    for y in range(min_years, min(max_years + 1, 21))  # Cap at 20 years for performance
                )
            except (ValueError, TypeError):
                exp_match = False

        row["college_match"] = college_match
        row["industry_match"] = industry_match
        row["location_match"] = loc_match
        row["exp_years_match"] = exp_match
        return row

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Apply feature flags
        df = df.apply(self._feature_flags, axis=1)
        
        # Apply scoring with error handling
        try:
            scoring_results = df.apply(
                lambda r: pd.Series(explain_score(r)), axis=1
            )
            df[["score", "category_wise_score", "strengths", "weaknesses", "recommendation"]] = scoring_results
        except Exception as e:
            print(f"Scoring error: {e}")
            # Create default scores if scoring fails
            df["score"] = 3  # Default middle score
            df["category_wise_score"] = "Default scoring applied"
            df["strengths"] = "Profile contains relevant information"
            df["weaknesses"] = "Detailed analysis not available"
            df["recommendation"] = "Manual review recommended"
        
        return df.sort_values("score", ascending=False).head(20)