import pandas as pd

FEATURE_WEIGHTS = {
    "college_match": 4,
    "exp_years_match": 3,
    "industry_match": 2,
    "location_match": 1,
}

def explain_score(row: pd.Series) -> tuple[int, str, str, str, str]:
    score = 0
    reasons = []
    strengths = []
    weaknesses = []
    
    # Calculate score and basic reasoning
    for feature, weight in FEATURE_WEIGHTS.items():
        if row.get(feature):
            score += weight
            reasons.append(f"{feature.replace('_', ' ')} (+{weight})")
    
    # Generate detailed strengths
    if row.get("college_match"):
        strengths.append("Elite Educational Pedigree: Graduating from a top-tier institution (IIT) demonstrates exceptional academic capabilities and technical foundation essential for advanced AI/ML roles.")
    
    if row.get("industry_match"):
        strengths.append("Relevant Industry Experience: Background in technology sector provides domain expertise and understanding of industry challenges and requirements.")
    
    if row.get("location_match"):
        strengths.append("Geographic Alignment: Located in target region, ensuring easier coordination, cultural fit, and reduced relocation complexities.")
    
    if row.get("exp_years_match"):
        strengths.append("Experience Level Match: Years of experience align with role requirements, indicating appropriate seniority and skill development.")
    
    # Generate detailed weaknesses
    if not row.get("college_match"):
        weaknesses.append("Educational Background: May lack the rigorous technical foundation typically expected from top-tier institutions for advanced AI/ML positions.")
    
    if not row.get("industry_match"):
        weaknesses.append("Industry Experience Gap: Limited exposure to technology sector may require additional time to understand domain-specific challenges and requirements.")
    
    if not row.get("location_match"):
        weaknesses.append("Geographic Mismatch: Location differences may pose coordination challenges and potential relocation requirements.")
    
    if not row.get("exp_years_match"):
        weaknesses.append("Experience Level Mismatch: Years of experience may not align optimally with the specific requirements and expectations of this role.")
    
    # Generate recommendation
    if score >= 7:
        recommendation = "High Priority: This candidate demonstrates exceptional qualifications and should be prioritized for immediate outreach. The combination of strong educational background and relevant experience makes them an ideal fit for the role."
    elif score >= 5:
        recommendation = "Medium Priority: Strong candidate with good potential. Recommend detailed evaluation of specific projects and technical depth before proceeding with next steps."
    elif score >= 3:
        recommendation = "Low Priority: Some relevant qualifications but may require significant evaluation to determine fit. Consider as backup option if primary candidates are unavailable."
    else:
        recommendation = "Not Recommended: Limited alignment with role requirements. Significant gaps in key qualifications make this candidate unsuitable for the current position."
    
    # Compile explanations
    basic_explanation = "; ".join(reasons) if reasons else "No strong match"
    strength_text = " | ".join(strengths) if strengths else "No significant strengths identified from available profile information."
    weakness_text = " | ".join(weaknesses) if weaknesses else "No significant weaknesses identified from available profile information."
    
    return score, basic_explanation, strength_text, weakness_text, recommendation
