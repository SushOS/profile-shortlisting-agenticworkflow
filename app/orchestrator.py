from langgraph.graph import StateGraph
from typing import TypedDict, Any, Optional, Dict, List
import pandas as pd
from datetime import datetime
import logging
from app.agents.prompt_analyzer import PromptAnalyzerAgent
from app.agents.search_agent import SearchAgent
from app.agents.enrich_agent import EnrichAgent
from app.agents.scoring_agent import ScoringAgent
from app.agents.sheet_writer import SheetWriterAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(TypedDict):
    user_prompt: str
    spec: dict
    snippets: list
    df: Any
    scores: Any
    result: str
    # New fields for conditional logic
    error: Optional[Dict]
    quality_metrics: Optional[Dict]
    routing_history: List[str]
    retry_count: int
    processing_mode: str
    reprocess_count: int  # Add counter to prevent infinite loops

class Orchestration:
    """
    Enhanced orchestration with conditional edges and intelligent routing.
    """

    def __init__(self):
        self.graph = StateGraph(State)
        
        # Core nodes
        self.graph.add_node("analyze", self._analyze_node)
        self.graph.add_node("search", self._search_node)
        self.graph.add_node("enrich", self._enrich_node)
        self.graph.add_node("score", self._score_node)
        self.graph.add_node("sheet", self._sheet_node)
        
        # Conditional nodes
        self.graph.add_node("retry_search", self._retry_search_node)
        self.graph.add_node("manual_review", self._manual_review_node)
        self.graph.add_node("quality_filter", self._quality_filter_node)
        self.graph.add_node("contact_enrichment", self._contact_enrichment_node)
        self.graph.add_node("senior_scoring", self._senior_scoring_node)
        self.graph.add_node("ai_scoring", self._ai_scoring_node)
        self.graph.add_node("wait_and_retry", self._wait_and_retry_node)
        self.graph.add_node("fallback_search", self._fallback_search_node)

        # Sequential start
        self.graph.add_edge("analyze", "search")
        
        # Conditional routing after search
        self.graph.add_conditional_edges(
            "search",
            self._search_quality_router,
            {
                "sufficient": "enrich",
                "insufficient": "retry_search",
                "no_results": "fallback_search",
                "error": "wait_and_retry"
            }
        )
        
        # Retry and fallback paths
        self.graph.add_edge("retry_search", "enrich")
        self.graph.add_edge("wait_and_retry", "search")
        self.graph.add_edge("fallback_search", "manual_review")
        
        # Conditional routing after enrichment
        self.graph.add_conditional_edges(
            "enrich",
            self._enrichment_quality_router,
            {
                "good_quality": "score",
                "needs_contact_info": "contact_enrichment",
                "poor_quality": "quality_filter"
            }
        )
        
        self.graph.add_edge("contact_enrichment", "score")
        self.graph.add_edge("quality_filter", "score")
        
        # Role-based scoring routing
        self.graph.add_conditional_edges(
            "score",
            self._scoring_strategy_router,
            {
                "senior_role": "senior_scoring",
                "ai_role": "ai_scoring",
                "standard": "sheet"
            }
        )
        
        # Final validation routing with loop prevention
        self.graph.add_conditional_edges(
            "senior_scoring",
            self._final_validation_router,
            {
                "high_quality": "sheet",
                "needs_review": "manual_review",
                "reprocess": "quality_filter",
                "force_complete": "sheet"  # Add forced completion
            }
        )
        
        self.graph.add_conditional_edges(
            "ai_scoring",
            self._final_validation_router,
            {
                "high_quality": "sheet",
                "needs_review": "manual_review",
                "reprocess": "quality_filter",
                "force_complete": "sheet"  # Add forced completion
            }
        )
        
        # Set finish points correctly
        self.graph.add_edge("sheet", "__end__")
        self.graph.add_edge("manual_review", "__end__")
        
        self.graph.set_entry_point("analyze")
        
        # Compile without config parameter - LangGraph handles recursion internally
        self.app = self.graph.compile()

    # ============ ROUTER FUNCTIONS ============
    
    def _search_quality_router(self, state: State) -> str:
        """Route based on search result quality and errors"""
        self._log_routing_decision("search_quality_router", state)
        
        # Check for errors first
        if state.get("error"):
            return "error"
            
        snippets = state.get("snippets", [])
        
        if len(snippets) == 0:
            logger.warning("No search results found")
            return "no_results"
        elif len(snippets) < 10:
            logger.info(f"Insufficient results ({len(snippets)}), will retry")
            return "insufficient"
        else:
            # Calculate quality metrics
            quality_metrics = self._calculate_search_quality(snippets)
            state["quality_metrics"] = quality_metrics
            
            if quality_metrics["valid_profile_ratio"] > 0.7:
                logger.info("Search quality sufficient, proceeding to enrichment")
                return "sufficient"
            else:
                logger.info("Search quality insufficient, retrying with broader parameters")
                return "insufficient"

    def _enrichment_quality_router(self, state: State) -> str:
        """Route based on data enrichment quality"""
        self._log_routing_decision("enrichment_quality_router", state)
        
        df = state.get("df")
        if df is None or df.empty:
            return "poor_quality"
        
        # Calculate enrichment quality metrics
        total_rows = len(df)
        missing_emails = df["emails"].isna().sum() + (df["emails"] == "").sum()
        short_snippets = (df["snippet"].str.len() < 100).sum()
        
        email_ratio = (total_rows - missing_emails) / total_rows
        content_ratio = (total_rows - short_snippets) / total_rows
        
        if email_ratio < 0.3:
            logger.info("Low email extraction rate, enhancing contact information")
            return "needs_contact_info"
        elif content_ratio < 0.5:
            logger.info("Poor content quality, applying quality filters")
            return "poor_quality"
        else:
            logger.info("Data quality acceptable, proceeding to scoring")
            return "good_quality"

    def _scoring_strategy_router(self, state: State) -> str:
        """Route to different scoring strategies based on role requirements"""
        self._log_routing_decision("scoring_strategy_router", state)
        
        spec = state["spec"]
        keywords = " ".join(spec.get("keywords", [])).lower()
        
        if any(term in keywords for term in ["senior", "lead", "principal", "architect"]):
            logger.info("Routing to senior role scoring strategy")
            return "senior_role"
        elif any(term in keywords for term in ["ml", "ai", "machine learning", "artificial intelligence", "data scientist"]):
            logger.info("Routing to AI/ML specialized scoring strategy")
            return "ai_role"
        else:
            logger.info("Using standard scoring strategy")
            return "standard"

    def _final_validation_router(self, state: State) -> str:
        """Route based on final scoring results with loop prevention"""
        self._log_routing_decision("final_validation_router", state)
        
        # Check reprocess count to prevent infinite loops
        reprocess_count = state.get("reprocess_count", 0)
        if reprocess_count >= 2:
            logger.warning(f"Max reprocess attempts ({reprocess_count}) reached, forcing completion")
            return "force_complete"
        
        df = state.get("scores")
        if df is None or df.empty:
            if reprocess_count == 0:
                state["reprocess_count"] = reprocess_count + 1
                return "reprocess"
            else:
                return "needs_review"
        
        # Analyze score distribution
        high_scores = len(df[df["score"] >= 7]) if "score" in df.columns else 0
        medium_scores = len(df[df["score"] >= 5]) if "score" in df.columns else 0
        total_candidates = len(df)
        
        if high_scores >= 3:  # Lowered threshold from 5 to 3
            logger.info(f"Found {high_scores} high-quality candidates, finalizing results")
            return "high_quality"
        elif medium_scores >= 2 and total_candidates >= 5:  # Lowered thresholds
            logger.info("Moderate quality candidates found, sending for review")
            return "needs_review"
        elif reprocess_count == 0:
            logger.warning("Insufficient quality candidates, reprocessing once")
            state["reprocess_count"] = reprocess_count + 1
            return "reprocess"
        else:
            logger.warning("Reprocess limit reached, completing with available candidates")
            return "high_quality"  # Force completion

    # ============ CORE NODE FUNCTIONS ============

    def _analyze_node(self, state: State) -> State:
        """Enhanced analysis with error handling"""
        try:
            spec = PromptAnalyzerAgent().run(state["user_prompt"])
            state["spec"] = spec
            state["routing_history"] = ["analyze"]
            state["retry_count"] = 0
            state["reprocess_count"] = 0  # Initialize reprocess counter
            state["processing_mode"] = "standard"
            logger.info(f"Analysis completed: {spec}")
        except Exception as e:
            state["error"] = {"type": "analysis_error", "message": str(e)}
            logger.error(f"Analysis failed: {e}")
        return state

    def _search_node(self, state: State) -> State:
        """Enhanced search with error handling"""
        try:
            snippets = SearchAgent().run(state["spec"])
            state["snippets"] = snippets
            state["routing_history"].append("search")
            logger.info(f"Search completed: {len(snippets)} results found")
            
            # Clear any previous errors
            if "error" in state:
                del state["error"]
                
        except Exception as e:
            state["error"] = {"type": "search_error", "message": str(e)}
            logger.error(f"Search failed: {e}")
        return state

    def _enrich_node(self, state: State) -> State:
        """Enhanced enrichment with quality tracking"""
        try:
            df = EnrichAgent().run(state["snippets"])
            state["df"] = df
            state["routing_history"].append("enrich")
            logger.info(f"Enrichment completed: {len(df)} profiles processed")
        except Exception as e:
            state["error"] = {"type": "enrichment_error", "message": str(e)}
            logger.error(f"Enrichment failed: {e}")
        return state

    def _score_node(self, state: State) -> State:
        """Standard scoring node"""
        try:
            scores = ScoringAgent(state["spec"]).run(state["df"])
            state["scores"] = scores
            state["routing_history"].append("score")
            logger.info(f"Scoring completed: {len(scores)} candidates scored")
        except Exception as e:
            state["error"] = {"type": "scoring_error", "message": str(e)}
            logger.error(f"Scoring failed: {e}")
        return state

    def _sheet_node(self, state: State) -> State:
        """Final output with success tracking"""
        try:
            scores = state.get("scores")
            if scores is None or scores.empty:
                # Create a minimal result if no scores available
                state["result"] = "No candidates found matching the criteria"
                logger.warning("No scores available, creating minimal result")
            else:
                result = SheetWriterAgent().run(scores)
                state["result"] = result
                logger.info("Results successfully written to output")
            
            state["routing_history"].append("sheet")
        except Exception as e:
            state["error"] = {"type": "output_error", "message": str(e)}
            logger.error(f"Output failed: {e}")
            # Still mark as completed to prevent loops
            state["result"] = f"Output generation failed: {str(e)}"
            state["routing_history"].append("sheet")
        return state

    # ============ CONDITIONAL NODE FUNCTIONS ============

    def _retry_search_node(self, state: State) -> State:
        """Retry search with broader parameters"""
        try:
            spec = state["spec"].copy()
            state["retry_count"] += 1
            
            # Broaden search criteria progressively
            if state["retry_count"] == 1:
                # First retry: remove location constraint
                if spec.get("location"):
                    spec["location"] = []
                    logger.info("Retry 1: Removed location constraints")
            elif state["retry_count"] == 2:
                # Second retry: broaden education requirements
                if len(spec.get("education", [])) > 1:
                    spec["education"] = spec["education"][:1]
                    logger.info("Retry 2: Reduced education filters")
            else:
                # Final retry: use only core keywords
                spec["education"] = []
                spec["location"] = []
                logger.info("Final retry: Using only core keywords")
            
            # Update state with modified spec
            state["spec"] = spec
            snippets = SearchAgent().run(spec)
            state["snippets"] = snippets
            state["routing_history"].append("retry_search")
            logger.info(f"Retry search completed: {len(snippets)} results")
            
        except Exception as e:
            state["error"] = {"type": "retry_search_error", "message": str(e)}
            logger.error(f"Retry search failed: {e}")
        return state

    def _quality_filter_node(self, state: State) -> State:
        """Apply advanced quality filtering"""
        try:
            df = state["df"].copy()
            initial_count = len(df)
            
            # Apply progressive quality filters
            df_filtered = df[
                (df["snippet"].str.len() > 50) &  # Minimum content requirement
                (df["headline"].str.len() > 10) &  # Valid headline
                (df["profile_url"].str.contains("linkedin.com", na=False))  # Valid LinkedIn URL
            ]
            
            # If too much data lost, apply lighter filters
            if len(df_filtered) < initial_count * 0.3:
                df_filtered = df[df["snippet"].str.len() > 20]
                logger.warning("Applied lighter quality filters to preserve data")
            
            state["df"] = df_filtered
            state["routing_history"].append("quality_filter")
            logger.info(f"Quality filtering: {initial_count} -> {len(df_filtered)} profiles")
            
        except Exception as e:
            state["error"] = {"type": "quality_filter_error", "message": str(e)}
            logger.error(f"Quality filtering failed: {e}")
        return state

    def _contact_enrichment_node(self, state: State) -> State:
        """Enhanced contact information extraction"""
        try:
            df = state["df"].copy()
            
            # Simple email extraction without external dependency
            import re
            
            def enhanced_email_extraction(text):
                # Try multiple email patterns
                patterns = [
                    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                    r"[a-zA-Z0-9_.+-]+\s*\[\s*at\s*\]\s*[a-zA-Z0-9-]+\s*\[\s*dot\s*\]\s*[a-zA-Z0-9-.]+",
                    r"[a-zA-Z0-9_.+-]+\s*@\s*[a-zA-Z0-9-]+\s*\.\s*[a-zA-Z0-9-.]+"
                ]
                
                emails = set()
                for pattern in patterns:
                    found = re.findall(pattern, str(text), re.IGNORECASE)
                    emails.update(found)
                return list(emails)
            
            # Re-extract emails with enhanced patterns
            df["emails"] = df["snippet"].apply(
                lambda x: ", ".join(enhanced_email_extraction(str(x)))
            )
            
            state["df"] = df
            state["routing_history"].append("contact_enrichment")
            logger.info("Enhanced contact information extraction completed")
            
        except Exception as e:
            state["error"] = {"type": "contact_enrichment_error", "message": str(e)}
            logger.error(f"Contact enrichment failed: {e}")
        return state

    def _senior_scoring_node(self, state: State) -> State:
        """Specialized scoring for senior roles"""
        try:
            df = state["df"].copy()
            
            # Add leadership indicators
            def detect_leadership(row):
                text = (str(row.get("headline", "")) + " " + str(row.get("snippet", ""))).lower()
                leadership_terms = ["lead", "manager", "director", "vp", "cto", "head of", "principal", "senior", "architect"]
                return any(term in text for term in leadership_terms)
            
            df["leadership_indicators"] = df.apply(detect_leadership, axis=1)
            
            # Apply senior-specific scoring using existing XAI scoring
            from app.tools.xai_scoring import explain_score
            
            # Add the leadership flag to each row and then score
            def score_with_leadership(row):
                return pd.Series(explain_score(row))
            
            scoring_results = df.apply(score_with_leadership, axis=1)
            df[["score", "category_wise_score", "strengths", "weaknesses", "recommendation"]] = scoring_results
            
            # Boost scores for leadership indicators
            df.loc[df["leadership_indicators"], "score"] += 1
            
            state["scores"] = df.sort_values("score", ascending=False).head(20)
            state["routing_history"].append("senior_scoring")
            logger.info("Senior role scoring completed")
            
        except Exception as e:
            state["error"] = {"type": "senior_scoring_error", "message": str(e)}
            logger.error(f"Senior scoring failed: {e}")
        return state

    def _ai_scoring_node(self, state: State) -> State:
        """Specialized scoring for AI/ML roles"""
        try:
            df = state["df"].copy()
            
            # Add AI/ML specific features
            def detect_ai_skills(row):
                text = (str(row.get("headline", "")) + " " + str(row.get("snippet", ""))).lower()
                ai_terms = ["machine learning", "deep learning", "ai", "tensorflow", "pytorch", "nlp", "computer vision", "data science"]
                return sum(1 for term in ai_terms if term in text)
            
            df["ai_skill_count"] = df.apply(detect_ai_skills, axis=1)
            
            # Apply AI-specific scoring using existing XAI scoring
            from app.tools.xai_scoring import explain_score
            
            def score_with_ai_skills(row):
                return pd.Series(explain_score(row))
            
            scoring_results = df.apply(score_with_ai_skills, axis=1)
            df[["score", "category_wise_score", "strengths", "weaknesses", "recommendation"]] = scoring_results
            
            # Boost scores for AI skills
            df["score"] += df["ai_skill_count"] * 0.5  # Add 0.5 points per AI skill
            
            state["scores"] = df.sort_values("score", ascending=False).head(20)
            state["routing_history"].append("ai_scoring")
            logger.info("AI/ML role scoring completed")
            
        except Exception as e:
            state["error"] = {"type": "ai_scoring_error", "message": str(e)}
            logger.error(f"AI scoring failed: {e}")
        return state

    def _manual_review_node(self, state: State) -> State:
        """Generate manual review report"""
        try:
            df = state.get("scores", pd.DataFrame())
            routing_history = state.get("routing_history", [])
            
            # Create comprehensive review report
            review_summary = {
                "total_candidates": len(df),
                "avg_score": df["score"].mean() if not df.empty and "score" in df.columns else 0,
                "top_candidates": len(df[df["score"] >= 5]) if not df.empty and "score" in df.columns else 0,
                "processing_path": " -> ".join(routing_history),
                "retry_count": state.get("retry_count", 0),
                "quality_metrics": state.get("quality_metrics", {}),
                "recommendation": "Manual review required due to insufficient automated results"
            }
            
            state["result"] = f"Manual Review Required:\n{review_summary}"
            state["routing_history"].append("manual_review")
            logger.warning("Routing to manual review")
            
        except Exception as e:
            state["error"] = {"type": "manual_review_error", "message": str(e)}
            logger.error(f"Manual review generation failed: {e}")
        return state

    def _wait_and_retry_node(self, state: State) -> State:
        """Handle API rate limits and errors"""
        try:
            import time
            
            error = state.get("error", {})
            wait_time = min(5 * (state.get("retry_count", 0) + 1), 30)  # Progressive backoff
            
            logger.info(f"Waiting {wait_time} seconds before retry due to: {error.get('message', 'Unknown error')}")
            time.sleep(wait_time)
            
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["routing_history"].append("wait_and_retry")
            
            # Clear error to allow retry
            if "error" in state:
                del state["error"]
                
        except Exception as e:
            state["error"] = {"type": "wait_retry_error", "message": str(e)}
            logger.error(f"Wait and retry failed: {e}")
        return state

    def _fallback_search_node(self, state: State) -> State:
        """Fallback search with minimal constraints"""
        try:
            spec = state["spec"].copy()
            
            # Use only essential keywords
            keywords = spec.get("keywords", [])
            if keywords:
                minimal_spec = {
                    "keywords": keywords[:2],  # Only first 2 keywords
                    "years_of_experience": [0, 20],  # Very broad experience range
                    "education": [],
                    "location": []
                }
            else:
                minimal_spec = {
                    "keywords": ["engineer", "developer"],
                    "years_of_experience": [0, 20],
                    "education": [],
                    "location": []
                }
            
            snippets = SearchAgent().run(minimal_spec)
            state["snippets"] = snippets
            state["spec"] = minimal_spec
            state["routing_history"].append("fallback_search")
            logger.info(f"Fallback search completed: {len(snippets)} results")
            
        except Exception as e:
            state["error"] = {"type": "fallback_search_error", "message": str(e)}
            logger.error(f"Fallback search failed: {e}")
        return state

    # ============ UTILITY FUNCTIONS ============

    def _calculate_search_quality(self, snippets: list) -> dict:
        """Calculate quality metrics for search results"""
        if not snippets:
            return {"valid_profile_ratio": 0, "avg_content_length": 0, "linkedin_ratio": 0}
        
        valid_profiles = sum(1 for s in snippets if s.get("title") and s.get("link"))
        linkedin_profiles = sum(1 for s in snippets if "linkedin.com" in s.get("link", ""))
        avg_content_length = sum(len(s.get("snippet", "")) for s in snippets) / len(snippets)
        
        return {
            "valid_profile_ratio": valid_profiles / len(snippets),
            "linkedin_ratio": linkedin_profiles / len(snippets),
            "avg_content_length": avg_content_length,
            "total_results": len(snippets)
        }

    def _log_routing_decision(self, router_name: str, state: State):
        """Log routing decisions for monitoring and optimization"""
        routing_info = {
            "timestamp": datetime.now().isoformat(),
            "router": router_name,
            "current_path": state.get("routing_history", []),
            "retry_count": state.get("retry_count", 0),
            "reprocess_count": state.get("reprocess_count", 0),
            "data_size": len(state.get("snippets", [])),
            "processing_mode": state.get("processing_mode", "standard")
        }
        logger.info(f"Routing decision: {routing_info}")

    def run(self, user_prompt: str):
        """Run the orchestration with enhanced state management"""
        initial_state = State(
            user_prompt=user_prompt,
            spec={},
            snippets=[],
            df=None,
            scores=None,
            result="",
            error=None,
            quality_metrics=None,
            routing_history=[],
            retry_count=0,
            processing_mode="standard",
            reprocess_count=0  # Initialize reprocess counter
        )
        
        try:
            logger.info(f"Starting orchestration for: {user_prompt}")
            # Use invoke with config parameter instead
            result = self.app.invoke(initial_state, config={"recursion_limit": 50})
            logger.info(f"Orchestration completed. Path: {' -> '.join(result.get('routing_history', []))}")
            return result
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "result": f"Orchestration failed: {str(e)}",
                "error": {"type": "orchestration_error", "message": str(e)},
                "routing_history": initial_state.get("routing_history", [])
            }