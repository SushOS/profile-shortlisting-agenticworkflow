from langgraph.graph import StateGraph
from typing import TypedDict, Any
from app.agents.prompt_analyzer import PromptAnalyzerAgent
from app.agents.search_agent import SearchAgent
from app.agents.enrich_agent import EnrichAgent
from app.agents.scoring_agent import ScoringAgent
from app.agents.sheet_writer import SheetWriterAgent

class State(TypedDict):
    user_prompt: str
    spec: dict
    snippets: list
    df: Any
    scores: Any
    result: str

class Orchestration:
    """
    Sequential orchestration pattern[9] implemented with LangGraph[14].
    """

    def __init__(self):
        self.graph = StateGraph(State)
        self.graph.add_node("analyze", self._analyze_node)
        self.graph.add_node("search", self._search_node)
        self.graph.add_node("enrich", self._enrich_node)
        self.graph.add_node("score", self._score_node)
        self.graph.add_node("sheet", self._sheet_node)

        # Edges
        self.graph.add_edge("analyze", "search")
        self.graph.add_edge("search", "enrich")
        self.graph.add_edge("enrich", "score")
        self.graph.add_edge("score", "sheet")
        
        self.graph.set_entry_point("analyze")
        self.graph.set_finish_point("sheet")

        self.app = self.graph.compile()

    def _analyze_node(self, state: State) -> State:
        spec = PromptAnalyzerAgent().run(state["user_prompt"])
        state["spec"] = spec
        return state

    def _search_node(self, state: State) -> State:
        snippets = SearchAgent().run(state["spec"])
        state["snippets"] = snippets
        return state

    def _enrich_node(self, state: State) -> State:
        df = EnrichAgent().run(state["snippets"])
        state["df"] = df
        return state

    def _score_node(self, state: State) -> State:
        scores = ScoringAgent(state["spec"]).run(state["df"])
        state["scores"] = scores
        return state

    def _sheet_node(self, state: State) -> State:
        result = SheetWriterAgent().run(state["scores"])
        state["result"] = result
        return state

    def run(self, user_prompt: str):
        initial_state = State(
            user_prompt=user_prompt,
            spec={},
            snippets=[],
            df=None,
            scores=None,
            result=""
        )
        return self.app.invoke(initial_state)
