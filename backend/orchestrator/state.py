from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator
from langchain_core.messages import BaseMessage


class CounsellingState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    candidate_id: str
    student_id: str  # backward-compatible alias for existing RAG code

    # Remote candidate-specific memory
    memory_context: List[Dict[str, Any]]
    memory_reused: bool
    memory_similarity: float
    repeat_candidate: bool
    memory_saved: bool

    # Shared reusable Q&A cache (non-personal answers only)
    global_cache_hit: bool
    cached_response: Optional[str]

    # Supervisor understanding
    query_summary: str
    established_facts: List[str]
    reasonable_inferences: List[str]
    unresolved_questions: List[str]
    selected_specialists: List[str]
    supervisor_can_answer: bool

    # Collaborative resolution
    retrieved_docs: List[Dict[str, Any]]
    resolved_questions: List[str]
    remaining_questions: List[str]
    information_sufficient: bool
    needs_user_clarification: bool
    clarification_question: Optional[str]
    is_incomplete: bool
    missing_info: Optional[str]

    # Observable internal collaboration (conclusions only; no hidden CoT)
    group_chat_log: Annotated[List[Dict[str, Any]], operator.add]
