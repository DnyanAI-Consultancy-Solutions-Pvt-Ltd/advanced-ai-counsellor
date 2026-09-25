from langgraph.graph import StateGraph, END
from backend.orchestrator.state import CounsellingState
from backend.orchestrator.memory_nodes import memory_lookup_node, memory_save_node, cached_response_node
from backend.orchestrator.supervisor import supervisor_node
from backend.agents.group_chat import specialist_panel_node, information_sufficiency_node, main_counsellor_node


def route_after_memory(state: dict) -> str:
    # Only a high-confidence, reusable GLOBAL Q&A cache hit bypasses LLM agents.
    if state.get("global_cache_hit") and state.get("cached_response"):
        return "cached_response"
    return "supervisor"


def route_after_supervisor(state: dict) -> str:
    if state.get("supervisor_can_answer") and not state.get("selected_specialists"):
        return "main_counsellor"
    return "specialist_panel"


builder = StateGraph(CounsellingState)
builder.add_node("memory_lookup", memory_lookup_node)
builder.add_node("cached_response", cached_response_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("specialist_panel", specialist_panel_node)
builder.add_node("information_sufficiency", information_sufficiency_node)
builder.add_node("main_counsellor", main_counsellor_node)
builder.add_node("memory_save", memory_save_node)

builder.set_entry_point("memory_lookup")
builder.add_conditional_edges(
    "memory_lookup", route_after_memory,
    {"cached_response": "cached_response", "supervisor": "supervisor"},
)
builder.add_edge("cached_response", END)
builder.add_conditional_edges(
    "supervisor", route_after_supervisor,
    {"main_counsellor": "main_counsellor", "specialist_panel": "specialist_panel"},
)
builder.add_edge("specialist_panel", "information_sufficiency")
builder.add_edge("information_sufficiency", "main_counsellor")
builder.add_edge("main_counsellor", "memory_save")
builder.add_edge("memory_save", END)

counsellor_graph = builder.compile()