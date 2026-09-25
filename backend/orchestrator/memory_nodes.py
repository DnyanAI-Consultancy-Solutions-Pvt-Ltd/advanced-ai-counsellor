import logging
import re
from langchain_core.messages import AIMessage, HumanMessage
from backend.tools.memory_tool import candidate_memory

logger = logging.getLogger("memory_orchestrator")
REPEAT_SIMILARITY_THRESHOLD = 0.88
GLOBAL_DIRECT_THRESHOLD = 0.90


def _is_explicit_memory_recall(text: str) -> bool:
    n = " ".join((text or "").lower().strip().split())
    if not n:
        return False
    patterns = [
        r"\b(?:do|can) you remember\b", r"\bwhat do you (?:remember|know)\b",
        r"\bwhat (?:have|did) i (?:tell|share|mention|say)\b",
        r"\bwhat (?:have|did) we (?:discuss|talk about)\b",
        r"\bwhich .+ did i (?:tell|say|mention)\b", r"\bwhat was my\b",
        r"\bwhat is my .+ based on (?:our|previous)\b", r"\bremind me\b",
        r"^who am i[?.! ]*$", r"\bwhat(?:'s| is) my name\b",
    ]
    return any(re.search(p, n) for p in patterns)


def _to_context(items, memory_kind: str, limit: int = 5):
    return [{
        "previous_user_message": x.get("user_message", ""),
        "previous_counsellor_response": x.get("counsellor_response", ""),
        "query_summary": x.get("query_summary", ""),
        "established_facts": x.get("established_facts") or [],
        "similarity": x.get("similarity", 0.0),
        "created_at": x.get("created_at"),
        "memory_kind": memory_kind,
    } for x in items[:limit]]


def _looks_personal_or_recall(text: str) -> bool:
    n = " ".join((text or "").lower().split())
    if _is_explicit_memory_recall(n):
        return True
    personal = [" my ", " i am ", " i'm ", " i have ", " i feel ", " should i ", " for me ", " about me "]
    padded = f" {n} "
    return any(x in padded for x in personal)


def _globally_reusable_question(text: str) -> bool:
    """Conservative gate: only generic informational questions may bypass agents."""
    n = " ".join((text or "").lower().strip().split())
    if not n or _looks_personal_or_recall(n):
        return False
    return bool(re.match(
        r"^(what is|what are|who is|define|explain|how does|how do|why does|why do|"
        r"what does|difference between|what's the difference|tell me about)\b", n
    ))


def memory_lookup_node(state: dict) -> dict:
    candidate_id = state.get("candidate_id") or state.get("student_id") or "default"
    messages = state.get("messages", [])
    latest_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    # 1) Explicit personal recall: durable facts + recent episodes for THIS candidate.
    if _is_explicit_memory_recall(latest_user):
        facts = candidate_memory.facts(candidate_id, limit=50)
        rows = candidate_memory.recent(candidate_id, limit=20)
        context = _to_context(rows, "candidate_history", limit=8)
        if facts:
            context.insert(0, {
                "previous_user_message": "[Durable candidate facts]",
                "previous_counsellor_response": "",
                "query_summary": "Candidate facts retrieved from long-term memory",
                "established_facts": [x.get("fact_text", "") for x in facts if x.get("fact_text")],
                "similarity": 1.0,
                "created_at": facts[0].get("updated_at") if facts else None,
                "memory_kind": "candidate_facts",
            })
        found = bool(facts or rows)
        log = {
            "type": "memory_check", "sender": "Memory", "recipient": "Supervisor",
            "status": "candidate_history_found" if found else "no_candidate_history",
            "message": f"Explicit recall: loaded {len(facts)} durable fact(s) and {len(rows)} candidate history item(s)." if found else "Explicit recall requested, but no candidate memory was found.",
            "memory_reused": found, "repeat_candidate": False,
        }
        logger.info("MEMORY CHECK | %s", log)
        return {"candidate_id": candidate_id, "memory_context": context, "memory_reused": found,
                "memory_similarity": 1.0 if facts else 0.0, "repeat_candidate": False,
                "global_cache_hit": False, "cached_response": None, "group_chat_log": [log]}

    # 2) Shared cache: only generic reusable questions are eligible for direct reuse.
    if _globally_reusable_question(latest_user):
        global_matches = candidate_memory.search_global(latest_user, limit=3)
        best_global = global_matches[0] if global_matches else None
        global_similarity = float(best_global.get("similarity", 0.0)) if best_global else 0.0
        if best_global and global_similarity >= GLOBAL_DIRECT_THRESHOLD:
            candidate_memory.increment_global_usage(best_global.get("id"))
            log = {
                "type": "global_cache", "sender": "Global Memory", "recipient": "User",
                "status": "cache_hit", "message": f"Reused a validated global answer; similarity={global_similarity:.2f}.",
                "memory_reused": True, "repeat_candidate": False,
            }
            logger.info("GLOBAL CACHE | %s", log)
            return {"candidate_id": candidate_id, "memory_context": [], "memory_reused": True,
                    "memory_similarity": global_similarity, "repeat_candidate": False,
                    "global_cache_hit": True, "cached_response": best_global.get("answer", ""),
                    "group_chat_log": [log]}

    # 3) Normal counselling continuity:
    # Always load durable facts for THIS candidate, then add semantic episode matches.
    # This makes factual recall reliable even when the wording of the current query
    # is not semantically close enough to an older conversation turn.
    facts = candidate_memory.facts(candidate_id, limit=50)
    matches = candidate_memory.search(candidate_id, latest_user, limit=5)

    best = matches[0] if matches else None
    similarity = float(best.get("similarity", 0.0)) if best else 0.0
    repeated = similarity >= REPEAT_SIMILARITY_THRESHOLD
    context = _to_context(matches, "semantic", limit=3)

    if facts:
        context.insert(0, {
            "previous_user_message": "[Durable candidate facts]",
            "previous_counsellor_response": "",
            "query_summary": "Candidate facts retrieved from long-term memory",
            "established_facts": [
                x.get("fact_text", "") for x in facts if x.get("fact_text")
            ],
            "similarity": 1.0,
            "created_at": facts[0].get("updated_at") if facts else None,
            "memory_kind": "candidate_facts",
        })

    found = bool(facts or matches)
    if facts and matches:
        status = "candidate_context_found"
        message = (
            f"Loaded {len(facts)} durable candidate fact(s) and "
            f"{len(matches)} semantic history match(es); "
            f"best semantic similarity={similarity:.2f}."
        )
    elif facts:
        status = "candidate_facts_found"
        message = f"Loaded {len(facts)} durable candidate fact(s); no semantic history match was required/found."
    elif matches:
        status = "relevant_history_found"
        message = (
            f"Found {len(matches)} candidate semantic match(es); "
            f"best similarity={similarity:.2f}."
        )
    else:
        status = "no_relevant_history"
        message = "No durable candidate facts or semantically relevant candidate history found."

    log = {
        "type": "memory_check",
        "sender": "Memory",
        "recipient": "Supervisor",
        "status": status,
        "message": message,
        "memory_reused": found,
        "repeat_candidate": repeated,
    }
    logger.info("MEMORY CHECK | %s", log)
    return {
        "candidate_id": candidate_id,
        "memory_context": context,
        "memory_reused": found,
        "memory_similarity": similarity,
        "repeat_candidate": repeated,
        "global_cache_hit": False,
        "cached_response": None,
        "group_chat_log": [log],
    }


def cached_response_node(state: dict) -> dict:
    answer = (state.get("cached_response") or "").strip()
    return {"messages": [AIMessage(content=answer)]} if answer else {}


def _safe_for_global_cache(state: dict, user_text: str) -> bool:
    if not _globally_reusable_question(user_text):
        return False
    selected = set(state.get("selected_specialists") or [])
    # Never publish personalized counselling domains into shared memory.
    if selected - {"academic_agent"}:
        return False
    if state.get("needs_user_clarification") or state.get("is_incomplete"):
        return False
    return True


def memory_save_node(state: dict) -> dict:
    candidate_id = state.get("candidate_id") or state.get("student_id") or "default"
    messages = state.get("messages", [])
    latest_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    latest_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    if not latest_user or not latest_ai:
        return {}

    facts = state.get("established_facts", []) or []
    episode_ok = candidate_memory.save({
        "candidate_id": candidate_id, "user_message": latest_user,
        "counsellor_response": latest_ai, "query_summary": state.get("query_summary", ""),
        "established_facts": facts,
    })
    facts_ok = candidate_memory.save_facts(candidate_id, facts) if facts else False
    global_ok = False
    if _safe_for_global_cache(state, latest_user):
        global_ok = candidate_memory.save_global(latest_user, latest_ai, category="general")

    log = {
        "type": "memory_write", "sender": "Head Counsellor", "recipient": "Memory",
        "status": "saved" if episode_ok else "not_configured",
        "message": (
            f"Saved candidate episode; durable facts={'saved' if facts_ok else 'none/new save skipped'}; "
            f"global reusable Q&A={'saved' if global_ok else 'not eligible'} ."
            if episode_ok else "Remote memory is unavailable; conversation was not persisted."
        ),
    }
    return {"memory_saved": episode_ok, "group_chat_log": [log]}
