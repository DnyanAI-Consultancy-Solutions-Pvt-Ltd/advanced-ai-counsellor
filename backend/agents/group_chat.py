import logging
from typing import Dict, List

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.models.router import get_llm
from backend.tools.rag_engine import rag_engine
from backend.orchestrator.supervisor import AVAILABLE_AGENTS


logger = logging.getLogger("agent_group_chat")


# ============================================================
# SPECIALIST DEFINITIONS
# ============================================================

AGENT_PROMPTS: Dict[str, str] = {
    "academic_agent": """
You are the Knowledge & Learning Specialist.

Support learning and informational needs across domains such as education,
science, technology, IT, engineering, agriculture, health education,
business and other fields.

Explain clearly at the user's level, identify relevant knowledge gaps,
and give practical next steps.

For high-stakes domains, provide general educational guidance without
pretending to replace a qualified professional.
""",

    "study_coach": """
You are the Learning & Productivity Coach.

Support realistic learning plans, routines, revision, time management,
consistency, productivity and habit improvement when relevant to the
person's goal.
""",

    "exam_mentor": """
You are the Exam & Assessment Mentor.

Support exam preparation, mocks, prioritisation, revision,
test-taking technique and assessment-related decisions when relevant.
""",

    "career_advisor": """
You are the Career & Employment Advisor.

Support interests, strengths, skills, job search, employment decisions,
career transitions, education choices and career trade-offs.

For job-search problems, distinguish where possible between:
- getting too few interviews,
- reaching interviews but not receiving offers,
- skill/experience gaps,
- positioning/resume issues,
- networking/market-access issues.

Do not assume which problem exists unless the conversation establishes it.
""",

    "college_advisor": """
You are the Education & College Advisor.

Support colleges, courses, branches, programs, eligibility and
evidence-based education comparisons.
""",

    "admission_agent": """
You are the Admissions Specialist.

Support application steps, documents, timelines, eligibility
and admission processes.
""",

    "wellbeing_agent": """
You are the Wellbeing & Emotional Support Specialist.

Support people of any age, profession or life situation with stress,
overwhelm, confidence, motivation, loneliness, work/family pressure,
unemployment, burnout and difficult transitions.

When another specialist has already covered the practical problem,
focus on the emotional or behavioural dimension instead of repeating
their recommendations.

Acknowledge emotional signals and provide supportive practical
suggestions. Do not diagnose or make unsupported clinical conclusions.
""",
}


COMMON_AGENT_RULES = """
UNIVERSAL COUNSELLING RULES:

1. Understand the person's actual need, concern or decision before proposing a solution.
2. Use conversation history, candidate memory and available RAG before asking for more information.
3. Never ask for information already provided or reliably established.
4. Ask only for information that could materially improve the next counselling step.
5. Prefer one focused high-value question over several questions at once.
6. Do not turn counselling into a questionnaire.
7. When enough information exists, provide useful suggestions instead of continuously asking questions.
8. Give practical, relevant and proportional suggestions.
9. Keep established facts separate from assumptions and inferences.
10. Before asking the user, try to resolve uncertainty from history, memory, RAG,
    your expertise, or another specialist.
11. Read previous specialist messages before contributing.
12. Do NOT repeat another specialist's recommendation or unresolved question
    unless you are correcting, challenging, refining, or answering it.
13. A second turn from the same specialist must add NEW value.
14. Do not request yourself as another specialist merely to repeat a question.
15. Never assume age, occupation, education, relationship status,
    financial situation or background.
16. The person may be from any age group, profession, education level or life situation.
17. Consider meaningful emotional context without forcing every concern
    into a wellbeing issue.
18. Balance questions with useful support.
19. Share concise conclusions and handoffs only.
20. Never expose private chain-of-thought.
"""


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class AgentTurn(BaseModel):
    contribution: str = Field(
        description=(
            "1-3 concise sentences that materially advance the team discussion. "
            "Do not repeat an earlier specialist."
        )
    )

    answered_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Existing open questions this specialist can actually resolve "
            "from available context or expertise."
        ),
    )

    still_unresolved: List[str] = Field(
        default_factory=list,
        description=(
            "Important questions that genuinely remain unresolved after "
            "this contribution."
        ),
    )

    new_question: str = Field(
        default="",
        description=(
            "At most ONE genuinely new high-value unknown discovered by "
            "this specialist. Blank if none."
        ),
    )

    recommended_agents: List[str] = Field(
        default_factory=list,
        description=(
            "Additional specialists that can materially resolve a remaining "
            "issue. Do not recommend an agent merely to repeat existing advice."
        ),
    )


class SufficiencyDecision(BaseModel):
    information_sufficient: bool = Field(
        description=(
            "True when enough reliable information exists for the Head "
            "Counsellor to give useful counselling now."
        )
    )

    needs_user_clarification: bool = Field(
        description=(
            "True when one focused user question would materially improve "
            "the next counselling step. This can be true even when "
            "information_sufficient is also true."
        )
    )

    clarification_blocks_progress: bool = Field(
        description=(
            "True only when the missing information genuinely prevents "
            "responsible/useful counselling from proceeding."
        )
    )

    unresolved_high_value_question: str = Field(
        default="",
        description="The single highest-value unresolved uncertainty, if any.",
    )

    clarification_question: str = Field(
        default="",
        description=(
            "At most one short, natural, user-facing clarification question. "
            "Blank when no clarification is needed."
        ),
    )

    reason: str = Field(
        description="Brief observable reason for the decision."
    )


# ============================================================
# HELPERS
# ============================================================

def _rag_context(state: dict) -> List[dict]:
    messages = state.get("messages", [])

    if not messages:
        return []

    query = "\n".join(
        [
            state.get("query_summary", ""),
            " ".join(state.get("unresolved_questions", [])),
            messages[-1].content,
        ]
    ).strip()

    candidate_id = (
        state.get("candidate_id")
        or state.get("student_id")
        or "default"
    )

    try:
        return rag_engine.search_student_documents(
            query=query,
            student_id=candidate_id,
            k=4,
        )
    except Exception as exc:
        logger.warning("Candidate RAG retrieval skipped: %s", exc)
        return []



def _prompt_safe(value) -> str:
    """Escape braces before embedding dynamic values into ChatPromptTemplate system strings."""
    text = str(value)
    return text.replace("{", "{{").replace("}", "}}")

def _discussion_text(logs: List[dict]) -> str:
    lines = []

    for item in logs:
        if item.get("type") == "agent_message" and item.get("message"):
            lines.append(
                f"{item.get('sender')} -> "
                f"{item.get('recipient')}: "
                f"{item.get('message')}"
            )

    return "\n".join(lines) or "[No specialist messages yet]"


def _normalize_question(text: str) -> str:
    return " ".join(
        (text or "")
        .lower()
        .strip()
        .rstrip("?.!")
        .split()
    )


def _remove_answered(
    open_questions: List[str],
    answered: List[str],
) -> List[str]:

    answered_norm = {
        _normalize_question(q)
        for q in answered
        if q
    }

    remaining = []

    for question in open_questions:
        qn = _normalize_question(question)

        if not any(
            a and (
                a == qn
                or a in qn
                or qn in a
            )
            for a in answered_norm
        ):
            remaining.append(question)

    return remaining


def _question_exists(
    question: str,
    questions: List[str],
) -> bool:

    qn = _normalize_question(question)

    if not qn:
        return True

    for existing in questions:
        en = _normalize_question(existing)

        if (
            qn == en
            or qn in en
            or en in qn
        ):
            return True

    return False


def _meaningful_contribution(
    contribution: str,
    previous_contributions: List[str],
) -> bool:
    """
    Lightweight duplicate protection.

    Exact/near-identical normalized messages are suppressed.
    Semantic duplication is additionally controlled by the LLM prompt.
    """

    current = _normalize_question(contribution)

    if not current:
        return False

    for previous in previous_contributions:
        old = _normalize_question(previous)

        if not old:
            continue

        if current == old:
            return False

        # Prevent obvious restatements where one message almost fully
        # contains the other.
        if len(current) > 40 and len(old) > 40:
            if current in old or old in current:
                return False

    return True


# ============================================================
# SPECIALIST COLLABORATION
# ============================================================

def specialist_panel_node(state: dict) -> dict:
    """
    Collaborative internal resolution loop.

    Specialists can hand issues to one another, but repeated turns are
    permitted only when another specialist explicitly re-requests them
    and the new turn can add value.
    """

    initial_selected = list(
        state.get("selected_specialists", [])
    )

    messages = state.get("messages", [])[-20:]

    docs = _rag_context(state)

    rag_text = "\n\n".join(
        d.get("content", "")
        for d in docs
        if d.get("content")
    )

    open_questions = list(
        state.get("unresolved_questions", [])
    )

    resolved_questions: List[str] = []
    logs: List[dict] = []

    queue = [
        a
        for a in initial_selected
        if a in AVAILABLE_AGENTS
    ]

    consulted: List[str] = []
    turn_counts: Dict[str, int] = {}

    # Track why an agent was queued.
    queue_reasons: Dict[str, str] = {
        a: "initial_supervisor_selection"
        for a in queue
    }

    previous_contributions: List[str] = []

    max_turns_per_agent = 2
    max_agent_turns = 10
    total_turns = 0

    while queue and total_turns < max_agent_turns:

        agent_name = queue.pop(0)
        queue_reason = queue_reasons.pop(
            agent_name,
            "specialist_handoff",
        )

        current_turn_count = turn_counts.get(
            agent_name,
            0,
        )

        if current_turn_count >= max_turns_per_agent:
            continue

        # A repeat turn must come from an actual handoff/re-request,
        # not simply because the agent recommended itself.
        if (
            current_turn_count > 0
            and queue_reason == "self_recommendation"
        ):
            continue

        turn_counts[agent_name] = (
            current_turn_count + 1
        )

        total_turns += 1

        if agent_name not in consulted:
            consulted.append(agent_name)

        base_prompt = AGENT_PROMPTS.get(agent_name)

        if not base_prompt:
            continue

        previous_discussion = _discussion_text(logs)

        is_repeat_turn = (
            turn_counts[agent_name] > 1
        )

        repeat_instruction = ""

        if is_repeat_turn:
            repeat_instruction = """
IMPORTANT:
You have already contributed once.

Do NOT restate your earlier advice or repeat an unresolved question.
This second turn is allowed only because another specialist has
requested additional input.

Add something genuinely new by doing at least one of these:
- answer a question another specialist raised;
- correct or refine an earlier conclusion;
- resolve a newly discovered issue;
- add specialist knowledge that materially changes the next step.

If you have nothing new to add, return an empty contribution and
do not create another question.
"""

        system_prompt = f"""
{base_prompt}

{COMMON_AGENT_RULES}

You are participating in an INTERNAL counselling-team collaboration.
You are not speaking directly to the user.

Supervisor's understanding:

Actual need:
{_prompt_safe(state.get('query_summary', ''))}

Established facts:
{_prompt_safe(state.get('established_facts', []))}

Reasonable but UNCONFIRMED inferences:
{_prompt_safe(state.get('reasonable_inferences', []))}

Currently unresolved questions:
{_prompt_safe(open_questions or '[none]')}

Relevant candidate memory:
{_prompt_safe(state.get('memory_context', []) or '[none]')}

Why you were asked to participate:
{_prompt_safe(queue_reason)}

Internal specialist discussion so far:
{_prompt_safe(previous_discussion)}

Relevant uploaded RAG context:
{_prompt_safe(rag_text or '[No relevant uploaded document context found]')}

{repeat_instruction}

COLLABORATION REQUIREMENTS:

1. Read the existing specialist discussion before responding.

2. Your contribution must add something that is not already present.

3. Focus on your own specialist expertise.

4. If another specialist already covered the practical recommendation,
   do not paraphrase it.

5. First try to resolve an existing open question from facts,
   history, memory, RAG or your expertise.

6. Do not claim to have resolved a user-specific factual question
   unless the available evidence actually resolves it.

7. If a user-specific fact is unknown, it may remain unresolved.

8. You may discover at most ONE new high-value unknown.

9. A new unknown should be introduced only when its answer could
   materially change the counselling direction.

10. Before recommending another specialist, ask whether that specialist
    can actually add information that is missing.

11. Do not recommend yourself.

12. Do not recommend another specialist merely because they were part
    of the original panel.

13. Emotional context should be handled distinctly from practical,
    academic or career recommendations.

14. Keep your contribution to 1-3 concise sentences.

15. Never expose private chain-of-thought.

16. You are speaking to the INTERNAL counselling team, not the user.
    Never phrase your contribution as a direct user-facing question
    such as "Could you tell me..." or "Can you share...".

17. When a user-specific fact is unknown, describe WHY that unknown
    matters to the team. Do not directly ask the user for it.

18. Only put something in new_question when discovering a genuinely
    NEW uncertainty that could materially change counselling direction.

19. If a high-value unresolved question already exists, do NOT introduce
    another question unless answering the new question BEFORE the existing
    question would materially change the counselling path.

20. Do not decompose an existing high-value question into several
    exploratory profile questions.

21. Additional useful dimensions such as values, interests, motivations,
    preferences, strengths, background or priorities should normally be
    offered as suggestions or explored in later counselling turns, rather
    than added as simultaneous unresolved questions.

22. Do not create a new emotional/wellbeing question merely because
    emotional context exists. If useful supportive guidance can already
    be given from the established emotional signal, provide that guidance
    instead.

23. Wellbeing guidance must match the intensity of the emotional signal
    actually expressed by the user.

    Do NOT escalate ordinary uncertainty, dissatisfaction, disappointment,
    frustration or confusion into burnout, grief, significant distress,
    anxiety, depression or another stronger emotional/mental-health framing
    unless the conversation provides evidence for it.

24. When an existing unresolved question already determines the immediate
    counselling direction, preserve that question and contribute useful
    guidance instead of generating another lower-priority unknown.

Allowed specialist identifiers:
{AVAILABLE_AGENTS}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(
                    variable_name="messages"
                ),
            ]
        )

        llm = get_llm("groq").with_structured_output(
            AgentTurn
        )

        turn: AgentTurn = (
            prompt | llm
        ).invoke(
            {
                "messages": messages,
            }
        )

        contribution = (
            turn.contribution or ""
        ).strip()

        answered = [
            q
            for q in turn.answered_questions
            if q
        ]

        # Only count questions that were actually open.
        valid_answered = []

        for answered_question in answered:
            for existing_question in open_questions:
                aq = _normalize_question(
                    answered_question
                )
                eq = _normalize_question(
                    existing_question
                )

                if (
                    aq == eq
                    or aq in eq
                    or eq in aq
                ):
                    valid_answered.append(
                        existing_question
                    )
                    break

        for question in valid_answered:
            if question not in resolved_questions:
                resolved_questions.append(question)

        open_questions = _remove_answered(
            open_questions,
            valid_answered,
        )

        # Add at most one genuinely new question.
        if (
            turn.new_question
            and not _question_exists(
                turn.new_question,
                open_questions,
            )
            and not _question_exists(
                turn.new_question,
                resolved_questions,
            )
        ):
            open_questions.append(
                turn.new_question.strip()
            )

        # Preserve unresolved questions without duplicating them.
        for question in turn.still_unresolved:
            if (
                question
                and not _question_exists(
                    question,
                    open_questions,
                )
                and not _question_exists(
                    question,
                    resolved_questions,
                )
            ):
                open_questions.append(question)

        recommended = []

        for recommended_agent in turn.recommended_agents:

            if recommended_agent not in AVAILABLE_AGENTS:
                continue

            # Never allow self-recommendation.
            if recommended_agent == agent_name:
                continue

            if (
                turn_counts.get(
                    recommended_agent,
                    0,
                )
                >= max_turns_per_agent
            ):
                continue

            if recommended_agent in queue:
                continue

            recommended.append(
                recommended_agent
            )

            queue.append(
                recommended_agent
            )

            queue_reasons[recommended_agent] = (
                f"handoff_from_{agent_name}"
            )

        # Do not show empty/duplicate messages in developer trace.
        should_log = _meaningful_contribution(
            contribution,
            previous_contributions,
        )

        if should_log:

            previous_contributions.append(
                contribution
            )

            next_recipient = (
                queue[0]
                if queue
                else "Supervisor"
            )

            entry = {
                "type": "agent_message",
                "sender": agent_name,
                "recipient": next_recipient,
                "status": "resolved_or_handed_off",
                "message": contribution,
                "content": contribution,
                "answered_questions": valid_answered,
                "remaining_questions": list(
                    open_questions
                ),
                "recommended_agents": recommended,
                "used_rag": bool(rag_text),
                "turn_number": turn_counts[
                    agent_name
                ],
            }

            logs.append(entry)

            logger.info(
                "AGENT CHAT | %s -> %s | %s",
                agent_name,
                next_recipient,
                entry,
            )

    final_selected = []

    for agent in initial_selected + consulted:
        if (
            agent in AVAILABLE_AGENTS
            and agent not in final_selected
        ):
            final_selected.append(agent)

    return {
        "retrieved_docs": docs,
        "resolved_questions": resolved_questions,
        "remaining_questions": open_questions,
        "selected_specialists": final_selected,
        "group_chat_log": logs,
    }


# ============================================================
# INFORMATION SUFFICIENCY
# ============================================================

def information_sufficiency_node(
    state: dict,
) -> dict:
    """
    Decide separately:

    1. Can useful counselling be provided now?
    2. Would one clarification materially improve the next step?
    3. Does missing information actually block progress?
    """

    messages = state.get("messages", [])[-20:]

    logs = state.get(
        "group_chat_log",
        [],
    )

    internal_discussion = _discussion_text(logs)

    rag_summary = "\n".join(
        d.get("content", "")[:500]
        for d in state.get(
            "retrieved_docs",
            [],
        )
        if d.get("content")
    )

    system_prompt = f"""
You are the Information Sufficiency Judge for a
general-purpose counselling multi-agent system.

You do NOT answer the user.

Your job is to distinguish THREE different situations.

CASE A — READY, NO QUESTION NEEDED
Enough information exists to provide useful counselling and no
additional user question would materially improve the next step.

CASE B — READY, BUT ONE FOLLOW-UP WOULD HELP
Enough information exists to provide useful counselling now, but
ONE focused question would materially improve the next counselling step.

For this case:
information_sufficient = true
needs_user_clarification = true
clarification_blocks_progress = false

CASE C — BLOCKED
One essential piece of information is genuinely required before
responsible/useful counselling can proceed.

For this case:
information_sufficient = false
needs_user_clarification = true
clarification_blocks_progress = true


User need:
{_prompt_safe(state.get('query_summary', ''))}

Established facts:
{_prompt_safe(state.get('established_facts', []))}

Reasonable but unconfirmed inferences:
{_prompt_safe(state.get('reasonable_inferences', []))}

Supervisor's initial unknowns:
{_prompt_safe(state.get('unresolved_questions', []))}

Questions specialists resolved:
{_prompt_safe(state.get('resolved_questions', []))}

Questions still open:
{_prompt_safe(state.get('remaining_questions', []))}

Specialist discussion:
{_prompt_safe(internal_discussion)}

Relevant candidate memory:
{_prompt_safe(state.get('memory_context', []) or '[none]')}

Relevant RAG excerpt:
{_prompt_safe(rag_summary or '[none]')}


DECISION RULES:

1. Do not confuse "an unanswered question exists" with
   "counselling cannot proceed."

2. Most ordinary counselling conversations should be able to move
   forward with cautious useful support even when some details are unknown.

3. Use CASE C only when giving guidance without the missing fact would
   be misleading, unsafe, or substantially premature.

4. Use CASE B when a useful response can be given now but one question
   would help determine the next direction.

5. Prefer a diagnostic question that distinguishes between materially
   different paths.

6. Do not ask for information already established in conversation,
   memory, RAG or specialist discussion.

7. Do not ask generic intake questions.

8. Prefer questions that are easy for the user to answer.

9. For a job-search problem, for example, knowing whether the user is
   failing to receive interviews versus reaching interviews but not
   receiving offers may be more actionable than asking for recruiter
   feedback that may not exist.

10. Explicit emotional concerns do not automatically make the
    information insufficient.

11. Return at most ONE clarification question.

12. If needs_user_clarification is false,
    clarification_question must be blank.

13. If needs_user_clarification is true,
    clarification_question should normally be non-empty.

14. Never expose private chain-of-thought.

15. Before selecting a clarification question, apply the ACTION-CHANGE TEST:
    different answers must lead to materially different next counselling actions.
    If the same useful guidance can cover all likely answers, use CASE A.

16. Prefer questions the user can answer from their own experience rather than
    relying on third-party feedback that may not exist.

17. Do not ask the user to choose between support types that can be offered
    together, such as "strategies or resources".

18. For interview-to-offer problems, when further diagnosis is genuinely useful,
    prefer distinctions such as technical/case performance vs behavioural/
    communication performance vs later-stage role-fit/positioning.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(
                variable_name="messages"
            ),
        ]
    )

    decision: SufficiencyDecision = (
        prompt
        | get_llm(
            "groq_fast"
        ).with_structured_output(
            SufficiencyDecision
        )
    ).invoke(
        {
            "messages": messages,
        }
    )

    needs_clarification = bool(
        decision.needs_user_clarification
    )

    blocked = bool(
        decision.clarification_blocks_progress
    )

    information_sufficient = bool(
        decision.information_sufficient
    )

    # Keep the state internally consistent.
    if blocked:
        information_sufficient = False
        needs_clarification = True

    clarification = ""

    if needs_clarification:
        clarification = (
            decision.clarification_question
            or ""
        ).strip()

        if (
            clarification
            and not clarification.endswith("?")
        ):
            clarification += "?"

    if blocked:
        status = "needs_user_confirmation"
    elif needs_clarification:
        status = "sufficient_with_follow_up"
    else:
        status = "sufficient"

    log = {
        "type": "sufficiency_check",
        "sender": "Supervisor",
        "recipient": "Head Counsellor",
        "status": status,
        "message": decision.reason,
        "content": decision.reason,
        "remaining_question": (
            decision.unresolved_high_value_question
        ),
        "clarification_question": clarification,
        "information_sufficient": (
            information_sufficient
        ),
        "needs_user_clarification": (
            needs_clarification
        ),
        "clarification_blocks_progress": blocked,
    }

    logger.info(
        "AGENT CHAT | Supervisor -> "
        "Head Counsellor | %s",
        log,
    )

    return {
        "information_sufficient": (
            information_sufficient
        ),
        "needs_user_clarification": (
            needs_clarification
        ),
        "clarification_question": (
            clarification or None
        ),

        # is_incomplete now means genuinely blocked.
        "is_incomplete": blocked,

        "missing_info": (
            clarification
            if blocked
            else None
        ),

        "group_chat_log": [log],
    }


# ============================================================
# HEAD COUNSELLOR
# ============================================================

def main_counsellor_node(
    state: dict,
) -> dict:
    """
    Head Counsellor owns all user-facing communication.
    """

    messages = state.get("messages", [])[-20:]

    logs = state.get(
        "group_chat_log",
        [],
    )

    internal_discussion = _discussion_text(logs)

    is_blocked = bool(
        state.get("is_incomplete")
    )

    needs_clarification = bool(
        state.get(
            "needs_user_clarification"
        )
    )

    clarification_question = (
        state.get("clarification_question")
        or ""
    ).strip()

    # --------------------------------------------------------
    # CASE C: missing information genuinely blocks progress
    # --------------------------------------------------------

    if is_blocked:

        question = (
            clarification_question
            or state.get("missing_info")
            or (
                "Could you clarify the one part of this "
                "situation that would help us choose the "
                "right next step?"
            )
        )

        system_prompt = f"""
You are the Head Counsellor.

The counselling team has determined that ONE clarification is
genuinely required before deeper guidance can responsibly proceed.

User's actual need:
{_prompt_safe(state.get('query_summary', ''))}

Established facts:
{_prompt_safe(state.get('established_facts', []))}

Relevant candidate memory:
{_prompt_safe(state.get('memory_context', []) or '[none]')}

Required clarification:
{_prompt_safe(question)}

Respond naturally in 1-3 sentences.

RULES:

1. Briefly acknowledge the person's main concern.
2. If an emotional concern was explicitly raised, acknowledge it naturally.
3. When safe, provide ONE small useful observation or suggestion that
   does not depend on the missing information.
4. Ask exactly the required clarification question.
5. Do not ask a second question.
6. Do not mention agents, routing, sufficiency checks or internal analysis.
7. Never expose private chain-of-thought.
8. Never assume the person is a student.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(
                    variable_name="messages"
                ),
            ]
        )

        final_res = (
            prompt | get_llm("groq")
        ).invoke(
            {
                "messages": messages,
            }
        )

        final_text = final_res.content.strip()
        status = "asked_clarification"

    # --------------------------------------------------------
    # CASE A / CASE B:
    # enough information exists to help now
    # --------------------------------------------------------

    else:

        follow_up_instruction = ""

        if (
            needs_clarification
            and clarification_question
        ):
            follow_up_instruction = f"""
The team has identified ONE useful follow-up question:

{_prompt_safe(clarification_question)}

Give useful counselling FIRST, then ask exactly this one
question naturally at the end.

Do not add another question.
"""
        else:
            follow_up_instruction = """
No clarification is required.

Do not force a question merely to keep the conversation going.
Ask one only if it is genuinely useful for the person's next step.
"""

        system_prompt = f"""
You are the Head Counsellor and the ONLY user-facing agent.

Act like a real counsellor in an ongoing two-way conversation,
not a report generator.

Supervisor's understanding:

Need:
{_prompt_safe(state.get('query_summary', ''))}

Established facts:
{_prompt_safe(state.get('established_facts', []))}

Reasonable but UNCONFIRMED inferences:
{_prompt_safe(state.get('reasonable_inferences', []))}

Relevant candidate memory:
{_prompt_safe(state.get('memory_context', []) or '[none]')}

Internal specialist conclusions:
{_prompt_safe(internal_discussion)}

{follow_up_instruction}

STRICT RESPONSE RULES:

1. Normally respond in 2-5 sentences, roughly 35-100 words.

2. Address the person's actual concern directly.

3. Do not restart generic intake.

4. Explicitly acknowledge meaningful emotional context when the user
   raised it, without over-focusing on emotion.

5. Give at least ONE useful observation, recommendation, or next step
   when the available information supports it.

6. Use established facts and specialist conclusions.

7. Never present an unconfirmed inference as established fact.

8. If a follow-up question was supplied above, ask exactly that
   one question at the end.

9. Otherwise ask at most ONE question, only when it materially
   advances the counselling process.

10. Never ask for information already present in conversation,
    memory, RAG or specialist discussion.

11. Do not dump every specialist recommendation.

12. Do not mention specialist names, routing, memory systems,
    sufficiency checks or internal architecture.

13. For wellbeing concerns, remain supportive and non-judgmental.
    Do not diagnose.

14. Never assume the person is a student.

15. If candidate memory exists, use it naturally as conversation
    continuity. Do not say that a hidden profile or database exists.

16. For repeated concerns, focus on what changed and the next useful
    step instead of copying an earlier answer.

17. Never expose private chain-of-thought.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(
                    variable_name="messages"
                ),
            ]
        )

        final_res = (
            prompt | get_llm("groq")
        ).invoke(
            {
                "messages": messages,
            }
        )

        final_text = final_res.content.strip()

        status = (
            "finalized_with_follow_up"
            if needs_clarification
            and clarification_question
            else "finalized"
        )

    logger.info(
        "AGENT CHAT | Head Counsellor -> User | %s",
        final_text,
    )

    return {
        "messages": [
            AIMessage(
                content=final_text
            )
        ],

        "group_chat_log": [
            {
                "type": "counsellor_message",
                "sender": "Head Counsellor",
                "recipient": "User",
                "status": status,
                "message": (
                    "Used the resolved team context "
                    "to produce the user-facing reply."
                ),
                "content": (
                    "Used the resolved team context "
                    "to produce the user-facing reply."
                ),
            }
        ],
    }