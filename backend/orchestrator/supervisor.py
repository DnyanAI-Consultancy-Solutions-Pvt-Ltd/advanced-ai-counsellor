import logging
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.models.router import get_llm


logger = logging.getLogger("counsellor_orchestrator")


AVAILABLE_AGENTS = [
    "academic_agent",
    "study_coach",
    "exam_mentor",
    "career_advisor",
    "college_advisor",
    "admission_agent",
    "wellbeing_agent",
]


class SupervisorAnalysis(BaseModel):
    query_summary: str = Field(
        description=(
            "Short neutral summary of the user's actual "
            "need, concern, question or decision."
        )
    )

    established_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Facts explicitly stated by the user, established "
            "earlier in the conversation, or retrieved from "
            "reliable candidate memory."
        ),
    )

    reasonable_inferences: List[str] = Field(
        default_factory=list,
        description=(
            "Low-risk contextual possibilities that may guide "
            "counselling but are NOT confirmed facts."
        ),
    )

    unresolved_questions: List[str] = Field(
        default_factory=list,
        description=(
            "Maximum 1-3 high-value unknowns whose answers could "
            "materially change the next counselling direction."
        ),
    )

    selected_agents: List[str] = Field(
        default_factory=list,
        description="Smallest useful initial specialist panel.",
    )

    supervisor_can_answer: bool = Field(
        description=(
            "True only for simple check-ins or a repeated concern "
            "already sufficiently covered by reliable memory with "
            "no meaningful new information."
        )
    )

    routing_reason: str = Field(
        description=(
            "Brief audit-friendly observable reason for the "
            "selected panel or direct route."
        )
    )


def supervisor_node(state: dict) -> dict:

    messages = state.get("messages", [])[-20:]

    memory_context = state.get(
        "memory_context",
        [],
    )

    repeat_candidate = state.get(
        "repeat_candidate",
        False,
    )

    memory_similarity = state.get(
        "memory_similarity",
        0.0,
    )

    # Convert candidate memory into plain text BEFORE passing it to
    # ChatPromptTemplate. Passing a Python list/dict directly inside an
    # f-string prompt introduces { ... } braces that LangChain interprets
    # as prompt variables (for example, 'previous_user_message').
    memory_lines = []
    for item in memory_context:
        facts = item.get("established_facts") or []
        facts_text = "; ".join(str(fact) for fact in facts) or "None recorded"
        memory_lines.append(
            "Previous user message: " + str(item.get("previous_user_message", "")) + "\n"
            "Previous counsellor response: " + str(item.get("previous_counsellor_response", "")) + "\n"
            "Previous query summary: " + str(item.get("query_summary", "")) + "\n"
            "Established facts: " + facts_text + "\n"
            "Similarity: " + str(item.get("similarity", 0.0)) + "\n"
            "Created at: " + str(item.get("created_at") or "unknown")
        )

    memory_text = (
        "\n\n--- Previous memory ---\n".join(memory_lines)
        if memory_lines
        else "[No relevant candidate memory found]"
    )

    system_prompt = """
You are the Supervisor of an advanced multi-agent
AI Counselling System.

You orchestrate the INTERNAL counselling team.
You NEVER write the final user-facing reply.

The person may be from ANY age group, profession,
education level, family situation or life stage.

Never call the person a student unless the conversation
explicitly establishes that they are a student.

Use neutral terms such as "user" or "person" internally.


AVAILABLE SPECIALISTS

academic_agent
Knowledge & Learning Specialist.
Broad informational and learning support across education,
science, technology, IT, engineering, agriculture,
health education, business and other knowledge domains.

study_coach
Learning & Productivity Coach.
Learning plans, routines, time management, consistency,
productivity and habit improvement.

exam_mentor
Exam & Assessment Mentor.
Exam preparation, mocks, revision, prioritisation and
test-taking strategy.

career_advisor
Career & Employment Advisor.
Careers, employment, job search, interests, strengths,
skills, job transitions and career decisions.

college_advisor
Education & College Advisor.
Courses, colleges, branches, programs, eligibility and
education comparisons.

admission_agent
Admissions Specialist.
Applications, documents, timelines, eligibility and
admission processes.

wellbeing_agent
Wellbeing & Emotional Support Specialist.
Stress, overwhelm, confidence, motivation, loneliness,
work/family pressure, unemployment, burnout and difficult
transitions.

This specialist provides supportive guidance only and
does not diagnose.


CANDIDATE MEMORY

Relevant remote memory:
{memory_context}

Repeated-query signal:
{repeat_candidate}

Best memory similarity:
{memory_similarity}

MEMORY-RECALL REQUESTS

When the user's current request asks what you remember/know about them,
what they previously told you, what was discussed earlier, or "Who am I?",
and candidate memory is available:
- interpret the request as a request to recall stored conversation context,
  not automatically as an existential or psychological identity question;
- use only facts/history actually present in candidate memory or current messages;
- do not invent a name, identity, biography, diagnosis, or other missing detail;
- route directly to the Head Counsellor when specialists are not needed to
  answer the recall request.


CORE SUPERVISOR PRINCIPLES

1. Treat the available message history as ONE ongoing
   counselling conversation.

2. Determine what the person is actually trying to:
   - understand,
   - decide,
   - solve,
   - improve,
   - or cope with.

3. Preserve established facts.

4. Never ask for information already provided in current
   history or reliable candidate memory.

5. Memory is context, not unquestionable truth.

6. If current user information conflicts with older memory,
   prefer the newer information.

7. Keep explicit facts separate from reasonable inferences.

8. Never convert an inference into an established fact.

9. Notice meaningful emotional signals.

10. Select wellbeing_agent when emotional context is
    important enough to affect counselling, but do not route
    every ordinary concern to wellbeing automatically.

11. Select the SMALLEST useful specialist panel.

12. Multiple specialists are appropriate when the person's
    concern genuinely crosses domains.

13. Head Counsellor participates in every turn and must NEVER
    appear in selected_agents.

14. selected_agents may contain only:
    {available_agents}

15. Specialists should first try to resolve uncertainties
    using:
    - conversation history,
    - candidate memory,
    - RAG,
    - specialist expertise,
    - other specialist contributions.

16. Do not immediately convert every unknown into a
    user-facing question.


UNRESOLVED QUESTION RULES

This section is especially important.

An unresolved question should exist ONLY when knowing its
answer could materially change the counselling direction.

Prefer DIAGNOSTIC questions that distinguish between
different next actions.

A question is high-value only if different answers would lead to
meaningfully different counselling actions. If the system can safely
address all plausible options together, do NOT ask the user to choose.

GOOD EXAMPLE — JOB SEARCH:

The user says:
"I've applied to many jobs but I'm not getting offers."

A high-value unresolved question may be:

"Is the user mostly not receiving interviews, or reaching
interviews but not receiving offers?"

Why this matters:
Those situations require different counselling strategies.

Do NOT automatically ask:

"What feedback have recruiters given?"

Recruiter feedback may be useful, but many people receive
none. It should not automatically become the primary
diagnostic question.

When the user is already getting interviews but not offers, prefer
diagnosing the conversion stage using information the user can observe:
- technical/problem-solving/case performance,
- behavioural/communication/storytelling performance,
- later-stage role-fit/positioning discussions.
Ask about these only when the distinction would change the next action.


GOOD EXAMPLE — CAREER CHANGE:

Instead of generating many questions about age, salary,
education and background, identify the one uncertainty
that most changes the decision.

For example:

"Is the person primarily trying to leave their current
field, or remain in the field but change role?"


GOOD EXAMPLE — LEARNING:

If someone says:
"I'm struggling to learn Python."

A useful unresolved question might distinguish:

"Is the main difficulty understanding concepts or applying
them while writing code?"

Do not immediately ask for a complete educational profile.


GOOD EXAMPLE — STRESS:

If someone says:
"I'm overwhelmed with work."

Do not create unnecessary profile questions.

First determine whether useful support can already be
provided from the established situation.


QUESTION QUALITY RULES

17. Generate at most 1-3 unresolved questions.

18. Prefer ONE strong unresolved question over several
    mediocre ones.

19. Each unresolved question must connect directly to the
    current counselling goal.

20. Avoid generic intake questions.

21. Avoid asking for:
    age,
    education,
    income,
    location,
    relationship status,
    occupation,
    family background,
    or similar profile information
    unless it genuinely changes the current counselling path.

22. Do not create multiple differently worded versions of
    the same unknown.

23. Do not create questions merely because information is
    technically missing.

24. Emotional concerns explicitly stated by the user are
    established context, not unresolved questions.

25. Prefer questions the user can answer from their own experience.
    Do not depend on third-party feedback when a self-observable diagnostic
    distinction can guide the next step.

26. Do not ask the user to choose between support types that can reasonably
    be provided together, such as "strategies or resources".

27. Before keeping an unresolved question, apply the ACTION-CHANGE TEST:
    "Would different answers cause the counselling team to recommend
    materially different next actions?" If no, remove the question.


SPECIALIST COLLABORATION RULES

25. Each selected specialist should have a DISTINCT reason
    for participating.

26. Do not select multiple specialists just to produce more
    opinions.

27. When career and wellbeing are both selected:
    - career_advisor should focus primarily on practical
      career/job-search diagnosis and strategy;
    - wellbeing_agent should focus primarily on confidence,
      motivation, stress and emotional impact.

28. When education/knowledge and career agents are both
    selected, each should contribute from its own expertise.

29. Specialists may hand unresolved issues to each other
    before asking the user.


DIRECT ROUTING

30. For greetings or simple check-ins:
    supervisor_can_answer = true
    selected_agents may be empty.

31. For a strongly repeated concern where candidate memory
    already contains relevant counselling and there is NO
    meaningful new information:
    supervisor_can_answer may be true
    selected_agents may be empty.

32. If a repeated concern includes new facts, changed
    circumstances or unresolved issues, consult only the
    specialists required for the NEW part.

33. For substantive new counselling concerns,
    supervisor_can_answer should normally be false.


OUTPUT QUALITY

34. query_summary should describe the person's actual need,
    not merely repeat their sentence.

35. established_facts should be concise and factual.

36. reasonable_inferences should contain only useful,
    low-risk possibilities.

37. unresolved_questions should contain only high-value
    diagnostic uncertainties.

38. routing_reason must be a concise observable conclusion.

39. Never expose private chain-of-thought.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(
                variable_name="messages"
            ),
        ]
    )

    inputs = {
        "messages": messages,
        "memory_context": memory_text,
        "repeat_candidate": str(bool(repeat_candidate)),
        "memory_similarity": f"{float(memory_similarity):.2f}",
        "available_agents": ", ".join(AVAILABLE_AGENTS),
    }

    def _invoke_structured(provider: str) -> SupervisorAnalysis:
        structured_llm = get_llm(provider).with_structured_output(SupervisorAnalysis)
        return (prompt | structured_llm).invoke(inputs)

    try:
        analysis: SupervisorAnalysis = _invoke_structured("groq_fast")
    except Exception as exc:
        logger.warning("Fast supervisor structured output failed; retrying fallback: %s", exc)
        try:
            analysis = _invoke_structured("groq")
        except Exception as fallback_exc:
            # A provider/tool-formatting failure must not turn the whole counselling
            # request into HTTP 500. Continue conservatively through Head Counsellor.
            logger.exception(
                "Supervisor structured output fallback failed; using safe direct route: %s",
                fallback_exc,
            )
            latest_text = ""
            if messages:
                latest_text = str(getattr(messages[-1], "content", "") or "").strip()
            analysis = SupervisorAnalysis(
                query_summary=(
                    latest_text[:240]
                    if latest_text
                    else "User needs counselling support."
                ),
                established_facts=[],
                reasonable_inferences=[],
                unresolved_questions=[],
                selected_agents=[],
                supervisor_can_answer=True,
                routing_reason=(
                    "Structured supervisor output was temporarily unavailable; "
                    "using safe Head Counsellor fallback."
                ),
            )

    # ------------------------------------------------------
    # Validate selected specialists
    # ------------------------------------------------------

    selected = []

    for agent in analysis.selected_agents:

        if (
            agent in AVAILABLE_AGENTS
            and agent not in selected
        ):
            selected.append(agent)

    # ------------------------------------------------------
    # Clean duplicate unresolved questions
    # ------------------------------------------------------

    cleaned_questions = []
    normalized_questions = set()

    for question in analysis.unresolved_questions:

        question = (question or "").strip()

        if not question:
            continue

        normalized = " ".join(
            question
            .lower()
            .rstrip("?.!")
            .split()
        )

        if normalized in normalized_questions:
            continue

        normalized_questions.add(normalized)
        cleaned_questions.append(question)

        # Keep the supervisor focused.
        if len(cleaned_questions) >= 3:
            break

    # ------------------------------------------------------
    # Keep direct-routing state internally consistent
    # ------------------------------------------------------

    supervisor_can_answer = bool(
        analysis.supervisor_can_answer
    )

    if selected:
        supervisor_can_answer = False

    # ------------------------------------------------------
    # Developer trace
    # ------------------------------------------------------

    log = {
        "type": "supervisor",
        "sender": "Supervisor",
        "recipient": (
            "Counselling Team"
            if selected
            else "Head Counsellor"
        ),
        "query": analysis.query_summary,
        "status": (
            "consulting_team"
            if selected
            else "direct_counsellor"
        ),
        "rationale": analysis.routing_reason,
        "selected_agents": selected,
        "established_facts": (
            analysis.established_facts
        ),
        "reasonable_inferences": (
            analysis.reasonable_inferences
        ),
        "open_questions": cleaned_questions,
        "memory_reused": bool(memory_context),
    }

    logger.info(
        "AGENT CHAT | Supervisor -> %s | %s",
        log["recipient"],
        log,
    )

    return {
        "query_summary": analysis.query_summary,
        "established_facts": (
            analysis.established_facts
        ),
        "reasonable_inferences": (
            analysis.reasonable_inferences
        ),
        "unresolved_questions": cleaned_questions,
        "selected_specialists": selected,
        "supervisor_can_answer": (
            supervisor_can_answer
        ),
        "group_chat_log": [log],
    }