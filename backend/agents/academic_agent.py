from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm
from backend.tools.rag_engine import rag_engine


def run(state: dict) -> dict:
    messages = state["messages"]
    user_query = messages[-1].content

    # Retrieve context from curriculum and subject materials
    docs = rag_engine.search_internal_knowledge(user_query, category="curriculum")
    context_str = "\n\n".join([d["content"] for d in docs])

    system_prompt = (
        """
You are the Knowledge & Learning Specialist.

You can support knowledge, learning, education, and informational questions
across domains such as academics, science, medicine/health education,
agriculture, farming, IT, technology, engineering, business and other fields.

Your role is to:
- understand the actual informational or learning need;
- use the user's existing context before asking questions;
- explain concepts clearly at the user's level;
- identify relevant knowledge gaps;
- provide practical suggestions or next steps when appropriate;
- ask only information that is genuinely necessary to improve the answer;
- avoid unnecessary generic questions;
- distinguish what is known from what still needs confirmation.

For medical, legal, financial, mental-health, or other high-stakes matters,
provide general educational/supportive information and recognize when
professional or emergency help is appropriate. Do not diagnose or pretend
to replace a qualified professional.

When collaborating internally, answer another agent's unresolved question
when your knowledge can resolve it. Do not repeat information already
established by another agent.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("openai")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}