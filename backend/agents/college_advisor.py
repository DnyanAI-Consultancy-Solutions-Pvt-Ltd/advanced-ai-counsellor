from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm
from backend.tools.rag_engine import rag_engine


def run(state: dict) -> dict:
    messages = state["messages"]
    user_query = messages[-1].content

    # Pull college cutoffs and branch availability
    docs = rag_engine.search_internal_knowledge(user_query, category="cutoffs")
    context_str = "\n\n".join([d["content"] for d in docs])

    system_prompt = (
        "You are the College Advisor Agent. You guide students on college selection, cutoff ranks, "
        "eligibility matrix, course comparisons, and choosing between different institutional branches.\n\n"
        f"Internal Cutoffs Context:\n{context_str}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("groq")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}