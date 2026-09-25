from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm


def run(state: dict) -> dict:
    messages = state["messages"]

    system_prompt = (
        "You are the Career Advisor Agent. You assist students in exploring career domains, "
        "building skill roadmaps, mapping personal interests to future industry roles, "
        "and evaluating long-term job market opportunities."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("groq")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}