from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm
from backend.tools.web_search import web_search_tool


def run(state: dict) -> dict:
    messages = state["messages"]
    user_query = messages[-1].content

    # Query latest updates regarding exams
    search_results = web_search_tool.search_educational_web(f"{user_query} exam dates updates")
    search_context = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])

    system_prompt = (
        "You are the Exam Mentor Agent. You provide strategic advice for competitive and "
        "board exams, previous year paper analysis, time-allocation strategies during tests, "
        "and mock test guidance.\n\n"
        f"Real-time search findings:\n{search_context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("groq")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}