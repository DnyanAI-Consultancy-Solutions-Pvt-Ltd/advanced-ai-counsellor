from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm


def run(state: dict) -> dict:
    messages = state["messages"]

    system_prompt = (
        "You are the Study Coach Agent. Your domain includes personalized study timetables, "
        "time management strategies (e.g., Pomodoro, Active Recall), revision strategies, "
        "and productivity techniques tailored to student daily schedules."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("groq")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}