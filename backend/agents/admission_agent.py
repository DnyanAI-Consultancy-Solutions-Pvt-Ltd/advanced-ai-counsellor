from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from backend.models.router import get_llm


def run(state: dict) -> dict:
    messages = state["messages"]

    system_prompt = (
        "You are the Admission Agent. You help students navigate college application processes, "
        "understand document verification criteria, keep track of application deadlines, and prepare "
        "Statements of Purpose (SOPs) or application forms."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = get_llm("groq")
    chain = prompt | llm
    response = chain.invoke({"messages": messages})

    return {"messages": [AIMessage(content=response.content)]}