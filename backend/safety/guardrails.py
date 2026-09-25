from langchain_core.prompts import ChatPromptTemplate
from backend.models.router import get_llm

def check_safety_and_wellbeing(user_query: str) -> dict:
    """Detects self-harm, extreme distress, or policy violations."""
    llm = get_llm("groq", temperature=0.0)
    
    system_prompt = """Analyze the user's query for severe mental distress, self-harm signals, or explicit toxic content.
    Return JSON format: {"flagged": boolean, "category": "wellbeing" | "toxicity" | "safe", "reason": "string"}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({"query": user_query})
    return res.content