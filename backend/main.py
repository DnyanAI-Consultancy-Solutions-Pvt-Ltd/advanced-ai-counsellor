import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from backend.orchestrator.graph import counsellor_graph
from backend.tools.document_parser import document_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_api")

app = FastAPI(title="Advanced AI Counsellor", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[MessageItem]
    candidate_id: str = Field(default="default")


class ChatResponse(BaseModel):
    response: str
    is_incomplete: bool
    missing_info: Optional[str] = None
    selected_specialists: List[str] = Field(default_factory=list)
    query_summary: str = ""
    group_chat_log: List[Dict[str, Any]]
    rag_documents_used: int = 0
    memory_reused: bool = False
    memory_saved: bool = False
    memory_similarity: float = 0.0


@app.get("/")
def read_root():
    return {"status": "online", "system": "Advanced AI Counsellor", "version": "3.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        formatted_messages = []
        for msg in request.messages:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AIMessage(content=msg.content))

        if not formatted_messages or not any(m.role == "user" for m in request.messages):
            raise HTTPException(status_code=422, detail="At least one user message is required.")

        initial_state = {
            "messages": formatted_messages,
            "candidate_id": request.candidate_id,
            "student_id": request.candidate_id,  # backward-compatible RAG alias
            "memory_context": [],
            "memory_reused": False,
            "memory_similarity": 0.0,
            "repeat_candidate": False,
            "memory_saved": False,
            "query_summary": "",
            "missing_info": None,
            "is_incomplete": False,
            "supervisor_can_answer": False,
            "selected_specialists": [],
            "established_facts": [],
            "reasonable_inferences": [],
            "unresolved_questions": [],
            "resolved_questions": [],
            "remaining_questions": [],
            "information_sufficient": False,
            "clarification_question": None,
            "retrieved_docs": [],
            "group_chat_log": [],
        }

        final_state = counsellor_graph.invoke(initial_state)
        last_message = final_state["messages"][-1]
        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

        return ChatResponse(
            response=response_text,
            is_incomplete=final_state.get("is_incomplete", False),
            missing_info=final_state.get("missing_info"),
            selected_specialists=final_state.get("selected_specialists", []),
            query_summary=final_state.get("query_summary", ""),
            group_chat_log=final_state.get("group_chat_log", []),
            rag_documents_used=len(final_state.get("retrieved_docs", [])),
            memory_reused=final_state.get("memory_reused", False),
            memory_saved=final_state.get("memory_saved", False),
            memory_similarity=float(final_state.get("memory_similarity", 0.0) or 0.0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error executing multi-agent graph")
        raise HTTPException(status_code=500, detail=f"Internal Graph Execution Error: {exc}")


@app.post("/api/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    candidate_id: str = Form("default"),
    doc_type: str = Form("candidate_document"),
):
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Currently only PDF files are supported.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = await document_parser.process_and_index_file(
            file_bytes=content,
            filename=file.filename,
            student_id=candidate_id,
            doc_type=doc_type,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG document upload failed")
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {exc}")
