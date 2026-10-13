import os
import tempfile
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.tools.rag_engine import rag_engine
from backend.tools.memory_tool import candidate_memory
from backend.models.router import get_llm


class ExtractedFacts(BaseModel):
    facts: List[str] = Field(default_factory=list)


class DocumentParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])

    def _extract_durable_facts(self, text: str) -> List[str]:
        if not text.strip(): return []
        prompt = """Extract only durable, explicit facts about the candidate from this candidate-owned document.
Examples: education, current/previous role, skills, stated strengths/weaknesses, career goals, learning preferences, stable constraints.
Do not infer, recommend, diagnose, or turn document advice into a candidate fact. Return concise standalone facts. Maximum 20.

DOCUMENT:
""" + text[:12000]
        try:
            result = get_llm("groq_fast").with_structured_output(ExtractedFacts).invoke(prompt)
            return [" ".join(x.split()) for x in result.facts if x and x.strip()][:20]
        except Exception:
            return []

    async def process_and_index_file(self, file_bytes: bytes, filename: str, student_id: str, doc_type: str, scope: str = "candidate") -> Dict[str, Any]:
        scope = (scope or "candidate").strip().lower()
        if scope not in {"candidate", "global"}:
            raise ValueError("scope must be 'candidate' or 'global'")
        if scope == "candidate" and not student_id:
            raise ValueError("candidate_id is required for candidate-scoped documents")
        suffix = os.path.splitext(filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes); temp_path = temp_file.name
        try:
            if not filename.lower().endswith(".pdf"):
                raise ValueError(f"Unsupported file extension: {suffix}")
            raw_docs = PyPDFLoader(temp_path).load()
            for doc in raw_docs:
                doc.metadata.update({"source_filename": filename, "doc_type": doc_type, "scope": scope})
            chunks = self.text_splitter.split_documents(raw_docs)
            facts_saved = False; facts = []
            if scope == "global":
                rag_engine.add_global_document_chunks(chunks, doc_type=doc_type)
            else:
                rag_engine.add_student_document_chunks(chunks, student_id=student_id, doc_type=doc_type)
                if doc_type in {"candidate_profile", "resume", "cv", "assessment", "candidate_document"}:
                    facts = self._extract_durable_facts("\n".join(d.page_content for d in raw_docs))
                    if facts:
                        facts_saved = candidate_memory.save_facts(student_id, facts, source_type="candidate_document", source_reference=filename)
            return {"filename": filename, "status": "success", "scope": scope, "candidate_id": student_id if scope == "candidate" else None, "chunks_indexed": len(chunks), "durable_facts_extracted": len(facts), "durable_facts_saved": facts_saved}
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)


document_parser = DocumentParser()
