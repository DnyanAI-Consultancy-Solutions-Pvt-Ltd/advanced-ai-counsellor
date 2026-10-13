import os
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.config import CHROMA_PERSIST_DIR


class RAGEngine:
    """Candidate-private and shared-global RAG using local HuggingFace embeddings."""
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.internal_db = Chroma(collection_name="internal_knowledge", embedding_function=self.embeddings, persist_directory=os.path.join(self.persist_directory, "internal"))
        self.student_docs_db = Chroma(collection_name="student_documents", embedding_function=self.embeddings, persist_directory=os.path.join(self.persist_directory, "student_docs"))
        self.global_docs_db = Chroma(collection_name="global_documents", embedding_function=self.embeddings, persist_directory=os.path.join(self.persist_directory, "global_docs"))

    def search_internal_knowledge(self, query: str, category: Optional[str] = None, k: int = 4) -> List[Dict[str, Any]]:
        kwargs = {"k": k}
        if category: kwargs["filter"] = {"category": category}
        return self._format_documents(self.internal_db.similarity_search(query, **kwargs))

    def add_internal_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.internal_db.add_texts(texts=texts, metadatas=metadatas)

    def search_student_documents(self, query: str, student_id: str, k: int = 3) -> List[Dict[str, Any]]:
        docs = self.student_docs_db.similarity_search(query, k=k, filter={"student_id": student_id})
        return self._format_documents(docs)

    def search_global_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self._format_documents(self.global_docs_db.similarity_search(query, k=k))

    def add_student_document_chunks(self, chunks: List[Document], student_id: str, doc_type: str) -> None:
        for chunk in chunks:
            chunk.metadata.update({"student_id": student_id, "candidate_id": student_id, "doc_type": doc_type, "scope": "candidate"})
        self.student_docs_db.add_documents(chunks)

    def add_global_document_chunks(self, chunks: List[Document], doc_type: str) -> None:
        for chunk in chunks:
            chunk.metadata.update({"doc_type": doc_type, "scope": "global"})
        self.global_docs_db.add_documents(chunks)

    @staticmethod
    def _format_documents(docs: List[Document]) -> List[Dict[str, Any]]:
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


rag_engine = RAGEngine()
