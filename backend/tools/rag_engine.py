import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.config import CHROMA_PERSIST_DIR


class RAGEngine:
    """RAG Engine powered by free local HuggingFace Embeddings."""

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory

        # Local CPU embeddings model (runs offline & 100% free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize vector stores
        self.internal_db = Chroma(
            collection_name="internal_knowledge",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(self.persist_directory, "internal")
        )

        self.student_docs_db = Chroma(
            collection_name="student_documents",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(self.persist_directory, "student_docs")
        )

    def search_internal_knowledge(
        self, query: str, category: Optional[str] = None, k: int = 4
    ) -> List[Dict[str, Any]]:
        search_kwargs = {"k": k}
        if category:
            search_kwargs["filter"] = {"category": category}

        docs = self.internal_db.similarity_search(query, **search_kwargs)
        return self._format_documents(docs)

    def add_internal_documents(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        self.internal_db.add_texts(texts=texts, metadatas=metadatas)

    def search_student_documents(
        self, query: str, student_id: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        docs = self.student_docs_db.similarity_search(
            query, k=k, filter={"student_id": student_id}
        )
        return self._format_documents(docs)

    def add_student_document_chunks(
        self, chunks: List[Document], student_id: str, doc_type: str
    ) -> None:
        for chunk in chunks:
            chunk.metadata["student_id"] = student_id
            chunk.metadata["doc_type"] = doc_type

        self.student_docs_db.add_documents(chunks)

    @staticmethod
    def _format_documents(docs: List[Document]) -> List[Dict[str, Any]]:
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


rag_engine = RAGEngine()