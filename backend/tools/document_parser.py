import os
import tempfile
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.tools.rag_engine import rag_engine


class DocumentParser:
    """Parses PDF marksheets, notes, and certificates into chunks and indexes them."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    async def process_and_index_file(
        self, file_bytes: bytes, filename: str, student_id: str, doc_type: str
    ) -> Dict[str, Any]:
        """Saves file bytes to temp storage, extracts text chunks, and adds to Chroma Vector Store."""
        suffix = os.path.splitext(filename)[-1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                raw_docs = loader.load()
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")

            # Inject source metadata before splitting
            for doc in raw_docs:
                doc.metadata["source_filename"] = filename
                doc.metadata["doc_type"] = doc_type

            chunks = self.text_splitter.split_documents(raw_docs)

            # Store in student_documents collection
            rag_engine.add_student_document_chunks(
                chunks=chunks,
                student_id=student_id,
                doc_type=doc_type
            )

            return {
                "filename": filename,
                "status": "success",
                "chunks_indexed": len(chunks)
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


document_parser = DocumentParser()