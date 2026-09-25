import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
logger = logging.getLogger("candidate_memory")


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv(
        "COUNSELLING_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ))


class CandidateMemoryStore:
    """Supabase memory: private candidate episodes/facts + shared reusable Q&A."""

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")
        self.table = os.getenv("COUNSELLING_MEMORY_TABLE", "counselling_memories")
        self.rpc_name = os.getenv("COUNSELLING_MEMORY_RPC", "match_counselling_memories")
        self.match_threshold = float(os.getenv("COUNSELLING_MEMORY_MATCH_THRESHOLD", "0.55"))
        self.facts_table = os.getenv("COUNSELLING_FACTS_TABLE", "candidate_facts")
        self.global_table = os.getenv("COUNSELLING_GLOBAL_QA_TABLE", "global_qa_memory")
        self.global_rpc = os.getenv("COUNSELLING_GLOBAL_QA_RPC", "match_global_qa_memory")
        self.global_threshold = float(os.getenv("COUNSELLING_GLOBAL_QA_THRESHOLD", "0.90"))

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self, prefer: str = "") -> Dict[str, str]:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def embed(self, text: str) -> Optional[List[float]]:
        text = " ".join((text or "").split())
        if not text:
            return None
        try:
            vector = _embedding_model().encode(text, normalize_embeddings=True, show_progress_bar=False)
            return [float(v) for v in vector.tolist()]
        except Exception as exc:
            logger.warning("Memory embedding skipped: %s", exc)
            return None

    def recent(self, candidate_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        if not self.enabled or not candidate_id:
            return []
        try:
            params = {
                "candidate_id": f"eq.{candidate_id}",
                "select": "id,candidate_id,user_message,counsellor_response,query_summary,established_facts,created_at",
                "order": "created_at.desc",
                "limit": str(limit),
            }
            with httpx.Client(timeout=12.0) as client:
                r = client.get(f"{self.url}/rest/v1/{self.table}", params=params, headers=self._headers())
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("Candidate recent memory lookup skipped: %s", exc)
            return []

    def search(self, candidate_id: str, query: str, limit: int = 5,
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.enabled or not candidate_id:
            return []
        embedding = self.embed(query)
        if not embedding:
            return []
        payload = {
            "p_candidate_id": candidate_id,
            "query_embedding": embedding,
            "match_threshold": self.match_threshold if threshold is None else float(threshold),
            "match_count": max(1, min(int(limit), 20)),
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(f"{self.url}/rest/v1/rpc/{self.rpc_name}", headers=self._headers(), content=json.dumps(payload))
                r.raise_for_status()
                rows = r.json()
            for row in rows:
                row["similarity"] = round(float(row.get("similarity", 0.0)), 4)
            return rows
        except Exception as exc:
            logger.warning("Candidate semantic memory lookup skipped: %s", exc)
            return []

    def facts(self, candidate_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Read durable facts for this candidate only."""
        if not self.enabled or not candidate_id:
            return []
        try:
            params = {
                "candidate_id": f"eq.{candidate_id}",
                "select": "id,candidate_id,fact_text,source_memory_id,created_at,updated_at",
                "order": "updated_at.desc",
                "limit": str(limit),
            }
            with httpx.Client(timeout=12.0) as client:
                r = client.get(f"{self.url}/rest/v1/{self.facts_table}", params=params, headers=self._headers())
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("Candidate facts lookup skipped: %s", exc)
            return []

    def save_facts(self, candidate_id: str, facts: List[str]) -> bool:
        clean = []
        seen = set()
        for fact in facts or []:
            value = " ".join(str(fact).split()).strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                clean.append(value)
        if not self.enabled or not candidate_id or not clean:
            return False
        rows = [{"candidate_id": candidate_id, "fact_text": fact} for fact in clean]
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(
                    f"{self.url}/rest/v1/{self.facts_table}?on_conflict=candidate_id,fact_text",
                    headers=self._headers("resolution=merge-duplicates,return=minimal"),
                    content=json.dumps(rows),
                )
                r.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Candidate facts save skipped: %s", exc)
            return False

    def search_global(self, query: str, limit: int = 3,
                      threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search reusable, non-candidate-specific Q&A across all users."""
        if not self.enabled:
            return []
        embedding = self.embed(query)
        if not embedding:
            return []
        payload = {
            "query_embedding": embedding,
            "match_threshold": self.global_threshold if threshold is None else float(threshold),
            "match_count": max(1, min(int(limit), 10)),
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(f"{self.url}/rest/v1/rpc/{self.global_rpc}", headers=self._headers(), content=json.dumps(payload))
                r.raise_for_status()
                rows = r.json()
            for row in rows:
                row["similarity"] = round(float(row.get("similarity", 0.0)), 4)
            return rows
        except Exception as exc:
            logger.warning("Global Q&A lookup skipped: %s", exc)
            return []

    def save_global(self, question: str, answer: str, category: str = "general") -> bool:
        embedding = self.embed(question)
        if not self.enabled or not question or not answer or not embedding:
            return False
        record = {
            "question": question.strip(),
            "answer": answer.strip(),
            "embedding": embedding,
            "category": category,
            "reusable": True,
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(f"{self.url}/rest/v1/{self.global_table}", headers=self._headers("return=minimal"), content=json.dumps(record))
                r.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Global Q&A save skipped: %s", exc)
            return False

    def increment_global_usage(self, memory_id: Any) -> None:
        if not self.enabled or memory_id is None:
            return
        try:
            # RPC keeps increment atomic; failure must never block counselling.
            with httpx.Client(timeout=8.0) as client:
                r = client.post(
                    f"{self.url}/rest/v1/rpc/increment_global_qa_usage",
                    headers=self._headers(),
                    content=json.dumps({"p_id": memory_id}),
                )
                r.raise_for_status()
        except Exception as exc:
            logger.debug("Global Q&A usage increment skipped: %s", exc)

    def save(self, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        record = dict(payload)
        memory_text = "\n".join(p for p in [record.get("user_message", ""), record.get("query_summary", "")] if p).strip()
        embedding = self.embed(memory_text)
        if not embedding:
            logger.warning("Candidate memory save skipped because embedding generation failed.")
            return False
        record["embedding"] = embedding
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(f"{self.url}/rest/v1/{self.table}", headers=self._headers("return=representation"), content=json.dumps(record))
                r.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Candidate memory save skipped: %s", exc)
            return False


candidate_memory = CandidateMemoryStore()


@tool
def search_candidate_memory(candidate_id: str, query: str) -> str:
    """Semantically search only this candidate's remote counselling history."""
    return json.dumps(candidate_memory.search(candidate_id, query), ensure_ascii=False)


@tool
def save_candidate_memory(candidate_id: str, user_message: str, counsellor_response: str,
                          query_summary: str = "") -> str:
    """Persist one completed counselling turn with its semantic embedding."""
    ok = candidate_memory.save({
        "candidate_id": candidate_id,
        "user_message": user_message,
        "counsellor_response": counsellor_response,
        "query_summary": query_summary,
        "established_facts": [],
    })
    return "saved" if ok else "memory_not_configured_or_unavailable"