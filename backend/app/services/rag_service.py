"""
RAG (Retrieval-Augmented Generation) service for MemoryOS.

Retrieves relevant document chunks from the vector store and uses an LLM to generate
grounded, concise answers to user questions based strictly on retrieved context.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai

from app.core.config import settings
from app.services.embedding_service import get_embeddings
from app.services.vector_store import VectorStoreService, get_vector_store

logger = logging.getLogger(__name__)

NO_INFO_FALLBACK = "I could not find information about that in your stored memory."


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Build system and user prompts grounded in retrieved context.

    Parameters
    ----------
    question : str
        The user's question.
    chunks : List[Dict[str, Any]]
        List of retrieved chunk dicts from vector store.

    Returns
    -------
    Tuple[str, str]
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are MemoryOS, an AI assistant retrieving facts from the user's personal memory database.\n"
        "Instructions:\n"
        "1. Answer the user's question ONLY using the provided retrieved context below.\n"
        "2. Do NOT invent, assume, or extrapolate facts outside the context.\n"
        "3. If the retrieved context does not contain enough information to answer the question, clearly state:\n"
        f"   '{NO_INFO_FALLBACK}'\n"
        "4. Keep your answer concise, clear, and direct."
    )

    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        source = c.get("metadata", {}).get("source", "Document")
        chunk_idx = c.get("metadata", {}).get("chunk_index")
        idx_str = f" Chunk #{chunk_idx + 1}" if chunk_idx is not None else ""
        context_blocks.append(f"[{idx}] (Source: {source}{idx_str})\n{c['text']}")

    context_str = "\n\n".join(context_blocks)

    user_prompt = (
        f"Retrieved Context:\n"
        f"------------------\n"
        f"{context_str}\n"
        f"------------------\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return system_prompt, user_prompt


class RAGService:
    """
    Service wrapper for Retrieval-Augmented Generation.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_client: Optional[Callable[[str, str], str]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._vector_store = vector_store
        self._llm_client = llm_client
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name or settings.OPENAI_MODEL
        self.base_url = base_url or settings.OPENAI_BASE_URL or None

    def get_vector_store(self) -> VectorStoreService:
        if self._vector_store:
            return self._vector_store
        return get_vector_store()

    def generate_llm_answer(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send system and user prompts to the LLM.
        Uses custom callable llm_client if provided (e.g. for testing/mocking),
        otherwise uses the official OpenAI client.
        """
        if self._llm_client is not None:
            return self._llm_client(system_prompt, user_prompt)

        if not self.api_key:
            raise RuntimeError(
                "LLM API key is not configured. "
                "Please set OPENAI_API_KEY in your environment or .env file."
            )

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = openai.OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content or ""
        return answer.strip()

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Process a user question through the complete RAG pipeline:
        1. Generate query embedding
        2. Retrieve top-k relevant chunks from vector store
        3. Build grounded RAG prompt
        4. Query LLM
        5. Return grounded answer and source metadata

        Parameters
        ----------
        question : str
            The user's natural language question.
        top_k : int, default=5
            Number of top matching chunks to retrieve.

        Returns
        -------
        Dict[str, Any]
            Result dictionary containing:
            - 'question': str
            - 'answer': str
            - 'sources': List[Dict]
            - 'total_sources': int
            - 'prompt': Optional[str]
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty or whitespace.")

        clean_question = question.strip()
        logger.info("RAG query received: %r (top_k=%d)", clean_question, top_k)

        # 1. Embed question
        query_embedding = get_embeddings([clean_question])[0]

        # 2. Query vector store
        store = self.get_vector_store()
        chunks = store.query_similar(query_embedding=query_embedding, n_results=top_k)

        # Handle case where vector store has no relevant chunks
        if not chunks:
            logger.info("No matching chunks retrieved for question: %r", clean_question)
            return {
                "question": clean_question,
                "answer": NO_INFO_FALLBACK,
                "sources": [],
                "total_sources": 0,
                "prompt": None,
            }

        # 3. Build grounded prompt
        system_prompt, user_prompt = build_rag_prompt(clean_question, chunks)

        # 4. Generate LLM answer
        answer = self.generate_llm_answer(system_prompt, user_prompt)

        # 5. Format sources
        sources = [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c.get("metadata", {}),
                "distance": c.get("distance"),
            }
            for c in chunks
        ]

        return {
            "question": clean_question,
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources),
            "prompt": user_prompt,
        }


# Convenience module-level function
def answer_question(
    question: str,
    top_k: int = 5,
    vector_store: Optional[VectorStoreService] = None,
    llm_client: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to answer a question using the RAG service.
    """
    service = RAGService(vector_store=vector_store, llm_client=llm_client)
    return service.answer_question(question=question, top_k=top_k)
