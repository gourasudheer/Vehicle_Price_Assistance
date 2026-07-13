"""
Production RAG query pipeline:
  1. Embed the user's question (same local embedder used at ingest time).
  2. Retrieve the top-K most relevant chunks from ChromaDB.
  3. Build a grounded prompt and call an LLM via OpenRouter for the answer.

Usage:
    from rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    result = rag.ask("What is the on-road price and mileage of Honda Activa 6G?")
    print(result["answer"])
"""

import logging
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag_pipeline")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about bike and \
car prices and mileage, using ONLY the CONTEXT provided below.

Rules:
- Always state On-Road Price, Ex-Showroom Price and Mileage when they are asked \
  about or available in the context, clearly labelled.
- If multiple matching vehicles appear in the context, list each one separately.
- If the answer is not present in the context, say you don't have that vehicle in \
  the current dataset — do NOT make up numbers.
- Be concise and use a clean bullet-point format for specs.
"""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    category: str
    distance: float


@dataclass
class RAGResult:
    answer: str
    chunks: list = field(default_factory=list)


class RAGPipeline:
    def __init__(self):
        config.require_openrouter_key()

        self._chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        try:
            self._collection = self._chroma_client.get_collection(
                name=config.CHROMA_COLLECTION_NAME, embedding_function=self._embed_fn
            )
        except Exception as e:
            raise RuntimeError(
                "ChromaDB collection not found. Run `python ingest.py` first to "
                "build the vector database from your PDF data."
            ) from e

        self._llm = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    # -------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = None) -> list[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        results = self._collection.query(query_texts=[query], n_results=top_k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                category=meta.get("category", "unknown"),
                distance=dist,
            ))
        return chunks

    # -------------------------------------------------------------------
    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n---\n\n".join(c.text for c in chunks)
        return (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"Answer using only the CONTEXT above."
        )

    # -------------------------------------------------------------------
    def ask(self, query: str, top_k: int = None, model: str = None) -> dict:
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return {"answer": "I couldn't find anything relevant in the dataset.", "chunks": []}

        user_prompt = self._build_prompt(query, chunks)
        model = model or config.OPENROUTER_MODEL

        log.info("Calling OpenRouter model=%s with %d retrieved chunks", model, len(chunks))
        response = self._llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": config.OPENROUTER_SITE_URL,
                "X-Title": config.OPENROUTER_APP_NAME,
            },
        )
        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "chunks": [
                {"text": c.text, "source": c.source, "category": c.category, "distance": c.distance}
                for c in chunks
            ],
        }


if __name__ == "__main__":
    rag = RAGPipeline()
    while True:
        q = input("\nAsk about a bike/car (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = rag.ask(q)
        print("\n--- ANSWER ---")
        print(result["answer"])
