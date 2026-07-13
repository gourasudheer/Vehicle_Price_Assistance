"""
Ingestion pipeline: reads PDF(s) from DATA_DIR, splits them into chunks,
embeds each chunk with a local sentence-transformers model, and stores
them (with metadata) in a persistent local ChromaDB collection.

Run once whenever your data changes:
    python ingest.py

This is idempotent: it recreates the collection from scratch each run,
so re-running after editing/adding PDFs is always safe.
"""

import logging
import re
from pathlib import Path

import chromadb
import pdfplumber
from chromadb.utils import embedding_functions

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# 1. Extract text from PDFs
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts plain text from every page of a PDF."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# 2. Chunk text
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping chunks on paragraph/sentence-friendly
    boundaries where possible, falling back to a hard character split.
    """
    # Prefer splitting on blank-line-separated "records" (our PDFs are
    # built one-vehicle-per-block, so this keeps each vehicle intact).
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    chunks = []
    buffer = ""
    for block in blocks:
        if len(buffer) + len(block) + 1 <= chunk_size:
            buffer = f"{buffer}\n{block}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(block) <= chunk_size:
                buffer = block
            else:
                # block itself too big -> hard split with overlap
                start = 0
                while start < len(block):
                    end = start + chunk_size
                    chunks.append(block[start:end])
                    start = end - overlap
                buffer = ""
    if buffer:
        chunks.append(buffer)

    return chunks


# ---------------------------------------------------------------------------
# 3. Build / persist the ChromaDB collection
# ---------------------------------------------------------------------------
def build_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    # Fresh start every run so stale data never lingers.
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s", config.DATA_DIR)
        return

    all_ids, all_docs, all_metas = [], [], []
    for pdf_path in pdf_files:
        log.info("Reading %s", pdf_path.name)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        category = "bike" if "bike" in pdf_path.stem.lower() else (
            "car" if "car" in pdf_path.stem.lower() else "unknown"
        )
        log.info("  -> %d chunks (category=%s)", len(chunks), category)

        for i, chunk in enumerate(chunks):
            all_ids.append(f"{pdf_path.stem}_{i}")
            all_docs.append(chunk)
            all_metas.append({"source": pdf_path.name, "category": category, "chunk_index": i})

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
    log.info("Ingested %d chunks into collection '%s' at %s",
              len(all_ids), config.CHROMA_COLLECTION_NAME, config.CHROMA_PERSIST_DIR)


if __name__ == "__main__":
    build_collection()
