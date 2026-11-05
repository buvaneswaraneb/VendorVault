"""Document chunking and pgvector ingestion pipeline for 10-K filings."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.services.embedding import embed_texts_batch
from app.services.parser import parse_pdf
from app.services.vector_store import (
    get_chunk_stats,
    get_indexed_pages_for_document,
    init_vector_store,
    is_document_already_indexed,
    store_chunks,
)

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chars: int = 4500, overlap: int = 300) -> list[str]:
    """Clean text and split into overlapping chunks while preserving meaning."""
    clean = " ".join(text.split())
    if not clean or len(clean) < 20:
        return []

    if len(clean) <= max_chars:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start += max_chars - overlap

    return chunks


def extract_chunks_from_pdf(
    pdf_path: Path | str,
    max_chars: int = 4500,
    overlap: int = 300,
) -> tuple[int, list[dict[str, Any]]]:
    """Parse a PDF filing and chunk every page with citation metadata."""
    path = Path(pdf_path)
    pages = parse_pdf(path)
    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_chunks = chunk_text(page.text, max_chars=max_chars, overlap=overlap)
        for idx, chunk_str in enumerate(page_chunks):
            chunks.append(
                {
                    "document_name": path.name,
                    "document_path": str(path),
                    "page_number": page.page_number,
                    "heading": page.heading,
                    "chunk_index": idx,
                    "content": chunk_str,
                    "token_count": max(1, len(chunk_str) // 4),
                }
            )

    return len(pages), chunks


def index_document(
    pdf_path: Path | str,
    settings: Settings | None = None,
    batch_size: int = 15,
    sleep_between_batches: float = 14.0,
    force_reindex: bool = False,
    max_chars: int = 4500,
) -> dict[str, Any]:
    """Parse, chunk, embed, and store a PDF filing in pgvector incrementally."""
    s = settings or get_settings()
    path = Path(pdf_path)

    init_vector_store(s.database_url)

    total_pages, chunks = extract_chunks_from_pdf(path, max_chars=max_chars)
    if not chunks:
        return {"document": path.name, "pages": total_pages, "chunks": 0, "status": "empty"}

    if not force_reindex:
        already_indexed = get_indexed_pages_for_document(path.name, s.database_url)
        if len(already_indexed) >= total_pages and is_document_already_indexed(path.name, s.database_url):
            logger.info("Filing '%s' fully indexed (%d pages), skipping.", path.name, len(already_indexed))
            return {
                "document": path.name,
                "status": "already_indexed",
                "pages": len(already_indexed),
                "chunks": len(chunks),
            }
        remaining_chunks = [c for c in chunks if c["page_number"] not in already_indexed]
    else:
        remaining_chunks = chunks

    logger.info(
        "Filing '%s': %d total pages, %d chunks to embed (batch size %d).",
        path.name,
        total_pages,
        len(remaining_chunks),
        batch_size,
    )

    stored_total = 0
    total_batches = (len(remaining_chunks) + batch_size - 1) // batch_size if remaining_chunks else 0

    for b_idx in range(0, len(remaining_chunks), batch_size):
        batch = remaining_chunks[b_idx : b_idx + batch_size]
        batch_num = (b_idx // batch_size) + 1
        logger.info(
            "Embedding batch %d/%d (%d chunks) for '%s'...",
            batch_num,
            total_batches,
            len(batch),
            path.name,
        )

        texts = [c["content"] for c in batch]
        embeddings = embed_texts_batch(
            texts=texts,
            settings=s,
            batch_size=batch_size,
            sleep_between_batches=0,
        )

        for chunk, emb in zip(batch, embeddings, strict=False):
            chunk["embedding"] = emb

        stored_count = store_chunks(batch, database_url=s.database_url)
        stored_total += stored_count

        logger.info(
            "Saved batch %d/%d (%d chunks) to pgvector. Total stored for this file: %d.",
            batch_num,
            total_batches,
            stored_count,
            stored_total,
        )

        if b_idx + batch_size < len(remaining_chunks) and sleep_between_batches > 0:
            logger.info("Sleeping %.1fs between batches...", sleep_between_batches)
            time.sleep(sleep_between_batches)

    stats = get_chunk_stats(s.database_url)
    return {
        "document": path.name,
        "pages": total_pages,
        "chunks": stored_total,
        "status": "indexed",
        "current_total_pages": stats["total_pages"],
        "current_total_chunks": stats["total_chunks"],
    }


def index_all_documents(
    pdf_dir: Path | str = "data/pdf",
    target_pages: int = 500,
    settings: Settings | None = None,
    batch_size: int = 15,
    sleep_between_batches: float = 14.0,
    max_chars: int = 4500,
) -> dict[str, Any]:
    """Index filings until at least target_pages pages are stored in pgvector."""
    s = settings or get_settings()
    dir_path = Path(pdf_dir)
    pdf_files = sorted(dir_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory {dir_path}")

    logger.info("Found %d PDF filings in %s", len(pdf_files), dir_path)

    processed_docs = []
    current_stats = get_chunk_stats(s.database_url)
    total_pages_indexed = current_stats["total_pages"]

    for pdf in pdf_files:
        if total_pages_indexed >= target_pages:
            logger.info(
                "Target reached: %d/%d pages already indexed.",
                total_pages_indexed,
                target_pages,
            )
            break

        result = index_document(
            pdf_path=pdf,
            settings=s,
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
            max_chars=max_chars,
        )
        processed_docs.append(result)
        current_stats = get_chunk_stats(s.database_url)
        total_pages_indexed = current_stats["total_pages"]
        logger.info(
            "Progress: %d/%d total pages indexed so far in pgvector.",
            total_pages_indexed,
            target_pages,
        )

    return {
        "target_pages": target_pages,
        "total_pages_indexed": total_pages_indexed,
        "total_chunks_indexed": current_stats["total_chunks"],
        "processed_documents": processed_docs,
        "summary": current_stats,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    res = index_all_documents(target_pages=target)
    print("Indexing completed successfully!")
    print(res)
