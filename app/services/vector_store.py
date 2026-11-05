"""pgvector storage service for chunked SEC 10-K filings with metadata."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    distinct,
    func,
    select,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.services.usage import Base, get_engine, get_session

logger = logging.getLogger(__name__)


class DocumentChunk(Base):
    """Database record for a chunk extracted from a filing page."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    document_path: Mapped[str] = mapped_column(String(500), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    def to_citation_dict(self) -> dict[str, Any]:
        return {
            "document": self.document_name,
            "page": self.page_number,
            "heading": self.heading,
            "snippet": self.content[:300] + ("..." if len(self.content) > 300 else ""),
        }


def init_vector_store(database_url: str | None = None) -> None:
    """Initialize vector extension, tables, and HNSW index if supported."""
    engine = get_engine(database_url)
    is_postgres = engine.dialect.name == "postgresql"

    if is_postgres:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

    Base.metadata.create_all(engine)

    if is_postgres:
        with engine.connect() as conn:
            try:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
                        "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
                    )
                )
                conn.commit()
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not create HNSW index: %s", exc)


def store_chunks(
    chunks: list[dict[str, Any]],
    database_url: str | None = None,
) -> int:
    """Persist multiple document chunks to pgvector."""
    if not chunks:
        return 0

    init_vector_store(database_url)

    records = [
        DocumentChunk(
            document_name=c["document_name"],
            document_path=c.get("document_path", ""),
            page_number=c["page_number"],
            heading=c.get("heading"),
            chunk_index=c.get("chunk_index", 0),
            content=c["content"],
            token_count=c.get("token_count", max(1, len(c["content"]) // 4)),
            embedding=c.get("embedding"),
        )
        for c in chunks
    ]

    with get_session(database_url) as session:
        session.add_all(records)
        session.commit()

    return len(records)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure python cosine similarity fallback for non-pgvector backends."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    database_url: str | None = None,
    document_filter: str | None = None,
) -> list[DocumentChunk]:
    """Retrieve top-k similar chunks using cosine distance in pgvector or python fallback."""
    init_vector_store(database_url)
    engine = get_engine(database_url)

    with get_session(database_url) as session:
        stmt = select(DocumentChunk)
        if document_filter:
            stmt = stmt.where(DocumentChunk.document_name.ilike(f"%{document_filter}%"))

        if engine.dialect.name == "postgresql":
            # Native pgvector cosine distance operator <=>
            stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(top_k)
            return list(session.scalars(stmt).all())

        # Dialect fallback for SQLite in CI/tests
        all_chunks = list(session.scalars(stmt).all())
        scored = []
        for chunk in all_chunks:
            chunk_vec = chunk.embedding
            if chunk_vec is not None:
                if hasattr(chunk_vec, "tolist"):
                    chunk_vec = chunk_vec.tolist()
                score = _cosine_similarity(query_embedding, list(chunk_vec))
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


def get_chunk_stats(database_url: str | None = None) -> dict[str, Any]:
    """Return counts of total chunks, unique pages, and documents indexed."""
    init_vector_store(database_url)
    with get_session(database_url) as session:
        total_chunks = session.scalar(select(func.count(DocumentChunk.id))) or 0
        total_pages = (
            session.scalar(
                select(
                    func.count(
                        distinct(
                            DocumentChunk.document_name + ":" + func.cast(DocumentChunk.page_number, String)
                        )
                    )
                )
            )
            or 0
        )

        doc_rows = session.execute(
            select(
                DocumentChunk.document_name,
                func.count(distinct(DocumentChunk.page_number)),
                func.count(DocumentChunk.id),
            ).group_by(DocumentChunk.document_name)
        ).all()

        documents = [
            {
                "document": row[0],
                "pages": row[1],
                "chunks": row[2],
            }
            for row in doc_rows
        ]

    return {
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "documents": documents,
    }


def is_document_already_indexed(
    document_name: str,
    database_url: str | None = None,
) -> bool:
    """Check if a filing document is already indexed in pgvector."""
    init_vector_store(database_url)
    with get_session(database_url) as session:
        count = (
            session.scalar(
                select(func.count(DocumentChunk.id)).where(DocumentChunk.document_name == document_name)
            )
            or 0
        )
        return count > 0


def get_indexed_pages_for_document(
    document_name: str,
    database_url: str | None = None,
) -> set[int]:
    """Return set of page numbers already stored in pgvector for a given document."""
    init_vector_store(database_url)
    with get_session(database_url) as session:
        pages = session.scalars(
            select(DocumentChunk.page_number)
            .distinct()
            .where(DocumentChunk.document_name == document_name)
        ).all()
        return set(pages)

