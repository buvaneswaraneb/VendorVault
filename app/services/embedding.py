"""Embedding service using Google GenAI with batching, retry, and rate-limit handling."""

from __future__ import annotations

import logging
import re
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import Settings, get_settings
from app.services.llm import get_genai_client

logger = logging.getLogger(__name__)


def _extract_retry_delay(exc: Exception) -> float:
    """Extract recommended retry delay from API error if present, else return default."""
    exc_str = str(exc)
    match = re.search(r"retry in ([\d\.]+)s", exc_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    match_sec = re.search(r"retryDelay':\s*'(\d+)s'", exc_str)
    if match_sec:
        return float(match_sec.group(1)) + 2.0
    return 25.0


def embed_text(
    text: str,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> list[float]:
    """Generate a 768-dimensional embedding vector for a single text."""
    results = embed_texts_batch([text], settings=settings, client=client, batch_size=1)
    if not results:
        raise RuntimeError("Embedding generation returned empty result.")
    return results[0]


def embed_texts_batch(
    texts: list[str],
    settings: Settings | None = None,
    client: genai.Client | None = None,
    batch_size: int = 15,
    sleep_between_batches: float = 0.5,
) -> list[list[float]]:
    """Generate embeddings for a list of texts with batching and rate-limit handling."""
    if not texts:
        return []

    s = settings or get_settings()
    genai_client = client or get_genai_client(s)
    model = s.embedding_model
    target_dim = s.embedding_dimensions

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Clean and truncate texts to keep token count safely below TPM limit
        batch_cleaned = [
            t[:1500] if t.strip() else "empty document section" for t in batch
        ]

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                resp = genai_client.models.embed_content(
                    model=model,
                    contents=batch_cleaned,
                    config=types.EmbedContentConfig(output_dimensionality=target_dim),
                )
                embeddings_data = resp.embeddings
                for emb in embeddings_data:
                    values = getattr(emb, "values", None) or []
                    all_embeddings.append(list(values))
                break
            except (APIError, Exception) as exc:
                is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if is_rate_limit:
                    wait_sec = _extract_retry_delay(exc)
                else:
                    wait_sec = 5.0 * (attempt + 1)

                logger.warning(
                    "Embedding batch %d-%d failed (attempt %d/%d): %s. Waiting %.1fs...",
                    i,
                    min(i + batch_size, len(texts)),
                    attempt + 1,
                    max_attempts,
                    exc,
                    wait_sec,
                )
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to generate embeddings after {max_attempts} attempts: {exc}"
                    ) from exc
                time.sleep(wait_sec)

        if i + batch_size < len(texts) and sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    return all_embeddings
