"""LLM service handling generation, retry logic, timeout, fallback, streaming, and cost tracking."""

import json
import logging
import time
from collections.abc import Generator
from typing import Any

from google import genai
from google.genai.errors import APIError

from app.config import Settings, get_settings
from app.schemas import QueryResponse
from app.services.usage import calculate_cost, record_usage

logger = logging.getLogger(__name__)


def get_genai_client(settings: Settings | None = None) -> genai.Client:
    s = settings or get_settings()
    timeout_ms = int(max(s.llm_timeout_seconds, 10.0) * 1000)
    return genai.Client(api_key=s.gemini_api_key, http_options={"timeout": timeout_ms})


def _extract_usage(response: Any, query: str, answer: str) -> tuple[int, int, int]:
    """Extracts prompt, completion and total token counts from response or estimates them."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )

    # Fallback estimation if metadata is absent
    if total_tokens == 0:
        prompt_tokens = max(1, len(query) // 4)
        completion_tokens = max(1, len(answer) // 4)
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def generate_chat_response(
    query: str,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> QueryResponse:
    """Generates a chat response with retry on primary model and fallback to secondary model."""
    s = settings or get_settings()
    genai_client = client or get_genai_client(s)

    primary_model = s.primary_model
    secondary_model = s.secondary_model
    max_retries = s.llm_max_retries

    # 1. Try primary model with retries
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = genai_client.models.generate_content(
                model=primary_model,
                contents=query,
            )
            answer = response.text or ""
            p_tokens, c_tokens, t_tokens = _extract_usage(response, query, answer)
            cost_usd, cost_inr = calculate_cost(
                primary_model, p_tokens, c_tokens, inr_rate=s.usd_to_inr_rate
            )

            record_usage(
                model=primary_model,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                query=query,
                database_url=s.database_url,
                inr_rate=s.usd_to_inr_rate,
            )

            return QueryResponse(
                query=query,
                answer=answer,
                model_used=primary_model,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                cost_usd=cost_usd,
                cost_inr=cost_inr,
            )
        except (APIError, TimeoutError, Exception) as exc:
            last_exception = exc
            logger.warning(
                "Primary model %s attempt %d/%d failed: %s",
                primary_model,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt < max_retries:
                time.sleep(0.5 * (2**attempt))

    # 2. Primary failed, trigger fallback to secondary model
    logger.info("Falling back to secondary model: %s", secondary_model)
    try:
        response = genai_client.models.generate_content(
            model=secondary_model,
            contents=query,
        )
        answer = response.text or ""
        p_tokens, c_tokens, t_tokens = _extract_usage(response, query, answer)
        cost_usd, cost_inr = calculate_cost(
            secondary_model, p_tokens, c_tokens, inr_rate=s.usd_to_inr_rate
        )

        record_usage(
            model=secondary_model,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            query=query,
            database_url=s.database_url,
            inr_rate=s.usd_to_inr_rate,
        )

        return QueryResponse(
            query=query,
            answer=answer,
            model_used=secondary_model,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            cost_usd=cost_usd,
            cost_inr=cost_inr,
        )
    except Exception as secondary_exc:
        logger.error(
            "Secondary model %s failed after primary failed: %s",
            secondary_model,
            secondary_exc,
        )
        raise RuntimeError(
            f"All models failed. Primary ({primary_model}): {last_exception}; "
            f"Secondary ({secondary_model}): {secondary_exc}"
        ) from secondary_exc


def generate_chat_stream(
    query: str,
    settings: Settings | None = None,
    client: genai.Client | None = None,
) -> Generator[str, None, None]:
    """Generates SSE chunks for streaming chat with fallback model support."""
    s = settings or get_settings()
    genai_client = client or get_genai_client(s)

    primary_model = s.primary_model
    secondary_model = s.secondary_model

    chosen_model = primary_model
    stream_iter = None

    try:
        stream_iter = genai_client.models.generate_content_stream(
            model=primary_model,
            contents=query,
        )
        # Check first chunk to catch connection/model errors early
        first_chunk = next(stream_iter, None)
    except Exception as exc:
        logger.warning(
            "Streaming with primary model %s failed: %s. Falling back to %s",
            primary_model,
            exc,
            secondary_model,
        )
        chosen_model = secondary_model
        try:
            stream_iter = genai_client.models.generate_content_stream(
                model=secondary_model,
                contents=query,
            )
            first_chunk = next(stream_iter, None)
        except Exception as fallback_exc:
            error_data = json.dumps({"error": f"Streaming failed: {fallback_exc}"})
            yield f"data: {error_data}\n\n"
            return

    collected_text = []

    if first_chunk is not None:
        chunk_text = getattr(first_chunk, "text", "") or ""
        if chunk_text:
            collected_text.append(chunk_text)
            yield f"data: {json.dumps({'text': chunk_text, 'model': chosen_model})}\n\n"

    if stream_iter:
        for chunk in stream_iter:
            chunk_text = getattr(chunk, "text", "") or ""
            if chunk_text:
                collected_text.append(chunk_text)
                yield f"data: {json.dumps({'text': chunk_text, 'model': chosen_model})}\n\n"

    full_answer = "".join(collected_text)
    p_tokens = max(1, len(query) // 4)
    c_tokens = max(1, len(full_answer) // 4)

    record_usage(
        model=chosen_model,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        query=query,
        database_url=s.database_url,
        inr_rate=s.usd_to_inr_rate,
    )

    summary_event = json.dumps(
        {
            "done": True,
            "model_used": chosen_model,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
        }
    )
    yield f"data: {summary_event}\n\n"
    yield "data: [DONE]\n\n"


# For backward compatibility
def genrate_response(SETTINGS, QR):
    return generate_chat_response(query=QR.query, settings=SETTINGS)
