"""Main FastAPI application for VendorVault."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.schemas import AskRequest, AskResponse, QueryRequest, QueryResponse, UsageResponse
from app.services.llm import generate_chat_response, generate_chat_stream
from app.services.rag import answer_ask_query, answer_ask_stream
from app.services.usage import get_usage_summary
from app.services.vector_store import get_chunk_stats

app = FastAPI(
    title="VendorVault API",
    description="Cited answers from US cybersecurity companies' 10-K filings with prompt-injection defense",
    version="0.1.0",
)

SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.get("/health")
def health(settings: SettingsDep) -> dict:
    """Liveness check: process is up. Fast and does not call LLM or DB."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@app.get("/")
def root() -> dict:
    return {"message": "VendorVault API is running", "docs": "/docs", "version": "0.1.0"}


@app.post("/chat", response_model=QueryResponse)
def chat_with_llm(
    request: QueryRequest,
    settings: SettingsDep,
):
    """POST /chat calls an LLM and returns structured JSON or streams SSE chunks."""
    if request.stream:
        return StreamingResponse(
            generate_chat_stream(query=request.query, settings=settings),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        return generate_chat_response(query=request.query, settings=settings)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error generating chat response: {exc}",
        ) from exc


@app.get("/usage", response_model=UsageResponse)
def get_usage(settings: SettingsDep) -> UsageResponse:
    """GET /usage returns aggregated token consumption and estimated costs."""
    return get_usage_summary(database_url=settings.database_url)


@app.post("/ask", response_model=AskResponse)
def ask_filings(
    request: AskRequest,
    settings: SettingsDep,
):
    """POST /ask answers user questions over SEC 10-K filings with [document, page] citations."""
    if request.stream:
        return StreamingResponse(
            answer_ask_stream(
                query=request.query,
                top_k=request.top_k,
                document_filter=request.document_filter,
                settings=settings,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        return answer_ask_query(
            query=request.query,
            top_k=request.top_k,
            document_filter=request.document_filter,
            settings=settings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error generating answer for /ask: {exc}",
        ) from exc


@app.get("/documents/status")
def documents_status(settings: SettingsDep) -> dict:
    """GET /documents/status returns the number of pages, chunks, and filings indexed in pgvector."""
    return get_chunk_stats(database_url=settings.database_url)

