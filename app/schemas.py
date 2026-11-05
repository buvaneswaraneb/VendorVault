"""Request and response models for VendorVault API."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user prompt or query to send to the LLM")
    stream: bool = Field(default=False, description="Whether to stream the response as SSE chunks")


class QueryResponse(BaseModel):
    query: str
    answer: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_inr: float


class Citation(BaseModel):
    document: str = Field(..., description="The source document name")
    page: int = Field(..., description="The physical page number in the filing")
    heading: str | None = Field(default=None, description="The section heading if available")
    snippet: str = Field(..., description="Relevant excerpt from this page")


class AskRequest(BaseModel):
    query: str = Field(..., description="The user question about cybersecurity 10-K filings")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of document chunks to retrieve")
    document_filter: str | None = Field(
        default=None,
        description="Optional filter by document name or company",
    )
    stream: bool = Field(default=False, description="Whether to stream the response as SSE chunks")


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list,
        description="List of unique citations formatted as [document, page]",
    )
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_inr: float



class UsageResponse(BaseModel):
    total_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    estimated_cost_inr: float
    by_model: dict[str, Any] = Field(default_factory=dict)
    recent_calls: list[dict[str, Any]] = Field(default_factory=list)


# Alias for backward compatibility with earlier naming
UseageResponse = UsageResponse
