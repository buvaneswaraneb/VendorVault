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
