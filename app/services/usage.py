"""Usage and token cost tracking service."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings
from app.schemas import UsageResponse


class Base(DeclarativeBase):
    pass


class TokenUsage(Base):
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


# Rates per million tokens
MODEL_PRICING = {
    # Gemini 2.5 / 1.5 / 3.8 Flash: $0.075 per 1M input, $0.30 per 1M output
    "flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    # Flash Lite / Flash 8B: $0.0375 per 1M input, $0.15 per 1M output
    "lite": {"input_per_m": 0.0375, "output_per_m": 0.15},
    # Default fallback rate
    "default": {"input_per_m": 0.10, "output_per_m": 0.40},
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    inr_rate: float = 84.0,
) -> tuple[float, float]:
    """Calculates estimated cost in USD and INR for a given model and token count."""
    model_lower = model.lower()
    if "lite" in model_lower:
        rates = MODEL_PRICING["lite"]
    elif "flash" in model_lower:
        rates = MODEL_PRICING["flash"]
    else:
        rates = MODEL_PRICING["default"]

    cost_usd = (prompt_tokens * rates["input_per_m"] / 1_000_000) + (
        completion_tokens * rates["output_per_m"] / 1_000_000
    )
    cost_inr = cost_usd * inr_rate
    return round(cost_usd, 7), round(cost_inr, 5)


_engine_cache: dict[str, Any] = {}


def get_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    if url not in _engine_cache:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        engine = create_engine(url, connect_args=connect_args)
        Base.metadata.create_all(engine)
        _engine_cache[url] = engine
    return _engine_cache[url]


def get_session(database_url: str | None = None):
    engine = get_engine(database_url)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return session_factory()


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    query: str = "",
    database_url: str | None = None,
    inr_rate: float | None = None,
) -> TokenUsage:
    """Persists a new LLM call record with calculated costs."""
    settings = get_settings()
    rate = inr_rate if inr_rate is not None else settings.usd_to_inr_rate
    cost_usd, cost_inr = calculate_cost(model, prompt_tokens, completion_tokens, rate)
    total_tokens = prompt_tokens + completion_tokens

    record = TokenUsage(
        query=query[:200],  # store a truncated sample of the query for reference
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_inr=cost_inr,
    )

    with get_session(database_url) as session:
        session.add(record)
        session.commit()
        session.refresh(record)

    return record


def get_usage_summary(database_url: str | None = None) -> UsageResponse:
    """Aggregates all usage records into a structured summary."""
    with get_session(database_url) as session:
        records = session.scalars(
            select(TokenUsage).order_by(TokenUsage.created_at.desc())
        ).all()

    total_requests = len(records)
    total_tokens = sum(r.total_tokens for r in records)
    prompt_tokens = sum(r.prompt_tokens for r in records)
    completion_tokens = sum(r.completion_tokens for r in records)
    estimated_cost_usd = round(sum(r.cost_usd for r in records), 6)
    estimated_cost_inr = round(sum(r.cost_inr for r in records), 4)

    by_model: dict[str, dict[str, Any]] = {}
    for r in records:
        if r.model not in by_model:
            by_model[r.model] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_inr": 0.0,
            }
        m = by_model[r.model]
        m["requests"] += 1
        m["prompt_tokens"] += r.prompt_tokens
        m["completion_tokens"] += r.completion_tokens
        m["total_tokens"] += r.total_tokens
        m["cost_usd"] = round(m["cost_usd"] + r.cost_usd, 6)
        m["cost_inr"] = round(m["cost_inr"] + r.cost_inr, 4)

    recent_calls = [
        {
            "id": r.id,
            "model": r.model,
            "query": r.query,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "cost_usd": r.cost_usd,
            "cost_inr": r.cost_inr,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records[:10]
    ]

    return UsageResponse(
        total_requests=total_requests,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        estimated_cost_inr=estimated_cost_inr,
        by_model=by_model,
        recent_calls=recent_calls,
    )
