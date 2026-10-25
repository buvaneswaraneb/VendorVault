from datetime import datetime, timezone
from sqlalchemy import(create_engine, String, Integer, DateTime)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, sessionmaker)


engine = create_engine("sqlite:///gemini_usage.db")

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

class TokenUsage(Base):
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )
    model: Mapped[str] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


Base.metadata.create_all(engine)