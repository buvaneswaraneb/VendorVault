"""Tests for token usage tracking, cost computation, and GET /usage."""

from app.services.usage import calculate_cost, record_usage


def test_calculate_cost_flash():
    usd, inr = calculate_cost(
        model="gemini-3.8-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        inr_rate=84.0,
    )
    # flash input: $0.075, output: $0.30 -> total $0.375
    assert usd == 0.375
    assert round(inr, 2) == round(0.375 * 84.0, 2)


def test_calculate_cost_lite():
    usd, inr = calculate_cost(
        model="gemini-3.1-flash-lite",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        inr_rate=84.0,
    )
    # lite input: $0.0375, output: $0.15 -> total $0.1875
    assert usd == 0.1875
    assert round(inr, 2) == round(0.1875 * 84.0, 2)


def test_get_usage_empty(client):
    response = client.get("/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 0
    assert data["total_tokens"] == 0
    assert data["estimated_cost_usd"] == 0.0
    assert data["estimated_cost_inr"] == 0.0


def test_get_usage_after_records(client, test_settings):
    record_usage(
        model="gemini-3.8-flash",
        prompt_tokens=100,
        completion_tokens=200,
        query="Test query 1",
        database_url=test_settings.database_url,
    )
    record_usage(
        model="gemini-3.1-flash-lite",
        prompt_tokens=50,
        completion_tokens=50,
        query="Test query 2",
        database_url=test_settings.database_url,
    )

    response = client.get("/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 2
    assert data["total_tokens"] == 400
    assert data["prompt_tokens"] == 150
    assert data["completion_tokens"] == 250
    assert data["estimated_cost_usd"] > 0
    assert data["estimated_cost_inr"] > 0
    assert "gemini-3.8-flash" in data["by_model"]
    assert "gemini-3.1-flash-lite" in data["by_model"]
    assert len(data["recent_calls"]) == 2
