"""Tests for POST /chat endpoint, streaming, timeouts, retries, and fallback."""

from unittest.mock import MagicMock, patch

from google.genai.errors import APIError


class MockUsage:
    def __init__(self, prompt=12, candidates=25):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = prompt + candidates


class MockResponse:
    def __init__(self, text="Sample 10-K analysis", prompt=12, candidates=25):
        self.text = text
        self.usage_metadata = MockUsage(prompt, candidates)


class MockChunk:
    def __init__(self, text):
        self.text = text


def test_chat_success_primary_model(client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MockResponse(
        text="CrowdStrike reported incident details in Item 1C.", prompt=15, candidates=30
    )

    with patch("app.services.llm.get_genai_client", return_value=mock_client):
        payload = {"query": "Explain CrowdStrike 10-K Item 1C."}
        response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Explain CrowdStrike 10-K Item 1C."
    assert "CrowdStrike" in data["answer"]
    assert data["model_used"] == "gemini-3.8-flash"
    assert data["prompt_tokens"] == 15
    assert data["completion_tokens"] == 30
    assert data["total_tokens"] == 45
    assert data["cost_usd"] > 0
    assert data["cost_inr"] > 0


def test_chat_fallback_when_primary_fails(client):
    mock_client = MagicMock()

    # Primary model throws APIError; Secondary succeeds
    def mock_generate(model, contents):
        if model == "gemini-3.8-flash":
            raise APIError("503 Service Unavailable")
        return MockResponse(text="Fallback response from Zscaler data", prompt=10, candidates=20)

    mock_client.models.generate_content.side_effect = mock_generate

    with patch("app.services.llm.get_genai_client", return_value=mock_client):
        payload = {"query": "What are Zscaler cybersecurity risks?"}
        response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "gemini-3.1-flash-lite"
    assert data["answer"] == "Fallback response from Zscaler data"
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 20


def test_chat_all_models_fail(client):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("All backends unreachable")

    with patch("app.services.llm.get_genai_client", return_value=mock_client):
        payload = {"query": "Will fail"}
        response = client.post("/chat", json=payload)

    assert response.status_code == 502
    assert "Error generating chat response" in response.json()["detail"]


def test_chat_streaming(client):
    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter(
        [MockChunk("CrowdStrike "), MockChunk("Item "), MockChunk("1C summary.")]
    )

    with patch("app.services.llm.get_genai_client", return_value=mock_client):
        payload = {"query": "Stream test", "stream": True}
        response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data: " in content
    assert "CrowdStrike" in content
    assert "[DONE]" in content
