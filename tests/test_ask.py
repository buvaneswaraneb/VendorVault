"""Tests for POST /ask endpoint, pgvector retrieval, and [document, page] citations."""

from unittest.mock import MagicMock, patch

from google.genai.errors import APIError

from app.services.vector_store import init_vector_store, store_chunks


class MockUsage:
    def __init__(self, prompt=25, candidates=50):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = prompt + candidates


class MockResponse:
    def __init__(self, text="Sample answer", prompt=25, candidates=50):
        self.text = text
        self.usage_metadata = MockUsage(prompt, candidates)


class MockChunk:
    def __init__(self, text):
        self.text = text


def test_ask_success_with_citations(client, test_settings):
    init_vector_store(test_settings.database_url)
    store_chunks(
        [
            {
                "document_name": "Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf",
                "document_path": "data/pdf/Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf",
                "page_number": 42,
                "heading": "ITEM 1A. RISK FACTORS",
                "content": "The July 19 incident caused customer disruptions and credit incentives.",
                "embedding": [0.2] * 768,
            }
        ],
        database_url=test_settings.database_url,
    )

    mock_client = MagicMock()
    # Mock embedding response
    mock_emb = MagicMock()
    mock_emb.values = [0.2] * 768
    mock_emb_resp = MagicMock()
    mock_emb_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_emb_resp

    # Mock chat completion response
    answer_text = (
        "CrowdStrike described the July 19 incident as leading to customer disruptions and credits "
        "[Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf, page 42]."
    )
    mock_client.models.generate_content.return_value = MockResponse(
        text=answer_text, prompt=35, candidates=45
    )

    with patch("app.services.rag.get_genai_client", return_value=mock_client):
        response = client.post(
            "/ask",
            json={"query": "How did CrowdStrike describe the July 19 incident in its 10-K?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How did CrowdStrike describe the July 19 incident in its 10-K?"
    assert "[Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf, page 42]" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document"] == "Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf"
    assert data["citations"][0]["page"] == 42
    assert data["citations"][0]["heading"] == "ITEM 1A. RISK FACTORS"
    assert "[Form 10-K for Crowdstrike Holdings INC filed 03:10:2025.pdf, page 42]" in data["sources"]
    assert data["model_used"] == "gemini-3.8-flash"
    assert data["prompt_tokens"] == 35
    assert data["completion_tokens"] == 45
    assert data["cost_usd"] > 0
    assert data["cost_inr"] > 0


def test_ask_streaming(client, test_settings):
    init_vector_store(test_settings.database_url)
    store_chunks(
        [
            {
                "document_name": "PANW_2024.pdf",
                "document_path": "data/pdf/PANW_2024.pdf",
                "page_number": 88,
                "heading": "ITEM 7",
                "content": "Palo Alto Networks Next-Gen Security ARR reached $4.2B.",
                "embedding": [0.1] * 768,
            }
        ],
        database_url=test_settings.database_url,
    )

    mock_client = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1] * 768
    mock_emb_resp = MagicMock()
    mock_emb_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_emb_resp

    mock_client.models.generate_content_stream.return_value = iter(
        [MockChunk("Palo Alto Networks NGS ARR "), MockChunk("grew 43% [PANW_2024.pdf, page 88].")]
    )

    with patch("app.services.rag.get_genai_client", return_value=mock_client):
        response = client.post(
            "/ask",
            json={"query": "What was PANW NGS ARR?", "stream": True},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "NGS ARR" in content
    assert "citations" in content
    assert "[DONE]" in content


def test_ask_fallback_on_primary_failure(client, test_settings):
    init_vector_store(test_settings.database_url)
    mock_client = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.3] * 768
    mock_emb_resp = MagicMock()
    mock_emb_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_emb_resp

    def mock_generate(model, contents):
        if model == "gemini-3.8-flash":
            raise APIError("503 Service Temporarily Unavailable")
        return MockResponse(text="Fallback cited answer [DocA, page 10].", prompt=20, candidates=30)

    mock_client.models.generate_content.side_effect = mock_generate

    with patch("app.services.rag.get_genai_client", return_value=mock_client):
        response = client.post("/ask", json={"query": "Compare Item 1C."})

    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "gemini-3.1-flash-lite"
    assert "Fallback cited answer" in data["answer"]


def test_ask_all_models_fail(client, test_settings):
    init_vector_store(test_settings.database_url)
    mock_client = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.3] * 768
    mock_emb_resp = MagicMock()
    mock_emb_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_emb_resp
    mock_client.models.generate_content.side_effect = RuntimeError("Service down")

    with patch("app.services.rag.get_genai_client", return_value=mock_client):
        response = client.post("/ask", json={"query": "Should fail"})

    assert response.status_code == 502
    assert "Error generating answer for /ask" in response.json()["detail"]


def test_documents_status_endpoint(client, test_settings):
    init_vector_store(test_settings.database_url)
    response = client.get("/documents/status")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "total_pages" in data
    assert "documents" in data
