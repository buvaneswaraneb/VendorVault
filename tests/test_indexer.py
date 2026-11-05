"""Unit tests for document chunking, indexing pipeline, and vector store operations."""

from unittest.mock import patch

from app.services.indexer import chunk_text, extract_chunks_from_pdf, index_document
from app.services.parser import ParsedPage
from app.services.vector_store import (
    get_chunk_stats,
    init_vector_store,
    search_similar_chunks,
    store_chunks,
)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []
    assert chunk_text("short") == []


def test_chunk_text_single_chunk():
    text = "CrowdStrike reported its annual cybersecurity revenue under Item 7."
    chunks = chunk_text(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_overlapping_chunks():
    sentence = "Cybersecurity risk management is reviewed by the board of directors. "
    text = sentence * 20  # ~1400 chars
    chunks = chunk_text(text, max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_extract_chunks_from_pdf(tmp_path):
    mock_pages = [
        ParsedPage(
            page_number=1,
            text="Item 1. Business Overview of cybersecurity operations.",
            heading="Item 1",
        ),
        ParsedPage(
            page_number=2,
            text="Item 1A. Risk factors regarding cloud security threats.",
            heading="Item 1A",
        ),
    ]

    with patch("app.services.indexer.parse_pdf", return_value=mock_pages):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.touch()
        total_pages, chunks = extract_chunks_from_pdf(pdf_path)

        assert total_pages == 2
        assert len(chunks) == 2
        assert chunks[0]["page_number"] == 1
        assert chunks[0]["heading"] == "Item 1"
        assert chunks[1]["page_number"] == 2
        assert chunks[1]["heading"] == "Item 1A"


def test_store_and_search_vector_store(test_settings):
    init_vector_store(test_settings.database_url)

    mock_chunks = [
        {
            "document_name": "Test_10K.pdf",
            "document_path": "data/pdf/Test_10K.pdf",
            "page_number": 5,
            "heading": "ITEM 1C",
            "content": "Palo Alto Networks Item 1C board oversight of cyber risk.",
            "embedding": [0.5] * 768,
        },
        {
            "document_name": "Test_10K.pdf",
            "document_path": "data/pdf/Test_10K.pdf",
            "page_number": 12,
            "heading": "ITEM 7",
            "content": "Annual recurring revenue ARR grew by 25 percent.",
            "embedding": [0.1] * 768,
        },
    ]

    stored = store_chunks(mock_chunks, database_url=test_settings.database_url)
    assert stored == 2

    stats = get_chunk_stats(database_url=test_settings.database_url)
    assert stats["total_chunks"] == 2
    assert stats["total_pages"] == 2
    assert len(stats["documents"]) == 1

    query_vec = [0.5] * 768
    results = search_similar_chunks(query_vec, top_k=1, database_url=test_settings.database_url)
    assert len(results) == 1
    assert results[0].page_number == 5
    assert results[0].heading == "ITEM 1C"


def test_index_document_mocked(tmp_path, test_settings):
    mock_pages = [
        ParsedPage(page_number=1, text="CrowdStrike Falcon platform revenue and metrics.", heading="ITEM 1"),
    ]

    with (
        patch("app.services.indexer.parse_pdf", return_value=mock_pages),
        patch("app.services.indexer.embed_texts_batch", return_value=[[0.2] * 768]),
    ):
        pdf_file = tmp_path / "CrowdStrike_10K.pdf"
        pdf_file.touch()

        res = index_document(
            pdf_path=pdf_file,
            settings=test_settings,
            batch_size=5,
            sleep_between_batches=0,
        )

        assert res["status"] == "indexed"
        assert res["pages"] == 1
        assert res["chunks"] == 1

        stats = get_chunk_stats(test_settings.database_url)
        assert stats["total_chunks"] >= 1
