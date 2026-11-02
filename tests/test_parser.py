"""Tests for page-level PDF parser metadata."""

from app.services.parser import ParsedPage, extract_heading


def test_extract_heading_recognises_sec_item_heading():
    assert extract_heading("PALO ALTO NETWORKS\nITEM 1. BUSINESS\nWe deliver security.") == "ITEM 1. BUSINESS"


def test_extract_heading_recognises_title_case_section():
    text = "Form 10-K\nManagement's Discussion and Analysis\nRevenue increased during the year."
    assert extract_heading(text) == "Management's Discussion and Analysis"


def test_parsed_page_keeps_physical_page_number_and_heading():
    page = ParsedPage(page_number=12, text="body", heading="ITEM 1A. RISK FACTORS")
    assert page.to_dict() == {
        "page_number": 12,
        "text": "body",
        "heading": "ITEM 1A. RISK FACTORS",
    }
