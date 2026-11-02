"""Tests for page-level PDF parser metadata."""

import sys
from types import SimpleNamespace

from app.services.parser import ParsedPage, extract_heading, parse_pdf


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


def test_parse_pdf_preserves_one_based_page_numbers_and_inherits_heading(tmp_path, monkeypatch):
    class FakePdfPage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [
                FakePdfPage("ITEM 1A. RISK FACTORS\nRisk disclosures."),
                FakePdfPage("More risk disclosures."),
            ]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    source = tmp_path / "filing.pdf"
    source.touch()

    pages = parse_pdf(source)

    assert [(page.page_number, page.heading) for page in pages] == [
        (1, "ITEM 1A. RISK FACTORS"),
        (2, "ITEM 1A. RISK FACTORS"),
    ]
