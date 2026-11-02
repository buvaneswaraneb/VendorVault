"""PDF extraction that preserves citation metadata for later RAG steps.

The parser deliberately returns one record per *PDF page*.  Splitting text into
chunks is a separate concern: keeping the original page number and the current
filing section here makes it possible for every future chunk to cite its source.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ITEM_OR_PART = re.compile(
    r"^(?:part\s+[ivxlcdm]+|item\s+\d{1,2}[a-z]?(?:\.|\s|$))", re.IGNORECASE
)
_NUMBERED_SECTION = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")
_TRAILING_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPage:
    """Text and source metadata extracted from a single PDF page."""

    page_number: int
    text: str
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data for storage as chunk metadata."""
        return asdict(self)


def _normalise_line(line: str) -> str:
    return " ".join(line.replace("\u00a0", " ").split())


def _looks_like_heading(line: str) -> bool:
    """Identify common SEC filing headings without treating prose as a heading."""
    if not line or len(line) > 140 or _TRAILING_PAGE_NUMBER.fullmatch(line):
        return False
    if _ITEM_OR_PART.match(line) or _NUMBERED_SECTION.match(line):
        return True

    # Filings use both all-caps labels ("RISK FACTORS") and title-case labels
    # ("Management's Discussion and Analysis").  Requiring short, punctuation-
    # free text prevents ordinary sentences from becoming section metadata.
    words = line.split()
    if not 2 <= len(words) <= 14 or line[-1:] in ".,;:!?":
        return False
    letters = "".join(char for char in line if char.isalpha())
    if not letters:
        return False
    if line.isupper():
        return True
    title_case_words = sum(word[:1].isupper() for word in words if word[:1].isalpha())
    return title_case_words >= max(2, len(words) - 2)


def _is_boilerplate(line: str) -> bool:
    """Exclude recurring document labels that appear above real headings."""
    compact = line.casefold().replace(" ", "")
    return compact.startswith("form10-") or compact.startswith("form10k")


def extract_heading(text: str) -> str | None:
    """Return the first section heading appearing near the top of a PDF page.

    Limiting detection to the first 30 nonblank lines avoids mistaking a table of
    contents entry or a title embedded in the body for the page's active section.
    """
    candidates = [_normalise_line(line) for line in text.splitlines()]
    checked = 0
    fallback_headings: list[str] = []
    for line in candidates:
        if not line:
            continue
        if _is_boilerplate(line):
            continue
        if _ITEM_OR_PART.match(line) or _NUMBERED_SECTION.match(line):
            return line
        if _looks_like_heading(line):
            fallback_headings.append(line)
        # Page headers and short boilerplate precede the real title occasionally;
        # after a modest prefix, text is much more likely to be body prose.
        checked += 1
        if checked >= 30:
            break

    # A filing's company name is often all-caps in the page header. Prefer a
    # later title-case heading, but keep an all-caps heading as a useful fallback.
    return next((line for line in fallback_headings if not line.isupper()), None) or (
        fallback_headings[0] if fallback_headings else None
    )


def parse_pdf(pdf_path: str | Path) -> list[ParsedPage]:
    """Extract a PDF into page records while retaining page number and heading.

    Page numbers are physical, one-based PDF page indexes rather than printed
    report numbers, so they always point to the page a user sees in the file.
    If a section continues onto the next page, its heading is inherited.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError("PDF parsing requires the 'pypdf' package.") from exc

    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")

    reader = PdfReader(str(source))
    pages: list[ParsedPage] = []
    current_heading: str | None = None
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        detected_heading = extract_heading(text)
        if detected_heading is not None:
            current_heading = detected_heading
        pages.append(
            ParsedPage(
                page_number=page_number,
                text=text,
                heading=current_heading,
            )
        )
    return pages


# Clear names for callers that prefer extraction terminology.
parse_document = parse_pdf
extract_pdf_pages = parse_pdf
