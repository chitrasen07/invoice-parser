"""Read invoice PDFs into text while keeping tables and nearby context.

pdfplumber supplies the page text and tables. PyMuPDF is the fallback when a
page has little or no extractable text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LABEL_HINTS: tuple[str, ...] = (
    "invoice no",
    "invoice number",
    "invoice date",
    "gstin",
    "bill to",
    "billed to",
    "ship to",
    "consignee",
    "hsn",
    "sac",
    "cgst",
    "sgst",
    "igst",
    "utgst",
    "taxable",
    "grand total",
    "round off",
    "discount",
    "place of supply",
    "irn",
    "buyer",
    "seller",
    "quantity",
    "total amount",
)


@dataclass
class ExtractedDocument:
    text: str
    page_count: int
    table_count: int
    char_count: int
    labels_seen: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _trim_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _render_table(table: list) -> str:
    lines: list[str] = []
    for row in table or []:
        if not row:
            continue
        cells = []
        for cell in row:
            cells.append("" if cell is None else " ".join(str(cell).split()))
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _labels_in(text: str) -> list[str]:
    lowered = text.lower()
    return [label for label in LABEL_HINTS if label in lowered]


def _page_text_pdfplumber(page: object) -> str:
    plain = page.extract_text() or ""
    laid_out = ""
    extract_text = getattr(page, "extract_text")
    try:
        laid_out = extract_text(layout=True) or ""
    except TypeError:
        laid_out = ""
    if laid_out and len(laid_out) < max(len(plain) * 4, 20000):
        return _trim_text(laid_out)
    return _trim_text(plain)


def _pymupdf_text(path: Path) -> tuple[str, int, list[str]]:
    import pymupdf

    warnings: list[str] = []
    doc = pymupdf.open(path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise RuntimeError("PDF is password protected")
        if doc.is_encrypted:
            warnings.append("PDF reported encryption but opened without a password")
        pages: list[str] = []
        for index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks") or []
            text_blocks = [block for block in blocks if len(block) >= 5 and str(block[4]).strip()]
            text_blocks.sort(key=lambda block: (round(float(block[1]), 1), float(block[0])))
            body = "\n".join(_trim_text(str(block[4])) for block in text_blocks)
            if not body.strip():
                body = _trim_text(page.get_text("text") or "")
            pages.append(f"=== PAGE {index} OF {doc.page_count} ===\n--- TEXT ---\n{body.strip()}")
        return "\n\n".join(pages).strip(), doc.page_count, warnings
    finally:
        doc.close()


def extract_pdf(path: str | Path) -> ExtractedDocument:
    import pdfplumber

    pdf_path = Path(path)
    warnings: list[str] = []
    page_sections: list[str] = []
    table_count = 0
    page_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for index, page in enumerate(pdf.pages, start=1):
                text = _page_text_pdfplumber(page).strip()
                tables = page.extract_tables() or []
                section = [f"=== PAGE {index} OF {page_count} ===", "--- TEXT ---", text]
                for table_index, table in enumerate(tables, start=1):
                    rendered = _render_table(table)
                    if not rendered:
                        continue
                    table_count += 1
                    section.append(f"--- TABLE {table_index} ---")
                    section.append(rendered)
                page_sections.append("\n".join(section).strip())
    except Exception as exc:
        warnings.append(f"pdfplumber could not read the PDF ({exc})")

    text = "\n\n".join(section for section in page_sections if section).strip()
    if len(text) < 40:
        try:
            fallback, fallback_pages, fallback_warnings = _pymupdf_text(pdf_path)
        except Exception as exc:
            warnings.append(f"PyMuPDF could not read the PDF ({exc})")
            fallback, fallback_pages, fallback_warnings = "", page_count, []
        warnings.extend(fallback_warnings)
        if len(fallback) > len(text):
            if text:
                warnings.append("Used PyMuPDF text because pdfplumber extracted very little")
            text = fallback
            page_count = fallback_pages or page_count

    if not text.strip():
        warnings.append("No extractable text. The PDF may be a scan, and OCR is not enabled.")

    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    labels = _labels_in(text)
    return ExtractedDocument(
        text=text,
        page_count=page_count,
        table_count=table_count,
        char_count=len(text),
        labels_seen=labels,
        warnings=warnings,
    )


def body_character_count(text: str) -> int:
    """Count invoice characters, ignoring the page and table markers we add."""

    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=== PAGE") or stripped.startswith("--- "):
            continue
        total += len(stripped)
    return total


def slim_document(text: str) -> str:
    """Drop repeated table dumps. The page text already contains those rows."""

    kept: list[str] = []
    skipping_table = False
    for line in text.splitlines():
        if line.startswith("--- TABLE"):
            skipping_table = True
            continue
        if line.startswith("=== PAGE") or line.startswith("--- TEXT"):
            skipping_table = False
        if skipping_table:
            continue
        kept.append(line.rstrip())
    compact = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def clip_document(text: str, limit: int) -> tuple[str, bool]:
    """Keep the start and the end of a very long document."""

    if limit < 1000:
        limit = 1000
    if len(text) <= limit:
        return text, False
    marker = "\n\n[... middle of document omitted by extractor ...]\n\n"
    head = int(limit * 0.7)
    tail = max(0, limit - head - len(marker))
    return text[:head] + marker + text[-tail:], True
