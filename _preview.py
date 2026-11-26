"""Write local text previews for sparse and representative invoices. No network calls."""

from pathlib import Path

import pymupdf

from pdf_reader import extract_pdf

out = Path("output") / "_preview"
out.mkdir(parents=True, exist_ok=True)
names = [
    "13203.pdf",
    "highlighted.pdf",
    "INVOICE3715.pdf",
    "6934.pdf",
    "GJ03BV6630.pdf",
    "GJ03BV6630 (1).pdf",
    "SalesBill_GT_5776.PDF",
    "Invoice_VGIL2526ND1798.pdf",
    "Sales_2946.pdf",
    "INV-1573.pdf",
]
for name in names:
    path = Path("Bills_Data") / name
    doc = extract_pdf(path)
    (out / f"{path.stem}.txt").write_text(doc.text, encoding="utf-8")
    pdf = pymupdf.open(path)
    images = sum(len(page.get_images()) for page in pdf)
    print(f"{name} pages={pdf.page_count} images={images} chars={doc.char_count} tables={doc.table_count}")
    pdf.close()
