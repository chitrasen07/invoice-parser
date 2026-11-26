"""Compare saved JSON with PDF text. Prints mismatches only, no secrets."""

import json
import re
from pathlib import Path

from pdf_reader import extract_pdf, slim_document

report = json.loads(Path("output/llm/invoices.json").read_text(encoding="utf-8"))
root = Path("Bills_Data")


def digits(value):
    if value is None:
        return ""
    return f"{float(value):.2f}"


def in_text(text, value):
    if not value:
        return True
    collapsed = re.sub(r"\s+", "", text).lower()
    token = re.sub(r"\s+", "", str(value)).lower()
    return token in collapsed


fails = 0
for entry in report["invoices"]:
    name = Path(entry["source_file"]).name
    data = entry.get("data")
    if entry["status"] == "failed" or not data:
        print("SKIP", name, entry["status"])
        continue
    text = slim_document(extract_pdf(root / name).text)
    problems = []
    if data.get("invoice_id") and not in_text(text, data["invoice_id"]):
        problems.append("invoice_id")
    for field in ("seller_gstin", "buyer_gstin"):
        if data.get(field) and not in_text(text, data[field]):
            problems.append(field)
    total = data.get("grand_total")
    if total is not None:
        plain = digits(total)
        indian = plain
        whole, frac = plain.split(".")
        groups = []
        while whole:
            groups.append(whole[-3:] if not groups else whole[-2:])
            whole = whole[: -3 if not groups[1:] else -2]
        # simpler presence check
        if plain not in text.replace(",", "") and str(int(float(total))) not in text.replace(",", ""):
            problems.append("grand_total")
    if not data.get("seller_name"):
        problems.append("seller_name_missing")
    if not data.get("buyer_name"):
        problems.append("buyer_name_missing")
    if not data.get("items"):
        problems.append("no_items")
    if problems:
        fails += 1
        print("MISMATCH", name, ",".join(problems), "items", len(data.get("items") or []))
    else:
        print("OK", name, "items", len(data["items"]))
print("mismatch_files", fails)
