"""Retry invoices that failed only because of the Groq rate limit."""

import json
import time
from pathlib import Path

from llm_extractor import (
    GroqInvoiceClient,
    InvoiceJobResult,
    build_report,
    load_api_key,
    persist,
    process_pdf,
    write_outputs,
    _elapsed_ms,
)
from database import ExtractionStore

RETRY = [
    "SalesBill_GT_12642.PDF",
    "Sales_2775 SANMAT.pdf",
    "Sales_2946.pdf",
]
input_dir = Path("Bills_Data").resolve()
output_dir = Path("output/llm").resolve()
report_path = output_dir / "extraction_report.json"
combined_path = output_dir / "invoices.json"
old_report = json.loads(report_path.read_text(encoding="utf-8"))
old_combined = json.loads(combined_path.read_text(encoding="utf-8"))
by_source = {entry["source_file"]: entry for entry in old_combined["invoices"]}

results = []
for entry in old_report["invoices"]:
    envelope = by_source[entry["source_file"]]
    results.append(
        InvoiceJobResult(
            source_file=entry["source_file"],
            status=entry["status"],
            warnings=list(entry.get("warnings") or []),
            error=entry.get("error"),
            extraction_time_ms=entry.get("extraction_time_ms"),
            payload=envelope.get("data"),
            model=envelope.get("model"),
            page_count=entry.get("page_count") or 0,
            table_count=entry.get("table_count") or 0,
            char_count=entry.get("char_count") or 0,
            labels_seen=list(entry.get("labels_seen") or []),
            duplicate_of=entry.get("duplicate_of"),
            file_sha256=entry.get("file_sha256"),
        )
    )

client = GroqInvoiceClient(load_api_key(), "openai/gpt-oss-120b", 120, 6)
client.resolve_model("openai/gpt-oss-120b")
started = time.perf_counter()
for name in RETRY:
    path = input_dir / name
    result = process_pdf(
        path,
        input_dir,
        client=client,
        dry_run=False,
        max_chars=100000,
        blocked_reason=None,
    )
    result.file_sha256 = next(item.file_sha256 for item in results if item.source_file.endswith(name))
    for index, item in enumerate(results):
        if item.source_file.endswith(name):
            results[index] = result
            break
    data = result.payload or {}
    print(name, result.status, data.get("invoice_id"), data.get("grand_total"), result.error)

report = build_report(
    results=results,
    input_dir=input_dir,
    output_dir=output_dir,
    model=client.model,
    dry_run=False,
    blocker=None,
)
previous = old_report.get("total_processing_ms") or 0
report["total_processing_ms"] = previous + _elapsed_ms(started)
report["average_processing_ms"] = int(report["total_processing_ms"] / report["counts"]["total"])
with ExtractionStore(Path("invoices.db")) as store:
    for result in results:
        if Path(result.source_file).name in RETRY:
            persist(store, result)
    write_outputs(results, output_dir, report)
print("counts", report["counts"])
print("retry_ms", _elapsed_ms(started))
