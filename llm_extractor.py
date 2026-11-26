"""LLM invoice extraction pipeline.

Reads every PDF under the input directory, asks Groq for a schema-constrained
invoice, validates it, and writes JSON plus a shared SQLite row. One failed
invoice does not stop the batch. Invoice text is sent to Groq only when
GROQ_API_KEY is set and --dry-run is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from database import ExtractionRecord, ExtractionStore
from pdf_reader import body_character_count, clip_document, extract_pdf, slim_document
from prompts import SYSTEM_PROMPT, build_user_prompt
from schema import (
    EXTRACTION_METHOD_LLM,
    INVOICE_FIELDS,
    INVOICE_JSON_SCHEMA,
    matches_schema,
)
from validator import InvoiceValidationError, normalize_invoice

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

logger = logging.getLogger("llm_extractor")


@dataclass
class InvoiceJobResult:
    source_file: str
    status: str
    output_file: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    extraction_time_ms: Optional[int] = None
    payload: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    page_count: int = 0
    table_count: int = 0
    char_count: int = 0
    labels_seen: list[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None
    file_sha256: Optional[str] = None


def load_api_key() -> Optional[str]:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not key or key.lower().startswith("your_"):
        return None
    return key


def find_pdfs(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.as_posix().lower(),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_duplicate_pdfs(pdfs: list[Path]) -> tuple[list[Path], dict[str, list[Path]]]:
    """Keep every file. Call the model once for each unique byte sequence."""

    unique: list[Path] = []
    copies: dict[str, list[Path]] = {}
    seen: dict[str, Path] = {}
    for path in pdfs:
        digest = _file_sha256(path)
        if digest in seen:
            copies.setdefault(digest, []).append(path)
        else:
            seen[digest] = path
            unique.append(path)
    return unique, copies


def source_id(path: Path, input_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(input_dir.resolve())
    except ValueError:
        relative = Path(path.name)
    return (Path(input_dir.name) / relative).as_posix()


def output_slug(source: str) -> str:
    slug = source
    if slug.lower().endswith(".pdf"):
        slug = slug[:-4]
    return slug.replace("/", "__").replace("\\", "__")


def field_inventory(results: list[InvoiceJobResult]) -> dict[str, list[str]]:
    label_sets = [set(result.labels_seen) for result in results if result.char_count]
    if not label_sets:
        return {"labels_in_all_invoices": [], "labels_in_some_invoices": []}
    common = set.intersection(*label_sets)
    union = set.union(*label_sets)
    return {
        "labels_in_all_invoices": sorted(common),
        "labels_in_some_invoices": sorted(union - common),
    }


class GroqInvoiceClient:
    def __init__(self, api_key: str, model: str, timeout: float, max_retries: int):
        from groq import Groq

        self.client = Groq(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.max_retries = max(1, max_retries)
        self._use_temperature = True
        self._use_reasoning = True

    def resolve_model(self, requested: str) -> str:
        try:
            listing = self.client.models.list()
            available = [model.id for model in listing.data]
        except Exception as exc:
            logger.warning("Could not list Groq models (%s). Using %s.", _safe_error(exc), requested)
            self.model = requested
            return requested

        if requested in available:
            self.model = requested
            logger.info("Groq model %s is available.", requested)
            return requested

        for candidate in FALLBACK_MODELS:
            if candidate in available:
                logger.warning("Model %s is not available. Using %s.", requested, candidate)
                self.model = candidate
                return candidate

        for model_id in available:
            if "gpt-oss" in model_id:
                logger.warning("Model %s is not available. Using %s.", requested, model_id)
                self.model = model_id
                return model_id

        preview = ", ".join(available[:30]) or "none"
        raise RuntimeError(f"Requested model {requested!r} is not available. Visible models: {preview}")

    def extract(self, document_text: str, source_name: str, truncated: bool) -> tuple[dict[str, Any], list[str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(document_text, source_name, truncated)},
        ]
        strict = True
        max_tokens = 3072
        delay = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._complete(messages, strict=strict, max_tokens=max_tokens)
                payload, warnings = normalize_invoice(_parse_json(content))
                return payload, warnings
            except (json.JSONDecodeError, InvoiceValidationError, ValueError) as exc:
                last_error = exc
                if "truncated" in _safe_error(exc) and max_tokens < 4096:
                    max_tokens = 4096
                if attempt == self.max_retries:
                    break
                messages = [
                    messages[0],
                    messages[1],
                    {
                        "role": "user",
                        "content": (
                            "The previous reply could not be validated "
                            f"({_safe_error(exc)}). Return the invoice object again. "
                            "Use null for unknown fields and do not invent values."
                        ),
                    },
                ]
                logger.warning("Invalid model output for %s. Retrying.", source_name)
            except Exception as exc:
                status = _status_code(exc)
                body = _safe_error(exc)
                last_error = exc
                if status in {401, 403}:
                    raise RuntimeError("Groq rejected the API key.") from exc
                if status == 400 and strict and _schema_rejected(body):
                    strict = False
                    logger.warning("Structured output was rejected. Retrying with JSON object mode.")
                    continue
                if status == 413:
                    raise RuntimeError("Invoice text exceeded the model request size.") from exc
                if status is not None and status not in RETRYABLE_STATUS:
                    raise
                if attempt == self.max_retries:
                    break
                logger.warning(
                    "Groq request failed for %s (attempt %s/%s): %s",
                    source_name,
                    attempt,
                    self.max_retries,
                    body,
                )
                time.sleep(min(_retry_delay(exc, delay), 30))
                delay = min(delay * 2, 20)
                if "truncated" in body and max_tokens < 4096:
                    max_tokens = 4096

        raise RuntimeError(f"Groq extraction failed: {_safe_error(last_error)}")

    def _complete(self, messages: list[dict[str, str]], *, strict: bool, max_tokens: int) -> str:
        response_format: dict[str, Any]
        if strict:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_extraction",
                    "strict": True,
                    "schema": _compact_schema(INVOICE_JSON_SCHEMA),
                },
            }
        else:
            response_format = {"type": "json_object"}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
            "max_completion_tokens": max_tokens,
        }
        if self._use_temperature:
            kwargs["temperature"] = 0
        if self._use_reasoning and ("gpt-oss" in self.model or self.model.startswith("qwen/")):
            kwargs["reasoning_effort"] = "low"
            kwargs["reasoning_format"] = "hidden"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError as exc:
            if "reasoning" not in str(exc):
                raise
            self._use_reasoning = False
            kwargs.pop("reasoning_effort", None)
            kwargs.pop("reasoning_format", None)
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._use_temperature and _status_code(exc) == 400 and "temperature" in _safe_error(exc).lower():
                self._use_temperature = False
                kwargs.pop("temperature", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise

        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError("model output was truncated")
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"model refusal: {refusal}")
        content = _message_text(message)
        if not content.strip():
            raise ValueError("model returned empty content")
        return content


def _compact_schema(schema: Any) -> Any:
    """Drop descriptions so the request stays inside the token-per-minute limit."""

    if isinstance(schema, dict):
        return {key: _compact_schema(value) for key, value in schema.items() if key != "description"}
    if isinstance(schema, list):
        return [_compact_schema(value) for value in schema]
    return schema


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return ""


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise InvoiceValidationError("model JSON was not an object")
    return data


def _status_code(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_delay(exc: Exception, default: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        if retry_after is not None:
            return min(max(float(retry_after), default), 45)
    except (TypeError, ValueError):
        pass
    match = re.search(r"try again in ([0-9.]+)", str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, 45)
    return min(default, 45)


def _schema_rejected(body: str) -> bool:
    lowered = body.lower()
    return "json_schema" in lowered or "response_format" in lowered or "schema" in lowered


def _safe_error(exc: Optional[BaseException]) -> str:
    if exc is None:
        return "unknown error"
    text = str(exc).replace("\n", " ").strip()
    if not text:
        text = exc.__class__.__name__
    secret = os.getenv("GROQ_API_KEY", "").strip()
    if secret:
        text = text.replace(secret, "[redacted]")
    return text[:500]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def process_pdf(
    path: Path,
    input_dir: Path,
    *,
    client: Optional[GroqInvoiceClient],
    dry_run: bool,
    max_chars: int,
    blocked_reason: Optional[str],
) -> InvoiceJobResult:
    started = time.perf_counter()
    source = source_id(path, input_dir)
    try:
        document = extract_pdf(path)
    except Exception as exc:
        return InvoiceJobResult(
            source_file=source,
            status="failed",
            error=f"Could not read PDF: {_safe_error(exc)}",
            extraction_time_ms=_elapsed_ms(started),
        )

    result = InvoiceJobResult(
        source_file=source,
        status="failed",
        warnings=list(document.warnings),
        extraction_time_ms=_elapsed_ms(started),
        page_count=document.page_count,
        table_count=document.table_count,
        char_count=document.char_count,
        labels_seen=list(document.labels_seen),
    )

    if blocked_reason:
        result.status = "blocked"
        result.error = blocked_reason
        result.extraction_time_ms = _elapsed_ms(started)
        return result

    if dry_run:
        result.status = "skipped"
        result.error = "Dry run: PDF text was extracted locally and was not sent to Groq."
        result.extraction_time_ms = _elapsed_ms(started)
        return result

    if body_character_count(document.text) < 40:
        result.status = "failed"
        result.error = (
            "No extractable invoice text. The PDF looks scanned, OCR is not available, "
            "and the model was not called."
        )
        result.extraction_time_ms = _elapsed_ms(started)
        return result

    if client is None:
        result.status = "blocked"
        result.error = "GROQ_API_KEY is not set. No invoice content was sent to Groq."
        result.extraction_time_ms = _elapsed_ms(started)
        return result

    clipped, truncated = clip_document(slim_document(document.text), max_chars)
    if truncated:
        result.warnings.append(f"Document text was truncated to {max_chars} characters")

    try:
        payload, validation_warnings = client.extract(clipped, source, truncated)
    except Exception as exc:
        result.status = "failed"
        result.error = _safe_error(exc)
        result.model = client.model
        result.extraction_time_ms = _elapsed_ms(started)
        return result

    result.warnings.extend(validation_warnings)
    result.payload = payload
    result.model = client.model
    result.status = "partial" if validation_warnings or truncated else "success"
    result.extraction_time_ms = _elapsed_ms(started)
    if not matches_schema(payload):
        result.status = "failed"
        result.error = "Normalized invoice did not match the shared schema"
        result.payload = None
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _envelope(result: InvoiceJobResult) -> dict[str, Any]:
    return {
        "source_file": result.source_file,
        "extraction_method": EXTRACTION_METHOD_LLM,
        "model": result.model,
        "status": result.status,
        "warnings": result.warnings,
        "error": result.error,
        "extraction_time_ms": result.extraction_time_ms,
        "data": result.payload,
    }


def write_outputs(results: list[InvoiceJobResult], output_dir: Path, report: dict[str, Any]) -> None:
    invoice_dir = output_dir / "invoices"
    combined: list[dict[str, Any]] = []
    for result in results:
        filename = output_slug(result.source_file) + ".json"
        destination = invoice_dir / filename
        _write_json(destination, _envelope(result))
        result.output_file = destination.relative_to(output_dir).as_posix()
        combined.append(_envelope(result))

    _write_json(
        output_dir / "invoices.json",
        {
            "extraction_method": EXTRACTION_METHOD_LLM,
            "model": report.get("model"),
            "invoices": combined,
        },
    )
    for entry in report["invoices"]:
        match = next(result for result in results if result.source_file == entry["source_file"])
        entry["output_file"] = match.output_file
    _write_json(output_dir / "extraction_report.json", report)


def build_report(
    *,
    results: list[InvoiceJobResult],
    input_dir: Path,
    output_dir: Path,
    model: Optional[str],
    dry_run: bool,
    blocker: Optional[str],
) -> dict[str, Any]:
    counts = {"total": len(results), "successful": 0, "partial": 0, "failed": 0, "blocked": 0, "skipped": 0}
    for result in results:
        key = {
            "success": "successful",
            "partial": "partial",
            "failed": "failed",
            "blocked": "blocked",
            "skipped": "skipped",
        }.get(result.status, "failed")
        counts[key] += 1

    data_payloads = [result.payload for result in results if result.payload is not None]
    consistent = all(matches_schema(payload) for payload in data_payloads)
    duplicate_groups: dict[str, list[str]] = {}
    for result in results:
        if result.file_sha256:
            duplicate_groups.setdefault(result.file_sha256, []).append(result.source_file)
    groups = [
        {"sha256": digest, "files": files}
        for digest, files in sorted(duplicate_groups.items())
        if len(files) > 1
    ]

    return {
        "extraction_method": EXTRACTION_METHOD_LLM,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "blocker": blocker,
        "counts": counts,
        "schema_fields": list(INVOICE_FIELDS),
        "schema_consistent": consistent,
        "duplicate_count": sum(len(group["files"]) - 1 for group in groups),
        "duplicate_groups": groups,
        "field_inventory": field_inventory(results),
        "invoices": [
            {
                "source_file": result.source_file,
                "status": result.status,
                "output_file": result.output_file,
                "extraction_time_ms": result.extraction_time_ms,
                "warnings": result.warnings,
                "error": result.error,
                "page_count": result.page_count,
                "table_count": result.table_count,
                "char_count": result.char_count,
                "labels_seen": result.labels_seen,
                "invoice_id": (result.payload or {}).get("invoice_id"),
                "grand_total": (result.payload or {}).get("grand_total"),
                "duplicate_of": result.duplicate_of,
                "file_sha256": result.file_sha256,
            }
            for result in results
        ],
    }


def persist(store: ExtractionStore, result: InvoiceJobResult) -> None:
    store.save(
        ExtractionRecord(
            source_file=result.source_file,
            extraction_method=EXTRACTION_METHOD_LLM,
            status=result.status,
            payload=result.payload,
            warnings=result.warnings,
            error=result.error,
            extraction_time_ms=result.extraction_time_ms,
            model=result.model,
        )
    )


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    input_dir = input_dir.resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 2

    pdfs = find_pdfs(input_dir)
    api_key = None if args.dry_run else load_api_key()
    blocker = None
    client: Optional[GroqInvoiceClient] = None

    if not args.dry_run and not api_key:
        blocker = "GROQ_API_KEY is not set. No invoice content was sent to Groq."
        if pdfs:
            logger.error(blocker)

    if pdfs and api_key and not args.dry_run:
        client = GroqInvoiceClient(api_key, args.model, args.timeout, args.retries)
        try:
            client.resolve_model(args.model)
        except Exception as exc:
            blocker = _safe_error(exc)
            logger.error(blocker)
            client = None

    workers = max(1, args.concurrency)
    batch_started = time.perf_counter()
    unique_paths, copies = _group_duplicate_pdfs(pdfs)
    results: list[InvoiceJobResult] = []
    if unique_paths:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_path = {
                pool.submit(
                    process_pdf,
                    path,
                    input_dir,
                    client=client,
                    dry_run=args.dry_run,
                    max_chars=args.max_chars,
                    blocked_reason=blocker,
                ): path
                for path in unique_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=args.timeout * args.retries + 60)
                except Exception as exc:
                    result = InvoiceJobResult(
                        source_file=source_id(path, input_dir),
                        status="failed",
                        error=_safe_error(exc),
                    )
                result.file_sha256 = _file_sha256(path)
                results.append(result)
                logger.info(
                    "%s %s in %sms",
                    result.status,
                    result.source_file,
                    result.extraction_time_ms,
                )
        for primary in list(results):
            for copy_path in copies.get(primary.file_sha256 or "", []):
                duplicate = replace(
                    primary,
                    source_file=source_id(copy_path, input_dir),
                    duplicate_of=primary.source_file,
                    warnings=[
                        *primary.warnings,
                        f"Byte-identical duplicate of {primary.source_file}. Extraction was reused.",
                    ],
                )
                results.append(duplicate)
                logger.info("duplicate %s of %s", duplicate.source_file, primary.source_file)

    results.sort(key=lambda item: item.source_file)
    report = build_report(
        results=results,
        input_dir=input_dir,
        output_dir=output_dir,
        model=client.model if client else None,
        dry_run=args.dry_run,
        blocker=blocker,
    )
    report["total_processing_ms"] = _elapsed_ms(batch_started)
    counted = report["counts"]["total"]
    report["average_processing_ms"] = int(report["total_processing_ms"] / counted) if counted else None

    with ExtractionStore(db_path) as store:
        for result in results:
            persist(store, result)
        write_outputs(results, output_dir, report)

    counts = report["counts"]
    print(f"Invoices: {counts['total']}")
    print(f"Successful: {counts['successful']}")
    print(f"Partial: {counts['partial']}")
    print(f"Failed: {counts['failed']}")
    if counts["blocked"]:
        print(f"Blocked: {counts['blocked']}")
    if counts["skipped"]:
        print(f"Skipped: {counts['skipped']}")
    print(f"Duplicates: {report['duplicate_count']}")
    print(f"Schema consistent: {report['schema_consistent']}")
    print(f"Total processing ms: {report['total_processing_ms']}")
    print(f"Average processing ms: {report['average_processing_ms']}")
    print(f"Report: {output_dir / 'extraction_report.json'}")
    if not pdfs:
        print(f"No PDF invoices found in {input_dir}")
    if blocker:
        print(f"Blocker: {blocker}")

    if blocker:
        return 2
    if counts["failed"]:
        return 1
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract invoice PDFs into JSON with Groq.")
    parser.add_argument("--input", default="Bills_Data", help="Directory of invoice PDFs")
    parser.add_argument("--output", default="output/llm", help="Directory for LLM JSON output")
    parser.add_argument("--db", default="invoices.db", help="Shared SQLite database path")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", DEFAULT_MODEL))
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum invoices processed at once")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Bounded retries for API and JSON failures")
    parser.add_argument("--max-chars", type=int, default=100000, help="Maximum document characters sent to Groq")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract PDF text locally and do not call Groq",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
