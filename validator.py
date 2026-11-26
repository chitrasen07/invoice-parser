"""Normalize extracted invoices and check GSTIN plus printed totals.

Normalization cleans formats. It does not invent missing values. Calculation
mismatches are warnings; the printed values are kept.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Optional

from pydantic import ValidationError

from schema import (
    INVOICE_FIELDS,
    ITEM_FIELDS,
    Invoice,
    invoice_to_dict,
)

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d/%b/%Y",
    "%d/%B/%Y",
    "%d-%b-%y",
    "%d/%b/%y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%B %d %Y",
)

_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_CURRENCY_ALIASES = {
    "INR": "INR",
    "RS": "INR",
    "RS.": "INR",
    "RUPEE": "INR",
    "RUPEES": "INR",
    "₹": "INR",
    "USD": "USD",
    "US$": "USD",
    "EUR": "EUR",
    "€": "EUR",
    "GBP": "GBP",
    "£": "GBP",
}
_MONEY_INVOICE = (
    "subtotal",
    "discount",
    "taxable_amount",
    "cgst",
    "sgst",
    "igst",
    "total_tax",
    "cess",
    "tcs",
    "round_off",
    "grand_total",
)
_MONEY_ITEM = ("unit_price", "discount", "taxable_amount", "tax_amount", "total_amount")
_STRING_INVOICE = (
    "invoice_id",
    "seller_name",
    "seller_address",
    "seller_gstin",
    "buyer_name",
    "buyer_address",
    "buyer_gstin",
    "currency",
)
_ABS_TOLERANCE = 1.0
_REL_TOLERANCE = 0.01
_SUMMARY_LINE = re.compile(
    r"^\s*(?:less\s*:?\s*)?(?:output\s+tax\s+)?(?:output\s+)?"
    r"(?:compensation\s+cess|cess|tcs|cgst|sgst|igst|utgst|round\s*off|grand\s+total|sub\s*total|total)\b",
    re.IGNORECASE,
)


class InvoiceValidationError(ValueError):
    """Raised when a model payload cannot be coerced into the shared schema."""


def gstin_checksum(first_14: str) -> str:
    total = 0
    factor = 2
    for char in reversed(first_14):
        code = _GSTIN_CHARS.index(char)
        product = factor * code
        total += (product // 36) + (product % 36)
        factor = 1 if factor == 2 else 2
    return _GSTIN_CHARS[(36 - (total % 36)) % 36]


def normalize_gstin(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value)).upper()
    return text or None


def gstin_problems(gstin: Optional[str]) -> list[str]:
    if gstin is None:
        return []
    problems: list[str] = []
    if not _GSTIN_RE.match(gstin):
        problems.append(f"GSTIN {gstin} does not match the 15-character format")
        return problems
    state_code = int(gstin[:2])
    if not 1 <= state_code <= 38:
        problems.append(f"GSTIN {gstin} has an unexpected state code")
    expected = gstin_checksum(gstin[:14])
    if gstin[14] != expected:
        problems.append(f"GSTIN {gstin} has checksum {gstin[14]}, expected {expected}")
    return problems


def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "null", "none", "na", "n/a", "-", "—", "nil"}:
            return None
        return text
    return value


def _as_float(value: Any, *, decimals: int, percent: bool = False) -> Optional[float]:
    value = _blank_to_none(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if percent:
            text = text.replace("%", "").strip()
        elif "%" in text:
            return None
        negative = False
        if text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1]
        text = (
            text.replace(",", "")
            .replace("₹", "")
            .replace("Rs.", "")
            .replace("Rs", "")
            .replace("INR", "")
            .replace("/-", "")
            .strip()
        )
        if text.startswith("-"):
            negative = True
            text = text[1:].strip()
        text = re.sub(r"[^0-9.]", "", text)
        if text.count(".") > 1 or text in {"", "."}:
            return None
        number = float(text)
        if negative:
            number = -number
    if not math.isfinite(number):
        return None
    return float(f"{number:.{decimals}f}")


def normalize_date(value: Any) -> tuple[Optional[str], Optional[str]]:
    value = _blank_to_none(value)
    if value is None:
        return None, None
    if isinstance(value, datetime):
        return value.date().isoformat(), None
    text = str(value).strip()
    iso_prefix = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if iso_prefix:
        try:
            parsed = datetime.strptime(iso_prefix.group(0), "%Y-%m-%d")
        except ValueError:
            return text, f"invoice_date {text!r} is not a valid date"
        return parsed.date().isoformat(), None

    cleaned = text.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        warning = None
        if fmt in {"%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"}:
            parts = re.split(r"[/.-]", cleaned)
            day, month = int(parts[0]), int(parts[1])
            if day <= 12 and month <= 12 and day != month:
                warning = "invoice_date was interpreted as day/month/year"
        return parsed.date().isoformat(), warning
    return text, f"invoice_date {text!r} was kept as printed because it could not be normalized"


def normalize_currency(value: Any) -> tuple[Optional[str], Optional[str]]:
    value = _blank_to_none(value)
    if value is None:
        return None, None
    text = str(value).strip().upper()
    alias = _CURRENCY_ALIASES.get(text) or _CURRENCY_ALIASES.get(text.replace(" ", ""))
    if alias:
        return alias, None
    compact = re.sub(r"[^A-Z]", "", text)
    if compact in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[compact], None
    if re.fullmatch(r"[A-Z]{3}", compact):
        return compact, None
    return text, f"currency {text!r} is not an ISO code and was kept as printed"


def _close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= max(_ABS_TOLERANCE, _REL_TOLERANCE * scale)


def _check_pair(warnings: list[str], label: str, left: Optional[float], right: Optional[float]) -> None:
    if left is None or right is None:
        return
    if not _close(left, right):
        warnings.append(f"{label}: printed {left} does not match {right}")


def _calculation_warnings(invoice: Invoice) -> list[str]:
    warnings: list[str] = []
    item_taxable: list[float] = []
    item_tax: list[float] = []
    item_totals: list[float] = []

    for index, item in enumerate(invoice.items, start=1):
        prefix = f"item {index}"
        if item.quantity is not None and item.unit_price is not None and item.taxable_amount is not None:
            gross = item.quantity * item.unit_price
            discount = item.discount or 0.0
            _check_pair(warnings, f"{prefix} taxable_amount", item.taxable_amount, gross - discount)
        if item.taxable_amount is not None and item.gst_rate is not None and item.tax_amount is not None:
            _check_pair(
                warnings,
                f"{prefix} tax_amount",
                item.tax_amount,
                item.taxable_amount * item.gst_rate / 100.0,
            )
        if item.taxable_amount is not None and item.tax_amount is not None and item.total_amount is not None:
            _check_pair(
                warnings,
                f"{prefix} total_amount",
                item.total_amount,
                item.taxable_amount + item.tax_amount,
            )
        if item.taxable_amount is not None:
            item_taxable.append(item.taxable_amount)
        if item.tax_amount is not None:
            item_tax.append(item.tax_amount)
        if item.total_amount is not None:
            item_totals.append(item.total_amount)

    lines_have_taxable = bool(invoice.items) and all(item.taxable_amount is not None for item in invoice.items)
    lines_have_tax = bool(invoice.items) and all(item.tax_amount is not None for item in invoice.items)
    if lines_have_taxable and invoice.taxable_amount is not None:
        _check_pair(warnings, "taxable_amount vs line items", invoice.taxable_amount, sum(item_taxable))
    if lines_have_taxable and invoice.subtotal is not None and invoice.discount is None:
        _check_pair(warnings, "subtotal vs line items", invoice.subtotal, sum(item_taxable))
    if lines_have_tax and invoice.total_tax is not None:
        _check_pair(warnings, "total_tax vs line items", invoice.total_tax, sum(item_tax))
    lines_have_totals = bool(invoice.items) and all(item.total_amount is not None for item in invoice.items)
    if lines_have_totals and invoice.grand_total is not None and invoice.total_tax is None:
        _check_pair(warnings, "grand_total vs line totals", invoice.grand_total, sum(item_totals))

    tax_parts = [value for value in (invoice.cgst, invoice.sgst, invoice.igst) if value is not None]
    if tax_parts and invoice.total_tax is not None:
        _check_pair(warnings, "total_tax vs cgst+sgst+igst", invoice.total_tax, sum(tax_parts))

    if invoice.subtotal is not None and invoice.discount is not None and invoice.taxable_amount is not None:
        _check_pair(
            warnings,
            "taxable_amount vs subtotal-discount",
            invoice.taxable_amount,
            invoice.subtotal - invoice.discount,
        )

    if invoice.taxable_amount is not None and invoice.grand_total is not None:
        tax = invoice.total_tax
        if tax is None and tax_parts:
            tax = sum(tax_parts)
        if tax is not None:
            expected = (
                invoice.taxable_amount
                + tax
                + (invoice.cess or 0.0)
                + (invoice.tcs or 0.0)
                + (invoice.round_off or 0.0)
            )
            _check_pair(warnings, "grand_total", invoice.grand_total, expected)

    if invoice.cgst is not None and invoice.sgst is not None and not _close(invoice.cgst, invoice.sgst):
        warnings.append(f"cgst {invoice.cgst} and sgst {invoice.sgst} are not equal")

    for label, gstin in (("seller_gstin", invoice.seller_gstin), ("buyer_gstin", invoice.buyer_gstin)):
        for problem in gstin_problems(gstin):
            warnings.append(f"{label}: {problem}")

    if invoice.invoice_id is None and invoice.seller_name is None and invoice.grand_total is None:
        warnings.append("invoice_id, seller_name, and grand_total are all missing")
    return warnings


def _string_field(value: Any) -> Optional[str]:
    value = _blank_to_none(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)):
        text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    else:
        text = str(value).strip()
    return text or None


def normalize_invoice(payload: Any) -> tuple[dict[str, Any], list[str]]:
    """Coerce a model response into the shared schema and collect warnings."""

    if not isinstance(payload, dict):
        raise InvoiceValidationError("model response was not a JSON object")

    warnings: list[str] = []
    cleaned: dict[str, Any] = {}
    for field in _STRING_INVOICE:
        cleaned[field] = _string_field(payload.get(field))

    invoice_date, date_warning = normalize_date(payload.get("invoice_date"))
    cleaned["invoice_date"] = invoice_date
    if date_warning:
        warnings.append(date_warning)

    currency, currency_warning = normalize_currency(cleaned.get("currency"))
    cleaned["currency"] = currency
    if currency_warning:
        warnings.append(currency_warning)

    for field in ("seller_gstin", "buyer_gstin"):
        cleaned[field] = normalize_gstin(cleaned.get(field))

    for field in _MONEY_INVOICE:
        number = _as_float(payload.get(field), decimals=2)
        if _blank_to_none(payload.get(field)) is not None and number is None:
            warnings.append(f"{field} value {payload.get(field)!r} could not be parsed and was set to null")
        cleaned[field] = number

    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise InvoiceValidationError("items must be a list")

    items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            warnings.append(f"item {index} was not an object and was skipped")
            continue
        item: dict[str, Any] = {
            "description": _string_field(raw_item.get("description")),
            "hsn_sac": _string_field(raw_item.get("hsn_sac")),
        }
        quantity = _as_float(raw_item.get("quantity"), decimals=4)
        if _blank_to_none(raw_item.get("quantity")) is not None and quantity is None:
            warnings.append(f"item {index} quantity {raw_item.get('quantity')!r} could not be parsed")
        item["quantity"] = quantity
        gst_rate = _as_float(raw_item.get("gst_rate"), decimals=2, percent=True)
        if _blank_to_none(raw_item.get("gst_rate")) is not None and gst_rate is None:
            warnings.append(f"item {index} gst_rate {raw_item.get('gst_rate')!r} could not be parsed")
        item["gst_rate"] = gst_rate
        for field in _MONEY_ITEM:
            number = _as_float(raw_item.get(field), decimals=2)
            if _blank_to_none(raw_item.get(field)) is not None and number is None:
                warnings.append(
                    f"item {index} {field} {raw_item.get(field)!r} could not be parsed and was set to null"
                )
            item[field] = number
        if all(item[field] is None for field in ITEM_FIELDS):
            warnings.append(f"item {index} was empty and was skipped")
            continue
        description = item.get("description") or ""
        if _SUMMARY_LINE.search(description):
            warnings.append(f"item {index} is a tax or total row and was not kept as a line item")
            continue
        items.append(item)
    cleaned["items"] = items

    for field in INVOICE_FIELDS:
        cleaned.setdefault(field, [] if field == "items" else None)

    try:
        invoice = Invoice.model_validate(cleaned)
    except ValidationError as exc:
        raise InvoiceValidationError(str(exc)) from exc

    warnings.extend(_calculation_warnings(invoice))
    # Preserve order and drop duplicates while keeping the first wording.
    unique_warnings = list(dict.fromkeys(warnings))
    return invoice_to_dict(invoice), unique_warnings
