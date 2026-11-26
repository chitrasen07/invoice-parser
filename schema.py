"""Shared invoice schema for deterministic and LLM extraction.

Both pipelines must emit these keys with the same types. Unknown values are
null. ``items`` is always a list and is empty when the invoice has no lines.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

ITEM_FIELDS: tuple[str, ...] = (
    "description",
    "hsn_sac",
    "quantity",
    "unit_price",
    "discount",
    "taxable_amount",
    "gst_rate",
    "tax_amount",
    "total_amount",
)

INVOICE_FIELDS: tuple[str, ...] = (
    "invoice_id",
    "invoice_date",
    "seller_name",
    "seller_address",
    "seller_gstin",
    "buyer_name",
    "buyer_address",
    "buyer_gstin",
    "items",
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
    "currency",
)

EXTRACTION_METHOD_DETERMINISTIC = "deterministic"
EXTRACTION_METHOD_LLM = "llm"


def _nullable_string(description: str) -> dict[str, Any]:
    return {"type": ["string", "null"], "description": description}


def _nullable_number(description: str) -> dict[str, Any]:
    return {"type": ["number", "null"], "description": description}


_ITEM_PROPERTIES: dict[str, dict[str, Any]] = {
    "description": _nullable_string("Product or service text printed on the line. Null if blank."),
    "hsn_sac": _nullable_string("HSN or SAC code exactly as printed, including leading zeros. Null if absent."),
    "quantity": _nullable_number("Quantity printed on the line. Null if absent."),
    "unit_price": _nullable_number("Unit price printed on the line, before tax. Null if absent."),
    "discount": _nullable_number("Line discount amount printed on the line. Null if absent. Do not convert a percent into an amount."),
    "taxable_amount": _nullable_number("Taxable value printed for the line. Null if absent."),
    "gst_rate": _nullable_number("GST rate in percent as printed, so 18% is 18. Null if absent."),
    "tax_amount": _nullable_number("Tax amount printed for the line. Null if absent."),
    "total_amount": _nullable_number("Line total printed for the line. Null if absent."),
}

ITEM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": _ITEM_PROPERTIES,
    "required": list(ITEM_FIELDS),
}

_INVOICE_PROPERTIES: dict[str, Any] = {
    "invoice_id": _nullable_string("Invoice number printed on the document. Null if absent."),
    "invoice_date": _nullable_string("Invoice date as YYYY-MM-DD. Null if absent or ambiguous."),
    "seller_name": _nullable_string("Name of the business that issued the invoice. Null if the issuer is unclear."),
    "seller_address": _nullable_string("Address of the issuer. Null if absent."),
    "seller_gstin": _nullable_string("15-character GSTIN of the issuer, copied exactly. Null if absent."),
    "buyer_name": _nullable_string("Name of the customer being billed. Null if the buyer is unclear."),
    "buyer_address": _nullable_string("Address of the buyer. Null if absent."),
    "buyer_gstin": _nullable_string("15-character GSTIN of the buyer, copied exactly. Null if absent."),
    "items": {
        "type": "array",
        "description": "Product and service rows only. Empty array when none are printed.",
        "items": ITEM_JSON_SCHEMA,
    },
    "subtotal": _nullable_number("Subtotal printed on the invoice, before invoice-level discount. Null if absent."),
    "discount": _nullable_number("Invoice-level discount amount printed on the invoice. Null if absent."),
    "taxable_amount": _nullable_number("Total taxable value printed on the invoice. Null if absent."),
    "cgst": _nullable_number("Total CGST amount printed on the invoice. Null if absent."),
    "sgst": _nullable_number("Total SGST or UTGST amount printed on the invoice. Null if absent."),
    "igst": _nullable_number("Total IGST amount printed on the invoice. Null if absent."),
    "total_tax": _nullable_number("GST only: CGST plus SGST plus IGST, or the printed total GST. Do not include compensation cess or TCS. Null if absent."),
    "cess": _nullable_number("Compensation cess or other cess amount printed on the invoice. Null if absent. Do not add it to CGST, SGST, or total_tax."),
    "tcs": _nullable_number("Tax collected at source (TCS) amount printed on the invoice. Null if absent."),
    "round_off": _nullable_number("Round-off amount printed on the invoice. Negative when printed as a reduction. Null if absent."),
    "grand_total": _nullable_number("Grand total or amount payable printed on the invoice. Null if absent."),
    "currency": _nullable_string("ISO 4217 currency code. Use INR when the rupee is shown. Null if unclear."),
}

INVOICE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": _INVOICE_PROPERTIES,
    "required": list(INVOICE_FIELDS),
}


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: Optional[str] = None
    hsn_sac: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    discount: Optional[float] = None
    taxable_amount: Optional[float] = None
    gst_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None


class Invoice(BaseModel):
    """Canonical invoice payload. Every extractor writes this object."""

    model_config = ConfigDict(extra="ignore")

    invoice_id: Optional[str] = None
    invoice_date: Optional[str] = None
    seller_name: Optional[str] = None
    seller_address: Optional[str] = None
    seller_gstin: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_gstin: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    discount: Optional[float] = None
    taxable_amount: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    total_tax: Optional[float] = None
    cess: Optional[float] = None
    tcs: Optional[float] = None
    round_off: Optional[float] = None
    grand_total: Optional[float] = None
    currency: Optional[str] = None


def empty_invoice() -> dict[str, Any]:
    payload = {field: None for field in INVOICE_FIELDS}
    payload["items"] = []
    return payload


def order_invoice(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the invoice dict with the shared key order and no extra keys."""

    items: list[dict[str, Any]] = []
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        items.append({field: raw_item.get(field) for field in ITEM_FIELDS})

    ordered: dict[str, Any] = {}
    for field in INVOICE_FIELDS:
        if field == "items":
            ordered[field] = items
        else:
            ordered[field] = payload.get(field)
    return ordered


def invoice_to_dict(invoice: Invoice) -> dict[str, Any]:
    return order_invoice(invoice.model_dump(mode="json"))


def matches_schema(payload: dict[str, Any]) -> bool:
    if list(payload.keys()) != list(INVOICE_FIELDS):
        return False
    items = payload.get("items")
    if not isinstance(items, list):
        return False
    return all(isinstance(item, dict) and list(item.keys()) == list(ITEM_FIELDS) for item in items)
