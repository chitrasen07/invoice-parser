# Invoice extraction

Two extractors share one schema and one SQLite database:

- Part 1, deterministic extraction, is separate and is not implemented here.
- Part 2, this pipeline, reads invoice PDFs and asks Groq for structured JSON.

LLM rows are stored with `extraction_method = "llm"`. Deterministic rows should use `extraction_method = "deterministic"`. The same PDF can have both.

## Setup

```bash
python -m pip install -r requirements.txt
copy .env.example .env
```

Put `GROQ_API_KEY` in `.env`. Place invoice PDFs in `Bills_Data/`.

Running the extractor sends extracted invoice text to Groq. Do that only when you are allowed to transmit those invoices. `.env`, PDFs, the database, and `output/` are gitignored.

## Run

```bash
python llm_extractor.py --input Bills_Data --output output/llm
```

Useful flags:

- `--dry-run` reads PDFs locally and does not call Groq.
- `--concurrency 2` limits parallel requests. The default is 2.
- `--model openai/gpt-oss-120b` is the default. If it is unavailable, the runner tries `openai/gpt-oss-20b`.

Outputs:

- `output/llm/invoices/<invoice>.json` — one file per PDF
- `output/llm/invoices.json` — every result
- `output/llm/extraction_report.json` — success, partial, failed, and blocked counts
- `invoices.db` — shared SQLite database

Missing values are `null`. Every successful payload uses the keys defined in `schema.py`. Dates are `YYYY-MM-DD` and money is a JSON number. GSTIN checksums and printed totals are checked after extraction; mismatches stay as printed and are reported as warnings (`partial`).

Scanned PDFs with no text layer fail with an OCR-not-enabled error and are not sent to the model.
