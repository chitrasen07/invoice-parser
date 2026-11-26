"""Local setup check. Prints no secrets."""

from pathlib import Path
import hashlib
import os

from dotenv import load_dotenv

load_dotenv(Path(".env"))
key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
print("key_loaded", bool(key), "length", len(key))
print("placeholder", (not key) or key.lower().startswith("your_"))

from groq import Groq

client = Groq(api_key=key, timeout=30)
models = [model.id for model in client.models.list().data]
print("model_count", len(models))
print("gpt_oss_120b", "openai/gpt-oss-120b" in models)
print("gpt_oss_20b", "openai/gpt-oss-20b" in models)
print("oss_models", ",".join(model for model in models if "oss" in model or "gpt" in model))

from pdf_reader import extract_pdf

root = Path("Bills_Data")
files = sorted(
    (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
    key=lambda path: path.name.lower(),
)
print("pdf_count", len(files))
digests: dict[str, list[str]] = {}
text_digests: dict[str, list[str]] = {}
for path in files:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:12]
    doc = extract_pdf(path)
    text_digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()[:12]
    digests.setdefault(digest, []).append(path.name)
    text_digests.setdefault(text_digest, []).append(path.name)
    print(
        f"{path.name}|pages={doc.page_count}|tables={doc.table_count}|chars={doc.char_count}|"
        f"file={digest}|text={text_digest}|labels={','.join(doc.labels_seen)}|warn={len(doc.warnings)}"
    )

print("BYTE_DUPES")
for digest, names in digests.items():
    if len(names) > 1:
        print(digest, " :: ".join(names))
print("TEXT_DUPES")
for digest, names in text_digests.items():
    if len(names) > 1:
        print(digest, " :: ".join(names))
