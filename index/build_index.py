"""
Build the vector index: fetch filings -> section text -> chunks -> Chroma.

Embeddings: Chroma's default (sentence-transformers all-MiniLM-L6-v2), which
runs locally. No extra API key, and good enough for retrieval over filing
prose. Swap the embedding_function here if you want OpenAI/Voyage later.

Each chunk stores metadata: ticker, form, filing_date, section, chunk_index.
That metadata is what lets the agent cite its sources.
"""

import os
import chromadb

from ingest.edgar import get_filings, download_filing
from ingest.parse import html_to_text, split_into_sections

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
COLLECTION = "sec_filings"

CHUNK_SIZE = 1200       # characters; sections are long, chunks stay readable
CHUNK_OVERLAP = 200     # keep context across boundaries


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION)


def build(ticker: str, form: str = "10-K", count: int = 2) -> int:
    """Ingest, parse, chunk, and index. Returns number of chunks added."""
    collection = get_collection()
    added = 0

    for filing in get_filings(ticker, form, count):
        print(f"  fetching {filing['form']} {filing['filing_date']} ...")
        html = download_filing(filing["url"])
        sections = split_into_sections(html_to_text(html))

        for section, body in sections.items():
            for ci, chunk in enumerate(_chunk(body)):
                uid = f"{filing['accession']}::{section}::{ci}"
                collection.add(
                    ids=[uid],
                    documents=[chunk],
                    metadatas=[{
                        "ticker": filing["ticker"],
                        "form": filing["form"],
                        "filing_date": filing["filing_date"],
                        "accession": filing["accession"],
                        "section": section,
                        "chunk_index": ci,
                    }],
                )
                added += 1
    print(f"  indexed {added} chunks for {ticker}")
    return added


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["NVDA"]
    for t in tickers:
        print(f"Building index for {t}")
        build(t, "10-K", 2)
