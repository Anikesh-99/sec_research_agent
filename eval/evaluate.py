"""
Retrieval eval — written before the app, on purpose.

The thing that actually determines answer quality in RAG is whether the right
passage gets retrieved. So we measure that directly: for each labeled question,
did a chunk from the expected (ticker, section) show up in the top-k?

This is a small, honest harness. Grow EVAL_QUESTIONS as you index more tickers.
The headline number to report is recall@k.

Usage:
  python eval/evaluate.py          # uses k=5
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from index.build_index import get_collection

# Each: question + the (ticker, section) a correct answer must come from.
# NOTE: section labels must match parse.py output ("Item 1A", "Item 7", ...).
# These are seed examples — verify/extend them against your indexed filings.
EVAL_QUESTIONS = [
    {"q": "What are NVDA's main risk factors?",                 "ticker": "NVDA", "section": "Item 1A"},
    {"q": "How does NVDA describe its liquidity and capital resources?", "ticker": "NVDA", "section": "Item 7"},
    {"q": "What legal proceedings is NVDA involved in?",        "ticker": "NVDA", "section": "Item 3"},
]


def recall_at_k(k: int = 5) -> None:
    collection = get_collection()
    hits = 0

    print(f"{'hit':<4} {'ticker':<6} {'section':<10} question")
    print("-" * 70)
    for item in EVAL_QUESTIONS:
        res = collection.query(
            query_texts=[item["q"]],
            n_results=k,
            where={"ticker": item["ticker"]},
        )
        metas = res.get("metadatas", [[]])[0]
        hit = any(m.get("section") == item["section"] for m in metas)
        hits += int(hit)
        print(f"{'  ✓' if hit else '  ✗':<4} {item['ticker']:<6} {item['section']:<10} {item['q'][:45]}")

    n = len(EVAL_QUESTIONS)
    print("-" * 70)
    print(f"recall@{k}: {hits}/{n} = {hits / n * 100:.1f}%")


if __name__ == "__main__":
    recall_at_k(k=5)
