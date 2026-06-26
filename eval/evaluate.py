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

from retrieval import search

# Each: question + the (ticker, section) a correct answer must come from.
# Section labels must match parse.py output ("Item 1A", "Item 7", ...).
#
# Labelling rule: questions target a section that genuinely exists in the
# indexed corpus, so this measures retrieval rather than missing data. The one
# exception is NVDA Item 3 (Legal Proceedings) — NVDA's is a one-line stub, so
# that question is a known, documented miss kept in for honesty.
EVAL_QUESTIONS = [
    # --- NVDA ---
    {"q": "What are the biggest risks Nvidia warns investors about?",                       "ticker": "NVDA", "section": "Item 1A"},
    {"q": "What supply-chain and manufacturing concentration risks does Nvidia describe?",  "ticker": "NVDA", "section": "Item 1A"},
    {"q": "How did Nvidia's data center revenue and gross margin trend this year?",          "ticker": "NVDA", "section": "Item 7"},
    {"q": "What business segments and products does Nvidia operate?",                        "ticker": "NVDA", "section": "Item 1"},
    {"q": "What is Nvidia's exposure to foreign currency and interest rate risk?",           "ticker": "NVDA", "section": "Item 7A"},
    {"q": "How does Nvidia govern and manage cybersecurity risk?",                           "ticker": "NVDA", "section": "Item 1C"},
    {"q": "What does Nvidia disclose about share repurchases and dividends?",                "ticker": "NVDA", "section": "Item 5"},
    {"q": "What did Nvidia's management conclude about internal control over financial reporting?", "ticker": "NVDA", "section": "Item 9A"},
    {"q": "What legal proceedings is Nvidia involved in?",                                   "ticker": "NVDA", "section": "Item 3"},  # known stub miss

    # --- AAPL ---
    {"q": "What are the main risks Apple discloses to shareholders?",                        "ticker": "AAPL", "section": "Item 1A"},
    {"q": "How did Apple's net sales by product category change year over year?",            "ticker": "AAPL", "section": "Item 7"},
    {"q": "What products, services, and segments make up Apple's business?",                 "ticker": "AAPL", "section": "Item 1"},
    {"q": "What market risks from interest rates and currency does Apple face?",             "ticker": "AAPL", "section": "Item 7A"},
    {"q": "How does Apple approach cybersecurity risk management and governance?",           "ticker": "AAPL", "section": "Item 1C"},
    {"q": "What does Apple say about its stock repurchase program and dividends?",           "ticker": "AAPL", "section": "Item 5"},
    {"q": "What legal proceedings or litigation is Apple party to?",                         "ticker": "AAPL", "section": "Item 3"},
    {"q": "What is management's assessment of Apple's internal control over financial reporting?", "ticker": "AAPL", "section": "Item 9A"},

    # --- MSFT ---
    {"q": "What key risk factors does Microsoft highlight?",                                 "ticker": "MSFT", "section": "Item 1A"},
    {"q": "How did Microsoft's segment revenue and operating income change year over year?", "ticker": "MSFT", "section": "Item 7"},
    {"q": "What are Microsoft's primary business segments and offerings?",                   "ticker": "MSFT", "section": "Item 1"},
    {"q": "What is Microsoft's exposure to foreign exchange and interest rate risk?",        "ticker": "MSFT", "section": "Item 7A"},
    {"q": "How does Microsoft manage and govern cybersecurity threats?",                     "ticker": "MSFT", "section": "Item 1C"},
    {"q": "What does Microsoft disclose about dividends and share buybacks?",                "ticker": "MSFT", "section": "Item 5"},
    {"q": "What did Microsoft conclude about the effectiveness of its internal controls?",   "ticker": "MSFT", "section": "Item 9A"},
    {"q": "Where does Microsoft describe its cloud and productivity businesses?",            "ticker": "MSFT", "section": "Item 1"},
]


def recall_at_k(k: int = 5, rerank='MMR') -> None:
    hits = 0

    print(f"{'hit':<4} {'ticker':<6} {'section':<10} question")
    print("-" * 70)
    for item in EVAL_QUESTIONS:
        results = search(item["q"], k=k, where={"ticker": item["ticker"]}, rerank=rerank)
        hit = any(h["metadata"].get("section") == item["section"] for h in results)
        hits += int(hit)
        print(f"{'  ✓' if hit else '  ✗':<4} {item['ticker']:<6} {item['section']:<10} {item['q'][:45]}")

    n = len(EVAL_QUESTIONS)
    print("-" * 70)
    print(f"recall@{k}: {hits}/{n} = {hits / n * 100:.1f}%")


if __name__ == "__main__":
    recall_at_k(k=5)
    recall_at_k(k=5, rerank="cross")
