"""
The one tool the agent has: semantic retrieval over the filing index.

It returns chunks *with* citation metadata baked into the text, so when the
model writes an answer it can quote the source ("Item 1A, NVDA 10-K filed
2025-02-26"). Citations the user can verify are the difference between a
research tool and a chatbot that sounds confident.
"""

from langchain_core.tools import tool

from retrieval import search


def _format_hit(doc: str, meta: dict) -> str:
    cite = f"[{meta['ticker']} {meta['form']} {meta['filing_date']} — {meta['section']}]"
    return f"{cite}\n{doc}"


@tool
def retrieve(query: str, k: int = 5) -> str:
    """Search indexed SEC filings for passages relevant to the query.

    Use this for any question about a company's filings. Returns the top
    passages, each prefixed with its citation (ticker, form, date, section).
    """
    hits = search(query, k=k)
    if not hits:
        return "No relevant passages found. The filing may not be indexed yet."

    return "\n\n---\n\n".join(_format_hit(h["document"], h["metadata"]) for h in hits)
