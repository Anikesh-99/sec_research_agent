"""
Turn a filing's raw HTML into clean, section-keyed text.

10-Ks and 10-Qs have a rigid "Item" structure (Item 1A Risk Factors,
Item 7 MD&A, ...). Splitting on those headers beats naive fixed-size chunking:
a retrieved chunk carries its section name, so answers can cite "Item 1A —
Risk Factors" instead of "chunk 47". That section metadata is the whole point.
"""

import re
from bs4 import BeautifulSoup

# Canonical 10-K items we care about. The regex is deliberately loose because
# filings format headers inconsistently ("ITEM 1A.", "Item 1A —", etc.).
ITEM_PATTERN = re.compile(
    r"\bITEM\s+(\d+[A-Z]?)\b[\.\:\-\—\s]*",
    re.IGNORECASE,
)


def html_to_text(html: str) -> str:
    """Strip tags, scripts, and styles; collapse whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines / spaces.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> dict[str, str]:
    """
    Split flat filing text into {item_label: body}.

    Falls back to a single 'full_document' section if no item headers are
    found (some 10-Qs and exhibits don't follow the pattern cleanly).
    """
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return {"full_document": text}

    sections: dict[str, str] = {}
    for idx, m in enumerate(matches):
        item_num = m.group(1).upper()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 200:
            # Likely a table-of-contents reference, not the real section.
            continue
        label = f"Item {item_num}"
        # Keep the longest body if an item appears twice (TOC + real section).
        if label not in sections or len(body) > len(sections[label]):
            sections[label] = body
    return sections or {"full_document": text}


if __name__ == "__main__":
    import sys
    from ingest.edgar import get_filings, download_filing

    f = get_filings(sys.argv[1] if len(sys.argv) > 1 else "NVDA", "10-K", 1)[0]
    html = download_filing(f["url"])
    secs = split_into_sections(html_to_text(html))
    for name, body in secs.items():
        print(f"{name:<12} {len(body):>8} chars")
