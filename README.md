# SEC filing research agent

Ask a natural-language question about a company's SEC filings and get an answer
that **cites the filing and section** it came from. For example:

> "What are the main risk factors in NVIDIA's latest 10-K, and how do they
> differ from the prior year?"

The agent retrieves the relevant passages from real EDGAR filings, reasons over
them, and answers with citations like `[NVDA 10-K 2025-02-26 — Item 1A]` that
you can verify against the source.

> **Status:** the ingest → parse → index → eval pipeline runs end-to-end on
> real NVDA filings (see results below). The agent + Streamlit layer are written
> but I haven't exercised them on a large corpus yet. Eval set is small (3 seed
> questions) and I'm expanding it.

## Why section-aware chunking (not fixed-size splits)

The obvious thing is to split each filing into 1,000-character chunks and embed
them. I started there and retrieval was mediocre: a question about *liquidity*
would pull a chunk that straddled the end of the risk-factors section and the
start of MD&A, and the model couldn't tell where it came from.

10-Ks have a rigid "Item" structure (Item 1A Risk Factors, Item 7 MD&A, ...),
so I split on those headers first, then chunk *within* each section. Every chunk
now carries its section label as metadata. Two payoffs:

1. **Citations** — answers point at "Item 1A", not "chunk 47".
2. **Filtered retrieval** — I can constrain a query to a section or ticker,
   which is also how the eval measures recall.

See [`ingest/parse.py`](ingest/parse.py) for the splitting and
[`index/build_index.py`](index/build_index.py) for chunking.

## Why a retrieval eval, and why I wrote it first

In RAG, answer quality is mostly downstream of *retrieval* quality — if the
right passage never gets retrieved, no prompt saves you. So before building the
UI I wrote [`eval/evaluate.py`](eval/evaluate.py), which measures **recall@k**:
for a labeled question, did a chunk from the expected (ticker, section) appear
in the top-k? That number is what I tune chunk size and overlap against.

### Results

Corpus: NVDA's two most recent 10-Ks (filed 2026-02-25 and 2025-02-26), 690
chunks.

| Metric | Value |
|--------|-------|
| recall@5 | 66.7% (2/3) |
| # eval questions | 3 (seed; expanding) |

**What the first run taught me.** Initial recall@5 was 33% (1/3). Inspecting
the index showed the section splitter was mis-attributing spans — Item 7 (MD&A)
had only 17 chunks while Item 9A (normally tiny) had 142 — because the header
regex matched *every* "Item N" string, including inline cross-references and the
table of contents. Anchoring the pattern to line-start headings fixed the
boundaries (Item 7 → 73 chunks, Item 9A → 8) and recall@5 rose to 67%.

The remaining miss is honest, not a bug: NVDA's Item 3 (Legal Proceedings) is a
one-line stub that defers to a financial-statements note, so there's no
substantive passage to retrieve. I'm keeping that question in the set rather
than deleting it to inflate the score.

## Architecture

```
ticker ──> EDGAR API ──> section-aware parse ──> chunk ──> Chroma (local embeddings)
                                                              │
question ──> agent (Claude + retrieve tool) ──> cited answer ─┘
```

| File | Job |
|------|-----|
| `ingest/edgar.py` | Map ticker → CIK, pull 10-K/10-Q from EDGAR |
| `ingest/parse.py` | HTML → section-keyed text |
| `index/build_index.py` | Chunk + embed → Chroma |
| `agent/tools.py` | `retrieve` tool with citation metadata |
| `agent/research_agent.py` | Tool-calling agent (Claude) |
| `eval/evaluate.py` | recall@k retrieval eval |
| `app.py` | Streamlit UI |

Embeddings run locally (Chroma's default `all-MiniLM-L6-v2`), so the only API
key you need is Anthropic's.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY and SEC_USER_AGENT

python -m index.build_index NVDA   # fetch + index NVDA's two latest 10-Ks
python eval/evaluate.py            # check retrieval recall
streamlit run app.py              # ask questions in the browser
```

SEC requires a descriptive `User-Agent` with contact info on every request —
set `SEC_USER_AGENT` in `.env` or EDGAR will block you.

## Known limitations

- Section splitting is regex-based on "Item N" headers. Some 10-Qs and older
  filings format headers oddly and fall back to a single `full_document`
  section — chunking still works but citations are coarser.
- The eval set is small (3 seed questions). recall@k is only as meaningful as
  the labels behind it; expanding this is the next priority.
- No cross-filing dedup: if you index overlapping amendments you may retrieve
  near-duplicate passages.

## Next

- [ ] Finalize eval set (~25 questions across tickers), report recall@5
- [ ] Tune chunk size / overlap against that number
- [ ] Add a sources panel in the UI that links to the EDGAR document
