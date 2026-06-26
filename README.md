# SEC filing research agent

Ask a natural-language question about a company's SEC filings and get an answer
that **cites the filing and section** it came from. For example:

> "What are the main risk factors in NVIDIA's latest 10-K, and how do they
> differ from the prior year?"

The agent retrieves the relevant passages from real EDGAR filings, reasons over
them, and answers with citations like `[NVDA 10-K 2025-02-26 — Item 1A]` that
you can verify against the source.

> **Status:** the ingest → parse → index → eval pipeline runs end-to-end on
> real filings from three companies (see results below). The agent + Streamlit
> layer are written but I haven't exercised them on a large corpus yet.

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

Corpus: two most recent 10-Ks each for NVDA, AAPL, and MSFT (~1,220 chunks).
Eval: 25 questions, each labelled with the section a correct answer must come
from. Metric is recall@5 — did a chunk from the expected section appear in the
top-5 retrieved?

| Metric | Value |
|--------|-------|
| recall@5 (overall) | **92.0% (23/25)** |
| AAPL | 8/8 |
| MSFT | 8/8 |
| NVDA | 7/9 |

**How I got here — three rounds of measure-then-fix:**

1. *Section splitting (33% → 67%).* The first run scored 33% on 3 NVDA
   questions. Inspecting the index showed mis-attributed spans — Item 7 (MD&A)
   had 17 chunks while Item 9A (normally tiny) had 142 — because the header
   regex matched *every* "Item N" string, including inline cross-references and
   the table of contents. Anchoring the regex to line-start headings fixed the
   boundaries (Item 7 → 73, Item 9A → 8) and recall rose to 67%.

2. *Expanded to 25 questions / 3 companies (76%).* A bigger set exposed
   **section imbalance**: NVDA scored worst (4/9) despite having the most chunks,
   because its huge Item 1A (226 chunks) crowded smaller sections out of the
   top-5. The bi-encoder kept returning near-duplicate Item 1A passages.

3. *MMR reranking (76% → 92%).* Added two-stage retrieval — over-fetch 40
   candidates, then re-rank with Maximal Marginal Relevance to trade off
   relevance against diversity (see [`retrieval.py`](retrieval.py)). NVDA rose
   4/9 → 7/9 and MSFT 7/8 → 8/8. Diversity reranking recovered the crowded-out
   sections without me optimizing the section label directly.

The two misses under MMR are both honest:
- **NVDA Item 1C (cybersecurity).** Its Item 1A talks about cybersecurity *risk*
  while the question asks about *governance* — a genuine semantic-overlap case
  the bi-encoder + MMR gets wrong.
- **NVDA Item 3 (Legal Proceedings).** A one-line stub deferring to a
  financial-statements note — nothing substantive to retrieve. Kept in the set
  rather than deleted to inflate the score.

### Cross-encoder reranking — the honest result

I then added an optional cross-encoder reranker (`rerank="cross"` in
[`retrieval.py`](retrieval.py)): over-fetch 40 candidates with the bi-encoder,
then re-score each *(query, chunk)* pair with a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) that reads query and chunk **together** with full
attention, and take the top 5. Unlike the bi-encoder it can't precompute a
reusable vector, so it only runs on the 40-candidate pool — the standard
two-stage retrieval pattern.

It did exactly what it was designed to: **the NVDA Item 1C semantic-overlap miss
flipped to a hit.** But aggregate recall@5 held at 92% — it *traded* Item 1C for
a new AAPL Item 5 (buybacks) miss. On a 25-question set a one-question swing is
within noise, so I can't claim the cross-encoder moved the metric; I can only
show it fixed the case it was built for. The real bottleneck is now eval size:
expanding the question set is the prerequisite to *proving* a reranker gain,
which is why it's the top item under *Next*.

## Architecture

```
ticker ──> EDGAR API ──> section-aware parse ──> chunk ──> Chroma (local embeddings)
                                                              │
question ──> agent (Claude) ──> retrieve ──> over-fetch 40 ──> rerank (MMR | cross-encoder) ──> top 5
                  │                                                            │
                  └──────────────────── cited answer ◄────────────────────────┘
```

| File | Job |
|------|-----|
| `ingest/edgar.py` | Map ticker → CIK, pull 10-K/10-Q from EDGAR |
| `ingest/parse.py` | HTML → section-keyed text |
| `index/build_index.py` | Chunk + embed → Chroma |
| `retrieval.py` | Two-stage search: over-fetch + rerank (MMR or cross-encoder; shared by eval and agent) |
| `agent/tools.py` | `retrieve` tool with citation metadata |
| `agent/research_agent.py` | Tool-calling agent (Claude) |
| `eval/evaluate.py` | recall@k retrieval eval |
| `app.py` | Streamlit UI |

The eval and the agent both retrieve through `retrieval.py`, so the number the
eval reports is the retrieval the agent actually gets — they can't drift.

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
- The eval set is 25 questions over 3 companies. recall@k is only as meaningful
  as the labels behind it; broadening coverage is ongoing.
- The 25-question eval is too small to resolve reranker differences: the
  cross-encoder fixes the NVDA Item 1C semantic-overlap miss but trades it for
  AAPL Item 5, so aggregate recall@5 is unchanged at 92%. Expanding the eval so
  the gain is statistically visible is the real next lever.
- MSFT's iXBRL primary doc extracts thinner than AAPL/NVDA's; MMR masks it in
  the eval but a cleaner source would be better.

## Next

- [x] Section-aware splitting + 25-question eval across 3 companies
- [x] MMR reranking to fix section imbalance (recall@5 76% → 92%)
- [x] Cross-encoder reranker (fixes NVDA Item 1C; eval too small to show an aggregate gain)
- [ ] Expand the eval set so reranker gains are statistically visible
- [ ] Fall back to the cleaner `.htm` exhibit when the primary doc is iXBRL
- [ ] Add a sources panel in the UI that links to the EDGAR document
