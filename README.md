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
Eval: 36 questions, each labelled with the section a correct answer must come
from. Metric is recall@5 — did a chunk from the expected section appear in the
top-5 retrieved? Default retrieval is MMR; the cross-encoder is a selectable
option (see below).

| Metric | MMR (default) | Cross-encoder |
|--------|--------------|---------------|
| recall@5 (overall) | **88.9% (32/36)** | 83.3% (30/36) |
| NVDA | 10/12 | 10/12 |
| AAPL | 11/12 | 10/12 |
| MSFT | 11/12 | 10/12 |

**How I got here — four rounds of measure-then-fix:**

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

3. *MMR reranking (76% → 92% on 25q).* Added two-stage retrieval — over-fetch 40
   candidates, then re-rank with Maximal Marginal Relevance to trade off
   relevance against diversity (see [`retrieval.py`](retrieval.py)). NVDA rose
   4/9 → 7/9 and MSFT 7/8 → 8/8. Diversity reranking recovered the crowded-out
   sections without me optimizing the section label directly.

4. *Cross-encoder + eval expansion (re-baselined to 88.9% on 36q).* Added an
   optional cross-encoder reranker, then doubled the eval — which revealed MMR
   still beats it. Details below.

### Cross-encoder reranking — and why expanding the eval mattered

I added an optional cross-encoder reranker (`rerank="cross"` in
[`retrieval.py`](retrieval.py)): over-fetch 40 candidates with the bi-encoder,
then re-score each *(query, chunk)* pair with a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) that reads query and chunk **together** with full
attention, and take the top 5. Unlike the bi-encoder it can't precompute a
reusable vector, so it only runs on the 40-candidate pool — the standard
two-stage retrieval pattern.

On the original 25-question set it looked like a wash: it fixed the **NVDA Item
1C** semantic-overlap miss but traded it for an **AAPL Item 5** miss, holding at
92%. A one-question swing on 25 questions is within noise — I couldn't tell
whether the cross-encoder helped, hurt, or did nothing. So I expanded the eval to
**36 questions** (second-angle questions on sections with real content, labelled
by 10-K convention and *not* filtered by whether they pass), and the bigger set
resolved it:

| Reranker | recall@5 (36q) | Behaviour |
|----------|---------------|-----------|
| **MMR (default)** | **88.9% (32/36)** | misses NVDA 1C, NVDA 3 (stub), AAPL & MSFT liquidity (Item 7) |
| Cross-encoder | 83.3% (30/36) | *fixes* NVDA 1C, but loses AAPL 5, NVDA liquidity, MSFT tax-notes |

**The cross-encoder is actually slightly worse here.** It nails the targeted
semantic-overlap case (1C) but, by chasing pure relevance, it discards the
*diversity* MMR provides — so on questions whose answer-section is crowded by a
dominant section it returns near-duplicate high-relevance chunks from the wrong
section and misses. So MMR stays the default and the cross-encoder remains a
selectable option. The evidence-motivated next experiment is to **chain** them
(cross-encoder for relevance → MMR for diversity) so the 1C fix doesn't cost the
diversity wins.

This is the whole point of a retrieval eval: the small set *hid* a real
regression, the larger one *surfaced* it, and the decision (keep MMR) is now
evidence-based rather than a guess. A separate, reranker-independent miss showed
up too — "liquidity & capital resources" questions (Item 7) get out-ranked by
cash-flow content in Item 8 under *both* rerankers, which points at
chunking/labeling rather than reranking.

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
- The eval set is 36 questions over 3 companies. recall@k is only as meaningful
  as the labels behind it; broadening coverage (more companies, more sections)
  is ongoing.
- "Liquidity & capital resources" questions (Item 7) miss under *both* rerankers
  — "liquidity" lexically matches the cash-flow statement (Item 8) and out-ranks
  the MD&A discussion. This is a chunking/labeling issue, not a reranker one.
- The cross-encoder fixes pure semantic-overlap (NVDA Item 1C) but underperforms
  MMR overall (83.3% vs 88.9%) by discarding diversity; chaining the two
  (relevance then diversity) is the next experiment.
- MSFT's iXBRL primary doc extracts thinner than AAPL/NVDA's; a cleaner source
  would be better.

## Next

- [x] Section-aware splitting + eval across 3 companies (now 36 questions)
- [x] MMR reranking to fix section imbalance (recall@5 76% → 92% on 25q)
- [x] Cross-encoder reranker (fixes NVDA Item 1C; underperforms MMR on the expanded eval)
- [x] Expand the eval to 36 questions — revealed MMR (88.9%) > cross-encoder (83.3%)
- [ ] Chain cross-encoder → MMR (relevance then diversity) to keep the 1C fix without losing diversity
- [ ] Fix the Item 7 "liquidity" miss (cash-flow content in Item 8 out-ranks the MD&A)
- [ ] Fall back to the cleaner `.htm` exhibit when the primary doc is iXBRL
- [ ] Add a sources panel in the UI that links to the EDGAR document
