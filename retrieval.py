"""
Two-stage retrieval: over-fetch by embedding similarity, then re-rank with MMR.

Why MMR (Maximal Marginal Relevance)? Plain top-k from a bi-encoder returns the
k chunks closest to the query — which, when one section dominates the corpus
(e.g. NVDA's Item 1A is 226 chunks), tends to be k near-duplicate chunks from
that section. A question about a smaller section (cybersecurity governance,
buybacks) gets crowded out even though the right chunk exists.

MMR fixes that by selecting each next chunk to maximize:

    lambda * relevance(query, chunk) - (1 - lambda) * max_similarity(chunk, already_selected)

Lower lambda = more diversity. Better section coverage falls out of reducing
redundancy — we're not optimizing the section label directly.

Both the eval and the agent's retrieve tool call search(), so they can't drift.
"""

import numpy as np
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from index.build_index import get_collection

CANDIDATES = 40       # pool fetched before re-ranking
LAMBDA_MULT = 0.5     # relevance vs. diversity trade-off

_ef = None

_ce = None

def _cross_encoder():
    global _ce
    if _ce is None:
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _ce

def _embed_query(text: str) -> np.ndarray:
    global _ef
    if _ef is None:
        _ef = embedding_functions.DefaultEmbeddingFunction()
    return np.asarray(_ef([text])[0], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)

def _mmr_search(query: str, k: int = 5, where: dict | None = None,
               candidates: int = CANDIDATES, lambda_mult: float = LAMBDA_MULT) -> list[dict]:
    """Return up to k hits as {document, metadata}, MMR-reranked for diversity."""
    res = get_collection().query(
        query_texts=[query],
        n_results=candidates,
        where=where,
        include=["documents", "metadatas", "embeddings"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    embs = res["embeddings"][0]
    if not docs:
        return []

    E = _normalize(np.asarray(embs, dtype=float))
    q = _normalize(_embed_query(query))
    relevance = E @ q  # cosine similarity to the query

    selected: list[int] = []
    remaining = list(range(len(docs)))
    while remaining and len(selected) < k:
        if not selected:
            chosen = remaining[int(np.argmax(relevance[remaining]))]
        else:
            sel_E = E[selected]
            best_score, chosen = -np.inf, remaining[0]
            for idx in remaining:
                redundancy = float(np.max(sel_E @ E[idx]))
                score = lambda_mult * relevance[idx] - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best_score, chosen = score, idx
        selected.append(chosen)
        remaining.remove(chosen)

    return [{"document": docs[i], "metadata": metas[i]} for i in selected]

def _cross_encoder_search(query: str, k: int = 5, where: dict | None = None,
               candidates: int = CANDIDATES, lambda_mult: float = LAMBDA_MULT) -> list[dict]:
    res = get_collection().query(
        query_texts=[query],
        n_results=candidates,
        where=where,
        include=["documents", "metadatas", "embeddings"]
    )
    ce = _cross_encoder()
    docs = res["documents"][0]
    meta = res["metadatas"][0]
    if not docs:
        return []
    pairs = [(query, doc) for doc in docs]
    scores = ce.predict(pairs)
    chosen = sorted(zip(docs, meta, scores), key=lambda x: x[2], reverse = True)[:k]
    chosen_docs, chosen_metas = [choice[0] for choice in chosen], [choice[1] for choice in chosen]
    return [{"document": chosen_docs[i], "metadata": chosen_metas[i]} for i in range(len(chosen))]

def search(query: str, k: int = 5, where: dict | None = None,
           candidates: int = CANDIDATES, lambda_mult: float = LAMBDA_MULT, rerank: str = "MMR") -> list[dict]:
    return _mmr_search(query, k, where, candidates, lambda_mult) if rerank == "MMR" else _cross_encoder_search(query, k, where, candidates, lambda_mult)