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

from index.build_index import get_collection

CANDIDATES = 40       # pool fetched before re-ranking
LAMBDA_MULT = 0.5     # relevance vs. diversity trade-off

_ef = None


def _embed_query(text: str) -> np.ndarray:
    global _ef
    if _ef is None:
        _ef = embedding_functions.DefaultEmbeddingFunction()
    return np.asarray(_ef([text])[0], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def search(query: str, k: int = 5, where: dict | None = None,
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
