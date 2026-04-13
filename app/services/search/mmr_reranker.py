"""
MMR (Maximal Marginal Relevance) Re-ranker

Standalone post-retrieval re-ranking that balances relevance and diversity.
Works with both SearchResult objects (UnifiedSearchService) and plain dicts
(RAGService).

Usage:
    reranker = MMRReranker(lambda_param=0.7)
    diverse_results = reranker.rerank(results, top_k=10)

Lambda parameter:
    1.0 = pure relevance (no diversity, same as sorted top-k)
    0.7 = default (70% relevance, 30% diversity)
    0.5 = equal balance
    0.0 = pure diversity (ignore relevance)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MMRResult:
    """Result of MMR re-ranking."""
    items: List[Any]              # Re-ranked items (same type as input)
    original_scores: List[float]  # Original relevance scores
    mmr_scores: List[float]       # Final MMR scores
    diversity_gains: List[float]  # Diversity component per item


class MMRReranker:
    """
    Lightweight MMR re-ranker for search results.

    Computes text similarity using TF-IDF vectors for diversity measurement.
    No external model calls — pure math on existing results.
    """

    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param
        self._vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def rerank(
        self,
        items: List[Any],
        top_k: int,
        score_fn: Callable[[Any], float] = None,
        text_fn: Callable[[Any], str] = None,
        lambda_param: Optional[float] = None,
    ) -> MMRResult:
        """
        Re-rank a list of search results using MMR.

        Args:
            items: Search results (SearchResult objects or dicts)
            top_k: Number of diverse results to return
            score_fn: Extract relevance score from an item.
                      Default: item.similarity_score or item['score']
            text_fn: Extract text content from an item for diversity calc.
                      Default: item.content or item['product_name'] + item['description']
            lambda_param: Override instance lambda (0-1)

        Returns:
            MMRResult with re-ranked items
        """
        if not items or top_k <= 0:
            return MMRResult([], [], [], [])

        lam = lambda_param if lambda_param is not None else self.lambda_param

        # Short-circuit: pure relevance mode or not enough items
        if lam >= 1.0 or len(items) <= top_k:
            scores = [self._default_score(item, score_fn) for item in items[:top_k]]
            return MMRResult(
                items=items[:top_k],
                original_scores=scores,
                mmr_scores=scores,
                diversity_gains=[0.0] * min(len(items), top_k),
            )

        # Extract scores and texts
        relevance_scores = [self._default_score(item, score_fn) for item in items]
        texts = [self._default_text(item, text_fn) for item in items]

        # Compute TF-IDF embeddings for diversity
        try:
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            # All texts empty or identical — fall back to top-k by score
            logger.warning("MMR: TF-IDF failed (empty/identical texts), falling back to top-k")
            top_items = sorted(
                range(len(items)), key=lambda i: relevance_scores[i], reverse=True
            )[:top_k]
            scores = [relevance_scores[i] for i in top_items]
            return MMRResult(
                items=[items[i] for i in top_items],
                original_scores=scores,
                mmr_scores=scores,
                diversity_gains=[0.0] * len(top_items),
            )

        # Greedy MMR selection
        selected: List[int] = []
        remaining = set(range(len(items)))
        mmr_scores: List[float] = []
        diversity_gains: List[float] = []

        # First pick: highest relevance
        best_first = max(remaining, key=lambda i: relevance_scores[i])
        selected.append(best_first)
        remaining.discard(best_first)
        mmr_scores.append(relevance_scores[best_first])
        diversity_gains.append(0.0)

        # Subsequent picks: MMR formula
        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = -float("inf")
            best_div = 0.0

            for idx in remaining:
                rel = relevance_scores[idx]

                # Max similarity to any already-selected item
                max_sim = max(sim_matrix[idx, s] for s in selected)
                div = 1.0 - max_sim

                mmr = lam * rel + (1.0 - lam) * div

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
                    best_div = div

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            mmr_scores.append(best_mmr)
            diversity_gains.append(best_div)

        return MMRResult(
            items=[items[i] for i in selected],
            original_scores=[relevance_scores[i] for i in selected],
            mmr_scores=mmr_scores,
            diversity_gains=diversity_gains,
        )

    @staticmethod
    def _default_score(item: Any, score_fn: Optional[Callable] = None) -> float:
        """Extract relevance score from item."""
        if score_fn:
            return score_fn(item)
        if hasattr(item, "similarity_score"):
            return item.similarity_score
        if isinstance(item, dict):
            return item.get("score", 0.0)
        return 0.0

    @staticmethod
    def _default_text(item: Any, text_fn: Optional[Callable] = None) -> str:
        """Extract text content from item for diversity calculation."""
        if text_fn:
            return text_fn(item)
        # SearchResult objects
        if hasattr(item, "content"):
            return item.content or ""
        # RAG service dicts
        if isinstance(item, dict):
            parts = []
            if item.get("product_name"):
                parts.append(item["product_name"])
            if item.get("description"):
                parts.append(item["description"])
            if item.get("content"):
                parts.append(item["content"])
            return " ".join(parts) if parts else ""
        return ""
