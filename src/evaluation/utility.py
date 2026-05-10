"""Automatic metrics for RAG (and generation) utility, including BERTScore."""

from typing import Any

import bert_score


class UtilityEvaluator:
    """BERTScore-based agreement between *candidate* and *reference* strings.

    BERTScore compares contextual embeddings (here under ``metric_model``) and
    returns precision / recall / F1 aggregated over the batch.

    Attributes:
        metric_model: Pretrained (HuggingFace) name passed to :mod:`bert_score`.
    """

    def __init__(self, metric_model: str = "roberta-large") -> None:
        """Cache the BERTScorer backbone name to avoid re-downloads per call.

        Args:
            metric_model: HuggingFace model id, e.g. ``roberta-large``.
        """
        self.metric_model = metric_model
        self._scorer = bert_score.BERTScorer(
            model_type=self.metric_model,
            lang="en",
            rescale_with_baseline=False,
        )

    def compute_bertscore(
        self, references: list[str], candidates: list[str]
    ) -> dict[str, Any]:
        """Compute mean P/R/F1 with a cached :class:`bert_score.BERTScorer`.

        Args:
            references: One gold string per row (e.g. original prompt text).
            candidates: Simulated "generations" (e.g. NN-retrieved text under DP noise).

        Returns:
            ``{"precision", "recall", "f1"}`` with scalar means over the micro-batch.
        """
        p, r, f1 = self._scorer.score(candidates, references)
        return {
            "precision": p.mean().item(),
            "recall": r.mean().item(),
            "f1": f1.mean().item(),
        }
