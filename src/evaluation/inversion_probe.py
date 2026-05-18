"""Linear-probe embedding inversion attack.

This attack trains a linear decoder from embedding space to a bag-of-words token
representation. At evaluation time, a possibly noisy embedding is decoded by
predicting token activations and returning the top predicted vocabulary items.
It is stronger than a pure nearest-neighbour lookup because it can compose tokens
that need not appear in a single retrieved training example, while remaining far
lighter than sequence-level attacks such as Vec2Text.
"""

from __future__ import annotations

import numpy as np

try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:  # pragma: no cover - exercised only without dependency.
    rouge_scorer = None

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge


class _RougeLScore:
    """Small score object matching rouge-score's ``.fmeasure`` attribute."""

    def __init__(self, fmeasure: float) -> None:
        self.fmeasure = float(fmeasure)


class _FallbackRougeScorer:
    """Minimal ROUGE-L F1 scorer used when ``rouge-score`` is unavailable."""

    @staticmethod
    def _lcs_len(a: list[str], b: list[str]) -> int:
        prev = [0] * (len(b) + 1)
        for tok_a in a:
            curr = [0]
            for j, tok_b in enumerate(b, start=1):
                if tok_a == tok_b:
                    curr.append(prev[j - 1] + 1)
                else:
                    curr.append(max(prev[j], curr[-1]))
            prev = curr
        return prev[-1]

    def score(self, target: str, prediction: str) -> dict[str, _RougeLScore]:
        target_tokens = str(target).lower().split()
        pred_tokens = str(prediction).lower().split()
        if not target_tokens or not pred_tokens:
            return {"rougeL": _RougeLScore(0.0)}
        lcs = self._lcs_len(target_tokens, pred_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(target_tokens)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return {"rougeL": _RougeLScore(f1)}


def _build_rouge_scorer():
    """Return the official ROUGE scorer when available, else a small fallback."""
    if rouge_scorer is not None:
        return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _FallbackRougeScorer()


class LinearProbeInversion:
    """Train a linear map from embeddings to bag-of-words token indicators.

    Attributes:
        max_features: Maximum vocabulary size for the bag-of-words target.
        top_k_tokens: Number of predicted tokens used in each reconstruction.
        vectorizer: Fitted ``CountVectorizer`` over the attack training corpus.
        probe: Fitted multi-output ``Ridge`` regressor from embeddings to BoW.
        rouge: ROUGE-L scorer for reconstruction quality.
    """

    def __init__(
        self,
        max_features: int = 2000,
        top_k_tokens: int = 20,
        alpha: float = 1.0,
    ) -> None:
        """Configure the probe attack.

        Args:
            max_features: Maximum number of vocabulary features to predict.
            top_k_tokens: Number of tokens emitted by ``reconstruct``.
            alpha: Ridge regularization strength.
        """
        if max_features <= 0:
            msg = "max_features must be positive."
            raise ValueError(msg)
        if top_k_tokens <= 0:
            msg = "top_k_tokens must be positive."
            raise ValueError(msg)
        if alpha < 0.0:
            msg = "alpha must be non-negative."
            raise ValueError(msg)

        self.max_features = int(max_features)
        self.top_k_tokens = int(top_k_tokens)
        self.vectorizer = CountVectorizer(max_features=self.max_features, binary=True)
        self.probe = Ridge(alpha=float(alpha))
        self.rouge = _build_rouge_scorer()
        self._is_fitted = False
        self._embedding_dim: int | None = None

    @staticmethod
    def _as_2d_embeddings(embeddings: np.ndarray, name: str) -> np.ndarray:
        """Convert embeddings to a finite 2-D float32 matrix."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            msg = f"{name} must be a 1-D vector or 2-D embedding matrix."
            raise ValueError(msg)
        if x.shape[0] == 0 or x.shape[1] == 0:
            msg = f"{name} must have non-zero rows and columns."
            raise ValueError(msg)
        if not np.isfinite(x).all():
            msg = f"{name} contains NaN or infinite values."
            raise ValueError(msg)
        return x

    def fit(self, corpus_texts: list[str], corpus_embeddings: np.ndarray) -> None:
        """Fit the vectorizer and linear probe on clean reference embeddings.

        Args:
            corpus_texts: Raw text strings aligned with ``corpus_embeddings`` rows.
            corpus_embeddings: ``(N, d)`` clean embeddings for attack training.
        """
        texts = list(corpus_texts)
        x = self._as_2d_embeddings(corpus_embeddings, "corpus_embeddings")
        if len(texts) != x.shape[0]:
            msg = (
                "corpus_texts length must match the number of embedding rows, "
                f"got {len(texts)} texts and {x.shape[0]} embeddings."
            )
            raise ValueError(msg)
        if not texts:
            msg = "corpus_texts must contain at least one document."
            raise ValueError(msg)

        bow = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        self.probe.fit(x, bow)
        self._embedding_dim = int(x.shape[1])
        self._is_fitted = True

    def reconstruct(self, embedding: np.ndarray) -> str:
        """Decode one embedding into a token-sequence reconstruction.

        Args:
            embedding: Length-``d`` or ``(1, d)`` possibly noisy embedding vector.

        Returns:
            Space-joined top predicted tokens from the learned vocabulary.
        """
        if not self._is_fitted or self._embedding_dim is None:
            msg = "LinearProbeInversion must be fitted before reconstruction."
            raise RuntimeError(msg)

        e = self._as_2d_embeddings(embedding, "embedding")
        if e.shape[0] != 1:
            msg = "reconstruct expects exactly one embedding row."
            raise ValueError(msg)
        if e.shape[1] != self._embedding_dim:
            msg = (
                f"Expected embedding width {self._embedding_dim}, got {e.shape[1]}."
            )
            raise ValueError(msg)

        bow_pred = np.asarray(self.probe.predict(e)[0], dtype=np.float32)
        vocab = self.vectorizer.get_feature_names_out()
        k = min(self.top_k_tokens, len(vocab))
        top_idx = np.argsort(bow_pred)[::-1][:k]
        return " ".join(str(vocab[i]) for i in top_idx)

    def score(
        self, embedding: np.ndarray, original_text: str
    ) -> dict[str, str | float]:
        """Reconstruct one embedding and compute ROUGE-L against gold text.

        Args:
            embedding: Length-``d`` or ``(1, d)`` possibly noisy embedding vector.
            original_text: Gold text for this sample.

        Returns:
            Dict with ``reconstructed_text`` and ``rouge_l_fmeasure``.
        """
        reconstructed = self.reconstruct(embedding)
        scores = self.rouge.score(str(original_text), reconstructed)
        return {
            "reconstructed_text": reconstructed,
            "rouge_l_fmeasure": float(scores["rougeL"].fmeasure),
        }
