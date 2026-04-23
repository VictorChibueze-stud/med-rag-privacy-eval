"""Heuristic and nearest-neighbor attacks from noisy embeddings to reference text."""

import faiss
import numpy as np
from rouge_score import rouge_scorer


class EmbeddingInversion:
    """FAISS nearest reference text plus ROUGE-L fidelity vs. a gold string.

    The reference set is built once: ``IndexFlatL2(384)`` on **L2** distances in
    embedding space (all rows should be in the same geometry as
    ``SentenceTransformer`` with ``normalize_embeddings=True`` for comparability to
    the RAG / DP experiments).

    Attributes:
        index: A FAISS index over the reference (clean) vectors.
        reference_corpus: Text aligned by row id with ``reference_embeddings`` when
            the index was built.
    """

    EMBED_DIM: int = 384

    def __init__(
        self,
        reference_corpus: list[str],
        reference_embeddings: np.ndarray,
    ) -> None:
        """Add the reference matrix to a fresh L2 index and wire ROUGE for scoring.

        Args:
            reference_corpus: One string per row, same order as the rows of the matrix.
            reference_embeddings: ``(N, 384)`` precomputed, typically unit L2.
        """
        self.reference_corpus = list(reference_corpus)
        mat = np.asarray(reference_embeddings, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.shape[1] != self.EMBED_DIM:
            msg = f"Expected embedding width {self.EMBED_DIM}, got {mat.shape[1]}"
            raise ValueError(msg)
        if len(self.reference_corpus) != mat.shape[0]:
            msg = "reference_corpus length must match the number of embedding rows."
            raise ValueError(msg)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.index.add(mat)
        # ROUGE-L for semantic overlap of retrieved string vs. the original.
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def nearest_neighbor_lookup(
        self, target_embedding: np.ndarray, original_text: str
    ) -> dict[str, str | float]:
        """1-NN in FAISS, then report text and ROUGE-L F-measure to ``original_text``.

        The ``(1, d)`` query comes from a possibly privacy-noised vector; the index
        row id maps back to the **clean** text line used as the inversion guess.

        Args:
            target_embedding: Length-``d`` (or a ``(1, d)``) query vector in R^d.
            original_text: Gold text for the same sample (e.g. pre-noise line).

        Returns:
            A dict with ``retrieved_text`` and ``rouge_l_fmeasure`` in ``[0, 1]``.
        """
        q = np.asarray(target_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.EMBED_DIM:
            msg = f"Query dim must be {self.EMBED_DIM}, got {q.shape[1]}"
            raise ValueError(msg)
        # ``search`` returns squared L2; we only need the argmin.
        _, idx = self.index.search(q, 1)
        ridx = int(idx[0, 0])
        text = self.reference_corpus[ridx]
        # Library convention: (reference, prediction) = (gold, generated).
        scores = self.rouge.score(original_text, text)
        f1 = float(scores["rougeL"].fmeasure)
        return {"retrieved_text": text, "rouge_l_fmeasure": f1}
