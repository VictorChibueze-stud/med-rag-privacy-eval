"""Non-private dense retrieval RAG baseline using SentenceTransformers and FAISS."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGBaseline:
    """FAISS-backed top-k retriever over a fixed corpus of texts.

    Embeddings are L2-normalized at encode time. For unit-norm *difference* of two
    embeddings, the L2 (Euclidean) sensitivity of the *sum* (or of an aggregate under
    a fixed counting query) is bounded by 2.0, which is the value used in the central
    DP module for Gaussian noise.

    Attributes:
        model_name: HuggingFace-style name for the ``SentenceTransformer`` encoder.
        model: Loaded sentence-transformer model.
        index: FAISS ``IndexFlatL2`` over 384-d unit vectors.
        corpus_map: Index-aligned list of the original text per FAISS row.
    """

    EMBED_DIM: int = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the encoder and create an empty L2 index in 384-D space.

        ``all-MiniLM-L6-v2`` has output dimension 384, matching
        ``faiss.IndexFlatL2(384)``.

        Args:
            model_name: Identifier passed to the sentence-transformers model loader.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Inner product in L2-normalized space relates to chord distance; FAISS L2
        # search on unit vectors is standard for this baseline.
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.corpus_map: list[str] = []

    def build_index(self, corpus: list[str]) -> None:
        """Encode the corpus with L2-normalized ST vectors and add them to the index.

        ``normalize_embeddings=True`` maps each row to the unit hypersphere, which is
        the step that lets us treat the embedding map as having controlled global
        sensitivity in downstream DP (bounded ``Delta_2 f`` for a suitable query).

        Args:
            corpus: List of text passages or documents to index.
        """
        if not corpus:
            self.corpus_map = []
            # Reset index: empty.
            self.index = faiss.IndexFlatL2(self.EMBED_DIM)
            return
        # SentenceTransformers: row-wise L2 norms become 1 (up to float error).
        emb = self.model.encode(
            list(corpus),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.EMBED_DIM:
            msg = f"Expected embedding dim {self.EMBED_DIM}, got {arr.shape[1]}"
            raise ValueError(msg)
        self.corpus_map = list(corpus)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.index.add(arr)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return the top-``k`` nearest corpus strings under squared L2 distance.

        The query is embedded on the same unit sphere as the index rows.

        Args:
            query: Natural-language query string.
            k: Number of neighbors to return from the FAISS index.

        Returns:
            The retrieved text strings, best match first, length up to ``k``.
        """
        if self.index.ntotal == 0 or not self.corpus_map:
            return []
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        qv = np.asarray(q, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        k_use = int(min(k, self.index.ntotal, len(self.corpus_map)))
        if k_use <= 0:
            return []
        _, neigh_idx = self.index.search(qv, k_use)
        return [
            self.corpus_map[int(i)]
            for i in neigh_idx[0]
            if 0 <= int(i) < len(self.corpus_map)
        ]
