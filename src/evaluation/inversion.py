import faiss
import numpy as np
from rouge_score import rouge_scorer

class EmbeddingInversion:
    EMBED_DIM: int = 384
    def __init__(self, reference_corpus: list[str], reference_embeddings: np.ndarray) -> None:
        self.reference_corpus = list(reference_corpus)
        mat = np.asarray(reference_embeddings, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.shape[1] != self.EMBED_DIM:
            msg = f'Expected embedding width {self.EMBED_DIM}, got {mat.shape[1]}'
            raise ValueError(msg)
        if len(self.reference_corpus) != mat.shape[0]:
            msg = 'reference_corpus length must match the number of embedding rows.'
            raise ValueError(msg)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.index.add(mat)
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def nearest_neighbor_lookup(self, target_embedding: np.ndarray, original_text: str) -> dict[str, str | float]:
        q = np.asarray(target_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.EMBED_DIM:
            msg = f'Query dim must be {self.EMBED_DIM}, got {q.shape[1]}'
            raise ValueError(msg)
        _, idx = self.index.search(q, 1)
        ridx = int(idx[0, 0])
        text = self.reference_corpus[ridx]
        scores = self.rouge.score(original_text, text)
        f1 = float(scores['rougeL'].fmeasure)
        return {'retrieved_text': text, 'rouge_l_fmeasure': f1}
