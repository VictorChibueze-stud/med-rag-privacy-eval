import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGBaseline:
    EMBED_DIM: int = 384
    def __init__(self, model_name: str='all-MiniLM-L6-v2') -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.corpus_map: list[str] = []

    def build_index(self, corpus: list[str]) -> None:
        if not corpus:
            self.corpus_map = []
            self.index = faiss.IndexFlatL2(self.EMBED_DIM)
            return
        emb = self.model.encode(list(corpus), normalize_embeddings=True, show_progress_bar=False)
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.EMBED_DIM:
            msg = f'Expected embedding dim {self.EMBED_DIM}, got {arr.shape[1]}'
            raise ValueError(msg)
        self.corpus_map = list(corpus)
        self.index = faiss.IndexFlatL2(self.EMBED_DIM)
        self.index.add(arr)

    def retrieve(self, query: str, k: int=5) -> list[str]:
        if self.index.ntotal == 0 or not self.corpus_map:
            return []
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        qv = np.asarray(q, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        k_use = int(min(k, self.index.ntotal, len(self.corpus_map)))
        if k_use <= 0:
            return []
        _, neigh_idx = self.index.search(qv, k_use)
        return [self.corpus_map[int(i)] for i in neigh_idx[0] if 0 <= int(i) < len(self.corpus_map)]
