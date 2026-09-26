"""
Step 02: compute and cache SBERT embeddings (Section 6.4, Stage 1, Steps 1.1-1.4).
Embeddings are cached to disk keyed by text hash so re-running later steps
never recomputes them.
"""
import hashlib
import os
import numpy as np


def _hash_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts_cached(texts, model, cache_dir: str, tag: str) -> np.ndarray:
    """
    texts: list[str] (order matters and defines row order of the output)
    model: a loaded SentenceTransformer
    tag: short label used in the cache filename, e.g. "requirements" or "changes"
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_texts(texts)
    cache_path = os.path.join(cache_dir, f"emb_{tag}_{key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    emb = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity = dot product
        convert_to_numpy=True,
    )
    np.save(cache_path, emb)
    return emb


def cosine_matrix(change_embs: np.ndarray, req_embs: np.ndarray) -> np.ndarray:
    """
    change_embs: (n_changes, d), req_embs: (n_reqs, d), both L2-normalised.
    Returns (n_changes, n_reqs) cosine similarity matrix via dot product.
    """
    return change_embs @ req_embs.T
