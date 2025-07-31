"""
Retrieve the most relevant memory records for a given user prompt.

This script loads the vector index and document metadata produced by
`build_index.py`. It attempts to use FAISS for fast nearest neighbour
search when available; otherwise, it falls back to a simple distance
computation over raw vectors. Sentence embeddings are generated using
SentenceTransformer when installed, with a hash-based fallback otherwise.
"""

import os, json, numpy as np

# Resolve base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INDICES_DIR = os.path.join(BASE_DIR, 'indices')

# Embedding function
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(text: str) -> np.ndarray:
        return _model.encode(text)
except Exception:
    vector_dim = 128
    def embed_text(text: str) -> np.ndarray:
        vec = np.zeros(vector_dim, dtype=np.float32)
        for token in text.split():
            h = abs(hash(token)) % vector_dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

docs_path    = os.path.join(INDICES_DIR, 'docs.npy')
vectors_path = os.path.join(INDICES_DIR, 'vectors.npy')

docs   = np.load(docs_path,    allow_pickle=True) if os.path.exists(docs_path) else np.array([],dtype=object)
vectors= np.load(vectors_path, allow_pickle=True) if os.path.exists(vectors_path) else None
results_count = len(docs)

# Attempt FAISS index
faiss_index = None
try:
    import faiss  # type: ignore
    idx_path = os.path.join(INDICES_DIR, 'memory.index')
    if os.path.exists(idx_path):
        faiss_index = faiss.read_index(idx_path)
except Exception:
    faiss_index = None

def _fallback_search(q_emb: np.ndarray, k: int) -> list[int]:
    vecs_path = os.path.join(INDICES_DIR, 'vectors.npy')
    if not os.path.exists(vecs_path):
        return []
    vectors = np.load(vecs_path)
    dists = np.linalg.norm(vectors - q_emb.reshape(1, -1), axis=1)
    return np.argsort(dists)[:k].tolist()

def get_relevant(prompt: str, k: int = 5, filter_tags: list[str] | None = None) -> list[dict]:
    if docs.size == 0:
        return []

    # 1) Filter by tag if requested
    idxs = list(range(len(docs)))
    if filter_tags:
        idxs = [
            i for i in idxs
            if any(tag in docs[i].get("tags", []) for tag in filter_tags)
        ]
        if not idxs:
            return []

    filtered_docs    = docs[idxs]
    filtered_vectors = vectors[idxs] if vectors is not None else None

    # 2) Embed query
    q_emb = embed_text(prompt).astype(np.float32).reshape(1, -1)

    # 3) Search
    if faiss_index and filtered_vectors is not None:
        # build a temporary FAISS index over only the filtered_vectors
        tmp = faiss.IndexFlatL2(filtered_vectors.shape[1])
        tmp.add(filtered_vectors)
        _, found = tmp.search(q_emb, k)
        sel = found[0].tolist()
    else:
        # fallback: Euclidean distance
        dists = np.linalg.norm(filtered_vectors - q_emb, axis=1)
        sel   = np.argsort(dists)[:k].tolist()

    # 4) Return the actual records
    return [filtered_docs[i] for i in sel]

if __name__ == '__main__':
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter query: ")
    print(json.dumps(get_relevant(query), indent=2))
