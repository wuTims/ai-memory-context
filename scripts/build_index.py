"""
This script builds a FAISS (or fallback) index over JSON memory records. It reads
all JSON files in the adjacent memory directory, normalises their filenames to
include a date prefix, generates sentence embeddings for each record, and
persists both the vector index and record metadata to disk.

The script attempts to import `sentence_transformers` and `faiss`. If either
dependency is unavailable, it falls back to simple alternatives: a basic
embedding function based on word hashing and an in-memory nearest neighbour
search. These fallbacks ensure the index can still be created in constrained
environments without internet access or pre-installed libraries.
"""

import os
import json
import numpy as np

# Resolve the base project directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
INDICES_DIR = os.path.join(BASE_DIR, 'indices')

# Try to import sentence-transformers for high-quality embeddings
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_text(text: str) -> np.ndarray:
        return _model.encode(text)
    vector_dim: int = _model.get_sentence_embedding_dimension()
except Exception:
    # Fallback embedding: hash tokens into a fixed-size vector
    vector_dim = 128
    def embed_text(text: str) -> np.ndarray:
        vec = np.zeros(vector_dim, dtype=np.float32)
        for token in text.split():
            h = abs(hash(token)) % vector_dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

# Attempt to import FAISS for fast vector indexing
try:
    import faiss  # type: ignore
    use_faiss = True
except Exception:
    faiss = None  # type: ignore
    use_faiss = False

records: list[dict] = []
vectors: list[np.ndarray] = []

# Read & normalise JSON files
for fn in os.listdir(MEMORY_DIR):
    if not fn.endswith('.json'):
        continue
    path = os.path.join(MEMORY_DIR, fn)
    try:
        rec = json.load(open(path, 'r'))
    except Exception:
        continue
    # skip mega-summary entries
    if rec.get('type') == 'mega-summary':
        continue

    # normalise filename to YYYY-MM-DD_id.json
    date_str = rec.get('date', '')
    rec_id = rec.get('id', os.path.splitext(fn)[0])
    date_prefix = date_str[:10] if date_str else 'unknown'
    target_name = f"{date_prefix}_{rec_id}.json"
    if fn != target_name:
        new_path = os.path.join(MEMORY_DIR, target_name)
        if not os.path.exists(new_path):
            os.rename(path, new_path)
            path = new_path

    # combine fields for embedding
    text = f"{rec.get('name','')}. {rec.get('description','')} {' '.join(rec.get('tags',[]))}"
    emb = embed_text(text).astype(np.float32)
    records.append(rec)
    vectors.append(emb)

# ensure indices folder exists
os.makedirs(INDICES_DIR, exist_ok=True)

if vectors:
    vect_array = np.stack(vectors)
    # save raw vectors for fallback search
    np.save(os.path.join(INDICES_DIR, 'vectors.npy'), vect_array)
    if use_faiss:
        index = faiss.IndexFlatL2(vector_dim)
        index.add(vect_array)
        faiss.write_index(index, os.path.join(INDICES_DIR, 'memory.index'))
    # persist record metadata
    np.save(os.path.join(INDICES_DIR, 'docs.npy'), np.array(records, dtype=object))
else:
    # no records: still write empty docs
    np.save(os.path.join(INDICES_DIR, 'docs.npy'), np.array([], dtype=object))
