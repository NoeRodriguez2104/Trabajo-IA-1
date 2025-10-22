#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import faiss

INPUT = 'music_data.json'
INDEX_FILE = 'music_faiss.index'
MAPPING_FILE = 'music_faiss_map.json'

def load_docs(path):
    p = Path(path)
    if not p.exists():
        print('No se encontró', path)
        return []
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_index(docs):
    ids = []
    embeddings = []
    metadatas = {}
    for d in docs:
        emb = d.get('embedding')
        did = d.get('doc_id')
        if emb and did:
            ids.append(did)
            embeddings.append(emb)
            metadatas[did] = {
                'title': d.get('title'),
                'artist': d.get('artist'),
                'url': d.get('url'),
                'metadata': d.get('metadata')
            }
    if not embeddings:
        print('No embeddings encontrados en', INPUT)
        return None, None, None

    mat = np.array(embeddings).astype('float32')
    # normalizar a unit vectors para usar inner product como similitud coseno
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    mat = mat / norms
    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    return index, ids, metadatas


def save_index(index, ids, metadatas):
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'metadatas': metadatas}, f, ensure_ascii=False, indent=2)
    print('Índice FAISS guardado en', INDEX_FILE)
    print('Mapping guardado en', MAPPING_FILE)


if __name__ == '__main__':
    docs = load_docs(INPUT)
    index, ids, metadatas = build_index(docs)
    if index is not None:
        save_index(index, ids, metadatas)
