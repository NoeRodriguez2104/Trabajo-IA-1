#!/usr/bin/env python3
"""
Pipeline que usa solo la API pública de Discogs para obtener releases, generar embeddings y guardar resultados.
"""
import os
import json
import time
import urllib.request
from pathlib import Path
from sentence_transformers import SentenceTransformer

from scrape_music_rag import save_to_chroma, EMBED_MODEL

BASE_URL = "https://api.discogs.com"
OUTPUT_FILE = "music_data_api_only.json"
PER_PAGE = 50


def api_get(path, params=None, token=None, timeout=15):
    q = ''
    if params:
        q = '?' + '&'.join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = BASE_URL + path + q
    headers = {"User-Agent": "discogs-api-pipeline/1.0"}
    if token:
        headers["Authorization"] = f"Discogs token={token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code} for {url}: {e.reason}")
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def fetch_release(release_id, token=None):
    data = api_get(f"/releases/{release_id}", token=token)
    if not data:
        return None
    m = {}
    m['_title_from_api'] = data.get('title')
    artists = []
    for a in data.get('artists', []):
        name = a.get('name')
        if name:
            artists.append(name)
    if artists:
        m['_artist_from_api'] = ' & '.join(artists)
    labs = [l.get('name') for l in data.get('labels', []) if l.get('name')]
    if labs:
        m['label'] = ', '.join(labs)
    fmts = []
    for f in data.get('formats', []):
        name = f.get('name') or ''
        desc = ' '.join(f.get('descriptions') or [])
        part = (name + ' ' + desc).strip()
        if part:
            fmts.append(part)
    if fmts:
        m['format'] = '; '.join(fmts)
    if data.get('country'):
        m['country'] = data.get('country')
    if data.get('released'):
        m['released'] = data.get('released')
    if data.get('genres'):
        m['genre'] = ', '.join(data.get('genres'))
    if data.get('styles'):
        m['style'] = ', '.join(data.get('styles'))
    if data.get('images'):
        first = data.get('images')[0]
        img = first.get('uri') or first.get('resource_url')
        if img:
            m['image'] = img
    # url
    m['url'] = f"https://www.discogs.com/release/{release_id}"
    return m


def search_releases(query='', page=1, per_page=PER_PAGE, token=None):
    params = {'type': 'release', 'page': page, 'per_page': per_page}
    if query:
        params['q'] = query
    return api_get('/database/search', params=params, token=token)


def build_docs_from_search(pages=1, token=None):
    docs = []
    for p in range(1, pages + 1):
        print(f"Searching page {p}...")
        res = search_releases(page=p, per_page=PER_PAGE, token=token)
        if not res:
            print("No results from search (possible auth or network issue).")
            break
        results = res.get('results', [])
        for r in results:
            rid = r.get('id')
            if not rid:
                continue
            meta = fetch_release(rid, token=token)
            if not meta:
                continue
            title = meta.get('_title_from_api', '')
            artist = meta.get('_artist_from_api', '')
            text_blob = ' | '.join(filter(None, [title, artist, meta.get('genre',''), meta.get('style',''), meta.get('country',''), meta.get('format',''), meta.get('label','')]))
            doc = {
                'doc_id': f"api_r{rid}_{int(time.time())}",
                'source': 'https://www.discogs.com',
                'title': title,
                'artist': artist,
                'url': meta.get('url'),
                'metadata': meta,
                'text': text_blob
            }
            docs.append(doc)
    return docs


def embed_and_save(docs, model_name=EMBED_MODEL, out_file=OUTPUT_FILE):
    if not docs:
        print('No docs to embed.')
        return []
    print('Generating embeddings with', model_name)
    model = SentenceTransformer(model_name)
    texts = [d.get('text','') for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, d in enumerate(docs):
        d['embedding'] = embeddings[i].tolist()
    # save
    p = Path(out_file)
    with p.open('w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print('Saved', len(docs), 'docs to', out_file)
    return docs


if __name__ == '__main__':
    pages = int(os.environ.get('API_PAGES', '1'))
    token = os.environ.get('DISCOGS_TOKEN')
    docs = build_docs_from_search(pages=pages, token=token)
    embedded = embed_and_save(docs, out_file=OUTPUT_FILE)
    # optional chroma
    if os.environ.get('USE_CHROMA') and 'save_to_chroma' in globals():
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR', './chroma_db')
        save_to_chroma(embedded, persist_dir=persist_dir)
    print('Done')
