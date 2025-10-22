#!/usr/bin/env python3
import json
from scrape_music_rag import save_to_chroma

with open('music_data.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)

save_to_chroma(docs, persist_dir='./chroma_db')
print('Indexing done')
