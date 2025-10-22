# Scraping Discogs — pipeline local

Este proyecto extrae metadata de releases en Discogs, genera embeddings con `sentence-transformers` y opcionalmente los indexa en una base de datos vectorial Chroma.

## Requisitos
- Python 3.10+ (probado con 3.12)
- Recomendado crear un entorno virtual:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

## Instalación (dependencias mínimas)
```bash
pip install -r requeriments.txt
# si quieres usar chromadb (opcional):
pip install chromadb
# instalar Playwright (si no está) y navegadores:
python -m playwright install
```

## Variables de entorno útiles
- `DISCOGS_TOKEN` (opcional): token de Discogs para aumentar cuota y autenticación en la API.
- `USE_CHROMA` (opcional): si está definida, el script intentará guardar embeddings en ChromaDB.
- `CHROMA_PERSIST_DIR` (opcional): directorio donde persiste Chroma (por defecto `./chroma_db`).

## Ejecutar el scraper
```bash
# ejemplo mínimo
python3 scrape_music_rag.py

# guardar también en Chroma (instala chromadb primero)
export USE_CHROMA=1
export CHROMA_PERSIST_DIR=./mi_chroma_db
python3 scrape_music_rag.py
```

## Salida
- `music_data.json`: contiene los documentos extraídos, cada uno con `title`, `artist`, `url`, `metadata`, `text` y `embedding`.
- Si `USE_CHROMA` está activo, Chroma contendrá la colección `music_releases` con vectores y metadatos.

## Notas
- Algunas páginas de Discogs están protegidas por Cloudflare. En esos casos el script usa la API pública de Discogs (`/releases/{id}`) como fallback. Para mejores resultados proporciona `DISCOGS_TOKEN`.
- El script es intencionalmente simple y pensado para uso local. Si planeas hacer scraping a gran escala respeta las políticas de Discogs y su API.

## Problemas y ajustes
- Si notas que faltan artistas/títulos o metadata, comparte ejemplos de URLs y ajusto selectores o la lógica de fallback.

Licencia: código de ejemplo, usa bajo tu propio riesgo.
