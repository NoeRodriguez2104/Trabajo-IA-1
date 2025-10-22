#!/usr/bin/env python3
# scrape_music_rag.py — versión estable sin ventanas, con detalles de artista y metadata

import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import re
import os
import urllib.request
import urllib.error
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# ------------------ CONFIG ------------------
BASE_URL = "https://www.discogs.com"
OUTPUT_FILE = "music_data.json"
MAX_PAGES = 1
EMBED_MODEL = "all-MiniLM-L6-v2"
HEADLESS = True                # ✅ Sin ventanas
SLOW_MO_MS = 200
# --------------------------------------------

def scrape_music_site(max_pages=MAX_PAGES):
    print("🎵 Iniciando scraping musical en Discogs...")
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        search_url = f"{BASE_URL}/search/?q=&type=release"
        print(f"🔍 Navegando a: {search_url}")
        page.goto(search_url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(3000)

        for page_idx in range(max_pages):
            print(f"\n📄 Procesando página {page_idx + 1}...")
            items = page.query_selector_all(".card, .search_result, article, .card_release, li")
            if not items:
                print("⚠️ No se encontraron resultados visibles.")
                break

            for idx, it in enumerate(items):
                try:
                    title_el = it.query_selector("h4, .card__title, .search_result_title, a.card_release_title")
                    title = title_el.inner_text().strip() if title_el else ""

                    artist_el = it.query_selector(".card__artist, .search_result_artist, .card_release_artist, .artist")
                    artist = artist_el.inner_text().strip() if artist_el else ""

                    anchor = it.query_selector("a")
                    href = anchor.get_attribute("href") if anchor else ""
                    if not href:
                        continue

                    # ✅ solo seguimos si es un release/master válido (con ID numérico)
                    # Excluir rutas como /release/add y otras páginas no-numéricas
                    if not re.search(r"/release/\d+|/master/\d+", href):
                        continue

                    # evitar entradas de UI o banners genéricos
                    if title and title.strip().lower() in ("welcome", "bienvenido"):
                        continue

                    url = href if href.startswith("http") else (BASE_URL + href)

                    # Intentar parsear la página de release para metadata más completa
                    meta = {}
                    try:
                        def parse_release_page(page_obj):
                            m = {}
                            # intentos por varios selectores comunes en Discogs
                            # perfil clave: pares label/value
                            rows = page_obj.query_selector_all(".releaseprofile, div.profile, .profile")
                            if rows:
                                # si hay un contenedor grande, buscar hijos label/value
                                pairs = page_obj.query_selector_all(".release .release-meta, .profile div, .release-profile div")
                                for i in range(0, len(pairs) - 1):
                                    try:
                                        label = pairs[i].inner_text().strip().replace(":", "")
                                        val = pairs[i + 1].inner_text().strip()
                                    except Exception:
                                        continue
                                    if not label:
                                        continue
                                    key = label.lower()
                                    if key in ["label", "series", "format", "country", "released", "genre", "style"]:
                                        m[key] = val

                            # fallback: buscar listas dt/dd o th/td
                            dts = page_obj.query_selector_all("dt")
                            dds = page_obj.query_selector_all("dd")
                            if dts and dds and len(dts) == len(dds):
                                for dt, dd in zip(dts, dds):
                                    k = dt.inner_text().strip().replace(":", "").lower()
                                    v = dd.inner_text().strip()
                                    if k in ["label", "series", "format", "country", "released", "genre", "style"]:
                                        m[k] = v

                            # imagen: diferentes selectores según plantilla
                            img = page_obj.query_selector("img.image_gallery_image") or page_obj.query_selector("img#large_image, .thumbnail img, .image_gallery img")
                            if img:
                                try:
                                    m["image"] = img.get_attribute("src") or img.get_attribute("data-src")
                                except Exception:
                                    pass

                            # intentar extraer artista/título desde la página de detalle si faltan
                            try:
                                if not title:
                                    t = page_obj.query_selector("h1, .title, .release-title")
                                    if t:
                                        m.setdefault("_title_from_detail", t.inner_text().strip())
                                if not artist:
                                    a = page_obj.query_selector("a.artist, .artist_name, .release-artist")
                                    if a:
                                        m.setdefault("_artist_from_detail", a.inner_text().strip())
                            except Exception:
                                pass

                            return m

                        page2 = context.new_page()
                        page2.goto(url, wait_until="domcontentloaded", timeout=15000)
                        # esperar un poco para que cargue contenido dinámico
                        page2.wait_for_timeout(500)
                        meta = parse_release_page(page2)
                        page2.close()
                    except Exception as e:
                        print(f"      ⚠️ Detalle omitido para {title or 'sin título'}: {e}")
                        # no continuar: queremos incluir el item aunque falte metadata

                    # --- Fall back: intentar la API pública de Discogs si no hay metadata útil ---
                    try:
                        def fetch_discogs_release(release_id, token=None):
                            api_url = f"https://api.discogs.com/releases/{release_id}"
                            headers = {"User-Agent": "discogs-scraper/1.0"}
                            if token:
                                headers["Authorization"] = f"Discogs token={token}"
                            req = urllib.request.Request(api_url, headers=headers)
                            try:
                                with urllib.request.urlopen(req, timeout=15) as resp:
                                    data = json.load(resp)
                            except Exception:
                                return None

                            mapi = {}
                            # título y artistas
                            if data.get("title"):
                                mapi.setdefault("_title_from_api", data.get("title"))
                            artists = []
                            for a in data.get("artists", []):
                                name = a.get("name")
                                if name:
                                    artists.append(name)
                            if artists:
                                mapi.setdefault("_artist_from_api", " & ".join(artists))

                            # labels
                            labs = [l.get("name") for l in data.get("labels", []) if l.get("name")]
                            if labs:
                                mapi.setdefault("label", ", ".join(labs))

                            # formats
                            fmts = []
                            for f in data.get("formats", []):
                                name = f.get("name") or ""
                                desc = " ".join(f.get("descriptions") or [])
                                part = (name + " " + desc).strip()
                                if part:
                                    fmts.append(part)
                            if fmts:
                                mapi.setdefault("format", "; ".join(fmts))

                            if data.get("country"):
                                mapi.setdefault("country", data.get("country"))
                            if data.get("released"):
                                mapi.setdefault("released", data.get("released"))
                            if data.get("genres"):
                                mapi.setdefault("genre", ", ".join(data.get("genres")))
                            if data.get("styles"):
                                mapi.setdefault("style", ", ".join(data.get("styles")))
                            if data.get("images"):
                                first = data.get("images")[0]
                                img = first.get("uri") or first.get("resource_url")
                                if img:
                                    mapi.setdefault("image", img)

                            return mapi

                        needs_api = (not meta) or (not artist) or (not title)
                        if needs_api:
                            m = re.search(r"/(?:release|master)/(\d+)", url)
                            if m:
                                rid = m.group(1)
                                token = os.environ.get("DISCOGS_TOKEN")
                                api_meta = fetch_discogs_release(rid, token=token)
                                if api_meta:
                                    print(f"      ℹ️ Metadata obtenida vía API para release {rid}")
                                    # no sobreescribir keys existentes; usar valores API para completar faltantes
                                    for k, v in api_meta.items():
                                        if k == "_title_from_api" and not title:
                                            title = v
                                        elif k == "_artist_from_api" and not artist:
                                            artist = v
                                        else:
                                            meta.setdefault(k, v)
                    except Exception as e:
                        print(f"      ⚠️ Fallback API falló para {url}: {e}")

                    # si artista o título están vacíos, intentar obtenerlos desde metadata recogida
                    if not title and meta.get("_title_from_detail"):
                        title = meta.pop("_title_from_detail")
                    if not artist and meta.get("_artist_from_detail"):
                        artist = meta.pop("_artist_from_detail")

                    text_blob = " | ".join(filter(None, [
                        title,
                        artist,
                        meta.get("genre", ""),
                        meta.get("style", ""),
                        meta.get("country", ""),
                        meta.get("format", ""),
                        meta.get("label", "")
                    ]))

                    doc_id = f"pg{page_idx}_i{idx}_{int(time.time())}"
                    results.append({
                        "doc_id": doc_id,
                        "source": BASE_URL,
                        "title": title,
                        "artist": artist,
                        "url": url,
                        "metadata": meta,
                        "text": text_blob
                    })
                except Exception as e:
                    print(f"   ⚠️ Error parseando item {idx}: {e}")
                    continue

            # --- PAGINACIÓN ---
            try:
                next_btn = page.query_selector('a[rel="next"], a.pagination_next, .pagination-next, .next')
                if next_btn:
                    next_href = next_btn.get_attribute("href")
                    if next_href:
                        next_url = next_href if next_href.startswith("http") else (BASE_URL + next_href)
                        print(f"   → Siguiente página: {next_url}")
                        page.goto(next_url, wait_until="networkidle", timeout=90000)
                        page.wait_for_timeout(2000)
                    else:
                        print("   🚫 No hay más páginas.")
                        break
                else:
                    print("   🚫 No hay más páginas.")
                    break
            except Exception as e:
                print(f"   ⚠️ Error en paginación: {e}")
                break

        browser.close()

    print(f"\n✅ Scraping finalizado. Total: {len(results)} elementos extraídos.")
    return results


def embed_music_data(docs, model_name=EMBED_MODEL):
    print("🧠 Generando embeddings con", model_name)
    model = SentenceTransformer(model_name)
    texts = [d.get("text", "") for d in docs]
    if not texts:
        print("⚠️ No hay textos a indexar.")
        return docs
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, d in enumerate(docs):
        d["embedding"] = embeddings[i].tolist()
    print("✅ Embeddings completados.")
    return docs


def save_json(data, filename=OUTPUT_FILE):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Datos guardados en {filename}")


def save_to_chroma(docs, persist_dir="./chroma_db"):
    """Guardar embeddings y metadatos en ChromaDB (opcional).

    Requiere que `chromadb` esté instalado. Cada documento debe tener clave `embedding`.
    """
    if not CHROMA_AVAILABLE:
        print("⚠️ chromadb no está instalado. Saltando guardado en Chroma.")
        return False

    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    collection = None
    name = "music_releases"
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)

    ids = []
    embeddings = []
    metadatas = []
    documents = []
    for d in docs:
        eid = d.get("doc_id")
        emb = d.get("embedding")
        if not eid or not emb:
            continue
        ids.append(eid)
        embeddings.append(emb)
        # guardar metadatos relevantes
        metadatas.append({"title": d.get("title"), "artist": d.get("artist"), "url": d.get("url"), "metadata": d.get("metadata")})
        documents.append(d.get("text", ""))

    if not ids:
        print("⚠️ No hay embeddings válidos para guardar en Chroma.")
        return False

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    client.persist()
    print(f"✅ {len(ids)} vectores guardados en ChromaDB en {persist_dir}")
    return True


def main():
    docs = scrape_music_site(max_pages=MAX_PAGES)
    if not docs:
        print("⚠️ No se extrajo ningún documento. Revisa los selectores.")
        return
    embedded = embed_music_data(docs)
    save_json(embedded, OUTPUT_FILE)
    # Si el usuario quiere guardar en Chroma y chromadb está instalado, hacerlo automáticamente
    use_chroma = os.environ.get("USE_CHROMA")
    if use_chroma and CHROMA_AVAILABLE:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
        save_to_chroma(embedded, persist_dir=persist_dir)

    print("🎶 Pipeline completado.")


if __name__ == "__main__":
    main()
