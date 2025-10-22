import json

# Rutas de entrada/salida
music_data_path = "music_data.json"
faiss_map_path = "music_faiss_map.json"
output_path = "music_faiss_map_with_embeddings.json"

# --- Cargar ambos archivos ---
with open(music_data_path, "r", encoding="utf-8") as f:
    music_data = json.load(f)

with open(faiss_map_path, "r", encoding="utf-8") as f:
    faiss_map = json.load(f)

# --- Crear índice rápido de embeddings por doc_id ---
embeddings_by_id = {item["doc_id"]: item["embedding"] for item in music_data if "embedding" in item}

# --- Añadir embeddings al faiss_map ---
merged_count = 0
for doc_id, meta in faiss_map["metadatas"].items():
    if doc_id in embeddings_by_id:
        meta["embedding"] = embeddings_by_id[doc_id]
        merged_count += 1

print(f"✅ Embeddings añadidos a {merged_count} canciones de {len(faiss_map['metadatas'])}")

# --- Guardar nuevo archivo ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(faiss_map, f, ensure_ascii=False, indent=2)

print(f"💾 Archivo guardado como: {output_path}")
