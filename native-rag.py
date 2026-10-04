import json
import urllib.request
import chromadb

# =====================================================================
# 1. MODERN OLLAMA EMBEDDING PROTOCOL (Using Specialized Embedding Model)
# =====================================================================
def get_local_embedding(text_to_embed: str) -> list:
    """Hits the current Ollama API using a specialized, ultra-fast embedding model."""
    url = "http://localhost:11434/api/embed"
    
    payload = {
        "model": "mxbai-embed-large",  # 🌟 FIX: Point strictly to the dedicated embedding model
        "input": text_to_embed
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        # 🌟 Extraction Fix: mxbai-embed-large via /api/embed yields a list of vectors
        # We grab the first element if it wraps it in an outer batch list matrix
        embeddings_list = result['embeddings']
        return embeddings_list[0] if isinstance(embeddings_list[0], list) else embeddings_list

# =====================================================================
# 2. DATA INGESTION PIPELINE (Parsing and Storing)
# =====================================================================
def ingest_documents():
    print("💾 Initializing Local Vector Database Storage Layer...")
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection(name="engineering_docs")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="engineering_docs")
    
    print("📖 Reading and chunking 'knowledge_base.txt'...")
    with open("knowledge_base.txt", "r") as f:
        content = f.read()
    
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    
    print(f"🧬 Generating Vector Embeddings for {len(chunks)} text chunks...")
    
    for index, chunk in enumerate(chunks):
        vector_coordinate = get_local_embedding(chunk)
        
        collection.add(
            embeddings=[vector_coordinate],
            documents=[chunk],
            metadatas=[{"source": "system_runbook", "doc_idx": index}],
            ids=[f"doc_{index}"]
        )
        print(f"   ↳ Indexed chunk {index+1}/{len(chunks)} into Vector Space.")
        
    return collection

# =====================================================================
# 3. SEMANTIC QUERY PLANE (Retrieval Engine)
# =====================================================================
def run_semantic_search(collection_engine, user_query: str):
    print(f"\n🔎 [USER QUERY]: '{user_query}'")
    
    query_vector = get_local_embedding(user_query)
    
    print("📐 Computing mathematical similarity distances inside ChromaDB index...")
    results = collection_engine.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    matched_document = results['documents'][0][0]
    distance_score = results['distances'][0][0]
    
    print(f"🎯 [MATCH FOUND] Geometric Coordinate Distance Score: {distance_score:.4f}")
    print(f"📄 [RETRIEVED TEXT LAYER]:\n{matched_document}\n")

# =====================================================================
# 4. EXECUTION RUNTIME
# =====================================================================
if __name__ == "__main__":
    # Execute Ingestion
    vector_db_collection = ingest_documents()
    
    # Execute semantic matching lookups
    run_semantic_search(vector_db_collection, "Why is token validation timeout error happening?")
