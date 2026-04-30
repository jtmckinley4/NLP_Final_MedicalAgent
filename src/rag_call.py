from tools.rag_stub import search_medical_kb, build_vector_db

# Optional: ensure DB exists (no-op if already built)
build_vector_db()

results = search_medical_kb("What are symptoms of diabetes?", top_k=5)
for r in results:
    print(r["score"], r["question"])