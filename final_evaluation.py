import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "v3_index": os.path.join(BASE_DIR, "faiss_index_v3.bin"),
    "v4_index": os.path.join(BASE_DIR, "faiss_index_v4.bin"),
    "v3_chunks" : os.path.join(BASE_DIR , "filtered_chunks_v3.pkl"),
    "v4_chunks": os.path.join(BASE_DIR, "combined_covid_chunks.pkl"),
    "v3_bm25": os.path.join(BASE_DIR, "v3_bm25.pkl"),
    "v4_bm25": os.path.join(BASE_DIR, "v4_bm25.pkl"),
}


# ==================== LOAD RETRIEVAL INDEXES ====================
print("Loading FAISS indexes...")
v3_index = faiss.read_index(PATHS["v3_index"])
v4_index = faiss.read_index(PATHS["v4_index"])


def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)


print("Loading chunks...")
v3_chunks = load_chunks(PATHS["v3_chunks"])
v4_chunks = load_chunks(PATHS["v4_chunks"])
print("v3 index size:", v3_index.ntotal, "| v3 chunks size:", len(v3_chunks))
print("v4 index size:", v4_index.ntotal, "| v4 chunks size:", len(v4_chunks))
# ==================== LOAD EMBEDDING MODEL ====================
print("Loading embedding model (BGE-small)...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

# ==================== LOAD GGUF LLM (CPU) ====================
print("\nLoading BioMistral model...")

# Download automatically (only the first time)
model_path = hf_hub_download(
    repo_id="MaziyarPanahi/BioMistral-7B-GGUF",
    filename="BioMistral-7B.Q4_K_M.gguf"
)

# Load GGUF model
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=4,      # Adjust based on your CPU
    n_batch=512,
    n_gpu_layers=0,   # CPU only
    verbose=False
)

print("Model loaded successfully!\n")
# ==================== LOAD BM25 INDEXES ====================
print("Loading BM25 indexes...")
with open(PATHS["v3_bm25"], "rb") as f:
    v3_bm25 = pickle.load(f)
with open(PATHS["v4_bm25"], "rb") as f:
    v4_bm25 = pickle.load(f)


def tokenize(text):
    return text.lower().split()

# ==================== HELPER FUNCTIONS ====================
def get_confidence(scores, method="avg_top3"):
    if method == "top1":
        return float(scores[0][0])
    elif method == "avg_top3":
        return float(scores[0].mean())
    else:
        raise ValueError(f"Unknown method: {method}")


def get_bm25_confidence(bm25, tokenized_query, top_k=3):
    scores = bm25.get_scores(tokenized_query)
    top_scores = sorted(scores, reverse=True)[:top_k]
    return sum(top_scores) / len(top_scores)

def retrieve(query, k=3):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)

    v3_scores, v3_ids = v3_index.search(query_embedding, k)
    v4_scores, v4_ids = v4_index.search(query_embedding, k)

    v3_confidence = get_confidence(v3_scores)
    v4_confidence = get_confidence(v4_scores)

    return {
        "v3": {"scores": v3_scores, "ids": v3_ids, "confidence": v3_confidence},
        "v4": {"scores": v4_scores, "ids": v4_ids, "confidence": v4_confidence},
    }


def hybrid_retrieve(query, k=3, bm25_weight=0.3):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    tokenized_query = tokenize(query)

    # Semantic scores
    v3_scores, v3_ids = v3_index.search(query_embedding, k)
    v4_scores, v4_ids = v4_index.search(query_embedding, k)
    v3_semantic_conf = get_confidence(v3_scores)
    v4_semantic_conf = get_confidence(v4_scores)

    # BM25 scores
    v3_bm25_conf = get_bm25_confidence(v3_bm25, tokenized_query, top_k=k)
    v4_bm25_conf = get_bm25_confidence(v4_bm25, tokenized_query, top_k=k)

    # Normalize BM25 scores relative to each other
    bm25_max = max(v3_bm25_conf, v4_bm25_conf, 1e-6)
    v3_bm25_norm = v3_bm25_conf / bm25_max
    v4_bm25_norm = v4_bm25_conf / bm25_max

    # Combined confidence
    v3_confidence = (1 - bm25_weight) * v3_semantic_conf + bm25_weight * v3_bm25_norm
    v4_confidence = (1 - bm25_weight) * v4_semantic_conf + bm25_weight * v4_bm25_norm

    if v3_confidence >= v4_confidence:
        winner = "v3"
        chunks = [v3_chunks[i] for i in v3_ids[0]]
        confidence = v3_confidence
    else:
        winner = "v4"
        chunks = [v4_chunks[i] for i in v4_ids[0]]
        confidence = v4_confidence
    return {
        "winner": winner,
        "confidence": confidence,
        "chunks": chunks,
        "v3_confidence": v3_confidence,
        "v4_confidence": v4_confidence,
        "v3_semantic": v3_semantic_conf,
        "v4_semantic": v4_semantic_conf,
        "v3_bm25": v3_bm25_conf,
        "v4_bm25": v4_bm25_conf,
    }

# =====================================================
# Prompt
# =====================================================
def build_prompt(query, retrieved_chunks, max_chunk_chars=500):

    context = "\n\n".join(
        f"[{i+1}] {r['text'][:max_chunk_chars]}"
        for i, r in enumerate(retrieved_chunks)
    )

    return f"""
[INST]

You are an expert life science assistant.

Answer ONLY using the provided context.

Instructions:
- Give a direct answer.
- Explain mechanisms if asked "how".
- Explain reasons if asked "why".
- Keep answers between 2 and 5 sentences.
- If the answer is missing from the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{query}

Answer:

[/INST]
"""


def generate_answer(prompt, max_new_tokens=300):
    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,      # deterministic (same as do_sample=False)
        stop=["</s>"],      # stop at end of sequence
        echo=False,         # don't include the prompt in the output
    )
    generated_text = output["choices"][0]["text"]
    return generated_text.strip()

def ask_hybrid(query, top_k=5, max_new_tokens=300, verbose=True):
    retrieval = hybrid_retrieve(query, k=top_k)
    prompt = build_prompt(query, retrieval["chunks"])
    answer = generate_answer(prompt, max_new_tokens=max_new_tokens)

    if verbose:
        print(f"Q: {query}\n")
        print(f"Routed to: {retrieval['winner']} (v3={retrieval['v3_confidence']:.3f}, v4={retrieval['v4_confidence']:.3f})\n")
        print(f"A: {answer}\n")

    return answer, retrieval 
    # ==================== DEBUG: INSPECT RETRIEVAL ====================
debug_query = "What are the common symptoms of COVID-19?"
retrieval = hybrid_retrieve(debug_query, k=5)
print(f"\nWinner: {retrieval['winner']}")
for i, c in enumerate(retrieval["chunks"]):
    print(f"--- chunk {i+1} ---")
    print(c["text"][:300])
    print()
# ==================== END DEBUG ====================

 
# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("BioMistral Hybrid RAG (CPU)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer, retrieval = ask_hybrid(
                query,
                top_k=5,
                max_new_tokens=300,
                verbose=True
            )
        except Exception as e:
            print(f"\nError: {e}")  