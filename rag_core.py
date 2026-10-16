import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "v3_index": os.path.join(BASE_DIR, "faiss_index_v3.bin"),
    "v4_index": os.path.join(BASE_DIR, "faiss_index_v4.bin"),
    "v3_chunks": os.path.join(BASE_DIR, "filtered_chunks_v3.pkl"),
    "v4_chunks": os.path.join(BASE_DIR, "combined_covid_chunks.pkl"),
    "v3_bm25": os.path.join(BASE_DIR, "v3_bm25.pkl"),
    "v4_bm25": os.path.join(BASE_DIR, "v4_bm25.pkl"),
}


# ==================== RESOURCE LOADING ====================
def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_resources(progress_cb=None):
    """
    Loads everything needed for retrieval + generation exactly once, and
    returns it as a dict. Pass progress_cb(str) to receive status strings
    (e.g. to feed st.spinner / st.status text) as loading proceeds.
    """
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("Loading FAISS indexes...")
    v3_index = faiss.read_index(PATHS["v3_index"])
    v4_index = faiss.read_index(PATHS["v4_index"])

    report("Loading chunks...")
    v3_chunks = load_chunks(PATHS["v3_chunks"])
    v4_chunks = load_chunks(PATHS["v4_chunks"])

    report("Loading embedding model (BGE-small)...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    report("Loading BM25 indexes...")
    with open(PATHS["v3_bm25"], "rb") as f:
        v3_bm25 = pickle.load(f)
    with open(PATHS["v4_bm25"], "rb") as f:
        v4_bm25 = pickle.load(f)

    report("Downloading / loading BioMistral GGUF model (first run can take a while)...")
    model_path = hf_hub_download(
        repo_id="MaziyarPanahi/BioMistral-7B-GGUF",
        filename="BioMistral-7B.Q4_K_M.gguf",
    )
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,      # adjust based on your CPU core count
        n_batch=512,
        n_gpu_layers=0,   # CPU only
        verbose=False,
    )

    report("All resources loaded.")

    return {
        "v3_index": v3_index,
        "v4_index": v4_index,
        "v3_chunks": v3_chunks,
        "v4_chunks": v4_chunks,
        "v3_bm25": v3_bm25,
        "v4_bm25": v4_bm25,
        "embed_model": embed_model,
        "llm": llm,
    }


def tokenize(text):
    return text.lower().split()


# ==================== RETRIEVAL HELPERS ====================
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
    all_max = max(scores.max(), 1e-6)  # normalize against this corpus's own max
    return (sum(top_scores) / len(top_scores)) / all_max


def hybrid_retrieve(resources, query, k=3, bm25_weight=0.3):
    embed_model = resources["embed_model"]
    v3_index, v4_index = resources["v3_index"], resources["v4_index"]
    v3_bm25, v4_bm25 = resources["v3_bm25"], resources["v4_bm25"]
    v3_chunks, v4_chunks = resources["v3_chunks"], resources["v4_chunks"]

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


# ==================== PROMPT + GENERATION ====================
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


def generate_answer(llm, prompt, max_new_tokens=300):
    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=["</s>"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def ask_hybrid(resources, query, top_k=5, max_new_tokens=300):
    retrieval = hybrid_retrieve(resources, query, k=top_k)
    prompt = build_prompt(query, retrieval["chunks"])
    answer = generate_answer(resources["llm"], prompt, max_new_tokens=max_new_tokens)
    return answer, retrieval


# ==================== DEBUG / MANUAL TEST ====================
if __name__ == "__main__":
    resources = load_resources(progress_cb=print)

    # 1. Check index/chunks alignment
    print("v3 index:", resources["v3_index"].ntotal, "| v3 chunks:", len(resources["v3_chunks"]))
    print("v4 index:", resources["v4_index"].ntotal, "| v4 chunks:", len(resources["v4_chunks"]))

    query = "What are the common symptoms of COVID-19?"

    # 2. Run hybrid retrieval and inspect routing/confidence
    retrieval = hybrid_retrieve(resources, query, k=5)

    print("\nv3_confidence:", retrieval["v3_confidence"],
          " (semantic:", retrieval["v3_semantic"], ", bm25:", retrieval["v3_bm25"], ")")
    print("v4_confidence:", retrieval["v4_confidence"],
          " (semantic:", retrieval["v4_semantic"], ", bm25:", retrieval["v4_bm25"], ")")
    print("Winner:", retrieval["winner"])

    print("\n--- Retrieved chunks (via hybrid_retrieve) ---")
    for i, c in enumerate(retrieval["chunks"]):
        print(f"[{i+1}] {c['text'][:200]}")

    # 3. Bypass routing entirely, check v4 raw retrieval
    query_embedding = resources["embed_model"].encode([query], normalize_embeddings=True)
    scores, ids = resources["v4_index"].search(query_embedding, 5)
    print("\n--- v4-only retrieval (bypassing router) ---")
    for i in ids[0]:
        print(resources["v4_chunks"][i]["text"][:200])
        print("---")

    # 4. Print the FULL prompt actually sent to the model
    prompt = build_prompt(query, retrieval["chunks"])
    print("\n--- FULL PROMPT SENT TO MODEL ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    # 5. Generate the final answer and show it alongside the prompt
    answer = generate_answer(resources["llm"], prompt, max_new_tokens=300)
    print("Final Answer:", answer)
    print("Routed to:", retrieval["winner"])

    #test
    test_prompt = """[INST]
Answer the question using only the context below.

Context:
Paris is the capital of France.

Question:
What is the capital of France?

Answer:
[/INST]"""

    test_answer = generate_answer(resources["llm"], test_prompt, max_new_tokens=50)
    print("\nTEST ANSWER:", test_answer)

