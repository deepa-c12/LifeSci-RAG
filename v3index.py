import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama
CHUNKS_PATH = "filtered_chunks_v3_new.pkl"
FAISS_INDEX_PATH = "faiss_index_v3.bin"
GGUF_PATH = "./models/BioMistral-7B.Q4_K_M.gguf"
N_THREADS = os.cpu_count()  # use all available CPU cores
N_CTX = 2048                # matches the max_length used in the Colab tokenizer call
 
 
# ---------------------------------------------------------------------
# Load embedding model, chunks, and FAISS index
# (index is loaded directly from disk, not rebuilt from raw embeddings)
# ---------------------------------------------------------------------
def load_retrieval_stack():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
 
    with open(CHUNKS_PATH, "rb") as f:
        combined_covid_chunks = pickle.load(f)
 
    index_new = faiss.read_index(FAISS_INDEX_PATH)
 
    print(f"Loaded {len(combined_covid_chunks)} chunks, FAISS index has {index_new.ntotal} vectors")
    return embed_model, combined_covid_chunks, index_new
 
 
# ---------------------------------------------------------------------
# BioMistral load — CPU via llama-cpp-python instead of bitsandbytes
# ---------------------------------------------------------------------
def load_biomistral_cpu(gguf_path=GGUF_PATH, n_ctx=N_CTX, n_threads=N_THREADS):
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    print("BioMistral-7B loaded (CPU, GGUF)")
    return llm
 
 
# ---------------------------------------------------------------------
# Retrieval (unchanged logic from cells 34-35)
# ---------------------------------------------------------------------
def semantic_search(query, index, embed_model, top_k=5):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]
 
 
def retrieve_context(query, index, chunks, embed_model, top_k=5):
    indices, _ = semantic_search(query, index, embed_model, top_k)
    return [chunks[i] for i in indices]
 
 
# ---------------------------------------------------------------------
# Prompt building — now with [INST] wrapping
# ---------------------------------------------------------------------
def build_prompt(query, retrieved_chunks):
    context_text = "\n\n".join([c["text"] for c in retrieved_chunks])
    return f"""[INST] Use the following context to answer the medical question. If the answer isn't in the context, say so honestly rather than guessing.
 
Context:
{context_text}
 
Question: {query} [/INST]"""
 
 
# ---------------------------------------------------------------------
# Generation — llama-cpp-python call instead of model.generate()
# ---------------------------------------------------------------------
def generate_answer(llm, prompt, max_new_tokens=300):
    output = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,   # deterministic, matches do_sample=False in the Colab version
        stop=["</s>", "[INST]"],
    )
    return output["choices"][0]["text"].strip()
 
 
# ---------------------------------------------------------------------
# Full pipeline (same shape as cell 49's ask())
# ---------------------------------------------------------------------
def ask(query, llm, index, chunks, embed_model, top_k=5, max_new_tokens=300, verbose=True):
    retrieved = retrieve_context(query, index, chunks, embed_model, top_k)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(llm, prompt, max_new_tokens=max_new_tokens)
 
    if verbose:
        print(f"Q: {query}\n")
        print(f"A: {answer}\n")
        print("-" * 60)
        print("Retrieved context used:")
        for i, c in enumerate(retrieved, 1):
            print(f"  [{i}] source={c['source']} | chapter={c.get('chapter','')} | section={c.get('section','')}")
 
    return answer, retrieved
 
 
if __name__ == "__main__":
    embed_model, combined_covid_chunks, index_new = load_retrieval_stack()
    llm = load_biomistral_cpu()
 
    answer, retrieved = ask(
        "What is the function of Mitochondria?",
        llm=llm,
        index=index_new,
        chunks=combined_covid_chunks,
        embed_model=embed_model,
        top_k=5,
    )
