import pickle
import faiss
 
index = faiss.read_index("faiss_index_v3.bin")
with open("filtered_chunks_v3_new.pkl", "rb") as f:
    chunks = pickle.load(f)
 
print(f"FAISS index size (ntotal): {index.ntotal}")
print(f"Chunks file length:        {len(chunks)}")
 
if index.ntotal != len(chunks):
    print("\n>>> MISMATCH CONFIRMED. Index and chunks file are out of sync.")
    print(">>> You need to rebuild faiss_index_v3.bin from the CURRENT filtered_chunks_v3_new.pkl.")
else:
    print("\n>>> Sizes match. Mismatch (if any) is an ordering issue, not a count issue.")
    print(">>> Spot-check: print a few chunks and sanity-read them manually:")
    for i in [0, len(chunks)//2, len(chunks)-1]:
        text = chunks[i].get("text", chunks[i]) if isinstance(chunks[i], dict) else chunks[i]
        print(f"\n--- chunk {i} ---")
        print(str(text)[:200])