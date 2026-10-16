 
import streamlit as st
from rag_core import load_resources, hybrid_retrieve, build_prompt, generate_answer
 
st.set_page_config(page_title="COVID-KOSHA", page_icon="🧬", layout="centered")
 
 
@st.cache_resource(show_spinner=False)
def get_resources():
    status = st.empty()
 
    def report(msg):
        status.info(msg)
 
    resources = load_resources(progress_cb=report)
    status.empty()
    return resources
 
 
st.title("🧬 COVID-KOSHA")
st.caption(
    "Hybrid FAISS + BM25 retrieval routing between v3 and v4 knowledge bases, "
    "answered by BioMistral-7B (CPU)."
)
 
with st.spinner("Loading models and indexes... this can take a few minutes on first run."):
    resources = get_resources()
 
if "history" not in st.session_state:
    st.session_state.history = []
 
# Sidebar controls
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)
    max_new_tokens = st.slider("Max answer tokens", min_value=50, max_value=500, value=300, step=50)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
 
# Replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Retrieval details"):
            r = turn["retrieval"]
            st.write(f"Routed to **{r['winner']}**")
            st.write(f"v3 confidence: `{r['v3_confidence']:.3f}` | v4 confidence: `{r['v4_confidence']:.3f}`")
            if show_chunks:
                for i, chunk in enumerate(r["chunks"]):
                    st.markdown(f"**[{i+1}]** {chunk['text'][:500]}")
 
# New query
query = st.chat_input("Ask a COVID-19 / life science question...")
 
if query:
    with st.chat_message("user"):
        st.write(query)
 
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating (can take a few minutes on CPU)..."):
            retrieval = hybrid_retrieve(resources, query, k=top_k)
            prompt = build_prompt(query, retrieval["chunks"])
            answer = generate_answer(resources["llm"], prompt, max_new_tokens=max_new_tokens)
 
        st.write(answer)
        with st.expander("Retrieval details"):
            st.write(f"Routed to **{retrieval['winner']}**")
            st.write(
                f"v3 confidence: `{retrieval['v3_confidence']:.3f}` | "
                f"v4 confidence: `{retrieval['v4_confidence']:.3f}`"
            )
            if show_chunks:
                for i, chunk in enumerate(retrieval["chunks"]):
                    st.markdown(f"**[{i+1}]** {chunk['text'][:500]}")
 
    st.session_state.history.append({"query": query, "answer": answer, "retrieval": retrieval})