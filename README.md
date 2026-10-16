# LifeSci-RAG
An AI-powered life science chatbot using RAG to provide context-aware answers from trusted biological knowledge sources.
#  Hybrid Retrieval-Augmented Generation (RAG) System for Computational Biology

A Hybrid RAG system that intelligently routes user queries between two specialized knowledge bases using **FAISS semantic search**, **BM25 keyword search**, and **BioMistral-7B (GGUF)** for answer generation.

---

##  Project Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** architecture capable of answering biology and COVID-19 related questions.

The system consists of two retrieval pipelines:

- **Version 3 (V3)**
  - Campbell Biology
  - Gale Encyclopedia of Medicine

- **Version 4 (V4)**
  - COVID-19 Knowledge Base
  - Biomedical literature

For every user query, both knowledge bases are searched simultaneously. A hybrid confidence score combining **semantic similarity (FAISS)** and **keyword relevance (BM25)** determines which knowledge base provides the most relevant context. The retrieved context is then passed to **BioMistral-7B** for final answer generation.

---

#  Features

- Hybrid Retrieval Architecture
- Semantic Search using FAISS
- Keyword Search using BM25
- Automatic Knowledge Base Routing
- BioMistral-7B GGUF Inference
- CPU Compatible
- Modular Python Implementation

---

#  Project Structure

```
final_evaluation/
│
├── final_evaluation.py          # Main application
├── README.md
├── requirements.txt
│
├── faiss_index_v3.bin
├── faiss_index_v4.bin
│
├── filtered_chunks_v3_new.pkl
├── combined_covid_chunks.pkl
│
├── v3_bm25.pkl
├── v4_bm25.pkl
│
└── BioMistral-7B.Q4_K_M.gguf
```

---

# Technologies Used

- Python 3.11+
- FAISS
- Sentence Transformers
- BM25
- Hugging Face Hub
- llama-cpp-python
- BioMistral-7B GGUF

---

#  Installation



Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Model Download

The BioMistral GGUF model is downloaded automatically during the first execution using Hugging Face Hub.

```python
model_path = hf_hub_download(
    repo_id="MaziyarPanahi/BioMistral-7B-GGUF",
    filename="BioMistral-7B.Q4_K_M.gguf"
)
```

The downloaded model is cached locally and does not need to be downloaded again.

---

#  Running the Project

```bash
python final_evaluation.py
```

Example:

```
============================================================
BioMistral Hybrid RAG
============================================================

Enter your question:
```

---

#  Retrieval Pipeline

1. User enters a question.
2. Question embedding is generated using BGE-Small.
3. Both knowledge bases are searched using:
   - FAISS
   - BM25
4. Hybrid confidence scores are computed.
5. The knowledge base with the highest confidence is selected.
6. Top retrieved chunks are used as context.
7. BioMistral generates the final answer.

---

#  Hybrid Confidence Score

The confidence score is calculated using:

```
Hybrid Score =
(1 − BM25 Weight) × Semantic Similarity
+
BM25 Weight × Normalized BM25 Score
```

Default BM25 weight:

```
0.3
```

---

#  Knowledge Bases

## Version 3

- Campbell Biology
- Gale Encyclopedia of Medicine

Suitable for:

- Cell Biology
- Genetics
- Molecular Biology
- Human Physiology
- General Medicine

---

## Version 4

Contains:

- COVID-19 literature
- Biomedical research
- COVID knowledge base

Suitable for:

- SARS-CoV-2
- COVID-19
- Vaccines
- Viral Infection
- Pandemic Research

---

# Example

**Question**

```
How does SARS-CoV-2 enter human cells?
```

**Output**

```
Routed to: V3

Confidence:
V3 = 0.786
V4 = 0.777

Answer:
SARS-CoV-2 enters human cells by binding to the ACE2 receptor through its spike protein, allowing viral entry and replication.
```

---

#  Workflow

```
                User Query
                     │
                     ▼
         Sentence Transformer (BGE)
                     │
     ┌───────────────┴───────────────┐
     ▼                               ▼
 FAISS Search (V3)             FAISS Search (V4)
     ▼                               ▼
 BM25 Search (V3)              BM25 Search (V4)
     ▼                               ▼
 Hybrid Confidence Calculation
                     │
                     ▼
          Best Knowledge Base
                     │
                     ▼
           Retrieved Context
                     │
                     ▼
        BioMistral-7B (GGUF)
                     │
                     ▼
              Generated Answer
```
