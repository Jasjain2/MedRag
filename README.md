# MedRAG — Cross-Lingual Medical Retrieval-Augmented Generation

A cross-lingual (English / Spanish / Marathi) medical question-answering system built on **Retrieval-Augmented Generation (RAG)** over 16,000+ NIH-sourced medical documents.

> ⚠️ Educational demonstration only — not medical advice.

## What it does
Ask a medical question in English, Spanish, or Marathi → the system retrieves relevant passages from a medical knowledge base and generates a grounded answer in the same language.

## Results (measured)
- **90% top-5 retrieval accuracy** (exact-source-match) on a 100-question evaluation
- **Cross-lingual retrieval:** Marathi 98.6% top-3, Spanish 94.4% top-3 (relevance)
- **Scale:** 16,407 documents → 52,685 chunks, 1024-dim embeddings
- **Latency:** ~122ms per query, ~9.8 queries/sec

## Architecture
- **Embeddings:** BAAI/bge-m3 (multilingual)
- **Retrieval:** dense vector search (cosine similarity)
- **Generation:** grounded answer from retrieved context
- **Cross-lingual (Marathi):** translate-in → answer in English → translate-out (NLLB-200)

## Key engineering findings
- Diagnosed retrieval failures through systematic experiments (embedding models, chunking, rerankers); identified long-document embedding dilution and resolved it with a higher-resolution embedder.
- Quantified a ~6x Devanagari (Marathi) tokenization penalty; a custom BPE tokenizer reduced Marathi token usage ~60%.
- Evaluated hybrid search (BM25 + dense) and 3 rerankers; all reduced accuracy on this data — shipped the simplest high-performing pipeline (evidence-based).
- Refuses to hallucinate unanswerable clinical details (e.g. pediatric drug dosages), deferring to a professional.

## Dataset
MedQuAD — medical Q&A from U.S. NIH sources (NIH, MedlinePlus, NIDDK, etc.).

## Tech stack
Python, Sentence-Transformers, BGE-M3, NLLB-200, PyTorch, Hugging Face, FAISS-style vector search, BM25.

## Note on large files
Embeddings and model weights are not included (size limits); the notebook regenerates them from the dataset.
