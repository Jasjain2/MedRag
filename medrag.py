"""
MedRAG — Cross-Lingual Medical Retrieval-Augmented Generation
=============================================================
Answers medical questions in English, Spanish, or Marathi by retrieving
relevant passages from an NIH-sourced medical corpus and generating a
grounded answer in the user's language.

Architecture:
    Query -> (translate to English if Marathi) -> dense retrieval (BGE-M3)
          -> generate grounded answer (English) -> (translate back if Marathi)

Educational demonstration only - not medical advice.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class MedRAG:
    def __init__(self, embeddings_path, chunks, embed_model="BAAI/bge-m3"):
        # Multilingual dense retriever
        self.embedder = SentenceTransformer(embed_model)
        # Answer generator (grounded on retrieved context)
        self.generator = pipeline(
            "text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16, device_map="auto"
        )
        # Translator (NLLB) for Marathi <-> English
        self.nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        # Precomputed knowledge base
        self.embeddings = np.load(embeddings_path)
        self.chunks = chunks

    # ---------- language handling ----------
    @staticmethod
    def detect_language(text):
        for ch in text:
            if "\u0900" <= ch <= "\u097F":   # Devanagari script -> Marathi
                return "marathi"
        if any(w in text.lower() for w in ["\u00bf", "cu\u00e1l", "s\u00edntomas", "qu\u00e9"]):
            return "spanish"
        return "english"

    def _translate(self, text, src, tgt):
        self.nllb_tok.src_lang = src
        inputs = self.nllb_tok(text, return_tensors="pt", truncation=True, max_length=400)
        bos = self.nllb_tok.convert_tokens_to_ids(tgt)
        out = self.nllb.generate(**inputs, forced_bos_token_id=bos, max_length=400)
        return self.nllb_tok.batch_decode(out, skip_special_tokens=True)[0]

    # ---------- retrieval ----------
    def retrieve(self, query, k=5):
        q_emb = self.embedder.encode(query, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0].numpy()
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.chunks[i] for i in top_k]

    # ---------- full pipeline ----------
    def answer(self, question, k=5):
        lang = self.detect_language(question)

        # Marathi queries: translate to English so the generator stays in its strength
        search_q = self._translate(question, "mar_Deva", "eng_Latn") if lang == "marathi" else question

        context = "\n\n".join(self.retrieve(search_q, k))
        prompt = (
            "You are a medical assistant. Answer concisely using ONLY the context. "
            "If the context lacks the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {search_q}\nAnswer:"
        )
        eng = self.generator(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"]
        eng = eng[len(prompt):].strip()

        # Translate answer back to Marathi if the user asked in Marathi
        return self._translate(eng, "eng_Latn", "mar_Deva") if lang == "marathi" else eng


if __name__ == "__main__":
    import pickle
    with open("full_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    rag = MedRAG("m3_chunk_embeddings.npy", chunks)

    for q in ["What are the symptoms of diabetes?",
              "\u092e\u0927\u0941\u092e\u0947\u0939\u093e\u091a\u0940 \u0932\u0915\u094d\u0937\u0923\u0947 \u0915\u094b\u0923\u0924\u0940 \u0906\u0939\u0947\u0924?"]:
        print("Q:", q)
        print("A:", rag.answer(q), "\n")
