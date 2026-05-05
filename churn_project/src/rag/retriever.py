# =============================================================================
# src/rag/retriever.py
# Similarity search across FAISS collections and context builder.
# Replaces ChromaDB retriever — no C++ compilation required.
# =============================================================================

import os
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import VECTOR_STORE, RAG_TOP_K
from .embedder import ChurnEmbedder


class FAISSRetriever:
    """
    Loads FAISS indexes from disk and performs similarity search across
    all three collections (customer_profiles, segments, features).

    Assembles a unified context string for the LLM on each query.
    """

    def __init__(self, persist_dir=VECTOR_STORE, top_k=RAG_TOP_K):
        self.persist_dir = str(persist_dir)
        self.top_k       = top_k
        self.embedder    = ChurnEmbedder()
        self._indexes    = {}   # collection_name -> faiss index
        self._docs       = {}   # collection_name -> list of doc dicts

    # --------------------------------------------------------------------------
    def connect(self):
        """Load all FAISS indexes and document stores from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        collections = ["customer_profiles", "segments", "features"]
        loaded = []

        for name in collections:
            index_path = os.path.join(self.persist_dir, f"{name}.index")
            docs_path  = os.path.join(self.persist_dir, f"{name}.json")

            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                print(f"[RETRIEVE] Warning: '{name}' not found — skipping. "
                      f"Run indexer.py first.")
                continue

            self._indexes[name] = faiss.read_index(index_path)
            with open(docs_path, "r", encoding="utf-8") as f:
                self._docs[name] = json.load(f)

            loaded.append(f"{name} ({self._indexes[name].ntotal} vectors)")

        self.embedder.load_embedding_model()
        print(f"[RETRIEVE] Loaded collections: {', '.join(loaded)}")
        return self

    # --------------------------------------------------------------------------
    def _query_collection(self, collection_name: str,
                          query_embedding: np.ndarray,
                          n_results: int = None) -> list:
        """
        Search a single FAISS collection.
        Returns list of document text strings for the top-k matches.
        """
        import faiss

        if collection_name not in self._indexes:
            return []

        index = self._indexes[collection_name]
        docs  = self._docs[collection_name]
        n     = min(n_results or self.top_k, index.ntotal)

        # Query vector must be float32 and normalised (cosine similarity)
        q = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        distances, indices = index.search(q, n)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(docs):
                results.append(docs[idx]["text"])
        return results

    # --------------------------------------------------------------------------
    def retrieve(self, query: str) -> dict:
        """
        Retrieve relevant documents from all collections for a query.

        Returns:
            dict with keys:
                customer_docs   : list of matched customer profile strings
                segment_docs    : list of matched segment summary strings
                feature_docs    : list of matched feature description strings
                context_string  : unified context block for the LLM prompt
        """
        query_embedding = self.embedder.embed([query])[0]

        customer_docs = self._query_collection(
            "customer_profiles", query_embedding, n_results=self.top_k
        )
        segment_docs  = self._query_collection(
            "segments", query_embedding, n_results=5
        )
        feature_docs  = self._query_collection(
            "features", query_embedding, n_results=3
        )

        # Build unified context string
        parts = []
        if segment_docs:
            parts.append(
                "=== SEGMENT INSIGHTS ===\n" + "\n".join(segment_docs)
            )
        if feature_docs:
            parts.append(
                "=== FEATURE DEFINITIONS ===\n" + "\n".join(feature_docs)
            )
        if customer_docs:
            parts.append(
                "=== SIMILAR CUSTOMER EXAMPLES ===\n" + "\n".join(customer_docs)
            )

        context_string = "\n\n".join(parts)

        return {
            "customer_docs":  customer_docs,
            "segment_docs":   segment_docs,
            "feature_docs":   feature_docs,
            "context_string": context_string,
        }

    # --------------------------------------------------------------------------
    def retrieve_by_segment(self, segment_col: str, segment_val) -> list:
        """
        Filtered retrieval: find segment documents matching a specific group.
        E.g. retrieve_by_segment('CreditRating', 6)
        """
        if "segments" not in self._docs:
            return []

        results = []
        for doc in self._docs["segments"]:
            meta = doc.get("metadata", {})
            if (meta.get("segment_col") == segment_col and
                    str(meta.get("segment_val")) == str(segment_val)):
                results.append(doc["text"])
        return results


# Keep old name as alias so existing imports still work
ChurnRetriever = FAISSRetriever


if __name__ == "__main__":
    retriever = FAISSRetriever()
    retriever.connect()
    result = retriever.retrieve("Why are new customers churning more?")
    print(result["context_string"][:1000])