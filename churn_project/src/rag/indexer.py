# =============================================================================
# src/rag/indexer.py
# Ingests customer documents, segment summaries, and feature descriptions
# into a persistent FAISS vector store (replaces ChromaDB).
# No C++ compilation required — faiss-cpu installs as a prebuilt wheel.
# =============================================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import VECTOR_STORE, DATA_ENGINEERED, OUTPUTS_DIR, TARGET_COL
from .embedder import ChurnEmbedder


class FAISSIndexer:
    """
    Builds and persists a FAISS vector store with three document collections:
        - customer_profiles : one document per customer (+ SHAP if available)
        - segments          : aggregate churn stats per segment group
        - features          : plain-English feature descriptions

    All collections are saved as flat files inside data/vector_store/:
        customer_profiles.index   (FAISS index binary)
        customer_profiles.json    (document texts + metadata)
        segments.index
        segments.json
        features.index
        features.json
    """

    def __init__(self, persist_dir=VECTOR_STORE):
        self.persist_dir = str(persist_dir)
        self.embedder    = ChurnEmbedder()
        os.makedirs(self.persist_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    def _build_and_save(self, collection_name: str, documents: list):
        """
        Embed documents, build a FAISS flat index, and save both the
        index binary and the document metadata JSON to disk.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        if not documents:
            print(f"[INDEX] No documents for '{collection_name}' — skipping.")
            return

        print(f"[INDEX] Building '{collection_name}' "
              f"({len(documents)} documents)...")

        # Load embedding model once
        self.embedder.load_embedding_model()

        texts      = [d["text"]     for d in documents]
        ids        = [d["id"]       for d in documents]
        metadatas  = [d["metadata"] for d in documents]

        # Embed in batches
        embeddings = self.embedder.embed(texts)             # (N, dim)
        embeddings = embeddings.astype(np.float32)

        # Normalise for cosine similarity (dot product on unit vectors = cosine)
        faiss.normalize_L2(embeddings)

        # Build flat inner-product index
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save FAISS index binary
        index_path = os.path.join(self.persist_dir, f"{collection_name}.index")
        faiss.write_index(index, index_path)

        # Save document texts + metadata as JSON
        docs_path = os.path.join(self.persist_dir, f"{collection_name}.json")
        payload   = [
            {"id": ids[i], "text": texts[i], "metadata": metadatas[i]}
            for i in range(len(documents))
        ]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[INDEX] '{collection_name}' saved  "
              f"({index.ntotal} vectors, dim={dim})")
        print(f"        Index : {index_path}")
        print(f"        Docs  : {docs_path}")

    # --------------------------------------------------------------------------
    def index_customers(self, df: pd.DataFrame,
                        shap_df: pd.DataFrame = None,
                        sample_size: int = 10000):
        """
        Index customer profile documents.
        Samples up to sample_size rows to keep the index manageable.
        """
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"[INDEX] Sampling {sample_size:,} / {len(df):,} customers")
        else:
            df_sample = df

        docs = ChurnEmbedder.build_customer_documents(df_sample, shap_df)
        self._build_and_save("customer_profiles", docs)
        return self

    # --------------------------------------------------------------------------
    def index_segments(self, df: pd.DataFrame):
        """Index segment-level aggregate summary documents."""
        docs = ChurnEmbedder.build_segment_documents(df)
        self._build_and_save("segments", docs)
        return self

    # --------------------------------------------------------------------------
    def index_features(self):
        """Index plain-English feature description documents."""
        docs = ChurnEmbedder.build_feature_documents()
        self._build_and_save("features", docs)
        return self

    # --------------------------------------------------------------------------
    def run(self, df: pd.DataFrame = None, shap_df: pd.DataFrame = None):
        """Full indexing pipeline — indexes all three collections."""
        print("\n" + "=" * 55)
        print("  RAG INDEXING PIPELINE  (FAISS)")
        print("=" * 55)

        if df is None:
            df = pd.read_csv(DATA_ENGINEERED)

        if shap_df is None:
            shap_path = os.path.join(str(OUTPUTS_DIR),
                                     "shap_customer_explanations.csv")
            if os.path.exists(shap_path):
                shap_df = pd.read_csv(shap_path)
                print(f"[INDEX] Loaded SHAP explanations: {len(shap_df):,} rows")

        self.index_customers(df, shap_df)
        self.index_segments(df)
        self.index_features()

        # List what was saved
        saved = [f for f in os.listdir(self.persist_dir)
                 if f.endswith(".index")]
        print(f"\n[INDEX] Done. Collections in {self.persist_dir}:")
        for s in saved:
            print(f"        {s}")
        print("=" * 55)
        return self


# Keep old name as alias so existing imports still work
ChromaIndexer = FAISSIndexer


if __name__ == "__main__":
    indexer = FAISSIndexer()
    indexer.run()