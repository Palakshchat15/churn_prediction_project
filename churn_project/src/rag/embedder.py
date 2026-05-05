# =============================================================================
# src/rag/embedder.py
# Converts customer profiles and segment summaries into text documents,
# then embeds them using sentence-transformers.
# CustomerID is used as the document ID and included in the text.
# =============================================================================

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DATA_ENGINEERED, OUTPUTS_DIR, EMBEDDING_MODEL, TARGET_COL


class ChurnEmbedder:
    """
    Builds rich text documents for each customer and key segments.
    CustomerID is embedded in every customer document so the LLM
    can reference and identify specific customers accurately.
    """

    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model_name = model_name
        self._model     = None

    # --------------------------------------------------------------------------
    def load_embedding_model(self):
        """Load the sentence-transformers model (~80MB, downloaded once)."""
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            print(f"[EMBED] Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print("[EMBED] Model ready.")
        return self

    # --------------------------------------------------------------------------
    def embed(self, texts: list) -> np.ndarray:
        """Embed a list of text strings into dense vectors."""
        if self._model is None:
            self.load_embedding_model()
        return self._model.encode(texts, show_progress_bar=True, batch_size=64)

    # --------------------------------------------------------------------------
    @staticmethod
    def build_customer_documents(df: pd.DataFrame,
                                  shap_df: pd.DataFrame = None) -> list:
        """
        Convert each customer row into a natural language document string.
        Uses CustomerID as the document identifier if present.
        Optionally enriches with SHAP explanation text.

        Returns:
            List of dicts: {id, text, metadata}
        """
        documents = []

        for i, row in df.iterrows():
            # Use real CustomerID if available, otherwise fall back to row index
            cid = int(row["CustomerID"]) if "CustomerID" in row.index else i

            churn_label = (
                "churned" if row.get(TARGET_COL, 0) == 1 else "retained"
            )

            text = (
                f"Customer ID {cid}: {churn_label.upper()}. "
                f"Monthly revenue: ${row.get('MonthlyRevenue', 0):.2f}. "
                f"Monthly minutes used: {row.get('MonthlyMinutes', 0):.0f}. "
                f"Months in service: {row.get('MonthsInService', 0):.0f}. "
                f"Credit rating: {row.get('CreditRating', 'N/A')}. "
                f"Retention calls received: {row.get('RetentionCalls', 0):.0f}. "
                f"Retention offers accepted: "
                f"{row.get('RetentionOffersAccepted', 0):.0f}. "
                f"Called retention team: "
                f"{'yes' if row.get('MadeCallToRetentionTeam', 0) == 1 else 'no'}. "
                f"Drop rate: {row.get('DropRate', 0):.4f}. "
                f"Revenue per minute: ${row.get('RevenuePerMinute', 0):.4f}. "
                f"Equipment age (years): {row.get('EquipmentAgeYears', 0):.1f}. "
                f"Income group: {row.get('IncomeGroup', 'N/A')}. "
                f"Owns computer: "
                f"{'yes' if row.get('OwnsComputer', 0) == 1 else 'no'}. "
                f"Has credit card: "
                f"{'yes' if row.get('HasCreditCard', 0) == 1 else 'no'}."
            )

            # Append SHAP explanation if available
            if shap_df is not None and not shap_df.empty:
                match = shap_df[shap_df["customer_idx"] == i]
                if not match.empty:
                    text += (
                        f" SHAP explanation: "
                        f"{match.iloc[0]['explanation_text']}"
                    )

            documents.append({
                "id": f"customer_{cid}",
                "text": text,
                "metadata": {
                    "type":              "customer",
                    "customer_id":       cid,
                    "churn":             int(row.get(TARGET_COL, -1)),
                    "monthly_revenue":   float(row.get("MonthlyRevenue", 0)),
                    "months_in_service": int(row.get("MonthsInService", 0)),
                    "credit_rating":     (
                        int(row.get("CreditRating", 0))
                        if pd.notna(row.get("CreditRating")) else 0
                    ),
                }
            })

        return documents

    # --------------------------------------------------------------------------
    @staticmethod
    def build_segment_documents(df: pd.DataFrame) -> list:
        """
        Build aggregate segment-level documents.
        Segments: by CreditRating, TenureBucket, MaritalStatus, IncomeGroup.
        """
        documents = []

        def segment_doc(group_col, group_val, group_df):
            churn_rate = group_df[TARGET_COL].mean()
            avg_rev    = (group_df["MonthlyRevenue"].mean()
                          if "MonthlyRevenue" in group_df else 0)
            avg_mins   = (group_df["MonthlyMinutes"].mean()
                          if "MonthlyMinutes" in group_df else 0)
            count      = len(group_df)
            risk       = ("HIGH"   if churn_rate > 0.35
                          else "MEDIUM" if churn_rate > 0.2 else "LOW")
            text = (
                f"Segment summary: {group_col}={group_val}. "
                f"Total customers: {count:,}. "
                f"Churn rate: {churn_rate:.1%}. "
                f"Average monthly revenue: ${avg_rev:.2f}. "
                f"Average monthly minutes: {avg_mins:.0f}. "
                f"Churn risk level: {risk}."
            )
            return {
                "id":   f"segment_{group_col}_{group_val}",
                "text": text,
                "metadata": {
                    "type":        "segment",
                    "segment_col": group_col,
                    "segment_val": str(group_val),
                    "churn_rate":  float(churn_rate),
                }
            }

        for col in ["CreditRating", "TenureBucket", "MaritalStatus",
                    "IncomeGroup"]:
            if col in df.columns:
                for val, grp in df.groupby(col):
                    documents.append(segment_doc(col, val, grp))

        return documents

    # --------------------------------------------------------------------------
    @staticmethod
    def build_feature_documents() -> list:
        """
        Plain English descriptions of each feature.
        Allows the RAG to answer 'what does X mean?' questions.
        """
        feature_descriptions = [
            ("CustomerID",
             "Unique identifier for each customer. "
             "Used to look up specific customers in the database."),
            ("MonthlyRevenue",
             "Total revenue the customer generates each month in dollars."),
            ("MonthlyMinutes",
             "Total minutes the customer used in the billing month."),
            ("TotalRecurringCharge",
             "Fixed monthly charge for the customer's plan."),
            ("DroppedCalls",
             "Number of calls dropped mid-conversation."),
            ("BlockedCalls",
             "Number of outgoing calls that could not connect."),
            ("CustomerCareCalls",
             "Number of times the customer contacted customer care."),
            ("RetentionCalls",
             "Number of proactive calls made by the retention team."),
            ("RetentionOffersAccepted",
             "Number of retention offers the customer accepted."),
            ("MonthsInService",
             "Total months the customer has been subscribed."),
            ("CreditRating",
             "Customer credit rating from 1 (Highest) to 6 (VeryLow)."),
            ("OverageMinutes",
             "Extra minutes used beyond the plan allowance."),
            ("DropRate",
             "Engineered: ratio of dropped calls to total minutes — "
             "proxy for call quality experience."),
            ("RevenuePerMinute",
             "Engineered: revenue generated per minute of usage."),
            ("RetentionEngagement",
             "Engineered: RetentionCalls x RetentionOffersAccepted — "
             "measures how engaged a customer is with retention efforts."),
            ("TenureBucket",
             "Engineered: tenure lifecycle stage — "
             "0=New (0-12m), 1=12-24m, 2=24-48m, 3=Loyal (48m+)."),
            ("EquipmentAgeYears",
             "Engineered: years since the customer last upgraded their handset."),
            ("ServiceArea_Freq",
             "Engineered: frequency encoding of service area — "
             "proportion of all customers in the same area."),
        ]
        return [
            {
                "id":   f"feature_{name}",
                "text": f"Feature '{name}': {desc}",
                "metadata": {"type": "feature_description", "feature": name}
            }
            for name, desc in feature_descriptions
        ]