# =============================================================================
# src/model/explainer.py
# SHAP-based explainability for the churn model.
# Generates global + per-customer explanations that feed into the RAG layer.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR, TARGET_COL

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


class SHAPExplainer:
    """
    Generates SHAP explanations for the XGBoost churn model.

    Three outputs:
        1. Global feature importance (summary plot, bar plot)
        2. Per-customer explanation CSV (used for RAG indexing)
        3. Saved SHAP explainer object for RAG inference
    """

    def __init__(self, model=None, X_test=None, feature_names=None):
        self.model         = model
        self.X_test        = X_test
        self.feature_names = feature_names
        self.explainer     = None
        self.shap_values   = None
        self._X_explained  = None

    # --------------------------------------------------------------------------
    def load_model(self, model_name="xgboost"):
        """Load a saved model from disk."""
        path = os.path.join(MODELS_DIR, f"{model_name}_churn.pkl")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[SHAP] Loaded model: {path}")
        return self

    # --------------------------------------------------------------------------
    def fit(self):
        """Fit SHAP TreeExplainer and compute SHAP values on test set."""
        try:
            import shap
        except ImportError:
            print("[SHAP] shap not installed. Run: pip install shap")
            return self

        print("[SHAP] Fitting TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

        # Subsample for speed if dataset is large
        X_sample = self.X_test
        if len(self.X_test) > 5000:
            X_sample = self.X_test.sample(n=5000, random_state=42)

        self.shap_values  = self.explainer.shap_values(X_sample)
        self._X_explained = X_sample
        print(f"[SHAP] Computed SHAP values for {len(X_sample):,} samples.")

        # Save explainer for RAG inference use
        path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.explainer, f)
        print(f"[SHAP] Explainer saved: {path}")
        return self

    # --------------------------------------------------------------------------
    def plot_global_summary(self):
        """SHAP beeswarm summary plot — shows feature impact distribution."""
        import shap
        shap.summary_plot(
            self.shap_values, self._X_explained,
            feature_names=self.feature_names,
            plot_type="dot", show=False, max_display=20,
        )
        plt.title("SHAP Global Summary — Top 20 Features",
                  fontweight="bold", fontsize=13)
        plt.tight_layout()
        _save(plt.gcf(), "18_shap_summary.png")
        plt.close("all")

    # --------------------------------------------------------------------------
    def plot_global_bar(self):
        """SHAP bar plot — mean absolute SHAP values per feature."""
        import shap
        shap.summary_plot(
            self.shap_values, self._X_explained,
            feature_names=self.feature_names,
            plot_type="bar", show=False, max_display=20,
        )
        plt.title("SHAP Mean |SHAP Value| — Feature Importance",
                  fontweight="bold")
        plt.tight_layout()
        _save(plt.gcf(), "19_shap_bar.png")
        plt.close("all")

    # --------------------------------------------------------------------------
    def export_per_customer_explanations(self,
                                         y_prob: np.ndarray = None
                                         ) -> pd.DataFrame:
        """
        Build a human-readable explanation string for each customer.
        This is the core document indexed into FAISS for RAG retrieval.

        Args:
            y_prob: array of churn probabilities. If None, uses SHAP sum
                    as a proxy so the function never crashes.

        Returns:
            DataFrame: [customer_idx, churn_prob, top_factors, explanation_text]
        """
        X_arr = (self._X_explained.values
                 if hasattr(self._X_explained, "values")
                 else self._X_explained)

        # If y_prob not supplied, derive a proxy from SHAP values
        # (sum of SHAP + base value approximates log-odds → sigmoid → prob)
        if y_prob is None:
            base_value = (self.explainer.expected_value
                          if not isinstance(self.explainer.expected_value,
                                            (list, np.ndarray))
                          else self.explainer.expected_value[0])
            raw_scores = self.shap_values.sum(axis=1) + base_value
            # sigmoid
            y_prob = 1 / (1 + np.exp(-raw_scores))
            print("[SHAP] y_prob not provided — derived from SHAP values.")

        # Align lengths — y_prob may come from full test set, shap from sample
        n = len(self._X_explained)
        if len(y_prob) != n:
            # If y_prob is longer, we can't slice safely — use SHAP-derived
            base_value = (self.explainer.expected_value
                          if not isinstance(self.explainer.expected_value,
                                            (list, np.ndarray))
                          else self.explainer.expected_value[0])
            raw_scores = self.shap_values.sum(axis=1) + base_value
            y_prob     = 1 / (1 + np.exp(-raw_scores))
            print("[SHAP] y_prob length mismatch — re-derived from SHAP.")

        records = []
        for i in range(n):
            row_shap = self.shap_values[i]
            top5_idx = np.argsort(np.abs(row_shap))[::-1][:5]

            top_factors = []
            for idx in top5_idx:
                fname     = self.feature_names[idx]
                fval      = X_arr[i, idx]
                sv        = row_shap[idx]
                direction = "increases" if sv > 0 else "decreases"
                top_factors.append(
                    f"{fname}={fval:.2f} "
                    f"({direction} churn risk by {abs(sv):.3f})"
                )

            prob       = float(y_prob[i])
            pred_label = "CHURNED" if prob > 0.5 else "RETAINED"

            explanation = (
                f"Customer prediction: {pred_label} "
                f"(churn probability: {prob:.2%}). "
                f"Top churn factors: {'; '.join(top_factors)}."
            )

            records.append({
                "customer_idx":     i,
                "churn_probability": prob,
                "prediction":        pred_label,
                "top_5_factors":     " | ".join(top_factors),
                "explanation_text":  explanation,
            })

        df_explanations = pd.DataFrame(records)
        path = os.path.join(OUTPUTS_DIR, "shap_customer_explanations.csv")
        df_explanations.to_csv(path, index=False)
        print(f"[SHAP] Per-customer explanations saved: {path}  "
              f"({len(df_explanations):,} rows)")
        return df_explanations

    # --------------------------------------------------------------------------
    def run(self, y_prob=None):
        """Full SHAP pipeline — fit, plot global, export per-customer."""
        print("[SHAP] Running Explainability Pipeline...")
        self.fit()
        self.plot_global_summary()
        self.plot_global_bar()
        df_exp = self.export_per_customer_explanations(y_prob)
        print("[SHAP] Done.\n")
        return df_exp


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DATA_ENGINEERED
    from sklearn.model_selection import train_test_split

    # Load engineered data
    df = pd.read_csv(DATA_ENGINEERED)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL].astype(int)

    # Recreate the same test split used during training
    # (same random_state=42 and test_size=0.15 as in trainer.py)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Fill any NaNs using column medians (mirrors clean_remaining_nans in trainer)
    nan_cols = X_test.columns[X_test.isnull().any()].tolist()
    for col in nan_cols:
        X_test[col] = X_test[col].fillna(X_test[col].median())

    # Load saved model and compute predictions
    model_path = os.path.join(MODELS_DIR, "xgboost_churn.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Run SHAP on a 1000-row sample of the test set
    sample_idx = X_test.sample(n=min(1000, len(X_test)),
                                random_state=42).index
    X_sample   = X_test.loc[sample_idx]
    y_sample   = y_prob[X_test.index.get_indexer(sample_idx)]

    explainer = SHAPExplainer(
        model=model,
        X_test=X_sample,
        feature_names=list(X.columns),
    )
    explainer.run(y_prob=y_sample)