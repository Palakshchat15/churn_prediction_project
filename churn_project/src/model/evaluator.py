# =============================================================================
# src/model/evaluator.py
# Comprehensive model evaluation: ROC-AUC, F1, Precision-Recall,
# Confusion Matrix, KS Statistic, Feature Importance plots.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TARGET_COL, PLOTS_DIR, OUTPUTS_DIR, MODELS_DIR
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from scipy import stats

sns.set_theme(style="whitegrid")
BLUE = "#2563EB"
RED  = "#EF4444"
DARK = "#1B3A6B"
GRAY = "#475569"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


class ModelEvaluator:
    """
    Evaluates trained models on the held-out test set.

    Args:
        models: dict of {name: fitted_model}
        X_test, y_test: test features and labels
        feature_names: list of feature column names
    """

    def __init__(self, models: dict, X_test, y_test, feature_names: list):
        self.models        = models
        self.X_test        = X_test
        self.y_test        = y_test
        self.feature_names = feature_names
        self.results       = {}

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_all(self):
        """Run full evaluation for every model."""
        print("\n[EVAL] Running Model Evaluation...")
        for name, model in self.models.items():
            print(f"\n  Model: {name.upper()}")
            self._evaluate_single(name, model)
        self._plot_roc_comparison()
        self._plot_pr_comparison()
        self._save_report()
        return self

    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_single(self, name, model):
        y_prob = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)

        auc       = roc_auc_score(self.y_test, y_prob)
        f1_macro  = f1_score(self.y_test, y_pred, average="macro")
        f1_churn  = f1_score(self.y_test, y_pred, average="binary")
        pr_auc    = average_precision_score(self.y_test, y_prob)
        ks        = self._ks_statistic(self.y_test, y_prob)
        report    = classification_report(self.y_test, y_pred,
                                          target_names=["Not Churned", "Churned"])

        self.results[name] = {
            "auc": auc, "f1_macro": f1_macro, "f1_churn": f1_churn,
            "pr_auc": pr_auc, "ks": ks,
            "y_prob": y_prob, "y_pred": y_pred, "report": report,
        }

        print(f"    ROC-AUC : {auc:.4f}")
        print(f"    PR-AUC  : {pr_auc:.4f}")
        print(f"    F1 Macro: {f1_macro:.4f}")
        print(f"    F1 Churn: {f1_churn:.4f}")
        print(f"    KS Stat : {ks:.4f}")

        self._plot_confusion_matrix(name, y_pred)
        self._plot_feature_importance(name, model)

    # ──────────────────────────────────────────────────────────────────────────
    def _ks_statistic(self, y_true, y_prob):
        """Kolmogorov-Smirnov statistic — industry standard for churn models."""
        prob_churn    = y_prob[y_true == 1]
        prob_no_churn = y_prob[y_true == 0]
        ks, _ = stats.ks_2samp(prob_churn, prob_no_churn)
        return ks

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_confusion_matrix(self, name, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Not Churned", "Churned"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix — {name.replace('_', ' ').title()}",
                     fontweight="bold", color=DARK, fontsize=13)
        plt.tight_layout()
        _save(fig, f"cm_{name}.png")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_feature_importance(self, name, model):
        """Plot top 20 feature importances (available for tree-based models)."""
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "named_steps"):
            # Pipeline (e.g. Logistic Regression)
            lr = model.named_steps.get("lr")
            if lr and hasattr(lr, "coef_"):
                importance = np.abs(lr.coef_[0])

        if importance is None or len(importance) != len(self.feature_names):
            return

        feat_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1],
                color=BLUE, alpha=0.8, edgecolor="white")
        ax.set_title(f"Top 20 Feature Importances — {name.replace('_', ' ').title()}",
                     fontweight="bold", color=DARK, fontsize=13)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        _save(fig, f"fi_{name}.png")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_roc_comparison(self):
        """ROC curves for all models on the same plot."""
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = [BLUE, RED, "#16A34A", "#9333EA"]

        for (name, res), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, res["y_prob"])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name.replace('_', ' ').title()} (AUC={res['auc']:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Model Comparison", fontweight="bold", color=DARK, fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, "16_roc_comparison.png")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_pr_comparison(self):
        """Precision-Recall curves — better for imbalanced classification."""
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = [BLUE, RED, "#16A34A", "#9333EA"]
        baseline = self.y_test.mean()

        for (name, res), color in zip(self.results.items(), colors):
            precision, recall, _ = precision_recall_curve(self.y_test, res["y_prob"])
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f"{name.replace('_', ' ').title()} (PR-AUC={res['pr_auc']:.3f})")

        ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1,
                   label=f"Baseline ({baseline:.2%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — Model Comparison",
                     fontweight="bold", color=DARK, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, "17_pr_comparison.png")

    # ──────────────────────────────────────────────────────────────────────────
    def _save_report(self):
        """Save a text model report."""
        path = os.path.join(OUTPUTS_DIR, "model_report.txt")
        with open(path, "w") as f:
            f.write("CELL2CELL CHURN MODEL EVALUATION REPORT\n")
            f.write("=" * 55 + "\n\n")
            for name, res in self.results.items():
                f.write(f"MODEL: {name.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC  : {res['auc']:.4f}\n")
                f.write(f"PR-AUC   : {res['pr_auc']:.4f}\n")
                f.write(f"F1 Macro : {res['f1_macro']:.4f}\n")
                f.write(f"F1 Churn : {res['f1_churn']:.4f}\n")
                f.write(f"KS Stat  : {res['ks']:.4f}\n\n")
                f.write(res["report"])
                f.write("\n\n")
        print(f"\n[EVAL] Report saved: {path}")


def load_and_evaluate(X_test, y_test, feature_names):
    """Helper — load saved models from disk and evaluate on test set."""
    model_files = {
        "xgboost": os.path.join(MODELS_DIR, "xgboost_churn.pkl"),
        "lightgbm": os.path.join(MODELS_DIR, "lightgbm_churn.pkl"),
        "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    }
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    evaluator = ModelEvaluator(models, X_test, y_test, feature_names)
    evaluator.evaluate_all()
    return evaluator
