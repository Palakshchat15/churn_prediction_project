# =============================================================================
# src/eda/multivariate.py
# Multivariate analysis: PCA, t-SNE, statistical tests (Chi-square, ANOVA),
# pair plots of top features.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TARGET_COL, PLOTS_DIR

sns.set_theme(style="whitegrid")
BLUE = "#2563EB"
RED  = "#EF4444"
DARK = "#1B3A6B"
GRAY = "#475569"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_pca(df, n_components=2):
    """
    PCA 2D scatter plot — reduce all features to 2 principal components,
    colour points by Churn to visualise class separation.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.drop(columns=[TARGET_COL], errors="ignore")
    y = numeric_df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_ * 100

    # Sample for speed (t-SNE / PCA on 10k is fast enough)
    sample_idx = np.random.choice(len(components), size=min(10000, len(components)), replace=False)
    comp_sample = components[sample_idx]
    y_sample    = y.iloc[sample_idx].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PCA — Feature Space Visualisation", fontsize=15, fontweight="bold", color=DARK)

    # Scatter by class
    for val, color, label in [(0, BLUE, "Not Churned"), (1, RED, "Churned")]:
        mask = y_sample == val
        axes[0].scatter(comp_sample[mask, 0], comp_sample[mask, 1],
                        c=color, alpha=0.3, s=10, label=label)
    axes[0].set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
    axes[0].set_title("PCA 2D — Class Separation")
    axes[0].legend(markerscale=3)

    # Scree plot — explained variance per component
    pca_full = PCA(n_components=min(20, X.shape[1]), random_state=42)
    pca_full.fit(X_scaled)
    axes[1].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_ * 100, color=BLUE, alpha=0.8)
    axes[1].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                 np.cumsum(pca_full.explained_variance_ratio_) * 100,
                 color=RED, marker="o", linewidth=2, label="Cumulative")
    axes[1].axhline(80, color=GRAY, linestyle="--", linewidth=1, label="80% threshold")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Explained Variance (%)")
    axes[1].set_title("Scree Plot")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "12_pca.png")
    return pca, var_explained


# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne(df, sample_size=5000):
    """
    t-SNE 2D visualisation on a sample of the data.
    Computationally expensive — keep sample_size ≤ 5000.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.drop(columns=[TARGET_COL], errors="ignore")
    y = numeric_df[TARGET_COL]

    # Sample for speed
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[idx]
    y_sample = y.iloc[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    print(f"  Running t-SNE on {len(X_sample):,} samples (this takes ~1-2 min)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    embed = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    for val, color, label in [(0, BLUE, "Not Churned"), (1, RED, "Churned")]:
        mask = y_sample.values == val
        ax.scatter(embed[mask, 0], embed[mask, 1],
                   c=color, alpha=0.4, s=12, label=label)
    ax.set_title(f"t-SNE Visualisation (n={len(X_sample):,} sample)",
                 fontsize=14, fontweight="bold", color=DARK)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3)
    plt.tight_layout()
    _save(fig, "13_tsne.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_statistical_tests(df):
    """
    Chi-square tests for binary/ordinal features vs Churn.
    ANOVA (F-test) for continuous features vs Churn.
    Saves a ranked results table as CSV and prints top findings.
    """
    results = []
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c != TARGET_COL]

    for col in feature_cols:
        col_data = df[col].dropna()
        target_aligned = df.loc[col_data.index, TARGET_COL]

        n_unique = col_data.nunique()
        try:
            if n_unique <= 10:
                # Chi-square for discrete / binary columns
                contingency = pd.crosstab(col_data, target_aligned)
                chi2, p, dof, _ = stats.chi2_contingency(contingency)
                results.append({
                    "Feature": col, "Test": "Chi-Square",
                    "Statistic": round(chi2, 4), "p_value": round(p, 6),
                    "Significant_0.05": p < 0.05
                })
            else:
                # One-way ANOVA for continuous columns
                group0 = col_data[target_aligned == 0]
                group1 = col_data[target_aligned == 1]
                f_stat, p = stats.f_oneway(group0, group1)
                results.append({
                    "Feature": col, "Test": "ANOVA",
                    "Statistic": round(f_stat, 4), "p_value": round(p, 6),
                    "Significant_0.05": p < 0.05
                })
        except Exception:
            continue

    results_df = pd.DataFrame(results).sort_values("p_value")
    out_path = os.path.join(PLOTS_DIR, "..", "statistical_tests.csv")
    results_df.to_csv(out_path, index=False)

    print("\n  TOP 20 STATISTICALLY SIGNIFICANT FEATURES (by p-value):")
    print(results_df[results_df["Significant_0.05"]].head(20).to_string(index=False))

    # Plot top 20 significant features by F/Chi2 statistic
    sig_df = results_df[results_df["Significant_0.05"]].head(20)
    if not sig_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [BLUE if t == "ANOVA" else RED for t in sig_df["Test"]]
        ax.barh(sig_df["Feature"][::-1], sig_df["Statistic"][::-1], color=colors[::-1], alpha=0.8)
        ax.set_title("Top Statistically Significant Features vs Churn\n(Blue=ANOVA, Red=Chi-Square)",
                     fontweight="bold", color=DARK)
        ax.set_xlabel("Test Statistic (F or Chi²)")
        plt.tight_layout()
        _save(fig, "14_statistical_tests.png")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
def plot_pairplot(df):
    """Pair plot of top 5 numeric features most correlated with Churn."""
    numeric_df = df.select_dtypes(include=[np.number])
    top_features = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    pair_cols = top_features + [TARGET_COL]

    sample = df[pair_cols].dropna().sample(n=min(3000, len(df)), random_state=42)
    sample[TARGET_COL] = sample[TARGET_COL].map({0: "Not Churned", 1: "Churned"})

    g = sns.pairplot(sample, hue=TARGET_COL, palette={"Not Churned": BLUE, "Churned": RED},
                     diag_kind="kde", plot_kws={"alpha": 0.3, "s": 15},
                     diag_kws={"fill": True})
    g.figure.suptitle("Pair Plot — Top 5 Churn-Correlated Features", y=1.01,
                       fontsize=13, fontweight="bold", color=DARK)
    _save(g.figure, "15_pairplot.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_multivariate(df: pd.DataFrame):
    """Run all multivariate analyses."""
    print("\n[EDA] Running Multivariate Analysis...")
    plot_pca(df)
    plot_tsne(df)
    run_statistical_tests(df)
    plot_pairplot(df)
    print("[EDA] Multivariate complete.\n")


if __name__ == "__main__":
    from config import DATA_ENGINEERED
    df = pd.read_csv(DATA_ENGINEERED)
    run_multivariate(df)
