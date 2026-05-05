# =============================================================================
# src/eda/bivariate.py
# Bivariate analysis: churn rates by feature, correlation heatmap,
# violin plots, chi-square tests, point-biserial correlations.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TARGET_COL, PLOTS_DIR

sns.set_theme(style="whitegrid")
BLUE  = "#2563EB"
RED   = "#EF4444"
DARK  = "#1B3A6B"
GRAY  = "#475569"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    """Pearson correlation heatmap of all numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(22, 18))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmax=1, vmin=-1,
                annot=False, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})

    # Highlight correlations with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", color=DARK, pad=20)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    _save(fig, "06_correlation_heatmap.png")

    # Also save top correlations with Churn as a bar chart
    top_n = 20
    top_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False).head(top_n)
    colors = [RED if corr[TARGET_COL][c] < 0 else BLUE for c in top_corr.index]

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1], edgecolor="white")
    ax2.set_title(f"Top {top_n} Features by |Correlation| with Churn",
                  fontsize=13, fontweight="bold", color=DARK)
    ax2.set_xlabel("|Pearson r| with Churn")
    ax2.axvline(0.1, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    _save(fig2, "07_top_churn_correlations.png")

    return target_corr


# ─────────────────────────────────────────────────────────────────────────────
def plot_churn_rate_by_segment(df):
    """Churn rate by key categorical segments."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Churn Rate by Customer Segment", fontsize=16, fontweight="bold", color=DARK)
    axes = axes.flatten()

    segments = [
        ("CreditRating", "Credit Rating (1=Best, 6=Worst)"),
        ("MaritalStatus", "Marital Status"),
        ("IncomeGroup", "Income Group"),
        ("Handsets", "Number of Handsets"),
        ("MadeCallToRetentionTeam", "Made Call to Retention Team"),
        ("NewCellphoneUser", "New Cellphone User"),
    ]

    for i, (col, title) in enumerate(segments):
        if col not in df.columns:
            axes[i].set_visible(False)
            continue
        ax = axes[i]
        grouped = df.groupby(col)[TARGET_COL].agg(["mean", "count"]).reset_index()
        grouped.columns = [col, "churn_rate", "count"]
        grouped[col] = grouped[col].astype(str)
        grouped = grouped.sort_values("churn_rate", ascending=False)

        bars = ax.bar(grouped[col], grouped["churn_rate"],
                      color=[RED if r > df[TARGET_COL].mean() else BLUE for r in grouped["churn_rate"]],
                      edgecolor="white", alpha=0.85)
        ax.axhline(df[TARGET_COL].mean(), color=DARK, linestyle="--",
                   linewidth=1.5, label=f"Overall mean ({df[TARGET_COL].mean():.1%})")

        for bar, (_, row) in zip(bars, grouped.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{row['churn_rate']:.1%}", ha="center", fontsize=8, fontweight="bold")

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, min(grouped["churn_rate"].max() * 1.3, 1.0))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "08_churn_rate_by_segment.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_violin_plots(df):
    """Violin plots for continuous features vs Churn."""
    key_cols = [
        "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
        "MonthsInService", "CurrentEquipmentDays", "OverageMinutes",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature Distribution by Churn (Violin Plots)", fontsize=15, fontweight="bold", color=DARK)
    axes = axes.flatten()

    for i, col in enumerate(key_cols):
        ax = axes[i]
        data_0 = df[df[TARGET_COL] == 0][col].dropna()
        data_1 = df[df[TARGET_COL] == 1][col].dropna()

        parts0 = ax.violinplot([data_0], positions=[0], showmedians=True, showextrema=False)
        parts1 = ax.violinplot([data_1], positions=[1], showmedians=True, showextrema=False)

        for pc in parts0["bodies"]:
            pc.set_facecolor(BLUE)
            pc.set_alpha(0.7)
        for pc in parts1["bodies"]:
            pc.set_facecolor(RED)
            pc.set_alpha(0.7)
        parts0["cmedians"].set_color("white")
        parts1["cmedians"].set_color("white")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Churned", "Churned"])
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel(col, fontsize=8)

        # Mann-Whitney U test p-value
        u_stat, p_val = stats.mannwhitneyu(data_0, data_1, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.set_xlabel(f"Mann-Whitney p={p_val:.4f} {sig}", fontsize=8, color=GRAY)

    plt.tight_layout()
    _save(fig, "09_violin_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_tenure_cohort_churn(df):
    """Churn rate by MonthsInService cohort (customer lifecycle analysis)."""
    if "MonthsInService" not in df.columns:
        return

    df = df.copy()
    bins  = [0, 12, 24, 48, float("inf")]
    labels = ["0-12m (New)", "12-24m", "24-48m", "48m+ (Loyal)"]
    df["TenureCohort"] = pd.cut(df["MonthsInService"], bins=bins, labels=labels, right=False)

    cohort_stats = df.groupby("TenureCohort", observed=True).agg(
        churn_rate=(TARGET_COL, "mean"),
        count=(TARGET_COL, "count"),
        avg_revenue=("MonthlyRevenue", "mean") if "MonthlyRevenue" in df.columns else (TARGET_COL, "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Lifecycle — Cohort Analysis", fontsize=14, fontweight="bold", color=DARK)

    # Churn rate by cohort
    colors = [RED if r > df[TARGET_COL].mean() else BLUE for r in cohort_stats["churn_rate"]]
    axes[0].bar(cohort_stats["TenureCohort"].astype(str), cohort_stats["churn_rate"],
                color=colors, edgecolor="white", alpha=0.85)
    axes[0].axhline(df[TARGET_COL].mean(), color=DARK, linestyle="--", linewidth=1.5,
                    label=f"Overall ({df[TARGET_COL].mean():.1%})")
    for j, (_, row) in enumerate(cohort_stats.iterrows()):
        axes[0].text(j, row["churn_rate"] + 0.005, f"{row['churn_rate']:.1%}",
                     ha="center", fontweight="bold")
    axes[0].set_title("Churn Rate by Tenure Cohort")
    axes[0].set_ylabel("Churn Rate")
    axes[0].legend()

    # Customer count by cohort
    axes[1].bar(cohort_stats["TenureCohort"].astype(str), cohort_stats["count"],
                color=BLUE, edgecolor="white", alpha=0.7)
    axes[1].set_title("Customer Count by Tenure Cohort")
    axes[1].set_ylabel("Number of Customers")
    for j, (_, row) in enumerate(cohort_stats.iterrows()):
        axes[1].text(j, row["count"] + 100, f"{row['count']:,}", ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "10_tenure_cohort.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_revenue_vs_churn(df):
    """Revenue distribution: churned vs retained customers."""
    if "MonthlyRevenue" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Monthly Revenue Analysis by Churn Status", fontsize=14, fontweight="bold", color=DARK)

    # Overlapping histograms
    for val, color, label in [(0, BLUE, "Not Churned"), (1, RED, "Churned")]:
        data = df[df[TARGET_COL] == val]["MonthlyRevenue"].dropna()
        axes[0].hist(data, bins=50, alpha=0.55, color=color, label=f"{label} (n={len(data):,})", density=True)
    axes[0].set_title("Revenue Distribution Overlap")
    axes[0].set_xlabel("Monthly Revenue")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Revenue bucket churn rate
    df2 = df.copy()
    df2["RevBucket"] = pd.qcut(df2["MonthlyRevenue"], q=5,
                               labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
                               duplicates="drop")
    bucket_churn = df2.groupby("RevBucket", observed=True)[TARGET_COL].mean()
    axes[1].bar(bucket_churn.index.astype(str), bucket_churn.values,
                color=[RED if r > df[TARGET_COL].mean() else BLUE for r in bucket_churn.values],
                edgecolor="white", alpha=0.85)
    axes[1].axhline(df[TARGET_COL].mean(), color=DARK, linestyle="--", linewidth=1.5)
    axes[1].set_title("Churn Rate by Revenue Quintile")
    axes[1].set_ylabel("Churn Rate")
    axes[1].set_xlabel("Revenue Quintile")

    plt.tight_layout()
    _save(fig, "11_revenue_vs_churn.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_bivariate(df: pd.DataFrame):
    """Run all bivariate analyses."""
    print("\n[EDA] Running Bivariate Analysis...")
    plot_correlation_heatmap(df)
    plot_churn_rate_by_segment(df)
    plot_violin_plots(df)
    plot_tenure_cohort_churn(df)
    plot_revenue_vs_churn(df)
    print("[EDA] Bivariate complete.\n")


if __name__ == "__main__":
    from config import DATA_ENGINEERED
    df = pd.read_csv(DATA_ENGINEERED)
    run_bivariate(df)
