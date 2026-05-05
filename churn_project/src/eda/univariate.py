# =============================================================================
# src/eda/univariate.py
# Univariate analysis: distributions, skewness, categorical frequencies,
# class balance. Saves all plots to outputs/plots/.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TARGET_COL, PLOTS_DIR

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Blues_d")
PALETTE   = ["#2563EB", "#EF4444"]
BLUE      = "#2563EB"
DARK      = "#1B3A6B"
GRAY      = "#475569"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_churn_balance(df):
    """Bar + pie chart showing class imbalance of the target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Distribution — Churn vs Not Churn", fontsize=15, fontweight="bold", color=DARK)

    counts = df[TARGET_COL].value_counts()
    labels = ["Not Churned (0)", "Churned (1)"]
    colors = ["#2563EB", "#EF4444"]

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Count by Class", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                     f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=11, fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportion", fontweight="bold")

    plt.tight_layout()
    _save(fig, "01_churn_balance.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_numeric_distributions(df):
    """
    Histogram + KDE for all numeric columns (excluding target).
    Plotted in a grid, coloured by churn.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    ncols = 4
    nrows = (len(numeric_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5))
    fig.suptitle("Numeric Feature Distributions", fontsize=16, fontweight="bold", color=DARK, y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for val, color, label in [(0, BLUE, "Not Churned"), (1, "#EF4444", "Churned")]:
            subset = df[df[TARGET_COL] == val][col].dropna()
            ax.hist(subset, bins=40, alpha=0.5, color=color, label=label, density=True)
            # KDE overlay
            if len(subset) > 10:
                kde_x = np.linspace(subset.min(), subset.max(), 200)
                try:
                    kde = stats.gaussian_kde(subset)
                    ax.plot(kde_x, kde(kde_x), color=color, linewidth=1.5)
                except Exception:
                    pass
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "02_numeric_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_skewness(df):
    """Horizontal bar chart showing skewness of all numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    skew_vals = df[numeric_cols].skew().sort_values(ascending=False)
    colors = ["#EF4444" if abs(v) > 1 else "#F59E0B" if abs(v) > 0.5 else BLUE
              for v in skew_vals]

    fig, ax = plt.subplots(figsize=(10, len(skew_vals) * 0.35 + 2))
    ax.barh(skew_vals.index, skew_vals.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(1, color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.7, label="|skew| > 1 (high)")
    ax.axvline(-1, color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_title("Feature Skewness (|skew| > 1 = high → consider log transform)",
                 fontsize=13, fontweight="bold", color=DARK)
    ax.set_xlabel("Skewness")
    ax.legend()
    plt.tight_layout()
    _save(fig, "03_skewness.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_boxplots(df):
    """Boxplots for key numeric columns, split by Churn."""
    key_cols = [
        "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
        "CurrentEquipmentDays", "MonthsInService", "CustomerCareCalls",
        "RetentionCalls", "OverageMinutes",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Key Feature Boxplots by Churn Status", fontsize=15, fontweight="bold", color=DARK)
    axes = axes.flatten()

    for i, col in enumerate(key_cols):
        ax = axes[i]
        data = [df[df[TARGET_COL] == 0][col].dropna(),
                df[df[TARGET_COL] == 1][col].dropna()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "white", "linewidth": 2})
        colors = [BLUE, "#EF4444"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(["Not Churned", "Churned"])
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel(col, fontsize=8)

    plt.tight_layout()
    _save(fig, "04_boxplots.png")


# ─────────────────────────────────────────────────────────────────────────────
def plot_categorical_frequencies(df):
    """Bar charts for key categorical (encoded) columns."""
    cat_cols = {
        "CreditRating": "Credit Rating (1=Highest, 6=VeryLow)",
        "PrizmCode_Rural": "Prizm: Rural",
        "PrizmCode_Suburban": "Prizm: Suburban",
        "Occupation_Professional": "Occupation: Professional",
        "MaritalStatus": "Marital Status (-1=Unknown, 0=No, 1=Yes)",
        "IncomeGroup": "Income Group",
        "Handsets": "Number of Handsets",
    }

    available = {k: v for k, v in cat_cols.items() if k in df.columns}
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Categorical Feature Frequencies by Churn", fontsize=15, fontweight="bold", color=DARK)
    axes = axes.flatten()

    for i, (col, title) in enumerate(available.items()):
        ax = axes[i]
        grouped = df.groupby(col)[TARGET_COL].agg(["sum", "count"]).reset_index()
        grouped["churn_rate"] = grouped["sum"] / grouped["count"]
        grouped[col] = grouped[col].astype(str)

        bars = ax.bar(grouped[col], grouped["count"], color=BLUE, alpha=0.6, label="Total")
        ax2 = ax.twinx()
        ax2.plot(grouped[col], grouped["churn_rate"], color="#EF4444",
                 marker="o", linewidth=2, label="Churn Rate")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Churn Rate", color="#EF4444", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#EF4444")

        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "05_categorical_frequencies.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_univariate(df: pd.DataFrame):
    """Run all univariate analyses."""
    print("\n[EDA] Running Univariate Analysis...")
    plot_churn_balance(df)
    plot_numeric_distributions(df)
    plot_skewness(df)
    plot_boxplots(df)
    plot_categorical_frequencies(df)
    print("[EDA] Univariate complete.\n")


if __name__ == "__main__":
    from config import DATA_ENGINEERED
    df = pd.read_csv(DATA_ENGINEERED)
    run_univariate(df)
