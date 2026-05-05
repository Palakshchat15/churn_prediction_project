# =============================================================================
# src/eda/report.py
# Generates a comprehensive auto-EDA HTML report using ydata-profiling.
# =============================================================================

import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DATA_ENGINEERED, EDA_REPORT


def generate_profile_report(df: pd.DataFrame = None, output_path: str = EDA_REPORT):
    """
    Generate a full ydata-profiling HTML report.

    Args:
        df: cleaned + engineered DataFrame. If None, loads from DATA_ENGINEERED.
        output_path: path to save HTML report.
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("  ydata-profiling not installed. Run: pip install ydata-profiling")
        return

    if df is None:
        df = pd.read_csv(DATA_ENGINEERED)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Generating EDA profile report on {df.shape[0]:,} rows × {df.shape[1]} columns...")
    print("  This may take 2-5 minutes for a dataset this size...")

    profile = ProfileReport(
        df,
        title="Cell2Cell Customer Churn — EDA Report",
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": True},
        },
        missing_diagrams={
            "bar": True,
            "matrix": True,
            "heatmap": True,
        },
        interactions={"continuous": True},
        n_obs_unique=10,
        vars={"num": {"low_categorical_threshold": 5}},
        progress_bar=True,
    )

    profile.to_file(output_path)
    print(f"  Profile report saved: {output_path}")
    return profile


if __name__ == "__main__":
    generate_profile_report()
