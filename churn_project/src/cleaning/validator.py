# =============================================================================
# src/cleaning/validator.py
# Post-cleaning validation checks. Run after DataCleaner to confirm quality.
# =============================================================================

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import TARGET_COL


def validate_cleaned_data(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Run a suite of quality checks on the cleaned DataFrame.

    Checks:
        1. No null values remain
        2. Target column is binary 0/1
        3. All columns are numeric (no remaining object dtypes)
        4. No duplicate rows
        5. No constant columns (zero variance)
        6. Target class distribution sanity check

    Returns:
        True if all checks pass, False otherwise.
    """
    passed = True
    results = []

    def check(name, condition, detail=""):
        nonlocal passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            passed = False
        results.append((status, name, detail))

    # 1. No nulls
    null_total = df.isnull().sum().sum()
    check("No null values", null_total == 0, f"{null_total} nulls found")

    # 2. Target is binary
    unique_target = set(df[TARGET_COL].unique())
    check("Target is binary 0/1", unique_target <= {0, 1},
          f"Unique values: {unique_target}")

    # 3. All numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    check("All columns numeric", len(non_numeric) == 0,
          f"Non-numeric cols: {non_numeric}")

    # 4. No duplicates
    n_dups = df.duplicated().sum()
    check("No duplicate rows", n_dups == 0, f"{n_dups} duplicate rows")

    # 5. No constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    check("No constant columns", len(constant_cols) == 0,
          f"Constant cols: {constant_cols}")

    # 6. Class distribution sanity
    churn_rate = df[TARGET_COL].mean()
    check("Churn rate in [0.05, 0.60]", 0.05 <= churn_rate <= 0.60,
          f"Churn rate = {churn_rate:.2%}")

    if verbose:
        print("\n" + "=" * 55)
        print("  DATA VALIDATION REPORT")
        print("=" * 55)
        for status, name, detail in results:
            icon = "✓" if status == "PASS" else "✗"
            print(f"  [{status}] {icon}  {name}")
            if detail:
                print(f"            → {detail}")
        print("=" * 55)
        if passed:
            print("  ALL CHECKS PASSED — dataset is clean and ready.\n")
        else:
            print("  SOME CHECKS FAILED — review above before proceeding.\n")

    return passed


if __name__ == "__main__":
    from config import DATA_CLEANED
    df = pd.read_csv(DATA_CLEANED)
    validate_cleaned_data(df)
