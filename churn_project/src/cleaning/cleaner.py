# =============================================================================
# src/cleaning/cleaner.py
# Main DataCleaner class — handles all cleaning steps for Cell2Cell dataset.
# CustomerID is KEPT in cleaned CSV for RAG customer lookup.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DATA_RAW, DATA_CLEANED,
    BINARY_COLS, ORDINAL_COLS, ONEHOT_COLS,
    HOMEOWNERSHIP_MAP, MARITAL_MAP,
    MEDIAN_IMPUTE_COLS, KNN_IMPUTE_COLS, MODE_IMPUTE_COLS,
    WINSORIZE_COLS, DROP_COLS, TARGET_COL, ID_COL,
)


class DataCleaner:
    """
    End-to-end cleaning pipeline for the Cell2Cell churn dataset.

    Steps (in order):
        1.  Load raw CSV
        2.  Encode target column (Churn: Yes->1, No->0)
        3.  Fix HandsetPrice (string->numeric, 'Unknown'->NaN)
        4.  Median imputation for numeric columns with small missingness
        5.  KNN imputation for AgeHH1 / AgeHH2
        6.  Mode imputation for ServiceArea
        7.  Drop single-row nulls (Handsets, HandsetModels, CurrentEquipmentDays)
        8.  Winsorize outliers (cap at 99th percentile)
        9.  Encode binary Yes/No columns -> 0/1
        10. Encode ordinal CreditRating, Homeownership, MaritalStatus
        11. Drop redundant columns (NOT CustomerID — kept for RAG lookup)
        12. Save cleaned CSV
    """

    def __init__(self, input_path=DATA_RAW, output_path=DATA_CLEANED):
        self.input_path  = input_path
        self.output_path = output_path
        self.df          = None
        self.cleaning_log = []

    # --------------------------------------------------------------------------
    def load(self):
        """Load raw dataset from CSV."""
        self.df = pd.read_csv(self.input_path)
        self._log(f"Loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self

    # --------------------------------------------------------------------------
    def encode_target(self):
        """Convert Churn 'Yes'/'No' -> 1/0 integer."""
        self.df[TARGET_COL] = self.df[TARGET_COL].map({"Yes": 1, "No": 0})
        churn_rate    = self.df[TARGET_COL].mean()
        churned_count = self.df[TARGET_COL].sum()
        retained      = len(self.df) - churned_count
        self._log(
            f"Target encoded. Churn rate: {churn_rate:.2%} "
            f"({churned_count:,} churned / {retained:,} retained)"
        )
        return self

    # --------------------------------------------------------------------------
    def fix_handset_price(self):
        """
        HandsetPrice is stored as object with values like '30', '150', 'Unknown'.
        Replace 'Unknown' with NaN, cast to float, then median-impute.
        """
        self.df["HandsetPrice"] = pd.to_numeric(
            self.df["HandsetPrice"].replace("Unknown", np.nan), errors="coerce"
        )
        median_val = self.df["HandsetPrice"].median()
        n_missing  = self.df["HandsetPrice"].isna().sum()
        self.df["HandsetPrice"] = self.df["HandsetPrice"].fillna(median_val)
        self._log(
            f"HandsetPrice: fixed type, imputed {n_missing:,} "
            f"'Unknown' values with median={median_val}"
        )
        return self

    # --------------------------------------------------------------------------
    def impute_median(self):
        """Median imputation for numeric columns with small-to-medium missingness."""
        for col in MEDIAN_IMPUTE_COLS:
            if col not in self.df.columns:
                continue
            n = self.df[col].isna().sum()
            if n > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                self._log(
                    f"Median imputed '{col}': {n:,} values -> {median_val:.4f}"
                )
        return self

    # --------------------------------------------------------------------------
    def impute_knn(self, n_neighbors=5):
        """
        KNN imputation for AgeHH1 and AgeHH2.
        Uses correlated numeric features as context (MonthsInService, etc.)
        Only runs KNN on the relevant subset to keep it fast.
        """
        knn_cols = [c for c in KNN_IMPUTE_COLS if c in self.df.columns]
        if not knn_cols:
            return self

        context_cols = [
            "MonthsInService", "IncomeGroup", "MonthlyRevenue",
            "ActiveSubs", "UniqueSubs",
        ] + knn_cols
        context_cols = [c for c in context_cols if c in self.df.columns]

        imputer = KNNImputer(n_neighbors=n_neighbors)
        subset  = self.df[context_cols].copy()
        imputed = imputer.fit_transform(subset)
        imputed_df = pd.DataFrame(imputed, columns=context_cols,
                                  index=self.df.index)

        for col in knn_cols:
            n = self.df[col].isna().sum()
            self.df[col] = imputed_df[col]
            self._log(
                f"KNN imputed '{col}': {n:,} missing values (k={n_neighbors})"
            )
        return self

    # --------------------------------------------------------------------------
    def impute_mode(self):
        """Mode imputation for ServiceArea."""
        for col in MODE_IMPUTE_COLS:
            if col not in self.df.columns:
                continue
            n = self.df[col].isna().sum()
            if n > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                self._log(f"Mode imputed '{col}': {n:,} values -> '{mode_val}'")
        return self

    # --------------------------------------------------------------------------
    def drop_single_null_rows(self):
        """
        Handsets, HandsetModels, CurrentEquipmentDays each have exactly 1 null.
        Dropping 1 row from 51,047 is negligible.
        """
        cols_with_one_null = ["Handsets", "HandsetModels", "CurrentEquipmentDays"]
        cols_present = [c for c in cols_with_one_null if c in self.df.columns]
        before  = len(self.df)
        self.df = self.df.dropna(subset=cols_present)
        dropped = before - len(self.df)
        self._log(f"Dropped {dropped} row(s) with single nulls in: {cols_present}")
        return self

    # --------------------------------------------------------------------------
    def winsorize(self, percentile=99):
        """
        Cap outliers at the specified percentile.
        Also clips PercChange columns to a fixed sensible range.
        """
        for col in WINSORIZE_COLS:
            if col not in self.df.columns:
                continue
            cap      = self.df[col].quantile(percentile / 100)
            n_capped = (self.df[col] > cap).sum()
            self.df[col] = self.df[col].clip(upper=cap)
            self._log(
                f"Winsorized '{col}' at {percentile}th pct "
                f"({cap:.2f}): {n_capped:,} values capped"
            )

        for col in ["PercChangeMinutes", "PercChangeRevenues"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(-500, 500)
                self._log(f"Clipped '{col}' to [-500, 500]")
        return self

    # --------------------------------------------------------------------------
    def encode_binary(self):
        """Encode all Yes/No binary columns to 1/0."""
        for col in BINARY_COLS:
            if col not in self.df.columns:
                continue
            self.df[col] = self.df[col].map({"Yes": 1, "No": 0})
            self._log(f"Binary encoded: '{col}'")
        return self

    # --------------------------------------------------------------------------
    def encode_ordinal(self):
        """Map ordinal string columns to ordered integers."""
        for col, mapping in ORDINAL_COLS.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
                self._log(f"Ordinal encoded: '{col}'")

        if "Homeownership" in self.df.columns:
            self.df["Homeownership"] = self.df["Homeownership"].map(
                HOMEOWNERSHIP_MAP
            )
            self._log("Encoded: 'Homeownership' -> Known=1, Unknown=0")

        if "MaritalStatus" in self.df.columns:
            self.df["MaritalStatus"] = self.df["MaritalStatus"].map(MARITAL_MAP)
            self._log("Encoded: 'MaritalStatus' -> Yes=1, No=0, Unknown=-1")
        return self

    # --------------------------------------------------------------------------
    def drop_columns(self):
        """
        Remove redundant and high-cardinality columns.

        CustomerID is intentionally KEPT — it is used as a customer identifier
        in the RAG vector store so the LLM can reference specific customers.
        It is excluded from model features in trainer.py.

        Dropped: NotNewCellphoneUser (inverse of NewCellphoneUser),
                 ServiceArea (747 unique values — frequency-encoded in FE step).
        """
        cols_to_drop = [
            c for c in DROP_COLS
            if c in self.df.columns and c != "CustomerID"
        ]
        self.df = self.df.drop(columns=cols_to_drop)
        self._log(f"Dropped columns: {cols_to_drop}")
        self._log("CustomerID retained for RAG customer identification.")
        return self

    # --------------------------------------------------------------------------
    def reorder_columns(self):
        """Move CustomerID and Churn to the front for readability."""
        priority = [c for c in ["CustomerID", TARGET_COL]
                    if c in self.df.columns]
        rest     = [c for c in self.df.columns if c not in priority]
        self.df  = self.df[priority + rest]
        return self

    # --------------------------------------------------------------------------
    def save(self):
        """Save cleaned dataframe to processed CSV."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        self._log(
            f"Saved cleaned data: {self.output_path}  ->  "
            f"{self.df.shape[0]:,} rows x {self.df.shape[1]} columns"
        )
        return self

    # --------------------------------------------------------------------------
    def run(self):
        """Run the full cleaning pipeline end-to-end."""
        print("=" * 60)
        print("  CELL2CELL DATA CLEANING PIPELINE")
        print("=" * 60)
        (
            self.load()
                .encode_target()
                .fix_handset_price()
                .impute_median()
                .impute_knn()
                .impute_mode()
                .drop_single_null_rows()
                .winsorize()
                .encode_binary()
                .encode_ordinal()
                .drop_columns()
                .reorder_columns()
                .save()
        )
        print("\n  CLEANING LOG:")
        for entry in self.cleaning_log:
            print(f"  +  {entry}")

        print("\n  NULL CHECK:")
        null_counts = self.df.isnull().sum()
        remaining   = null_counts[null_counts > 0]
        if remaining.empty:
            print("  + No nulls remaining — dataset is clean.")
        else:
            print(remaining)

        print("\n  COLUMNS KEPT:")
        print(f"  CustomerID present : {'CustomerID' in self.df.columns}")
        print(f"  Total columns      : {self.df.shape[1]}")
        print("=" * 60)
        return self.df

    # --------------------------------------------------------------------------
    def _log(self, message):
        self.cleaning_log.append(message)


if __name__ == "__main__":
    cleaner  = DataCleaner()
    df_clean = cleaner.run()
    print(f"\nFirst 3 rows:\n{df_clean[['CustomerID', 'Churn']].head(3)}")