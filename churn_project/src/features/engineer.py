# =============================================================================
# src/features/engineer.py
# Feature engineering pipeline — creates new predictive features,
# encodes categoricals, frequency-encodes high-cardinality columns.
# CustomerID flows through unchanged for RAG lookup.
# =============================================================================

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import DATA_CLEANED, DATA_ENGINEERED, TARGET_COL, ONEHOT_COLS


class FeatureEngineer:
    """
    Transforms the cleaned Cell2Cell dataset by:
        1.  Creating 10 new interaction/ratio features
        2.  Frequency-encoding ServiceArea (747 unique values)
        3.  One-hot encoding PrizmCode and Occupation
        4.  Creating tenure buckets
        5.  Final NaN sweep
        6.  Saving — CustomerID always placed as first column
    """

    def __init__(self, input_path=DATA_CLEANED, output_path=DATA_ENGINEERED):
        self.input_path  = input_path
        self.output_path = output_path
        self.df          = None
        self._service_area_freq_map = {}

    # --------------------------------------------------------------------------
    def load(self):
        self.df = pd.read_csv(self.input_path)
        has_id  = "CustomerID" in self.df.columns
        print(f"[FE] Loaded cleaned data: {self.df.shape[0]:,} rows x "
              f"{self.df.shape[1]} cols  (CustomerID present: {has_id})")
        return self

    # --------------------------------------------------------------------------
    def create_ratio_features(self):
        """
        Create call quality, usage efficiency and engagement ratio features.
        All denominators get +1 to avoid division-by-zero.
        CustomerID is skipped automatically (not numeric).
        """
        df = self.df

        df["DropRate"]     = df["DroppedCalls"]   / (df["MonthlyMinutes"] + 1)
        df["BlockRate"]    = df["BlockedCalls"]   / (df["MonthlyMinutes"] + 1)
        df["OverageRatio"] = df["OverageMinutes"] / (df["MonthlyMinutes"] + 1)

        df["RevenuePerMinute"] = df["MonthlyRevenue"] / (df["MonthlyMinutes"] + 1)
        df["CareCareRatio"]    = df["CustomerCareCalls"] / (df["MonthlyMinutes"] + 1)

        df["RetentionEngagement"] = (
            df["RetentionCalls"] * df["RetentionOffersAccepted"]
        )
        df["TotalCallVolume"]  = df["PeakCallsInOut"] + df["OffPeakCallsInOut"]
        df["EquipmentAgeYears"]= df["CurrentEquipmentDays"] / 365

        if "AgeHH1" in df.columns and "AgeHH2" in df.columns:
            df["AvgHouseholdAge"] = df[["AgeHH1", "AgeHH2"]].mean(axis=1)

        if "TotalRecurringCharge" in df.columns:
            df["EffectiveOverspend"] = (
                df["MonthlyRevenue"] - df["TotalRecurringCharge"]
            )

        print("[FE] Created 10 ratio/interaction features")
        self.df = df
        return self

    # --------------------------------------------------------------------------
    def create_tenure_buckets(self):
        """Bin MonthsInService into lifecycle stage integers."""
        if "MonthsInService" not in self.df.columns:
            return self

        bins   = [0, 12, 24, 48, float("inf")]
        labels = [0, 1, 2, 3]
        self.df["TenureBucket"] = pd.cut(
            self.df["MonthsInService"],
            bins=bins, labels=labels, right=False,
        ).astype(int)
        print("[FE] Created 'TenureBucket' (0=New -> 3=Loyal)")
        return self

    # --------------------------------------------------------------------------
    def encode_service_area(self):
        """Frequency-encode ServiceArea (747 unique values)."""
        if "ServiceArea" not in self.df.columns:
            return self

        freq_map = self.df["ServiceArea"].value_counts(normalize=True).to_dict()
        self._service_area_freq_map = freq_map
        self.df["ServiceArea_Freq"] = (
            self.df["ServiceArea"].map(freq_map).fillna(0)
        )
        self.df = self.df.drop(columns=["ServiceArea"])
        print("[FE] Frequency-encoded 'ServiceArea' -> 'ServiceArea_Freq'")
        return self

    # --------------------------------------------------------------------------
    def one_hot_encode(self):
        """One-hot encode low-cardinality nominal columns (PrizmCode, Occupation)."""
        cols = [c for c in ONEHOT_COLS if c in self.df.columns]
        if not cols:
            return self
        self.df = pd.get_dummies(self.df, columns=cols, drop_first=False,
                                 dtype=int)
        print(f"[FE] One-hot encoded: {cols}")
        return self

    # --------------------------------------------------------------------------
    def clip_engineered_features(self):
        """Cap ratio features at 99th percentile to handle outliers."""
        ratio_cols = [
            "DropRate", "BlockRate", "OverageRatio",
            "RevenuePerMinute", "CareCareRatio",
        ]
        for col in ratio_cols:
            if col in self.df.columns:
                cap = self.df[col].quantile(0.99)
                self.df[col] = self.df[col].clip(upper=cap)
        print("[FE] Clipped ratio features at 99th percentile")
        return self

    # --------------------------------------------------------------------------
    def fill_remaining_nans(self):
        """
        Final safety sweep — fill any NaNs introduced during feature derivation.
        Skips CustomerID (string/int identifier, should never be NaN).
        Uses median fill on numeric columns, 0 on everything else.
        """
        # Only check non-ID, non-target columns
        skip_cols   = {"CustomerID", TARGET_COL}
        feature_cols= [c for c in self.df.columns if c not in skip_cols]
        nan_counts  = self.df[feature_cols].isnull().sum()
        nan_cols    = nan_counts[nan_counts > 0]

        if nan_cols.empty:
            print("[FE] Final NaN check passed — no missing values.")
            return self

        print(f"[FE] Final NaN fill on {len(nan_cols)} column(s):")
        for col, cnt in nan_cols.items():
            if self.df[col].dtype in [np.float64, np.float32,
                                       np.int64, np.int32]:
                fill_val = self.df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0.0
            else:
                fill_val = 0
            self.df[col] = self.df[col].fillna(fill_val)
            print(f"     '{col}': {cnt} NaNs filled with {fill_val:.4f}")
        return self

    # --------------------------------------------------------------------------
    def save(self):
        """
        Save engineered dataframe.
        CustomerID is always placed as the first column for easy lookup.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Put CustomerID first, then Churn, then all features
        priority = [c for c in ["CustomerID", TARGET_COL]
                    if c in self.df.columns]
        rest     = [c for c in self.df.columns if c not in priority]
        self.df  = self.df[priority + rest]

        self.df.to_csv(self.output_path, index=False)
        print(f"[FE] Saved engineered data: {self.output_path}")
        print(f"     Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} features")
        print(f"     CustomerID present: {'CustomerID' in self.df.columns}")

        remaining = self.df.drop(
            columns=[c for c in ["CustomerID"] if c in self.df.columns]
        ).isnull().sum().sum()
        if remaining == 0:
            print("[FE] Saved file is NaN-free (excluding CustomerID).")
        else:
            print(f"[FE] WARNING: {remaining} NaNs still in saved file.")
        return self

    # --------------------------------------------------------------------------
    def run(self):
        """Run the full feature engineering pipeline end-to-end."""
        print("\n" + "=" * 55)
        print("  FEATURE ENGINEERING PIPELINE")
        print("=" * 55)
        (
            self.load()
                .create_ratio_features()
                .create_tenure_buckets()
                .encode_service_area()
                .one_hot_encode()
                .clip_engineered_features()
                .fill_remaining_nans()
                .save()
        )
        print("=" * 55)
        return self.df


if __name__ == "__main__":
    fe = FeatureEngineer()
    df = fe.run()
    print(f"\nFirst 3 rows (ID + target):\n{df[['CustomerID','Churn']].head(3)}")