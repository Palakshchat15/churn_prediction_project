# =============================================================================
# src/model/trainer.py
# Full model training pipeline.
# CustomerID is excluded from features but kept in the dataframe for reference.
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DATA_ENGINEERED, TARGET_COL, MODELS_DIR,
    XGBOOST_PATH, LIGHTGBM_PATH,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_STATE,
    XGB_PARAMS, LGB_PARAMS,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """
    Trains XGBoost, LightGBM, and Logistic Regression on Cell2Cell.

    CustomerID is excluded from model features (not predictive) but the
    split indices are preserved so SHAP explanations can be joined back
    to customer IDs for RAG indexing.
    """

    def __init__(self, data_path=DATA_ENGINEERED):
        self.data_path         = data_path
        self.df                = None
        self.X                 = None
        self.y                 = None
        self.X_train           = None
        self.X_val             = None
        self.X_test            = None
        self.y_train           = None
        self.y_val             = None
        self.y_test            = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.models            = {}
        self.feature_names     = []
        self._col_medians      = {}

    # --------------------------------------------------------------------------
    def load(self):
        self.df = pd.read_csv(self.data_path)
        self.y  = self.df[TARGET_COL].astype(int)

        # Exclude both target and CustomerID from features
        non_feature_cols = [TARGET_COL]
        if "CustomerID" in self.df.columns:
            non_feature_cols.append("CustomerID")

        self.X = self.df.drop(columns=non_feature_cols)
        self.feature_names = list(self.X.columns)

        print(f"[TRAIN] Loaded: {self.X.shape[0]:,} rows x "
              f"{self.X.shape[1]} features")
        print(f"        Churn rate    : {self.y.mean():.2%}")
        print(f"        CustomerID    : "
              f"{'present in df (excluded from X)' if 'CustomerID' in self.df.columns else 'not found'}")

        total_nans = self.X.isnull().sum().sum()
        if total_nans > 0:
            nan_summary = self.X.isnull().sum()
            nan_cols    = nan_summary[nan_summary > 0]
            print(f"[TRAIN] WARNING: {total_nans} NaNs in "
                  f"{len(nan_cols)} column(s) — will fix before SMOTE.")
        return self

    # --------------------------------------------------------------------------
    def split(self):
        """
        Stratified 70/15/15 train/val/test split.
        Uses df index so CustomerID can be retrieved later via index alignment.
        """
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=self.y,
        )
        val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
        print(f"[TRAIN] Split -> Train: {len(self.X_train):,}  "
              f"Val: {len(self.X_val):,}  Test: {len(self.X_test):,}")
        print(f"        Churn  -> Train: {self.y_train.mean():.2%}  "
              f"Val: {self.y_val.mean():.2%}  Test: {self.y_test.mean():.2%}")
        return self

    # --------------------------------------------------------------------------
    def clean_remaining_nans(self):
        """
        Fill any remaining NaNs before SMOTE using train-set medians only.
        Prevents data leakage from val/test into training statistics.
        """
        nan_cols = self.X_train.columns[self.X_train.isnull().any()].tolist()

        if not nan_cols:
            print("[TRAIN] NaN check passed — no missing values.")
            return self

        print(f"[TRAIN] Filling NaNs in {len(nan_cols)} column(s) "
              f"with train medians:")
        for col in nan_cols:
            median_val = self.X_train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            self._col_medians[col] = median_val

            n_train = self.X_train[col].isna().sum()
            n_val   = self.X_val[col].isna().sum()
            n_test  = self.X_test[col].isna().sum()

            self.X_train = self.X_train.copy()
            self.X_val   = self.X_val.copy()
            self.X_test  = self.X_test.copy()

            self.X_train[col] = self.X_train[col].fillna(median_val)
            self.X_val[col]   = self.X_val[col].fillna(median_val)
            self.X_test[col]  = self.X_test[col].fillna(median_val)

            print(f"     '{col}': train={n_train} / val={n_val} / "
                  f"test={n_test} NaNs  (median={median_val:.4f})")

        remaining = self.X_train.isnull().sum().sum()
        if remaining == 0:
            print("[TRAIN] All NaNs resolved — clean for SMOTE.")
        else:
            raise ValueError(
                f"[TRAIN] FATAL: {remaining} NaNs remain after fill."
            )
        return self

    # --------------------------------------------------------------------------
    def apply_smote(self):
        """SMOTE on training set only — never on val or test."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            print("[TRAIN] imbalanced-learn not installed — skipping SMOTE.")
            self.X_train_resampled = self.X_train
            self.y_train_resampled = self.y_train
            return self

        print(f"[TRAIN] SMOTE  (before: {len(self.X_train):,} rows, "
              f"churn={self.y_train.mean():.2%})...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        print(f"[TRAIN] SMOTE done -> {len(self.X_train_resampled):,} rows, "
              f"churn={self.y_train_resampled.mean():.2%}")
        return self

    # --------------------------------------------------------------------------
    def train_logistic_regression(self):
        from sklearn.pipeline import Pipeline
        print("[TRAIN] Training Logistic Regression...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE, C=0.1, solver="lbfgs",
            ))
        ])
        X_tr = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_tr = self.y_train_resampled if self.y_train_resampled is not None else self.y_train
        pipe.fit(X_tr, y_tr)
        self.models["logistic_regression"] = pipe
        print("        Logistic Regression done.")
        return self

    # --------------------------------------------------------------------------
    def train_xgboost(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("[TRAIN] xgboost not installed — skipping.")
            return self

        print("[TRAIN] Training XGBoost...")
        X_tr = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_tr = self.y_train_resampled if self.y_train_resampled is not None else self.y_train

        xgb = XGBClassifier(
            **XGB_PARAMS, early_stopping_rounds=50, verbosity=0,
        )
        xgb.fit(X_tr, y_tr,
                eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.models["xgboost"] = xgb
        print(f"        XGBoost done — best iter: {xgb.best_iteration}  "
              f"val AUC: {xgb.best_score:.4f}")
        return self

    # --------------------------------------------------------------------------
    def train_lightgbm(self):
        try:
            import lightgbm as lgb
        except ImportError:
            print("[TRAIN] lightgbm not installed — skipping.")
            return self

        print("[TRAIN] Training LightGBM...")
        X_tr = self.X_train_resampled if self.X_train_resampled is not None else self.X_train
        y_tr = self.y_train_resampled if self.y_train_resampled is not None else self.y_train

        lgbm = lgb.LGBMClassifier(
            **LGB_PARAMS,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        lgbm.fit(X_tr, y_tr, eval_set=[(self.X_val, self.y_val)])
        self.models["lightgbm"] = lgbm
        print(f"        LightGBM done — best iter: {lgbm.best_iteration_}")
        return self

    # --------------------------------------------------------------------------
    def train_all(self):
        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()
        print(f"\n[TRAIN] Models trained: {list(self.models.keys())}")
        return self

    # --------------------------------------------------------------------------
    def save_all(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path_map = {
            "xgboost":             XGBOOST_PATH,
            "lightgbm":            LIGHTGBM_PATH,
            "logistic_regression": os.path.join(MODELS_DIR,
                                                 "logistic_regression.pkl"),
        }
        for name, model in self.models.items():
            path = path_map.get(name, os.path.join(MODELS_DIR, f"{name}.pkl"))
            with open(path, "wb") as f:
                pickle.dump(model, f)
            print(f"[TRAIN] Saved: {path}")

        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
            pickle.dump(self.feature_names, f)

        with open(os.path.join(MODELS_DIR, "nan_fill_medians.pkl"), "wb") as f:
            pickle.dump(self._col_medians, f)

        print(f"[TRAIN] Feature names + NaN medians saved to: {MODELS_DIR}")
        return self

    # --------------------------------------------------------------------------
    def run(self):
        print("\n" + "=" * 55)
        print("  MODEL TRAINING PIPELINE")
        print("=" * 55)
        (
            self.load()
                .split()
                .clean_remaining_nans()
                .apply_smote()
                .train_all()
                .save_all()
        )
        print("=" * 55)
        return self


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()