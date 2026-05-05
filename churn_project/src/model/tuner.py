# =============================================================================
# src/model/tuner.py
# Optuna Bayesian HPO for XGBoost and LightGBM.
# Finds best params via cross-validation, then retrains final models
# on the full training set and saves them — replacing the old .pkl files.
# =============================================================================

import pandas as pd
import numpy as np
import pickle, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DATA_ENGINEERED, TARGET_COL, MODELS_DIR,
    XGBOOST_PATH, LIGHTGBM_PATH,
    RANDOM_STATE, OPTUNA_TRIALS, OPTUNA_CV_FOLDS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data():
    """Load engineered data and return train/val/test splits."""
    df = pd.read_csv(DATA_ENGINEERED)
    y  = df[TARGET_COL].astype(int)

    drop_cols = [TARGET_COL]
    if "CustomerID" in df.columns:
        drop_cols.append("CustomerID")
    X = df.drop(columns=drop_cols)

    # Fill any residual NaNs
    for col in X.columns[X.isnull().any()]:
        X[col] = X[col].fillna(X[col].median())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"[TUNE] Data loaded — Train: {len(X_train):,}  "
          f"Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── XGBoost ───────────────────────────────────────────────────────────────────
def tune_xgboost(X_train, y_train, X_val, y_val,
                 n_trials=OPTUNA_TRIALS) -> dict:
    try:
        import optuna
        from xgboost import XGBClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("pip install optuna xgboost")
        return {}

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 20),
            "gamma":           trial.suggest_float("gamma", 0, 5),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1, 10),
            "scale_pos_weight": 36336 / 14711,
            "random_state":    RANDOM_STATE,
            "verbosity":       0,
            "eval_metric":     "auc",
        }
        cv     = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
        scores = cross_val_score(XGBClassifier(**params), X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    print(f"\n[TUNE] XGBoost — {n_trials} trials x {OPTUNA_CV_FOLDS}-fold CV...")
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_churn",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[TUNE] XGBoost CV best AUC : {study.best_value:.4f}")
    print(f"       Best params         : {best}")

    # ── Retrain final model on full training set with best params ──────────
    print("\n[TUNE] Retraining XGBoost with best params + early stopping...")
    from xgboost import XGBClassifier
    final_model = XGBClassifier(
        **best,
        scale_pos_weight=36336 / 14711,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on val set
    val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    print(f"[TUNE] XGBoost final val AUC : {val_auc:.4f}")

    # Save — overwrites the old model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(XGBOOST_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"[TUNE] XGBoost saved         : {XGBOOST_PATH}")

    # Also save best params for reference
    with open(os.path.join(MODELS_DIR, "xgboost_best_params.pkl"), "wb") as f:
        pickle.dump(best, f)

    return best


# ── LightGBM ──────────────────────────────────────────────────────────────────
def tune_lightgbm(X_train, y_train, X_val, y_val,
                  n_trials=OPTUNA_TRIALS) -> dict:
    try:
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("pip install optuna lightgbm")
        return {}

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
            "class_weight":      "balanced",
            "metric":            "auc",
            "random_state":      RANDOM_STATE,
            "verbose":           -1,
        }
        cv     = StratifiedKFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
        scores = cross_val_score(lgb.LGBMClassifier(**params), X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    print(f"\n[TUNE] LightGBM — {n_trials} trials x {OPTUNA_CV_FOLDS}-fold CV...")
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_churn",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n[TUNE] LightGBM CV best AUC : {study.best_value:.4f}")
    print(f"       Best params          : {best}")

    # ── Retrain final model on full training set with best params ──────────
    print("\n[TUNE] Retraining LightGBM with best params + early stopping...")
    final_model = lgb.LGBMClassifier(
        **best,
        class_weight="balanced",
        metric="auc",
        random_state=RANDOM_STATE,
        verbose=-1,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    print(f"[TUNE] LightGBM final val AUC : {val_auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LIGHTGBM_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"[TUNE] LightGBM saved          : {LIGHTGBM_PATH}")

    with open(os.path.join(MODELS_DIR, "lightgbm_best_params.pkl"), "wb") as f:
        pickle.dump(best, f)

    return best


# ── Re-evaluate after tuning ──────────────────────────────────────────────────
def evaluate_tuned_models(X_test, y_test):
    """
    Quick evaluation of the newly saved tuned models on the test set.
    Prints ROC-AUC and classification report.
    """
    from sklearn.metrics import (
        roc_auc_score, classification_report,
        average_precision_score, f1_score,
    )

    print("\n" + "=" * 55)
    print("  POST-TUNING EVALUATION ON TEST SET")
    print("=" * 55)

    for name, path in [("XGBoost", XGBOOST_PATH),
                        ("LightGBM", LIGHTGBM_PATH)]:
        if not os.path.exists(path):
            print(f"[EVAL] {name} model not found at {path}")
            continue

        with open(path, "rb") as f:
            model = pickle.load(f)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc    = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1     = f1_score(y_test, y_pred, average="macro")
        f1_c   = f1_score(y_test, y_pred, pos_label=1)

        print(f"\n  Model : {name}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  PR-AUC   : {pr_auc:.4f}")
        print(f"  F1 Macro : {f1:.4f}")
        print(f"  F1 Churn : {f1_c:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Not Churned','Churned'])}")

    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  OPTUNA HYPERPARAMETER TUNING PIPELINE")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    tune_xgboost(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS)
    tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS)

    evaluate_tuned_models(X_test, y_test)

    print("\n[TUNE] Done. Tuned models saved and ready.")
    print("       Run main.py --skip-eda --skip-train to rebuild RAG index.")