# =============================================================================
# config.py
# Central configuration — all paths, constants, and hyperparameters live here.
# Import this module in every other file instead of hard-coding values.
# =============================================================================

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_RAW        = ROOT_DIR / "data" / "raw" / "cell2celltrain.csv"
DATA_CLEANED    = ROOT_DIR / "data" / "processed" / "cell2cell_cleaned.csv"
DATA_ENGINEERED = ROOT_DIR / "data" / "processed" / "cell2cell_engineered.csv"
VECTOR_STORE    = ROOT_DIR / "data" / "vector_store"

# ── Model Paths ───────────────────────────────────────────────────────────────
MODELS_DIR          = ROOT_DIR / "models"
XGBOOST_PATH        = MODELS_DIR / "xgboost_churn.pkl"
LIGHTGBM_PATH       = MODELS_DIR / "lightgbm_churn.pkl"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
EDA_REPORT  = OUTPUTS_DIR / "eda_report.html"
MODEL_REPORT= OUTPUTS_DIR / "model_report.txt"

# ── Target & ID Columns ───────────────────────────────────────────────────────
TARGET_COL = "Churn"
ID_COL     = "CustomerID"

# ── Column Definitions ────────────────────────────────────────────────────────

DROP_COLS = [
    "CustomerID",
    "NotNewCellphoneUser",
    "ServiceArea",
]

BINARY_COLS = [
    "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable",
    "TruckOwner", "RVOwner", "BuysViaMailOrder", "RespondsToMailOffers",
    "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard",
    "NewCellphoneUser", "OwnsMotorcycle", "MadeCallToRetentionTeam",
]

ORDINAL_COLS = {
    "CreditRating": {
        "1-Highest": 1, "2-High": 2, "3-Good": 3,
        "4-Medium": 4, "5-Low": 5, "6-VeryLow": 6,
        "7-Lowest": 7,    # add this line
    }
}

HOMEOWNERSHIP_MAP = {"Known": 1, "Unknown": 0}
MARITAL_MAP       = {"Yes": 1, "No": 0, "Unknown": -1}
ONEHOT_COLS       = ["PrizmCode", "Occupation"]

MEDIAN_IMPUTE_COLS = [
    "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
    "DirectorAssistedCalls", "OverageMinutes", "RoamingCalls",
    "PercChangeMinutes", "PercChangeRevenues",
]

KNN_IMPUTE_COLS  = ["AgeHH1", "AgeHH2"]
MODE_IMPUTE_COLS = ["ServiceArea"]

WINSORIZE_COLS = [
    "MonthlyRevenue", "MonthlyMinutes", "PeakCallsInOut",
    "OffPeakCallsInOut", "CurrentEquipmentDays", "RetentionCalls",
]

# ── Train / Val / Test Split ──────────────────────────────────────────────────
TRAIN_SIZE   = 0.70
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15
RANDOM_STATE = 42

# ── Imbalance ─────────────────────────────────────────────────────────────────
SCALE_POS_WEIGHT = 36336 / 14711  # ~2.47

# ── XGBoost Hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  SCALE_POS_WEIGHT,
    "eval_metric":       "auc",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

# ── LightGBM Hyperparameters ──────────────────────────────────────────────────
LGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "class_weight":     "balanced",
    "metric": "auc",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS   = 100
OPTUNA_CV_FOLDS = 5

# ── RAG Configuration ─────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
RAG_TOP_K         = 5

# ── LLM — Ollama (local, free, no API key needed) ─────────────────────────────
LLM_PROVIDER   = "ollama"
LLM_MODEL      = "llama3.2"          # run: ollama pull llama3.2
OLLAMA_BASE_URL= "http://localhost:11434"
LLM_MAX_TOKENS = 1024

# ── OpenAI (kept as fallback, leave blank if using Ollama) ───────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Anthropic (kept as fallback, leave blank if using Ollama) ────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")