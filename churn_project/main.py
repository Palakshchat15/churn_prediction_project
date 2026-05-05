# =============================================================================
# main.py
# Master pipeline orchestrator — runs all 4 stages in order:
#   Stage 1: Data Cleaning
#   Stage 2: Feature Engineering
#   Stage 3: EDA (can be skipped for speed)
#   Stage 4: Model Training + SHAP
#   Stage 5: RAG Indexing
#
# Run: python main.py [--skip-eda] [--skip-rag]
# =============================================================================

import argparse
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Cell2Cell Churn Prediction Pipeline")
    parser.add_argument("--skip-eda",   action="store_true", help="Skip EDA stage (saves time)")
    parser.add_argument("--skip-rag",   action="store_true", help="Skip RAG indexing stage")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training (use saved models)")
    return parser.parse_args()


def banner(title):
    print("\n" + "█" * 60)
    print(f"  {title}")
    print("█" * 60)


def main():
    args  = parse_args()
    start = time.time()

    # ── STAGE 1: DATA CLEANING ────────────────────────────────────────────────
    banner("STAGE 1 — DATA CLEANING")
    from src.cleaning.cleaner   import DataCleaner
    from src.cleaning.validator import validate_cleaned_data

    cleaner  = DataCleaner()
    df_clean = cleaner.run()
    validate_cleaned_data(df_clean)

    # ── STAGE 2: FEATURE ENGINEERING ─────────────────────────────────────────
    banner("STAGE 2 — FEATURE ENGINEERING")
    from src.features.engineer import FeatureEngineer

    fe = FeatureEngineer()
    df_engineered = fe.run()

    # ── STAGE 3: EDA ─────────────────────────────────────────────────────────
    if not args.skip_eda:
        banner("STAGE 3 — INDUSTRY-LEVEL EDA")
        from src.eda.univariate   import run_univariate
        from src.eda.bivariate    import run_bivariate
        from src.eda.multivariate import run_multivariate
        from src.eda.report       import generate_profile_report

        run_univariate(df_engineered)
        run_bivariate(df_engineered)
        run_multivariate(df_engineered)
        generate_profile_report(df_engineered)
    else:
        print("\n[SKIP] EDA stage skipped (--skip-eda flag set)")

    # ── STAGE 4: MODEL TRAINING ───────────────────────────────────────────────
    if not args.skip_train:
        banner("STAGE 4 — MODEL TRAINING + EVALUATION + SHAP")
        from src.model.trainer   import ModelTrainer
        from src.model.evaluator import ModelEvaluator
        from src.model.explainer import SHAPExplainer

        # Train
        trainer = ModelTrainer()
        trainer.run()

        # Evaluate on test set
        evaluator = ModelEvaluator(
            models=trainer.models,
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            feature_names=trainer.feature_names,
        )
        evaluator.evaluate_all()

        # SHAP explanations (use XGBoost as primary model)
        if "xgboost" in trainer.models:
            y_prob = trainer.models["xgboost"].predict_proba(trainer.X_test)[:, 1]
            shap_explainer = SHAPExplainer(
                model=trainer.models["xgboost"],
                X_test=trainer.X_test,
                feature_names=trainer.feature_names,
            )
            shap_df = shap_explainer.run(y_prob=y_prob)
        else:
            shap_df = None
    else:
        print("\n[SKIP] Training stage skipped (--skip-train flag set)")
        shap_df = None

    # ── STAGE 5: RAG INDEXING ─────────────────────────────────────────────────
    if not args.skip_rag:
        banner("STAGE 5 — RAG INDEXING (ChromaDB)")
        from src.rag.indexer import ChromaIndexer

        indexer = ChromaIndexer()
        indexer.run(df=df_engineered, shap_df=shap_df)

        print("\n✓ RAG system ready.")
        print("  To launch chatbot UI:  streamlit run app.py")
        print("  To use terminal mode:  python -c \"from src.rag.chatbot import ChurnChatbot; ChurnChatbot().connect().interactive()\"")
    else:
        print("\n[SKIP] RAG indexing skipped (--skip-rag flag set)")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner("PIPELINE COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Cleaned data:    data/processed/cell2cell_cleaned.csv")
    print(f"  Engineered data: data/processed/cell2cell_engineered.csv")
    print(f"  Models:          models/")
    print(f"  EDA plots:       outputs/plots/")
    print(f"  EDA report:      outputs/eda_report.html")
    print(f"  Model report:    outputs/model_report.txt")
    print(f"  SHAP exports:    outputs/shap_customer_explanations.csv")
    print(f"  Vector store:    data/vector_store/")
    print()


if __name__ == "__main__":
    main()
