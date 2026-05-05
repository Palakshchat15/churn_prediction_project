import sys
import os
from pathlib import Path

# Add the current directory to sys.path so we can import from src
# This ensures that even if run from inside 'churn_project', it can find 'src'
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from src.sql_engine import run_etl
from src.eda_analyst import run_analyst_eda

def main():
    """
    Main orchestrator for the End-to-End Data Analysis project.
    Flow: SQL ETL -> Statistical EDA -> Report Generation
    """
    print("\n" + "#"*60)
    print("  END-TO-END DATA ANALYSIS PROJECT: CUSTOMER CHURN")
    print("#"*60 + "\n")
    
    # 1. Run SQL-based ETL
    run_etl()
    
    # 2. Run Analyst EDA
    run_analyst_eda()
    
    print("\n" + "#"*60)
    print("  PROJECT EXECUTION COMPLETE")
    print("  1. ETL: DuckDB SQL pipeline (data/processed/cell2cell_engineered.csv)")
    print("  2. EDA: Statistical visualizations (outputs/analyst_plots/)")
    print("  3. BI:  Update Tableau with the cleaned CSV")
    print("  4. DOC: Review report/Full_Technical_Manual.md")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
