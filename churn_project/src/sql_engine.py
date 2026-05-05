import duckdb
import os
import time
from pathlib import Path

def run_etl():
    """
    Executes the SQL ETL pipeline using DuckDB.
    Uses Path-aware logic to work from any directory.
    """
    # Find the 'churn_project' root directory relative to this file
    # This file is in churn_project/src/sql_engine.py
    project_root = Path(__file__).resolve().parent.parent
    
    sql_file = project_root / "sql" / "etl_pipeline.sql"
    raw_csv = project_root / "data" / "raw" / "cell2celltrain.csv"
    output_path = project_root / "data" / "processed" / "cell2cell_engineered.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("  SQL-POWERED ETL PIPELINE (DUCKDB)")
    print("="*60)
    
    con = duckdb.connect()
    
    if not raw_csv.exists():
        print(f"  ERROR: Raw CSV not found at {raw_csv}")
        return

    print(f"  [1/2] Loading raw data into staging...")
    try:
        # Convert path to string for DuckDB
        raw_csv_str = str(raw_csv).replace("\\", "/")
        con.execute(f"CREATE OR REPLACE TABLE staging AS SELECT * FROM read_csv_auto('{raw_csv_str}', all_varchar=True)")
        count = con.execute("SELECT count(*) FROM staging").fetchone()[0]
        print(f"  [LOG] Staging loaded: {count:,} rows")
    except Exception as e:
        print(f"  ERROR during staging: {e}")
        return

    print(f"  [2/2] Running transformations...")
    try:
        with open(sql_file, 'r') as f:
            full_sql = f.read()
            if "FINAL CLEANING QUERY" in full_sql:
                transform_sql = full_sql.split("FINAL CLEANING QUERY")[1].strip()
                if transform_sql.startswith("--"):
                    transform_sql = "\n".join(transform_sql.split("\n")[1:])
            else:
                transform_sql = full_sql
        
        df = con.execute(transform_sql).df()
        
        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            print(f"  SUCCESS: Exported {df.shape[0]:,} rows x {df.shape[1]} columns")
            print(f"  Saved to: {output_path}")
        else:
            print("  ERROR: Transformation returned no data.")
            
    except Exception as e:
        print(f"  ERROR during transformation: {e}")
        
    print("="*60)

if __name__ == "__main__":
    run_etl()
