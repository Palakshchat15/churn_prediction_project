import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

def run_analyst_eda():
    """
    Generates high-level statistical visualizations for the analyst report.
    Uses Path-aware logic to work from any directory.
    """
    # Find the 'churn_project' root directory relative to this file
    project_root = Path(__file__).resolve().parent.parent
    
    input_path = project_root / "data" / "processed" / "cell2cell_engineered.csv"
    output_dir = project_root / "outputs" / "analyst_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("  DATA ANALYST EDA (STATISTICAL PROFILING)")
    print("="*60)
    
    if not input_path.exists():
        print(f"  ERROR: Cleaned CSV not found at {input_path}")
        print("  Please run the ETL pipeline first.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Churn Rate by Monthly Revenue (Bins)
    print("  [1/3] Analyzing Revenue vs Churn...")
    try:
        df['Revenue_Bin'] = pd.qcut(df['MonthlyRevenue'].fillna(0), 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Revenue_Bin', y='Churn', data=df, palette='viridis')
        plt.title('Churn Rate by Monthly Revenue Quintile')
        plt.ylabel('Churn Probability')
        plt.savefig(output_dir / "churn_by_revenue.png")
    except Exception as e:
        print(f"  Warning on Plot 1: {e}")
    
    # 2. Churn Rate by Handset Price
    print("  [2/3] Analyzing Handset Price vs Churn...")
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Churn', y='HandsetPrice', data=df)
        plt.title('Handset Price Distribution by Churn Status')
        plt.savefig(output_dir / "churn_by_handset.png")
    except Exception as e:
        print(f"  Warning on Plot 2: {e}")
    
    # 3. Correlation Matrix (Analyst focus)
    print("  [3/3] Generating Correlation Heatmap...")
    try:
        cols = ['Churn', 'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'OverageMinutes', 'HandsetPrice', 'AgeHH1']
        available_cols = [c for c in cols if c in df.columns]
        corr = df[available_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Key Correlation Matrix (Business Metrics)')
        plt.savefig(output_dir / "correlation_heatmap.png")
    except Exception as e:
        print(f"  Warning on Plot 3: {e}")
    
    print("-" * 60)
    print(f"  SUCCESS: Analyst plots saved to {output_dir}")
    print("="*60)

if __name__ == "__main__":
    run_analyst_eda()
