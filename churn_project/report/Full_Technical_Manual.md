# Full Technical Manual: End-to-End Churn Intelligence Platform

This document provides an exhaustive technical and business breakdown of the **SQL-Powered Churn Analysis Platform**. It explains the architecture, the transformation logic, and the analytical strategies used to drive subscriber retention.

---

## 1. Project Architecture & Operational Flow

The project follows a modern **ELT (Extract, Load, Transform)** architecture, optimized for performance and maintainability.

### 1.1 The Chain of Custody
1.  **Raw Ingestion**: `src/sql_engine.py` reads the 13MB raw CSV using DuckDB's high-speed scanner.
2.  **SQL Transformation**: `sql/etl_pipeline.sql` performs the heavy cleaning using temporary staging tables.
3.  **Statistical Profiling**: `src/eda_analyst.py` consumes the cleaned data to find churn drivers.
4.  **Executive Visualization**: Tableau connects to the final CSV for the business-facing dashboard.

---

## 2. Deep-Dive: The SQL Cleaning Pipeline (`etl_pipeline.sql`)

This pipeline is designed to be **Production-Hardened**. It replaces the old Python-based cleaning with a 100% SQL approach.

### 2.1 The "Flexible Ingestion" Layer
```sql
CREATE OR REPLACE TABLE staging AS 
SELECT * FROM read_csv('path/to/data.csv', all_varchar=True);
```
**The Logic**: By loading everything as `VARCHAR` (Text) first, we prevent the SQL engine from crashing if a numeric column accidentally contains a string like "Unknown" or "N/A." This is a "Safety First" approach used in professional data engineering.

### 2.2 Numerical Integrity & TRY_CAST
We use `TRY_CAST(column AS FLOAT)` instead of a standard `CAST`. 
- **The Rationale**: If `TRY_CAST` encounters a value it cannot convert (like an emoji or a typo), it returns `NULL` instead of failing the whole script. This ensures the pipeline remains stable even with "dirty" data.

### 2.3 SQL Imputation Strategy
```sql
COALESCE(MonthlyRevenue, (SELECT median(MonthlyRevenue) FROM base))
```
**The Logic**: We use the `COALESCE` function to fill missing values. If the specific row has a NULL, it "falls back" to the **Global Median** of that column. 
- **Analyst Benefit**: Median is used over Mean because it is "Outlier-Resistant." One billionaire customer won't skew the missing revenue values for average customers.

### 2.4 Feature Categorization
- **Binary Mapping**: Transforms `Yes/No` to `1/0`. This is required for Correlation Analysis (you cannot run a correlation on text).
- **Labeling for Tableau**: We create "Dual Columns" (e.g., `CreditRating` as a number for sorting and `CreditRating_Label` as text for display). This makes the Tableau dashboard significantly more user-friendly.

---

## 3. Deep-Dive: The Analyst EDA (`eda_analyst.py`)

This module moves beyond simple charts to find **Actionable Insights**.

### 3.1 Revenue Quintile Analysis (`pd.qcut`)
**The Strategy**: We divide customers into 5 equal-sized groups based on their spend.
- **The Insight**: If the "Very High" revenue quintile has a higher churn rate than the "Very Low" group, the business is losing its most profitable customers. This plot helps the retention team prioritize their "Save Calls" based on customer value.

### 3.2 Handset Price Distribution
**The Strategy**: A boxplot comparing handset prices for Churners vs. Retained customers.
- **The Insight**: If churners have significantly lower handset prices, it suggests that "Hardware Obsolescence" is a driver. The recommendation would be to offer handset upgrade discounts to prevent them from leaving for a competitor with a better phone deal.

### 3.3 The Business Correlation Heatmap
**The Strategy**: A statistical matrix showing the relationship between Churn and usage metrics.
- **The Insight**: We look for the "Darkest" squares. If `OverageMinutes` has a high positive correlation with `Churn`, it tells the business that "Surprise Charges" on bills are causing people to quit.

---

## 4. Key Metrics & Definitions

- **Churn Rate**: The percentage of the total base that leaves during the period.
- **ARPU (Avg Revenue Per User)**: A critical telecom KPI calculated from `MonthlyRevenue`.
- **Equipment Tenure**: Calculated from `CurrentEquipmentDays`, used to trigger upgrade marketing.

---

## 5. Conclusion: Why this Approach?

By moving the logic to **SQL** and focusing on **Statistical EDA**, the project is now:
1.  **Transparent**: Any SQL developer can audit the cleaning logic.
2.  **Scalable**: DuckDB can handle millions of rows far faster than traditional Pandas.
3.  **Business-Aligned**: Every chart and line of code is tied to a specific management decision.
