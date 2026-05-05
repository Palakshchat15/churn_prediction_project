# Technical Report: Customer Churn Analysis Platform (SQL + Tableau)

This document details the end-to-end analytical framework used to identify, analyze, and visualize subscriber churn for a major telecommunications provider using the Cell2Cell dataset.

---

## 1. Project Overview & Business Case

### The Problem
Customer churn is the single biggest threat to profitability in the telecom industry. Acquiring a new customer is **5x more expensive** than retaining an existing one.

### The Objective
To build a SQL-powered analytics pipeline that identifies high-risk customer segments and provides the executive team with a real-time Tableau dashboard for proactive retention strategies.

---

## 2. Data Architecture & ETL (SQL-Powered)

Unlike traditional data science projects that rely on complex Python libraries, this project utilizes **SQL (DuckDB)** as the primary transformation engine. This ensures high performance and scalability in a production database environment.

### ETL Pipeline Flow:
1.  **Ingestion**: Loading 51,000+ subscriber records from raw CSV.
2.  **Schema Enforcement**: Casting object types to appropriate Numeric (Float) and Categorical (String) types.
3.  **SQL-Based Imputation**: Handling missing values using **Window Functions** and Global Medians to ensure data consistency.
4.  **Categorical Mapping**: Standardizing "Yes/No" and Ordinal data (Credit Rating, Homeownership) into numeric scores for correlation analysis.
5.  **Outlier Handling**: Implementing **SQL-based Winsorization** (capping at the 99th percentile) to prevent extreme outliers from skewing average metrics.

---

## 3. Key Analytical Insights

### Churn Drivers Identified
Through statistical profiling, the following primary drivers were isolated:
- **Service Quality**: A direct correlation was found between "Dropped Calls" and churn probability.
- **Tenure Sensitivity**: Customers in their first 6 months show a significantly higher risk of churn compared to long-term subscribers.
- **Equipment Age**: Subscribers with equipment older than 2 years are more likely to churn, suggesting a need for upgrade incentives.

---

## 4. Visual Intelligence (Tableau Dashboard)

The final output is a multi-page **Executive Tableau Dashboard** designed for non-technical stakeholders.

### Dashboard Features:
- **Executive Summary**: Real-time Churn Rate, Total Revenue at Risk, and ARPU (Average Revenue Per User).
- **Segment Analysis**: Breakdown of churn by Credit Rating and Income Group.
- **Geographic Hotspots**: Map visualization of churn density to help target regional retention offers.
- **Risk Scorecards**: A list of high-value customers with "Early Warning" signals (e.g., high drop rates).

---

## 5. Conclusion & Recommendations

Based on the analysis, the following strategies are recommended:
1.  **Retention Team Focus**: Deploy specialized teams to contact subscribers with a high "Dropped Call" frequency within 24 hours.
2.  **Loyalty Incentives**: Launch a 24-month equipment upgrade program for high-value customers.
3.  **Segmented Pricing**: Adjust monthly recurring charges for low-income segments to improve retention.
