# AI-Based KPI Audit Framework for Promotion Fairness in Human Resources

This repository contains the graduation project titled **"AI-Based KPI Audit Framework for Promotion Fairness in Human Resources"** conducted at **Istanbul University, Faculty of Economics (Management Information Systems)**. 

The project introduces a data science and machine learning approach that positions Artificial Intelligence not as an autonomous decision-maker, but as a diagnostic auditing tool to evaluate the consistency, transparency, and fairness of historical promotion decisions against objective performance indicators.

## Executive Summary & Key Findings
* **The Gap:** While traditional HR analytics focuses merely on predicting past promotions (which often replicates human or organizational bias), this framework establishes an independent, merit-based standard using Key Performance Indicators (KPIs).
* **Core Discovery:** The trained Random Forest model successfully learned and aligned with KPI-based fair datasets with an accuracy of **77% - 80%** (ROC-AUC > 0.92). However, when applied to the full real HR dataset, alignment accuracy dropped to **52%** (ROC-AUC ~0.50). This gap provides a measurable metric showing that historical promotion decisions frequently rely on non-quantifiable or external organizational factors rather than objective performance metrics alone.
* **Audit Categories:** The framework translates technical classification errors into actionable HR insights by categorizing employees into: *Correct Recognition*, *Hidden Talent (Overlooked High-Potentials)*, and *Undeserved/Low Merit Promotions*.

## Project Structure & Files
The repository contains the data preprocessing, modeling pipeline, datasets, evaluation reports, and an interactive dashboard interface.

The core assets include:
* `graphs/`: Visualization outputs including Confusion Matrices and Audit Category distributions.
* `promotion_audit_model.joblib`: The final hyperparameter-tuned and trained Random Forest Classification model object.
* `Extended_Employee_Performance_and_Pr...`: The core processed dataset used for evaluation.
* `Fair_Promotion_Dataset.xlsx` & `synthetic_kpi_dataset3/4`: Filtered, KPI-consistent, and balanced training/validation subsets.
* `model_wKPI.py`, `model_wKPI2.py`, `model_wKPIlog.py`: Core machine learning scripts handles model training, 5-fold cross-validation, and SHAP analysis.
* `streamlit.py`: Script for the web-based interactive application.
* `analysis_with_the_new_dataset.py` & `to_create_deserved_or_not_promotions.py`: Diagnostic and auditing scripts.

## Interactive Streamlit Dashboard
An executive dashboard was developed to operationalize this framework for HR decision-makers. It enables users to upload datasets, query individual Employee IDs, inspect suitability confidence scores, and visually analyze an employee's multi-dimensional KPI breakdown.
* To launch the interface locally, run: `streamlit run streamlit.py`

## Strategic KPI Engineering & Model Logic
Instead of processing individual raw attributes, variables were engineered into **four normalized strategic KPI dimensions** to improve explainability:
1. **Performance KPI:** Performance scores, projects handled, and training hours.
2. **Efficiency KPI:** Weekly work hours, overtime hours, and an attendance score (derived inversely from sick leaves).
3. **Commitment KPI:** Employee satisfaction score, years at the company, and remote work frequency.
4. **Profile KPI:** Demographics and static traits (education level, age, resignation status).

Global model interpretability via **SHAP (SHapley Additive exPlanations)** validated that the model functions ethically: **Performance KPI** and **Efficiency KPI** yielded the highest impact weights, while **Profile KPI** had a near-zero contribution ($0.0029$), ensuring static demographic biases were eliminated.

## Academic Documentation
The comprehensive theoretical background, literature review, methodology, and extensive discussion are available in the official graduation report included in this repository.
* **Author:** İrem Güz YILDIZ
* **Supervisor:** Asst. Prof. Dr. Waqar BADSHAH
* **Date:** June, 2026
