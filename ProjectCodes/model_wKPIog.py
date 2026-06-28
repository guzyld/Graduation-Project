# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:35:19 2026

@author: HUAWEI
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import shap

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# STRATEGIC KPI COMPUTATION FUNCTION
def compute_strategic_kpis(df_input):
    df = df_input.copy()

    # Data Cleaning and Transformation
    edu_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    if df['Education_Level'].dtype == 'O':
        df['Education_Level'] = df['Education_Level'].map(edu_map)
    
    # Reverse sick_days for Attendance Score
    if 'Attendance_Score' not in df.columns:
        df['Attendance_Score'] = df['Sick_Days'].max() - df['Sick_Days']

    le = LabelEncoder()
    df['Resigned_Enc'] = le.fit_transform(df['Resigned'].astype(str))

    # Strategic KPI Mapping (Thesis Groups)
    kpi_map = {
        'Performance_KPI': ['Performance_Score', 'Projects_Handled', 'Training_Hours'],
        'Efficiency_KPI': ['Work_Hours_Per_Week', 'Overtime_Hours', 'Attendance_Score'],
        'Commitment_KPI': ['Employee_Satisfaction_Score', 'Years_At_Company', 'Remote_Work_Frequency'],
        'Profile_KPI': ['Education_Level', 'Age', 'Resigned_Enc']
    }

    scaler = MinMaxScaler()
    kpi_results = pd.DataFrame(index=df.index)

    for kpi_name, cols in kpi_map.items():
        scaled_data = scaler.fit_transform(df[cols].fillna(df[cols].median()))
        kpi_results[kpi_name] = scaled_data.mean(axis=1)

    return kpi_results

# TRAINING ON FAIR DATASET
print("Training model on Fair Promotion standards...")
df_fair = pd.read_csv('Fair_Promotion_Dataset.csv')
X_train = compute_strategic_kpis(df_fair)
y_train = df_fair['Promotions'].apply(lambda x: 1 if x > 0 else 0)

# - CROSS-VALIDATION ON FAIR TRAINING DATASET
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_accuracy = cross_val_score(base_model, X_train, y_train, cv=cv, scoring='accuracy')
cv_f1 = cross_val_score(base_model, X_train, y_train, cv=cv, scoring='f1_weighted')
cv_roc_auc = cross_val_score(base_model, X_train, y_train, cv=cv, scoring='roc_auc')

print("\n--- CROSS-VALIDATION RESULTS ON FAIR TRAINING DATASET ---")
print(f"5-Fold CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"5-Fold CV Weighted F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
print(f"5-Fold CV ROC-AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std():.4f})")


# - HYPERPARAMETER TUNING
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- HYPERPARAMETER TUNING RESULTS ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validated Weighted F1-Score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

model_filename = 'promotion_audit_model.joblib'
joblib.dump(model, model_filename)
print(f"Model başarıyla '{model_filename}' adıyla kaydedildi.")

# AUDIT ON SYNTHETIC DATASET
print("Auditing synthetic dataset for KPI alignment...")
df_syn = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')
X_test = compute_strategic_kpis(df_syn)
y_actual = df_syn['Promotions'].apply(lambda x: 1 if x > 0 else 0)
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_actual, y_prob)

# - SHAP EXPLAINABILITY ANALYSIS
print("\nRunning SHAP explainability analysis...")

import numpy as np

# Use a sample to reduce computation time
X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=42)

# TreeExplainer is suitable for Random Forest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Select SHAP values for the positive class: Promotion
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_class1 = shap_values[:, :, 1]
else:
    shap_values_class1 = shap_values

# Calculate mean absolute SHAP values manually
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

shap_importance = pd.Series(
    mean_abs_shap,
    index=X_shap.columns
).sort_values(ascending=True)

# Plot manual SHAP bar chart
plt.figure(figsize=(10, 5))
shap_importance.plot(kind='barh')

plt.title("SHAP Global Importance of Strategic KPI Groups")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Strategic KPI Group")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("shap_global_kpi_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("SHAP plot saved as shap_global_kpi_importance.png")
print("\nSHAP Global Importance Values:")
print(shap_importance.sort_values(ascending=False))

# EVALUATION & REPORTING
print("\n--- STRATEGIC KPI AUDIT PERFORMANCE REPORT ---")
print(f"Overall Alignment Accuracy: {accuracy_score(y_actual, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report (Reality vs. KPI Logic):")
print(classification_report(y_actual, y_pred, target_names=['No Promotion', 'Promotion']))

# VISUALIZATIONS

# Audit Category Analysis
def categorize_audit(row):
    if row['Predicted'] == row['Actual']:
        return 'Correct Recognition'
    elif row['Predicted'] == 1 and row['Actual'] == 0:
        return 'Hidden Talent (Overlooked)'
    else:
        return 'Undeserved (Low Merit)'

audit_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
audit_df['Audit_Category'] = audit_df.apply(categorize_audit, axis=1)

# 

plt.figure(figsize=(10, 6))
sns.countplot(data=audit_df, x='Audit_Category', 
              palette='viridis', 
              order=['Hidden Talent (Overlooked)', 'Correct Recognition', 'Undeserved (Low Merit)'])
plt.title('Audit of Promotion Decisions: Real vs. Fair System')
plt.xlabel('Audit Category')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_actual, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: No', 'Predicted: Yes'], 
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Audit Results: KPI Logic vs Dataset Decisions')
plt.show()

importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='darkblue')
plt.title('Impact of Strategic KPI Groups on Promotion Decisions')
plt.xlabel('Importance Weight')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()