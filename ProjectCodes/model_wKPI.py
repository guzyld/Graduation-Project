# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:35:19 2026

@author: HUAWEI
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# AUDIT ON SYNTHETIC DATASET
print("Auditing synthetic dataset for KPI alignment...")
df_syn = pd.read_csv('synthetic_kpi_dataset4.csv')
X_test = compute_strategic_kpis(df_syn)
y_actual = df_syn['Promotions'].apply(lambda x: 1 if x > 0 else 0)
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_actual, y_prob)

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

# Visual: Audit Category Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=audit_df, x='Audit_Category', 
              palette='viridis', 
              order=['Hidden Talent (Overlooked)', 'Correct Recognition', 'Undeserved (Low Merit)'])
plt.title('Audit of Promotion Decisions: Real vs. Fair System')
plt.xlabel('Audit Category')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.show()

# Visual: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_actual, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: No', 'Predicted: Yes'], 
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Audit Results: KPI Logic vs Dataset Decisions')
plt.show()

# Visual: Strategic KPI Importance
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='darkblue')
plt.title('Impact of Strategic KPI Groups on Promotion Decisions')
plt.xlabel('Importance Weight')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

####################################################

import joblib

model_filename = 'promotion_audit_model.joblib'
joblib.dump(model, model_filename)