# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:50:29 2026

@author: HUAWEI
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_fair = pd.read_csv('Fair_Promotion_Dataset.csv')

df_fair['Got_Promotion'] = df_fair['Promotions'].apply(lambda x: 1 if x > 0 else 0)
cols_to_drop = [
    'Employee_ID', 'Job_Title', 'Monthly_Salary', 'Hire_Date',
    'Promotions', 'Sick_Days', 'Merit_Score', 'Fairness_Label', 'Got_Promotion'
]
X = df_fair.drop(columns=cols_to_drop, errors='ignore')
y = df_fair['Got_Promotion']

categorical_cols = ['Department', 'Gender', 'Education_Level', 'Resigned']
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

results_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

results_df = results_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(results_df['Feature'], results_df['Importance'])
plt.xlabel('Importance Weight')
plt.ylabel('Feature')
plt.tight_layout()

plt.savefig('figure6_feature_importance.png', dpi=300, bbox_inches='tight')

plt.show()

print(results_df.sort_values(by='Importance', ascending=False))