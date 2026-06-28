# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:24:27 2026

@author: HUAWEI
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

df['Attendance_Score'] = df['Sick_Days'].max() - df['Sick_Days']

merit_features = [
    'Performance_Score', 'Projects_Handled', 
    'Training_Hours', 'Attendance_Score', 'Employee_Satisfaction_Score'
]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[merit_features]), columns=merit_features)
df['Merit_Score'] = df_scaled.mean(axis=1)

high_merit_threshold = df['Merit_Score'].quantile(0.75)
low_merit_threshold = df['Merit_Score'].quantile(0.25)

deserved_and_got = df[(df['Merit_Score'] >= high_merit_threshold) & (df['Promotions'] > 0)].copy()

not_deserved_and_not_got = df[(df['Merit_Score'] <= low_merit_threshold) & (df['Promotions'] == 0)].copy()

fair_dataset = pd.concat([deserved_and_got, not_deserved_and_not_got])
fair_dataset.to_csv('Fair_Promotion_Dataset.csv', index=False)

print(f"Yeni Veri Seti Boyutu: {len(fair_dataset)} çalışan")