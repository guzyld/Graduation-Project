import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_fair = pd.read_csv('Fair_Promotion_Dataset.csv')

df_fair['Got_Promotion'] = df_fair['Promotions'].apply(lambda x: 1 if x > 0 else 0)

cols_to_drop = [
    'Employee_ID', 'Job_Title', 'Monthly_Salary', 'Hire_Date', 
    'Promotions', 'Sick_Days', 'Merit_Score', 'Fairness_Label', 'Got_Promotion'
]
X = df_fair.drop(columns=cols_to_drop, errors='ignore')
y = df_fair['Got_Promotion']

le = LabelEncoder()
categorical_cols = ['Department', 'Gender', 'Education_Level', 'Resigned']
for col in categorical_cols:
    if col in X.columns:
        X[col] = le.fit_transform(X[col])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

results_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
kpi_map = {
    'Productivity & Output': ['Projects_Handled', 'Work_Hours_Per_Week', 'Overtime_Hours'],
    'Professional Growth': ['Training_Hours', 'Education_Level', 'Performance_Score'],
    'Engagement & Attendance': ['Employee_Satisfaction_Score', 'Attendance_Score', 'Remote_Work_Frequency', 'Resigned']
}
results_df['KPI Group'] = results_df['Feature'].apply(lambda x: next((k for k, v in kpi_map.items() if x in v), 'Other'))

kpi_impact = results_df.groupby('KPI Group')['Importance'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='KPI Group', data=kpi_impact, palette='rocket')
plt.title('Adil Sistemde KPI Gruplarının Terfi Üzerindeki Etkisi')
plt.show()

print("Analiz Başarıyla Tamamlandı.")
print(results_df.sort_values(by='Importance', ascending=False))