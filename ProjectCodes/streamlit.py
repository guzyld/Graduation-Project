import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# CORE LOGIC: STRATEGIC KPI COMPUTATION 
def compute_strategic_kpis(df_input):
    df = df_input.copy()
    edu_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    if 'Education_Level' in df.columns and df['Education_Level'].dtype == 'O':
        df['Education_Level'] = df['Education_Level'].map(edu_map)
    
    if 'Attendance_Score' not in df.columns and 'Sick_Days' in df.columns:
        df['Attendance_Score'] = df['Sick_Days'].max() - df['Sick_Days']

    if 'Resigned' in df.columns:
        le = LabelEncoder()
        df['Resigned_Enc'] = le.fit_transform(df['Resigned'].astype(str))
    else:
        df['Resigned_Enc'] = 0

    kpi_map = {
        'Performance_KPI': ['Performance_Score', 'Projects_Handled', 'Training_Hours'],
        'Efficiency_KPI': ['Work_Hours_Per_Week', 'Overtime_Hours', 'Attendance_Score'],
        'Commitment_KPI': ['Employee_Satisfaction_Score', 'Years_At_Company', 'Remote_Work_Frequency'],
        'Profile_KPI': ['Education_Level', 'Age', 'Resigned_Enc']
    }

    scaler = MinMaxScaler()
    kpi_results = pd.DataFrame(index=df.index)

    for kpi_name, cols in kpi_map.items():
        existing_cols = [c for c in cols if c in df.columns]
        if existing_cols:
            scaled_data = scaler.fit_transform(df[existing_cols].fillna(df[existing_cols].median()))
            kpi_results[kpi_name] = scaled_data.mean(axis=1)
        else:
            kpi_results[kpi_name] = 0

    return kpi_results

# STREAMLIT USER INTERFACE
st.set_page_config(page_title="Strategic KPI Audit System", layout="wide")

st.title("📊 Strategic KPI Audit: Promotion Intelligence")
st.markdown("Interactive dashboard for employee promotion suitability and KPI distribution.")

@st.cache_resource
def load_model():
    try:
        return joblib.load('promotion_audit_model.joblib')
    except:
        return None

model = load_model()

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Employee Dataset (CSV)", type="csv")

st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.info("Core Engine: **Random Forest Classifier**")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    X_processed = compute_strategic_kpis(df_raw)
    
    if 'Employee_ID' in df_raw.columns:
        ids = df_raw['Employee_ID'].unique()
        selected_id = st.selectbox("Select Employee ID to Audit:", ids)
        
        emp_row = df_raw[df_raw['Employee_ID'] == selected_id]
        emp_idx = emp_row.index[0]
        emp_name = emp_row['Employee_Name'].values[0] if 'Employee_Name' in df_raw.columns else "Unknown Employee"
        
        if model is not None:
            prediction = model.predict(X_processed.iloc[[emp_idx]])[0]
            prob = model.predict_proba(X_processed.iloc[[emp_idx]])[0][1]
            
            st.divider()
            st.header(f"Employee: {emp_name}")
            st.caption(f"Employee ID: {selected_id}")
            
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.markdown("### Audit Status")
                if prediction == 1:
                    st.success("✅ **RECOMMENDED FOR PROMOTION**")
                else:
                    st.warning("⚠️ **NOT RECOMMENDED FOR PROMOTION**")
                
                st.metric("Suitability Confidence", f"{prob*100:.2f}%")
                st.write("---")
                st.write(f"**Audit Engine:** Predictions are powered by a **Random Forest Classifier** trained on fair promotion standards.")

            with col2:
                # INTERACTIVE GRAPH SWITCHER 
                st.markdown("### Strategic KPI Analysis")
                
                # Switcher/Toggle for charts
                chart_type = st.radio("Choose Visualization Type:", 
                                     ["Bar Chart (Comparison)", "Pie Chart (KPI Composition)"], 
                                     horizontal=True)
                
                emp_kpis = X_processed.iloc[emp_idx]
                
                if "Bar Chart" in chart_type:
                    st.bar_chart(emp_kpis)
                else:
                    # Pie Chart using Plotly for better interactivity
                    fig = px.pie(values=emp_kpis.values, 
                                 names=emp_kpis.index, 
                                 title=f"KPI Breakdown for {emp_name}",
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig, use_container_width=True)
                
            st.subheader("Raw Strategic KPI Analysis")
            st.table(emp_kpis.to_frame().T)
            
        else:
            st.error("Error: Model file 'promotion_audit_model.joblib' not found.")
    else:
        st.error("The dataset must contain 'Employee_ID' column.")
else:
    st.info("Please upload a CSV file from the sidebar to begin.")