import streamlit as st
import pandas as pd
import requests
import os
import streamlit as st
import pandas as pd
import requests

# Reads API_URL from Render environment variables (falls back to localhost on your PC)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Prescriptive CDSS | Counterfactual AI", layout="wide")

# Helper functions for clinical stage annotations
def annotate_bmi(val):
    if val >= 30: return f"{val:.1f} (Obese)"
    elif val >= 25: return f"{val:.1f} (Overweight)"
    return f"{val:.1f} (Normal)"

def annotate_bp(val):
    val = int(round(val))
    if val >= 140: return f"{val} (Stage 2 HTN)"
    elif val >= 130: return f"{val} (Stage 1 HTN)"
    elif val >= 120: return f"{val} (Elevated)"
    return f"{val} (Normal Range)"

def annotate_chol(val):
    val = int(round(val))
    if val >= 240: return f"{val} (High)"
    elif val >= 200: return f"{val} (Borderline)"
    return f"{val} (Normal Range)"

def annotate_fbs(val):
    val = int(round(val))
    if val >= 126: return f"{val} (Diabetic Range)"
    elif val >= 100: return f"{val} (Prediabetic)"
    return f"{val} (Normal)"

# Main Header
st.title("🩺 Prescriptive Clinical Decision Support System")
st.markdown("""
**From Predictive to Prescriptive AI:** Standard explainability tools (like SHAP) only tell clinicians *why* a patient is at high risk. 
This engine uses **Constrained Counterfactual Optimization (DiCE)** to calculate the minimum, safest interventions on **modifiable vitals** while keeping demographic and genetic history strictly locked.
""")

# Sidebar: Patient Electronic Health Record Input
st.sidebar.header("📋 Patient Vitals & Profile")

st.sidebar.subheader("🔒 Immutable Features (Locked)")
age = st.sidebar.slider("Age", 30, 85, 62)
sex = st.sidebar.selectbox("Biological Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
fam_hist = st.sidebar.selectbox("Family History of Heart Disease", options=[(1, "Present"), (0, "None")], format_func=lambda x: x[1])[0]

st.sidebar.subheader("⚡ Actionable Vitals (Modifiable)")
bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=16.0, max_value=50.0, value=34.2, step=0.1)
sys_bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 210, 155)
chol = st.sidebar.slider("Total Cholesterol (mg/dL)", 120, 380, 260)
fbs = st.sidebar.slider("Fasting Blood Sugar (mg/dL)", 65, 280, 118)

num_paths = st.sidebar.slider("Alternative Treatment Paths to Generate", 2, 4, 3)

if st.button("🚀 Generate Counterfactual Treatment Plans", type="primary"):
    payload = {
        "Age": age,
        "Sex": sex,
        "Family_History": fam_hist,
        "BMI": bmi,
        "Systolic_BP": sys_bp,
        "Cholesterol": chol,
        "Fasting_Blood_Sugar": fbs,
        "total_cfs": num_paths
    }
    
    with st.spinner("Computing constrained counterfactual pathways via DiCE..."):
        try:
            response = requests.post(
            f"{API_URL}/generate_treatment_plans", 
            json=payload, 
            timeout=90  # Prevents timeout error when Render free tier wakes up from sleep
            )
            if response.status_code == 200:
                data = response.json()
                base_risk = data["baseline_risk"]
                cf_paths = data["counterfactual_paths"]
                
                # Top Summary Row
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(
                        label="Current Heart Failure Risk",
                        value=f"{base_risk}%",
                        delta="High Risk State" if base_risk >= 50 else "Moderate/Low Risk",
                        delta_color="inverse"
                    )
                with col2:
                    st.info("🔒 **Clinical Constraint Check:** `Age`, `Biological Sex`, and `Family History` were mathematically locked during optimization. All treatment paths below strictly modify actionable biomarkers within safe physiological limits.")

                # Build the Action Plan Table
                table_data = {
                    "Patient Vitals": [
                        "Age (Locked)",
                        "Biological Sex (Locked)",
                        "Family History (Locked)",
                        "BMI (kg/m²)",
                        "Systolic BP (mmHg)",
                        "Cholesterol (mg/dL)",
                        "Fasting Blood Sugar (mg/dL)",
                        "Heart Risk Score"
                    ],
                    "Current High-Risk State": [
                        f"{age}",
                        "Male" if sex == 1 else "Female",
                        "Present" if fam_hist == 1 else "None",
                        annotate_bmi(bmi),
                        annotate_bp(sys_bp),
                        annotate_chol(chol),
                        annotate_fbs(fbs),
                        f"🔴 {base_risk}% Risk"
                    ]
                }

                for i, path in enumerate(cf_paths):
                    # Determine the primary clinical focus of each path
                    drops = {
                        "Weight Loss": max(0, (bmi - path["BMI"]) / bmi),
                        "Blood Pressure": max(0, (sys_bp - path["Systolic_BP"]) / sys_bp),
                        "Cholesterol": max(0, (chol - path["Cholesterol"]) / chol),
                        "Glycemic Control": max(0, (fbs - path["Fasting_Blood_Sugar"]) / fbs)
                    }
                    primary_focus = max(drops, key=drops.get)
                    col_name = f"Target Path {chr(65+i)} (Focus: {primary_focus})"

                    bmi_str = f"{annotate_bmi(path['BMI'])} ⬇️" if path['BMI'] < bmi - 0.3 else f"{path['BMI']:.1f} (Unchanged)"
                    bp_str = f"{annotate_bp(path['Systolic_BP'])} ⬇️" if path['Systolic_BP'] < sys_bp - 2 else f"{int(path['Systolic_BP'])} (Unchanged)"
                    chol_str = f"{annotate_chol(path['Cholesterol'])} ⬇️" if path['Cholesterol'] < chol - 3 else f"{int(path['Cholesterol'])} (Unchanged)"
                    fbs_str = f"{annotate_fbs(path['Fasting_Blood_Sugar'])} ⬇️" if path['Fasting_Blood_Sugar'] < fbs - 2 else f"{int(path['Fasting_Blood_Sugar'])} (Unchanged)"

                    table_data[col_name] = [
                        f"{int(path['Age'])}",
                        "Male" if int(path['Sex']) == 1 else "Female",
                        "Present" if int(path['Family_History']) == 1 else "None",
                        bmi_str,
                        bp_str,
                        chol_str,
                        fbs_str,
                        f"🟢 {path['Risk_Score']}% Risk"
                    ]

                st.subheader("📊 Clinical Action Plan Table")
                result_df = pd.DataFrame(table_data)
                st.table(result_df.set_index("Patient Vitals"))
                
            else:
                st.error(f"API Error: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI Backend at http://127.0.0.1:8000. Error: {e}")