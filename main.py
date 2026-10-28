from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import dice_ml

app = FastAPI(title="Clinical Counterfactual Decision Support API")

# 1. Load trained LightGBM model and reference dataset
model = joblib.load("lgbm_clinical_model.pkl")
reference_df = pd.read_csv("clinical_reference_data.csv")

# 2. Define Feature Rules
CONTINUOUS_FEATURES = ['Age', 'BMI', 'Systolic_BP', 'Cholesterol', 'Fasting_Blood_Sugar']

# CRITICAL CLINICAL RULE: Only modifiable vitals are unlocked.
# Age, Sex, and Family_History are omitted so DiCE strictly locks them.
MUTABLE_FEATURES = ['BMI', 'Systolic_BP', 'Cholesterol', 'Fasting_Blood_Sugar']

# 3. Initialize DiCE Explainer
dice_data = dice_ml.Data(
    dataframe=reference_df,
    continuous_features=CONTINUOUS_FEATURES,
    outcome_name='Heart_Failure_Risk'
)
dice_model = dice_ml.Model(model=model, backend="sklearn")
explainer = dice_ml.Dice(dice_data, dice_model, method="random")

class PatientProfile(BaseModel):
    Age: int
    Sex: int
    Family_History: int
    BMI: float
    Systolic_BP: int
    Cholesterol: int
    Fasting_Blood_Sugar: int
    total_cfs: int = 3

@app.get("/")
def health_check():
    return {"status": "Clinical Counterfactual API is running"}

@app.post("/generate_treatment_plans")
def generate_treatment_plans(patient: PatientProfile):
    patient_dict = patient.model_dump(exclude={"total_cfs"})
    query_instance = pd.DataFrame([patient_dict])
    
    # Predict baseline risk probability
    baseline_prob = float(model.predict_proba(query_instance)[0][1])
    
    # Clinically Safe Permitted Ranges (Prevents suggesting worse vitals or impossible drops)
    safe_permitted_range = {
        'BMI': [18.5, max(24.9, float(patient.BMI))],
        'Systolic_BP': [95, max(120, int(patient.Systolic_BP))],
        'Cholesterol': [130, max(180, int(patient.Cholesterol))],
        'Fasting_Blood_Sugar': [75, max(99, int(patient.Fasting_Blood_Sugar))]
    }

    try:
        # Generate Diverse Counterfactuals
        cf_object = explainer.generate_counterfactuals(
            query_instance,
            total_CFs=patient.total_cfs,
            desired_class=0, # Flip patient from High Risk (1) to Low Risk (0)
            features_to_vary=MUTABLE_FEATURES,
            permitted_range=safe_permitted_range,
            random_seed=42
        )
        
        cf_df = cf_object.cf_examples_list[0].final_cfs_df
        if cf_df is None or cf_df.empty:
            raise ValueError("No valid counterfactuals found within safe clinical bounds.")
            
        # Calculate new risk score for each counterfactual path
        cf_features = cf_df[query_instance.columns]
        cf_probs = model.predict_proba(cf_features)[:, 1]
        
        paths = []
        for idx, (_, row) in enumerate(cf_features.iterrows()):
            path_dict = row.to_dict()
            path_dict['Risk_Score'] = round(float(cf_probs[idx]) * 100, 1)
            paths.append(path_dict)
            
        return {
            "baseline_risk": round(baseline_prob * 100, 1),
            "patient_state": patient_dict,
            "counterfactual_paths": paths
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Counterfactual Engine Error: {str(e)}")