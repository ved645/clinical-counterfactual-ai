import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

def generate_clinical_dataset(n_samples=6000):
    """
    Generates a synthetic electronic health record (EHR) cohort modeled on 
    framingham/cardiovascular risk distributions.
    """
    np.random.seed(42)
    
    # 1. Immutable Features (Cannot be changed by a doctor's intervention)
    age = np.random.randint(30, 85, n_samples)
    sex = np.random.choice([0, 1], n_samples) # 0: Female, 1: Male
    family_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # 2. Mutable / Actionable Features (Modifiable vitals & biomarkers)
    bmi = np.round(np.random.normal(28.5, 5.5, n_samples).clip(16.0, 50.0), 1)
    systolic_bp = np.random.normal(132, 22, n_samples).clip(90, 210).astype(int)
    cholesterol = np.random.normal(215, 45, n_samples).clip(120, 380).astype(int)
    fasting_bs = np.random.normal(105, 30, n_samples).clip(65, 280).astype(int)
    
    # 3. Ground-Truth Physiological Log-Odds Equation
    logit = (
        -11.5 
        + 0.045 * age 
        + 0.35 * sex 
        + 0.75 * family_history 
        + 0.095 * (bmi - 22) 
        + 0.042 * (systolic_bp - 110) 
        + 0.015 * (cholesterol - 170) 
        + 0.018 * (fasting_bs - 90)
    )
    
    # Convert log-odds to probability and sample binary clinical outcomes
    probs = 1 / (1 + np.exp(-logit))
    adverse_event = (np.random.uniform(0, 1, n_samples) < probs).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'Family_History': family_history,
        'BMI': bmi,
        'Systolic_BP': systolic_bp,
        'Cholesterol': cholesterol,
        'Fasting_Blood_Sugar': fasting_bs,
        'Heart_Failure_Risk': adverse_event
    })
    return df

def train_and_save():
    print("Step 1/4: Generating clinical cohort dataset (6,000 patients)...")
    df = generate_clinical_dataset(6000)
    
    X = df.drop('Heart_Failure_Risk', axis=1)
    y = df['Heart_Failure_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Step 2/4: Training LightGBM Classifier with Monotonic Constraints...")
    # Monotonic constraints (+1) guarantee that increasing BMI, BP, Cholesterol, or Blood Sugar 
    # never accidentally decreases predicted risk due to tree-splitting noise.
    # Order: [Age, Sex, Family_History, BMI, Systolic_BP, Cholesterol, Fasting_Blood_Sugar]
    feature_monotone = [1, 0, 1, 1, 1, 1, 1]
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        monotone_constraints=feature_monotone,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Step 3/4: Evaluating Clinical Risk Model...")
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_class = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, preds_proba)
    acc = accuracy_score(y_test, preds_class)
    
    print("-" * 50)
    print(f"Validation ROC-AUC Score : {roc_auc:.4f}")
    print(f"Validation Accuracy      : {acc * 100:.2f}%")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, preds_class, target_names=["Low Risk (0)", "High Risk (1)"]))
    
    # Quick sanity check on our 62-year-old demo patient
    sample_patient = pd.DataFrame([{
        'Age': 62, 'Sex': 1, 'Family_History': 1,
        'BMI': 34.2, 'Systolic_BP': 155, 'Cholesterol': 260, 'Fasting_Blood_Sugar': 118
    }])
    sample_risk = model.predict_proba(sample_patient)[0][1] * 100
    print(f"Demo Patient (Age 62, BP 155, BMI 34.2) Predicted Risk: {sample_risk:.1f}%\n")
    
    print("Step 4/4: Saving model and reference dataset for DiCE...")
    joblib.dump(model, "lgbm_clinical_model.pkl")
    df.to_csv("clinical_reference_data.csv", index=False)
    print("✅ Saved 'lgbm_clinical_model.pkl' and 'clinical_reference_data.csv' successfully!")

if __name__ == "__main__":
    train_and_save()