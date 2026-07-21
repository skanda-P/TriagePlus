import ast
import json
import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# Data paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/DDXPlus"))
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
MANIFEST_PATH = os.path.join(MODEL_DIR, "training_manifest.json")


def extract_base_evidence(evidences_str):
    try:
        evidences = ast.literal_eval(evidences_str)
        # Extract base evidence (e.g., 'E_54' from 'E_54_@_V_161')
        return list(set([e.split('_@_')[0] for e in evidences]))
    except Exception:
        return []


def main():
    print("Loading training data...")
    df = pd.read_csv(TRAIN_CSV)
    
    print("Processing features...")
    # Age
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').fillna(0)
    
    # Sex (M=1, F=0)
    df['SEX_BIN'] = df['SEX'].apply(lambda x: 1 if x == 'M' else 0)
    
    # Evidences (Symptoms)
    print("Parsing evidences...")
    df['parsed_evidences'] = df['EVIDENCES'].apply(extract_base_evidence)
    
    print("Binarizing evidences...")
    mlb = MultiLabelBinarizer(sparse_output=True)
    evidence_matrix = mlb.fit_transform(df['parsed_evidences'])
    
    # Target
    le = LabelEncoder()
    y = le.fit_transform(df['PATHOLOGY'])
    
    # Create feature matrix combining AGE, SEX, and EVIDENCES
    import scipy.sparse as sp
    
    print("Constructing feature matrix...")
    # Convert AGE and SEX to sparse matrix and horizontally stack
    age_sex_matrix = sp.csr_matrix(df[['AGE', 'SEX_BIN']].values)
    X = sp.hstack([age_sex_matrix, evidence_matrix])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    
    print("Training XGBoost model (CUDA accelerated if available)...")
    # Auto-detect CUDA device; if not available or GPU error, fall back to CPU
    device = os.getenv("XGBOOST_DEVICE", "cuda")
    clf = xgb.XGBClassifier(
        tree_method='hist',
        device=device,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(le.classes_),
        verbosity=1
    )
    
    try:
        clf.fit(X, y)
    except xgb.core.XGBoostError as e:
        err = str(e)
        if "cuda" in err.lower() or "gpu" in err.lower() or "cublas" in err.lower():
            print(f"WARNING: CUDA/GPU error ({err}). Falling back to CPU...")
            clf.set_params(device='cpu')
            clf.fit(X, y)
        else:
            raise
    
    print("Saving model and encoders...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the XGBoost model
    clf.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    
    # Save the encoders for prediction time
    with open(os.path.join(MODEL_DIR, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)
        
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Write training manifest for inference to restore constructor params
    manifest = {
        "mlb_classes": mlb.classes_.tolist(),
        "le_classes": le.classes_.tolist(),
        "num_class": len(le.classes_),
        "objective": "multi:softprob",
        "feature_order": ["AGE", "SEX_BIN"] + mlb.classes_.tolist(),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Training manifest saved to {MANIFEST_PATH}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()