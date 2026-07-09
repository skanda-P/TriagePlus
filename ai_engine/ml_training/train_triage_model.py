import os
import json
import ast
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

data_dir = Path(__file__).parent / "data" / "DDXPlus"
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def train_xgboost():
    logging.info("Loading release_evidences.json to determine feature space...")
    with open(data_dir / "release_evidences.json", "r", encoding="utf-8") as f:
        evidences_meta = json.load(f)
        
    # We can use the base evidence codes, plus expand categorical values
    # For simplicity and speed in this triage model, we will treat each unique string in EVIDENCES as a distinct feature.
    # We will build a vocabulary of all possible evidence strings from the training data, OR pre-build it.
    # Actually, to be robust, let's just fit a CountVectorizer or DictVectorizer on the lists.
    from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
    
    # Due to size (670MB, ~1M rows), let's read a sample to save time and memory for this run
    logging.info("Loading train.csv...")
    df_train = pd.read_csv(data_dir / "train.csv", usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"], nrows=200000)
    
    logging.info("Parsing evidences...")
    # EVIDENCES is a string "['E_48', 'E_50', ...]"
    # Use literal_eval
    df_train['EVIDENCES'] = df_train['EVIDENCES'].apply(ast.literal_eval)
    
    # We also want to include AGE and SEX in the features
    # Let's map SEX to binary: F=0, M=1
    df_train['SEX'] = df_train['SEX'].map({'F': 0, 'M': 1}).fillna(0)
    
    logging.info("Binarizing evidences...")
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_evidences = mlb.fit_transform(df_train['EVIDENCES'])
    
    # Save the MLB so we can binarize user inputs the same way
    with open(models_dir / "mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)
        
    logging.info(f"Evidence features shape: {X_evidences.shape}")
    
    import scipy.sparse as sp
    # Combine AGE, SEX, and X_evidences
    age_sex = df_train[['AGE', 'SEX']].values
    X = sp.hstack([age_sex, X_evidences]).tocsr()
    
    logging.info("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df_train['PATHOLOGY'])
    
    with open(models_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    logging.info(f"Number of classes: {len(le.classes_)}")
    
    logging.info("Training XGBoost model...")
    # Train XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist', # fast histogram method
        device='cuda', # use GPU if available
        n_jobs=-1
    )
    
    try:
        clf.fit(X, y)
    except xgb.core.XGBoostError as e:
        logging.warning(f"CUDA failed with {e}. Falling back to CPU...")
        clf.set_params(device='cpu')
        clf.fit(X, y)
        
    model_path = models_dir / "triage_xgb.json" # newer format is json
    clf.save_model(model_path)
    logging.info(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    train_xgboost()
