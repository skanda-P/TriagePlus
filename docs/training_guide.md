# TriagePlus Training & Embedding Guide

This guide outlines the steps required to regenerate the embeddings and retrain the machine learning models used by the TriagePlus AI Engine.

## Prerequisites

Ensure you have your environment set up with the required dependencies:
```bash
cd ai_engine/ml_training
pip install -r ../../requirements.txt
# Alternatively, ensure xgboost, scikit-learn, pandas, and numpy are installed.
```

## 1. Generating the Evaluation Set

The evaluation set is used to tune the thresholds of the ML models and validate performance.

Run the evaluation generation script:
```bash
python generate_eval_set.py
```
**What it does:** Randomly samples 50 patients from the `test.csv` dataset and generates `eval_set.json` which can be used for benchmarking.

## 2. Training the XGBoost Triage Model

The core predictive model that powers the LangGraph classification node is an XGBoost model trained on the DDXPlus synthetic patient records.

Run the training script:
```bash
python train_triage_model.py
```

**What it does:**
1. Loads the DDXPlus `release_evidences.json` to map out the complete feature space.
2. Loads `train.csv` (using pandas).
3. Extracts and binarizes the symptom evidence codes (`E_*`).
4. Uses a `MultiLabelBinarizer` for features and `LabelEncoder` for the target pathology.
5. Trains an `XGBClassifier` using the fast histogram method (`tree_method='hist'`). 
6. Saves the models to the `ai_engine/ml_training/models/` directory:
   - `triage_xgb.json` (The trained XGBoost model)
   - `mlb.pkl` (The binarizer used for inference)
   - `label_encoder.pkl` (The label encoder used to decode predictions)

*Note: The script is configured to attempt training on CUDA (`device='cuda'`) for speed. If no compatible NVIDIA GPU is detected, it automatically falls back to CPU training.*

## 3. (Legacy) RAG Embeddings

> **Note:** The current architecture primarily uses the deterministic NetworkX Knowledge Graph and the XGBoost classifier. FAISS embeddings from the previous architecture have been removed.

If you ever need to re-implement semantic search over a medical corpus (like MedQuAD), ensure you use the `NeuML/pubmedbert-base-embeddings` model as it is specifically tuned for biomedical vocabulary, whereas generic embedding models like `all-MiniLM-L6-v2` will severely degrade performance.
