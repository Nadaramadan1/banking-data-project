"""
Standalone verification script to compare Native Keras 3 predictions against
both the current and corrected NumPy implementations on identical preprocessed inputs.
Executed using .test_venv/Scripts/python.exe
"""
import os
import sys
import numpy as np
import pandas as pd
import keras

# Compatibility shim
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model_engine import engine

MODEL_PATH = os.path.join(PROJECT_ROOT, "large_nn_model.keras")

print("[1] Loading native Keras model from:", MODEL_PATH)
keras_model = keras.models.load_model(MODEL_PATH)
print("    Native Keras loaded successfully! Version:", keras.__version__)

# Define current vs corrected forward pass
def current_forward(X, eng):
    return eng.forward_pass(X)

def corrected_forward(X, eng):
    z1 = np.dot(X, eng.w0) + eng.b0
    a1 = eng._relu(z1)
    bn1 = eng._batch_norm(a1, eng.gamma0, eng.beta0, eng.mean0, eng.var0, epsilon=0.001)

    z2 = np.dot(bn1, eng.w1) + eng.b1
    a2 = eng._relu(z2)
    bn2 = eng._batch_norm(a2, eng.gamma1, eng.beta1, eng.mean1, eng.var1, epsilon=0.001)

    z3 = np.dot(bn2, eng.w2) + eng.b2
    a3 = eng._relu(z3)

    z4 = np.dot(a3, eng.w3) + eng.b3
    return eng._sigmoid(z4)

# Test 1: Production Archetypes
print("\n[2] Evaluating 5 Production Archetypes:")
samples = engine.get_sample_profiles()
df_samples = engine.prepare_dataframe([s['data'] for s in samples])
X_samples = engine.preprocessor.transform(df_samples)
if hasattr(X_samples, 'toarray'):
    X_samples = X_samples.toarray()

k_preds_s = keras_model.predict(X_samples, verbose=0).flatten()
cur_preds_s = current_forward(X_samples, engine).flatten()
corr_preds_s = corrected_forward(X_samples, engine).flatten()

print("-" * 105)
print(f"{'Profile ID':<18} | {'Native Keras':<14} | {'Current NumPy':<14} | {'Curr Diff':<12} | {'Corrected NumPy':<16} | {'Corr Diff'}")
print("-" * 105)
for idx, s in enumerate(samples):
    kp = float(k_preds_s[idx])
    cp = float(cur_preds_s[idx])
    corrp = float(corr_preds_s[idx])
    diff_c = abs(cp - kp)
    diff_corr = abs(corrp - kp)
    print(f"{s['id']:<18} | {kp:<14.6f} | {cp:<14.6f} | {diff_c:<12.6f} | {corrp:<16.6f} | {diff_corr:.8f}")
print("-" * 105)

# Test 2: 10 Synthetic Diverse Samples
synthetic_records = [
    {"age": 22, "balance": -350, "pdays": -1, "previous": 0, "job": "student", "marital": "single", "education": "secondary", "default": "no", "housing": "no", "loan": "no", "month": "may", "poutcome": "unknown"},
    {"age": 28, "balance": 1500, "pdays": 30, "previous": 1, "job": "technician", "marital": "single", "education": "tertiary", "default": "no", "housing": "yes", "loan": "no", "month": "jun", "poutcome": "failure"},
    {"age": 35, "balance": 0, "pdays": -1, "previous": 0, "job": "blue-collar", "marital": "married", "education": "primary", "default": "yes", "housing": "yes", "loan": "yes", "month": "jul", "poutcome": "unknown"},
    {"age": 42, "balance": 7800, "pdays": 90, "previous": 2, "job": "services", "marital": "married", "education": "secondary", "default": "no", "housing": "yes", "loan": "no", "month": "aug", "poutcome": "other"},
    {"age": 48, "balance": 45000, "pdays": 180, "previous": 4, "job": "management", "marital": "married", "education": "tertiary", "default": "no", "housing": "no", "loan": "no", "month": "oct", "poutcome": "success"},
    {"age": 55, "balance": 12000, "pdays": -1, "previous": 0, "job": "admin.", "marital": "divorced", "education": "secondary", "default": "no", "housing": "no", "loan": "no", "month": "nov", "poutcome": "unknown"},
    {"age": 63, "balance": 32000, "pdays": 45, "previous": 2, "job": "self-employed", "marital": "married", "education": "tertiary", "default": "no", "housing": "no", "loan": "no", "month": "feb", "poutcome": "success"},
    {"age": 68, "balance": 18500, "pdays": 120, "previous": 3, "job": "retired", "marital": "married", "education": "secondary", "default": "no", "housing": "no", "loan": "no", "month": "mar", "poutcome": "success"},
    {"age": 75, "balance": 85000, "pdays": 90, "previous": 5, "job": "retired", "marital": "divorced", "education": "tertiary", "default": "no", "housing": "no", "loan": "no", "month": "apr", "poutcome": "success"},
    {"age": 31, "balance": 2400, "pdays": -1, "previous": 0, "job": "entrepreneur", "marital": "single", "education": "tertiary", "default": "no", "housing": "yes", "loan": "yes", "month": "sep", "poutcome": "unknown"},
]

df_syn = engine.prepare_dataframe(synthetic_records)
X_syn = engine.preprocessor.transform(df_syn)
if hasattr(X_syn, 'toarray'):
    X_syn = X_syn.toarray()

k_preds_syn = keras_model.predict(X_syn, verbose=0).flatten()
cur_preds_syn = current_forward(X_syn, engine).flatten()
corr_preds_syn = corrected_forward(X_syn, engine).flatten()

# Overall Metrics across all 15 samples
all_k = np.concatenate([k_preds_s, k_preds_syn])
all_cur = np.concatenate([cur_preds_s, cur_preds_syn])
all_corr = np.concatenate([corr_preds_s, corr_preds_syn])

print("\n[3] OVERALL PARITY METRICS (15 samples across batch-1 and multi-row):")
print(f"Total evaluated samples: {len(all_k)}")
print(f"Keras probability range: [{np.min(all_k):.6f}, {np.max(all_k):.6f}]")
print(f"Current NumPy probability range: [{np.min(all_cur):.6f}, {np.max(all_cur):.6f}]")
print(f"Corrected NumPy probability range: [{np.min(all_corr):.6f}, {np.max(all_corr):.6f}]")

print("\n--- Current Uncorrected NumPy vs Native Keras ---")
diff_current = np.abs(all_cur - all_k)
max_ae_c = float(np.max(diff_current))
mean_ae_c = float(np.mean(diff_current))
print(f"Maximum Absolute Error (MaxAE): {max_ae_c:.6f}")
print(f"Mean Absolute Error (MAE):    {mean_ae_c:.6f}")
print(f"Samples outside 1e-4 tolerance: {int(np.sum(diff_current > 1e-4))} / {len(all_k)}")

print("\n--- Corrected NumPy vs Native Keras ---")
diff_corr = np.abs(all_corr - all_k)
max_ae_corr = float(np.max(diff_corr))
mean_ae_corr = float(np.mean(diff_corr))
print(f"Maximum Absolute Error (MaxAE): {max_ae_corr:.8f}")
print(f"Mean Absolute Error (MAE):    {mean_ae_corr:.8f}")
print(f"Samples outside 1e-4 tolerance: {int(np.sum(diff_corr > 1e-4))} / {len(all_k)}")
assert max_ae_corr <= 1e-4, f"Corrected forward pass exceeds 1e-4 tolerance: {max_ae_corr}"
print("\n[SUCCESS] Native Keras and Corrected NumPy implementations achieve full numerical parity within 1e-4 tolerance!")
