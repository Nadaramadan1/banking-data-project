"""
AURA-Term Phase 2 — Keras vs NumPy Inference Parity Test Suite
==============================================================
Validates whether the production NumPy forward-pass implementation in
model_engine.py is numerically equivalent to the original trained Keras model,
using the exact production preprocessing pipeline, feature ordering, and weights.

Covers:
  1. Keras model loading status and runtime environment checks
  2. Production preprocessing and feature-order verification (53 features)
  3. Batch-size-1 and multi-row batch inference tests
  4. Numerical parity comparison (MaxAE, MAE, tolerance checking)
  5. Layer-by-layer divergence diagnosis
"""
import io
import json
import os
import sys
import zipfile
import h5py
import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from model_engine import BankingModelEngine, engine

MODEL_PATH = os.path.join(PROJECT_ROOT, "large_nn_model.keras")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "preprocessor.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "feature_metadata.joblib")

# Numerical tolerance appropriate for float32 inference
TOLERANCE_FLOAT32 = 1e-4


# ---------------------------------------------------------------------------
# Independent Reference Evaluator (Reconstructed directly from .keras spec)
# ---------------------------------------------------------------------------

class KerasArtifactReferenceModel:
    """
    Independent reference implementation loaded directly from large_nn_model.keras.
    Attempts to load via native Keras 3 if an active ML backend exists; otherwise
    reconstructs the exact Sequential layer-by-layer execution from config.json
    and model.weights.h5 without relying on model_engine.py.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.keras_native_model = None
        self.keras_load_error = None
        self._try_load_native_keras()
        self._load_from_zip_spec()

    def _try_load_native_keras(self):
        try:
            import keras
            self.keras_native_model = keras.models.load_model(self.model_path)
        except Exception as e:
            self.keras_load_error = f"{type(e).__name__}: {str(e)}"

    def _load_from_zip_spec(self):
        with zipfile.ZipFile(self.model_path, 'r') as z:
            self.config = json.loads(z.read('config.json').decode('utf-8'))
            weights_bytes = z.read('model.weights.h5')
            with h5py.File(io.BytesIO(weights_bytes), 'r') as h5f:
                # Layer 1: Dense 13 (53 -> 128)
                self.w0 = h5f['layers/dense/vars/0'][:]
                self.b0 = h5f['layers/dense/vars/1'][:]
                # Layer 2: BatchNorm 4 (epsilon=0.001)
                self.gamma0 = h5f['layers/batch_normalization/vars/0'][:]
                self.beta0 = h5f['layers/batch_normalization/vars/1'][:]
                self.mean0 = h5f['layers/batch_normalization/vars/2'][:]
                self.var0 = h5f['layers/batch_normalization/vars/3'][:]

                # Layer 4: Dense 14 (128 -> 64)
                self.w1 = h5f['layers/dense_1/vars/0'][:]
                self.b1 = h5f['layers/dense_1/vars/1'][:]
                # Layer 5: BatchNorm 5 (epsilon=0.001)
                self.gamma1 = h5f['layers/batch_normalization_1/vars/0'][:]
                self.beta1 = h5f['layers/batch_normalization_1/vars/1'][:]
                self.mean1 = h5f['layers/batch_normalization_1/vars/2'][:]
                self.var1 = h5f['layers/batch_normalization_1/vars/3'][:]

                # Layer 7: Dense 15 (64 -> 32)
                self.w2 = h5f['layers/dense_2/vars/0'][:]
                self.b2 = h5f['layers/dense_2/vars/1'][:]

                # Layer 9: Dense 18 (32 -> 1)
                self.w3 = h5f['layers/dense_3/vars/0'][:]
                self.b3 = h5f['layers/dense_3/vars/1'][:]

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    @staticmethod
    def _batch_norm(x, gamma, beta, mean, var, eps=0.001):
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Executes inference according to the exact Keras Sequential layer sequence:
          Layer 0: Input(53)
          Layer 1: Dense(128, activation='relu')
          Layer 2: BatchNormalization(epsilon=0.001)
          Layer 3: Dropout(0.3) -> identity at test time
          Layer 4: Dense(64, activation='relu')
          Layer 5: BatchNormalization(epsilon=0.001)
          Layer 6: Dropout(0.25) -> identity at test time
          Layer 7: Dense(32, activation='relu')
          Layer 8: Dropout(0.15) -> identity at test time
          Layer 9: Dense(1, activation='sigmoid')
        """
        if self.keras_native_model is not None:
            raw = self.keras_native_model.predict(X, verbose=0)
            return np.asarray(raw, dtype=np.float64)

        X_f32 = np.asarray(X, dtype=np.float32)

        # Layer 1: Dense with relu
        z1 = np.dot(X_f32, self.w0) + self.b0
        a1 = self._relu(z1)

        # Layer 2: BatchNorm on ReLU activations
        bn1 = self._batch_norm(a1, self.gamma0, self.beta0, self.mean0, self.var0, eps=0.001)

        # Layer 4: Dense with relu
        z2 = np.dot(bn1, self.w1) + self.b1
        a2 = self._relu(z2)

        # Layer 5: BatchNorm on ReLU activations
        bn2 = self._batch_norm(a2, self.gamma1, self.beta1, self.mean1, self.var1, eps=0.001)

        # Layer 7: Dense with relu
        z3 = np.dot(bn2, self.w2) + self.b2
        a3 = self._relu(z3)

        # Layer 9: Dense with sigmoid
        z4 = np.dot(a3, self.w3) + self.b3
        return self._sigmoid(z4)


# Deterministic synthetic test samples (covering diverse demographics, balances, and categories)
DETERMINISTIC_SYNTHETIC_RECORDS = [
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


# ===========================================================================
# Step 1 & 5 — Runtime Environment & Keras Dependency Check
# ===========================================================================

class TestKerasEnvironmentAndLoading:
    """Checks whether the original Keras model can be loaded with installed dependencies."""

    def test_report_keras_loading_status(self):
        ref = KerasArtifactReferenceModel()
        if ref.keras_native_model is not None:
            print("\n[INFO] Native Keras 3 model loaded successfully via active backend.")
        else:
            print(f"\n[DEPENDENCY WARNING] Native Keras 3 could not run directly in Python 3.14: {ref.keras_load_error}")
            print("[INFO] Fallback to direct layer-by-layer specification evaluator enabled.")
            # Verify the underlying archive and weights were parsed cleanly
            assert hasattr(ref, 'w0')
            assert ref.w0.shape == (53, 128)
            assert ref.w3.shape == (32, 1)


# ===========================================================================
# Step 3 — Preprocessing and Feature Order Validation
# ===========================================================================

class TestPreprocessingAndFeatureOrder:
    """
    Validates that:
      - The production preprocessor produces exactly 53 features matching the model input.
      - Feature names, ordering, and categorical encodings match preprocessor.joblib.
      - Unknown categories or missing values conform to preprocessor behavior.
      - No feature is silently dropped or reordered.
    """
    def test_preprocessor_output_dimension_matches_model_input(self):
        df = engine.prepare_dataframe(DETERMINISTIC_SYNTHETIC_RECORDS)
        X = engine.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert X.shape[1] == 53, f"Expected 53 transformed features, got {X.shape[1]}"

    def test_feature_ordering_and_transformer_spec(self):
        prep = engine.preprocessor
        meta = engine.metadata
        transformers = {t[0]: t[2] for t in prep.transformers_}

        # Numerical pipeline verification
        num_cols = transformers['num']
        assert list(num_cols) == meta['numerical_features'], (
            f"Numerical feature mismatch: expected {meta['numerical_features']}, got {list(num_cols)}"
        )
        assert list(num_cols) == ['age', 'balance', 'pdays', 'previous']

        # Categorical pipeline verification
        cat_cols = transformers['cat']
        assert list(cat_cols) == meta['categorical_features'], (
            f"Categorical feature mismatch: expected {meta['categorical_features']}, got {list(cat_cols)}"
        )

        # Output feature names match 53 exact columns
        feature_names = prep.get_feature_names_out()
        assert len(feature_names) == 53
        assert feature_names[0] == "num__age"
        assert feature_names[1] == "num__balance"
        assert feature_names[2] == "num__pdays"
        assert feature_names[3] == "num__previous"
        assert feature_names[4] == "cat__job_admin."
        assert feature_names[-1] == "cat__balance_status_Zero"

    def test_unknown_categorical_values_handled_without_dimension_change(self):
        exotic_record = [{
            "age": 40, "balance": 1000, "pdays": -1, "previous": 0,
            "job": "astronaut",  # Unknown category
            "marital": "unspecified",
            "education": "doctorate",
            "default": "no", "housing": "no", "loan": "no",
            "month": "xyz",
            "poutcome": "unrecorded"
        }]
        df = engine.prepare_dataframe(exotic_record)
        X = engine.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert X.shape == (1, 53)
        assert np.all(np.isfinite(X))


# ===========================================================================
# Step 2 & 4 — Parity Comparison & Divergence Diagnosis
# ===========================================================================

class TestInferenceParityAndDiagnosis:
    """
    Evaluates whether model_engine.py's forward_pass produces outputs
    equivalent to the Keras model specification across batch-size-1 and multi-row batches.
    """
    @classmethod
    def setup_class(cls):
        cls.ref = KerasArtifactReferenceModel()

    def _evaluate_batch(self, records: list):
        df = engine.prepare_dataframe(records)
        X = engine.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()

        numpy_preds = engine.forward_pass(X).flatten()
        keras_preds = self.ref.predict_probabilities(X).flatten()

        abs_errors = np.abs(numpy_preds - keras_preds)
        max_ae = float(np.max(abs_errors))
        mean_ae = float(np.mean(abs_errors))
        outside_tol = int(np.sum(abs_errors > TOLERANCE_FLOAT32))

        return {
            "X": X,
            "numpy_preds": numpy_preds,
            "keras_preds": keras_preds,
            "abs_errors": abs_errors,
            "max_ae": max_ae,
            "mean_ae": mean_ae,
            "outside_tol": outside_tol,
            "total_samples": len(records)
        }

    def test_batch_size_one_output_properties(self):
        sample = [DETERMINISTIC_SYNTHETIC_RECORDS[0]]
        res = self._evaluate_batch(sample)
        assert res["numpy_preds"].shape == (1,)
        assert np.all(np.isfinite(res["numpy_preds"]))
        assert 0.0 <= res["numpy_preds"][0] <= 1.0
        assert np.all(np.isfinite(res["keras_preds"]))
        assert 0.0 <= res["keras_preds"][0] <= 1.0

    def test_multi_row_batch_output_properties(self):
        res = self._evaluate_batch(DETERMINISTIC_SYNTHETIC_RECORDS)
        assert res["numpy_preds"].shape == (len(DETERMINISTIC_SYNTHETIC_RECORDS),)
        assert np.all(np.isfinite(res["numpy_preds"]))
        assert np.all((res["numpy_preds"] >= 0.0) & (res["numpy_preds"] <= 1.0))

    def test_parity_divergence_measurement_and_diagnosis(self):
        """
        Measures the actual parity error between current model_engine.py and the Keras model.
        Reports the evidence:
          - Maximum Absolute Error (MaxAE)
          - Mean Absolute Error (MAE)
          - Failure count against TOLERANCE_FLOAT32 (1e-4)
          - Pinpoints the root cause: Activation Function vs. Batch-Normalization order.
        """
        all_samples = engine.get_sample_profiles()
        sample_records = [s['data'] for s in all_samples] + DETERMINISTIC_SYNTHETIC_RECORDS
        res = self._evaluate_batch(sample_records)

        print(f"\n==================== INFERENCE PARITY REPORT ====================")
        print(f"Total Samples Tested: {res['total_samples']}")
        print(f"Documented Tolerance: {TOLERANCE_FLOAT32} (float32 target)")
        print(f"Mean Absolute Error (MAE):    {res['mean_ae']:.6f}")
        print(f"Maximum Absolute Error (MaxAE): {res['max_ae']:.6f}")
        print(f"Predictions Outside Tolerance: {res['outside_tol']} / {res['total_samples']}")
        print(f"==================================================================")

        # DIAGNOSTIC PROOF OF DIVERGENCE:
        # In large_nn_model.keras, Layer 1 is Dense(128, activation='relu') followed by
        # Layer 2 BatchNormalization. In model_engine.py, forward_pass applies
        # BatchNorm FIRST, and ReLU SECOND.
        # Check moving mean: All 128 moving means in Layer 2 are strictly positive (min 0.00167),
        # demonstrating that BatchNorm was trained on post-ReLU non-negative activations.
        positive_mean0_count = int(np.sum(self.ref.mean0 >= 0))
        print(f"\n[DIAGNOSTIC] Layer 2 BatchNorm moving means >= 0: {positive_mean0_count} / {len(self.ref.mean0)}")
        print(f"[DIAGNOSTIC] Layer 5 BatchNorm moving means >= 0: {int(np.sum(self.ref.mean1 >= 0))} / {len(self.ref.mean1)}")
        assert positive_mean0_count == 128, "Trained BatchNorm moving means verify post-ReLU training distribution"

        # Record whether current implementation satisfies parity
        parity_established = (res["max_ae"] <= TOLERANCE_FLOAT32)
        print(f"[PARITY RESULT] Equivalence established: {parity_established}")
        assert parity_established, (
            f"Parity failed! MaxAE={res['max_ae']:.8f} exceeded tolerance {TOLERANCE_FLOAT32}. "
            f"MAE={res['mean_ae']:.8f}"
        )

    def test_native_keras_parity_via_isolated_venv(self):
        """
        Executes native Keras verification via isolated .test_venv if available.
        If isolated environment is not present, reports unverified status.
        """
        import subprocess
        venv_python = os.path.join(PROJECT_ROOT, ".test_venv", "Scripts", "python.exe")
        script_path = os.path.join(PROJECT_ROOT, "tests", "run_native_keras_check.py")
        if os.path.exists(venv_python) and os.path.exists(script_path):
            result = subprocess.run([venv_python, script_path], capture_output=True, text=True)
            assert result.returncode == 0, f"Native Keras verification failed: {result.stderr}"
            assert "[SUCCESS]" in result.stdout
        else:
            pytest.skip("Native Keras isolated environment not found; native parity remains unverified in this runner.")
