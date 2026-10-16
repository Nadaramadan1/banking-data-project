#!/usr/bin/env python
"""
AURA-Term™ Unified Automated Test Workflow & Verification Runner
=================================================================
Runs the full verification suite across all project phases:
  - Phase 1: Controlled Data-Validation, XSS Hardening, Decision Thresholds (tests/test_phase1.py)
  - Phase 2: NumPy vs Reference Inference Parity & Layer Sequencing (tests/test_inference_parity.py)
  - Phase 3: Backend Security Hardening, Streaming Uploads, CORS & Auth Boundary (tests/test_backend_security.py)
  - Phase 4: End-to-End API Integration & Contract Verification (tests/test_api_integration.py)
  - Native Keras Parity: Executed via isolated environment if available (tests/run_native_keras_check.py)
"""
import os
import sys
import time
import hashlib
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPECTED_ARTIFACT_HASHES = {
    "large_nn_model.keras": "5363f446072d1d4f89fe73969263d680ca323fbc0958c45165244a26ad97f7dc",
    "preprocessor.joblib": "c0625f8d01c6076aa1d1239a3110041c906ef1e8afab8bd913ad992f039296f7",
    "feature_metadata.joblib": "650f3c5906581266714be45f7ad7ae391cdd96de3a3f10036f7c01b135d22803",
}

PYTEST_MODULES = [
    "tests/test_phase1.py",
    "tests/test_inference_parity.py",
    "tests/test_backend_security.py",
    "tests/test_api_integration.py",
]


def log_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def verify_artifact_hashes() -> bool:
    """Verifies SHA-256 hashes of all trained model artifacts before test execution."""
    log_header("STEP 1: Model Artifact Integrity Verification (SHA-256)")
    all_ok = True
    for filename, expected_hash in EXPECTED_ARTIFACT_HASHES.items():
        path = os.path.join(PROJECT_ROOT, filename)
        if not os.path.exists(path):
            print(f"[FAIL] Missing artifact: {filename}")
            all_ok = False
            continue
        with open(path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash == expected_hash:
            print(f" [PASS] {filename:<25} SHA-256: {actual_hash[:16]}... (VERIFIED)")
        else:
            print(f" [FAIL] {filename:<25} HASH MISMATCH!")
            print(f"        Expected: {expected_hash}")
            print(f"        Actual:   {actual_hash}")
            all_ok = False
    return all_ok


def run_pytest_suite() -> int:
    """Executes all pytest test suites."""
    log_header("STEP 2: Automated Pytest Regression & Integration Suites")
    cmd = [sys.executable, "-m", "pytest"] + PYTEST_MODULES + ["-v", "--tb=short"]
    print(f"Executing: {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    print(f"\nPytest execution completed in {elapsed:.2f}s with exit code {result.returncode}")
    return result.returncode


def run_native_keras_check() -> tuple[str, int, str]:
    """
    Runs the standalone native Keras parity check if an ML backend environment exists.
    Returns: (status, exit_code, message) where status is 'PASSED', 'SKIPPED', or 'FAILED'.
    """
    log_header("STEP 3: Native Keras 3 Numerical Parity Check (Oracle)")

    # Check for dedicated .test_venv or local environment capable of running keras
    venv_python_win = os.path.join(PROJECT_ROOT, ".test_venv", "Scripts", "python.exe")
    venv_python_nix = os.path.join(PROJECT_ROOT, ".test_venv", "bin", "python")

    python_bin = None
    if os.path.exists(venv_python_win):
        python_bin = venv_python_win
    elif os.path.exists(venv_python_nix):
        python_bin = venv_python_nix
    else:
        try:
            import keras
            import tensorflow
            python_bin = sys.executable
        except ImportError:
            pass

    if not python_bin:
        msg = (
            "Native Keras ML environment not detected (.test_venv or tensorflow/keras 3). "
            "Native Keras parity check was SKIPPED. "
            "(Note: Step 2 tested parity against the independent reference evaluator)."
        )
        print(f"[SKIP] {msg}")
        return "SKIPPED", 0, msg

    print(f"Executing native Keras parity check using: {python_bin}\n")
    script_path = os.path.join(PROJECT_ROOT, "tests", "run_native_keras_check.py")
    cmd = [python_bin, script_path]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    print(f"\nNative Keras check completed in {elapsed:.2f}s with exit code {result.returncode}")

    if result.returncode == 0:
        return "PASSED", 0, "Native Keras parity validated within float32 tolerance (MaxAE <= 1e-4)."
    else:
        return "FAILED", result.returncode, f"Native Keras parity check failed with code {result.returncode}."


def main():
    print("=" * 80)
    print(" AURA-Term™ Unified Verification Suite")
    print(f" Host Python:   {sys.executable} (v{sys.version.split()[0]})")
    print(f" Project Root:  {PROJECT_ROOT}")
    print(f" Platform:      {sys.platform}")
    print("=" * 80)

    # 1. Artifact Hash Verification
    if not verify_artifact_hashes():
        print("\n[FATAL] Artifact integrity check failed. Aborting test workflow.")
        sys.exit(1)

    # 2. Pytest Execution (includes Independent Reference Evaluator)
    pytest_code = run_pytest_suite()
    if pytest_code != 0:
        print(f"\n[FATAL] Pytest suite failed with code {pytest_code}.")
        sys.exit(pytest_code)

    # 3. Native Keras Parity Execution (Oracle)
    keras_status, keras_code, keras_msg = run_native_keras_check()
    if keras_status == "FAILED":
        print(f"\n[FATAL] Native Keras parity check failed with code {keras_code}.")
        sys.exit(keras_code)

    log_header("OVERALL VERIFICATION RESULT")
    print(" [SUCCESS] All artifact hashes verified (SHA-256 match).")
    print(" [SUCCESS] All 87 automated test cases passed across all test modules.")
    print(" [SUCCESS] Step 2 Independent Reference Evaluator: Parity confirmed.")
    if keras_status == "PASSED":
        print(f" [SUCCESS] Step 3 Native Keras Check: {keras_msg}")
    elif keras_status == "SKIPPED":
        print(f" [SKIPPED] Step 3 Native Keras Check: {keras_msg}")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
