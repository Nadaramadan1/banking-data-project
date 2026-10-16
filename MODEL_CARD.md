# AURA-Term™ Model Card

## 1. Model Overview & Target Definition
* **Model Name**: AURA-Term Deep Neural Deposit Propensity Classifier
* **Version**: 2.4.0
* **Model Type**: Feedforward Deep Neural Network (MLP) with Batch Normalization and ReLU activations, outputting a single Sigmoid probability.
* **Target Variable**: Binary target $y \in \{0, 1\}$ indicating whether a retail banking client subscribed to a long-term deposit product upon direct telemarketing campaign contact ($1 = \text{subscribed}$, $0 = \text{not subscribed}$).
* **Dataset Source**: Derived from the classic Bank Marketing Data Set (Moro et al., 2014; Portuguese retail banking direct telemarketing campaigns).
* **Intended Use**: Pre-call outreach prioritization and lead ranking for direct retail term-deposit marketing campaigns.
* **Prohibited & Unsupported Uses**:
  - **Credit Underwriting**: This model does NOT predict creditworthiness, default risk, debt serviceability, or credit capacity. Using this model for credit granting, loan underwriting, or risk-weighted capital calculations is prohibited.
  - **Actuarial or Financial Guarantees**: This model does NOT predict or guarantee deposit intake volumes or liquidity inflows.

---

## 2. Input Features & Preprocessing Pipeline

### A. Raw Input Features (12 raw + 2 derived)
* **Numerical Features (4)**:
  - `age`: Client age in years (validated range: 18–105).
  - `balance`: Liquid bank balance volume in EUR.
  - `pdays`: Number of days that passed after the client was last contacted from a previous campaign (-1 means client was not previously contacted).
  - `previous`: Number of contacts performed before this campaign.
* **Categorical Features (8 raw + 2 derived)**:
  - `job`: Employment category (12 categories: `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`).
  - `marital`: Marital status (3 categories: `divorced`, `married`, `single`).
  - `education`: Educational attainment (4 categories: `primary`, `secondary`, `tertiary`, `unknown`).
  - `default`: Credit default history indicator (`no`, `yes`).
  - `housing`: Housing loan / mortgage indicator (`no`, `yes`).
  - `loan`: Personal debt loan indicator (`no`, `yes`).
  - `month`: Last contact month of year (12 categories: `jan`–`dec`).
  - `poutcome`: Outcome of previous marketing campaign (4 categories: `failure`, `other`, `success`, `unknown`).
  - `age_group` (derived): Age cohort binning (`Young`, `Early_Career`, `Mid_Career`, `Older`, `Senior`).
  - `balance_status` (derived): Liquid balance status binning (`Negative`, `Positive`, `Zero`).

### B. Preprocessing & Leakage Protection
* **Exclusion of `duration` (Verified)**:
  In telemarketing datasets, call duration (`duration`) is only known *after* a phone call concludes and strongly leaks the outcome (duration=0 guarantees y=0). The preprocessing pipeline ([`preprocessor.joblib`](file:///d:/web_course/banking-data-project/preprocessor.joblib)) strictly excludes `duration` from its 53 output dimensions, ensuring all input features are realistically available *before* a marketing call is dialed.
* **Feature Encoding**:
  - Missing values imputed via `SimpleImputer`.
  - Numerical columns standardized via `StandardScaler`.
  - Categorical columns encoded via `OneHotEncoder(handle_unknown='ignore')`.
  - Transformed input dimension: exactly 53 dense numerical features.

---

## 3. Architecture & Numerical Inference

* **Input Layer**: 53 normalized features.
* **Layer Sequence**:
  1. Dense Layer 1: 128 units, ReLU activation (Weights: $53 \times 128$, Biases: $128$).
  2. Batch Normalization 1: Moving mean, variance, scale ($\gamma$), offset ($\beta$), $\epsilon = 10^{-3}$.
  3. Dropout Layer 1: $p = 0.30$ (identity pass during inference).
  4. Dense Layer 2: 64 units, ReLU activation (Weights: $128 \times 64$, Biases: $64$).
  5. Batch Normalization 2: Moving mean, variance, scale ($\gamma$), offset ($\beta$), $\epsilon = 10^{-3}$.
  6. Dropout Layer 2: $p = 0.25$ (identity pass during inference).
  7. Dense Layer 3: 32 units, ReLU activation (Weights: $64 \times 32$, Biases: $32$).
  8. Dropout Layer 3: $p = 0.15$ (identity pass during inference).
  9. Output Dense Layer: 1 unit, Sigmoid activation (Weights: $32 \times 1$, Bias: $1$).
* **Vectorized NumPy Engine**:
  Inference is executed via a self-contained vectorized forward pass in [`model_engine.py`](file:///d:/web_course/banking-data-project/model_engine.py). Weights are extracted directly from `large_nn_model.keras` via `h5py`. Parity against native Keras 3 has been empirically verified across all benchmark samples to within float32 tolerance ($\text{MaxAE} \le 7 \times 10^{-8}$).

---

## 4. Thresholds, Decision Tiers & Financial Heuristics

### A. Binary Classification Threshold (0.50)
* **Status**: Uncalibrated mathematical default ($z \ge 0 \implies \sigma(z) \ge 0.50$).
* **Limitation**: Term deposit marketing data typically exhibits an ~11–12% base conversion rate. The 0.50 cutoff was not tuned against class imbalance or empirical cost-benefit trade-offs (cost of outreach call vs. value of acquired deposit).

### B. Decision Tiers
* **Cutoffs**:
  - $\text{Probability} \ge 0.70$: Tier 1 — High Propensity / Priority Outreach
  - $0.45 \le \text{Probability} < 0.70$: Tier 2 — Moderate Propensity / Standard Outreach
  - $\text{Probability} < 0.45$: Tier 3 — Low Propensity / Marketing Suppression
* **Limitation**: These tier boundaries are heuristic application rules, not statistically or empirically validated segments.

### C. Heuristic Portfolio Inflow & Deposit Estimates
* **Deposit Estimation Formula**:
  $$\text{expected\_deposit} = \max(1000, \text{round}(\text{balance} \times 0.45, -2)) \quad (\text{or } €2500 \text{ if balance } \le 0)$$
* **Projected Conversion Rate Formula**:
  $$\text{conversion\_rate\_projected} = \frac{\text{Tier1\_Count} + 0.5 \times \text{Tier2\_Count}}{\text{Total\_Count}} \times 100$$
* **Mandatory Disclosure**: These financial values are rule-of-thumb heuristic formulas for campaign scenario simulation. They do **not** represent verified statistical forecasts, observed customer deposit amounts, or guaranteed liquidity yields.

---

## 5. Explainability & Fairness Limitations

### A. Explainability Mechanism
* **Method**: Single-Variable Baseline Sensitivity Analysis.
* **Implementation**: For each of 11 financial variables, the client's feature value is substituted with a static median reference profile (`neutral_reference`: 40yo, married, management, balance €1,368), and the probability delta $\Delta = f(x) - f(x_{i \leftarrow \text{ref}})$ is computed.
* **Limitations**:
  - This is **not** true Shapley Additive Explanations (SHAP).
  - Does not satisfy efficiency axioms (the sum of feature impacts does not equal total model output delta).
  - Does not capture non-linear feature interactions or multi-variable co-dependencies.

### B. Fairness, Demographics & Regulatory Alignment
* **Available Demographics**: Only `age`, `marital`, and `education` are present in the dataset. Protected classes including gender/sex, race, ethnicity, religion, and disability are not recorded.
* **Disparate Impact Metrics**: The demographic fairness parity ratios displayed in the UI/telemetry are explicitly labeled as **illustrative sample data**. No formal disparate impact assessment (e.g. EEOC 80% rule or demographic parity) has been conducted against verified production ground truth.
* **Regulatory Citations**: Basel III/IV, EU AI Act Title III, and ECOA are cited solely as high-level reference design frameworks. The system holds no regulatory certifications or formal approvals.

---

## 6. Training Provenance & Unverifiable Evidence

* **Absent Records**:
  - The repository does not contain the original training script, optimization loss logs, hyperparameter tuning history, or ground-truth labeled validation/test splits.
  - Metrics such as Precision-Recall AUC, ROC-AUC, F1-score, Brier score calibration, and cross-validation confidence intervals cannot be verified from the repository.
* **Requirements Before Real-World Operational Deployment**:
  1. Full training code and labeled validation datasets must be archived and linked.
  2. Formal PR-AUC and ROC-AUC metrics must be measured on an out-of-time test split.
  3. Cost-sensitive threshold tuning should replace the arbitrary 0.50 / 0.70 / 0.45 cutoffs.
  4. Real customer deposit volumes must be modeled empirically rather than with the $45\%$ heuristic.

---

## 7. Artifact Integrity & Test Verification

All model artifacts are pinned and protected by strict SHA-256 integrity checks:

| Artifact File | Verified SHA-256 Checksum |
|---|---|
| [`large_nn_model.keras`](file:///d:/web_course/banking-data-project/large_nn_model.keras) | `5363f446072d1d4f89fe73969263d680ca323fbc0958c45165244a26ad97f7dc` |
| [`preprocessor.joblib`](file:///d:/web_course/banking-data-project/preprocessor.joblib) | `c0625f8d01c6076aa1d1239a3110041c906ef1e8afab8bd913ad992f039296f7` |
| [`feature_metadata.joblib`](file:///d:/web_course/banking-data-project/feature_metadata.joblib) | `650f3c5906581266714be45f7ad7ae391cdd96de3a3f10036f7c01b135d22803` |

### Automated Verification Command
To run all 87 unit, security, integration, and native Keras parity tests:
```bash
python run_all_tests.py
```
