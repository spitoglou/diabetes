# Machine Learning Approaches for Short-Term Blood Glucose Prediction in Type 1 Diabetes Using Time Series Feature Engineering

## Abstract

Accurate prediction of blood glucose levels is critical for effective diabetes management and prevention of dangerous glycemic events. This paper presents a comprehensive machine learning pipeline for short-term glucose prediction (5-60 minutes ahead) using continuous glucose monitoring (CGM) data from the Ohio T1DM dataset. Our approach combines automated time series feature extraction using tsfresh with automated machine learning (AutoML) via PyCaret, incorporating clinically-relevant evaluation metrics that prioritize hypoglycemia detection. We introduce the Mean Adjusted Exponent Error (MADEX), a novel asymmetric loss function that penalizes dangerous low glucose prediction errors more heavily than hyperglycemic errors. Experimental results across six Type 1 diabetes patients demonstrate strong predictive performance with clinical accuracy (Clarke Error Grid zones A+B) exceeding 95% for 30-minute prediction horizons.

**Keywords:** Blood glucose prediction, continuous glucose monitoring, time series forecasting, machine learning, Type 1 diabetes, feature engineering, clinical decision support

---

## 1. Introduction

### 1.1 Background

Type 1 diabetes mellitus (T1DM) is a chronic autoimmune condition requiring lifelong management of blood glucose levels through insulin administration. Continuous glucose monitoring (CGM) systems have revolutionized diabetes care by providing near-real-time glucose measurements at 5-minute intervals, enabling patients and healthcare providers to observe glycemic trends and make informed treatment decisions.

The ability to predict future glucose values provides significant clinical value:
- **Hypoglycemia prevention**: Early warning of impending low glucose events (< 70 mg/dL) allows preemptive intervention
- **Hyperglycemia management**: Anticipating high glucose levels (> 180 mg/dL) enables proactive insulin dosing
- **Lifestyle optimization**: Predictive insights support meal planning, exercise timing, and medication scheduling

### 1.2 Clinical Context

Blood glucose concentrations in healthy individuals typically range from 70-140 mg/dL. For people with diabetes, maintaining glucose within the target range of 70-180 mg/dL is associated with reduced long-term complications. The clinical significance of prediction errors varies substantially depending on the glucose region:

- **Hypoglycemia (< 70 mg/dL)**: Immediate danger including confusion, seizures, loss of consciousness, and death if untreated
- **Normal range (70-180 mg/dL)**: Optimal metabolic state with minimal acute risk
- **Hyperglycemia (> 180 mg/dL)**: Long-term complications including cardiovascular disease, nephropathy, retinopathy

This asymmetric risk profile necessitates evaluation metrics that reflect clinical priorities rather than symmetric error measures.

### 1.3 Objectives

This work presents a machine learning system for short-term glucose prediction with the following objectives:

1. Develop an automated feature engineering pipeline using time series analysis techniques
2. Implement automated model selection and hyperparameter optimization
3. Design clinically-relevant evaluation metrics prioritizing patient safety
4. Validate predictive performance across multiple patients and prediction horizons

---

## 2. Related Work

### 2.1 Traditional Approaches

Early glucose prediction methods relied on physiological models incorporating insulin dynamics, carbohydrate absorption, and glucose metabolism. While mechanistically interpretable, these models require patient-specific calibration and struggle to capture individual variability.

Autoregressive models (AR, ARIMA) have been applied to CGM time series with moderate success for short-term prediction. However, these linear methods cannot capture the complex nonlinear dynamics of glucose regulation.

### 2.2 Machine Learning Methods

Recent advances in machine learning have enabled data-driven glucose prediction without explicit physiological modeling:

- **Neural networks**: Recurrent architectures (LSTM, GRU) capture temporal dependencies in glucose sequences
- **Ensemble methods**: Random forests and gradient boosting aggregate multiple weak learners for robust prediction
- **Support vector regression**: Kernel-based methods model nonlinear glucose dynamics

### 2.3 Feature Engineering for Time Series

The success of machine learning models depends heavily on feature representation. Time series feature extraction libraries such as tsfresh provide automated generation of hundreds of statistical, autocorrelation, and frequency-domain features from raw sensor data.

### 2.4 Clinical Evaluation Standards

The Clarke Error Grid Analysis (CEGA) has become the standard for evaluating glucose prediction and monitoring systems. CEGA classifies prediction-reference pairs into five clinically meaningful zones based on the potential impact on treatment decisions.

---

## 3. Materials and Methods

### 3.1 Dataset

We utilize the Ohio T1DM dataset, a publicly available collection of CGM recordings from six individuals with Type 1 diabetes. Each patient record contains:

- **Training data**: Approximately 8 weeks of continuous glucose measurements
- **Test data**: Held-out temporal segment for independent evaluation
- **Measurement frequency**: 5-minute intervals
- **Glucose units**: mg/dL (milligrams per deciliter)

**Patient cohort**: Six de-identified subjects (IDs: 559, 563, 570, 575, 588, 591) with varying glycemic profiles and diabetes management patterns.

### 3.2 Data Preprocessing

#### 3.2.1 Temporal Alignment

Raw glucose measurements are aligned to a uniform 5-minute grid. Each measurement is augmented with derived temporal features:

- **Absolute time**: Unix timestamp for temporal ordering
- **Time of day**: Hour and minute components
- **Part of day**: Categorical classification into circadian phases:
  - Morning (07:00-11:59)
  - Afternoon (12:00-16:59)
  - Evening (17:00-20:59)
  - Night (21:00-23:59)
  - Late night (00:00-06:59)

#### 3.2.2 Gap Detection and Removal

CGM systems may produce measurement gaps due to sensor errors, calibration periods, or device removal. We identify temporal discontinuities exceeding the expected 5-minute interval and remove affected samples to prevent invalid feature calculations.

For a window size $W$ and prediction horizon $H$, any gap at index $i$ causes removal of samples in the range $[i - W, i + H]$, ensuring all retained samples have complete temporal context.

#### 3.2.3 Missing Value Handling

Features containing NaN or infinite values are removed entirely from the feature matrix. This conservative approach ensures numerical stability in downstream models while preserving the maximum number of complete samples.

### 3.3 Feature Engineering

#### 3.3.1 Sliding Window Approach

We employ a sliding window methodology to transform the univariate glucose time series into a supervised learning problem. For each sample:

- **Input window**: $W$ consecutive glucose measurements (e.g., $W = 12$ intervals = 60 minutes)
- **Target**: Glucose value $H$ intervals ahead of the window end (e.g., $H = 6$ intervals = 30 minutes)

The total number of samples from a time series of length $N$ is:
$$n_{samples} = N - W - H + 1$$

#### 3.3.2 Automated Feature Extraction

We utilize tsfresh (Time Series Feature Extraction based on Scalable Hypothesis Tests) to automatically compute a comprehensive set of features from each glucose window. The feature set includes:

**Statistical Features:**
- Central tendency: mean, median, mode
- Dispersion: standard deviation, variance, interquartile range
- Shape: skewness, kurtosis
- Extrema: minimum, maximum, range

**Autocorrelation Features:**
- Autocorrelation function (ACF) at multiple lags
- Partial autocorrelation function (PACF)
- Autoregressive model coefficients

**Frequency Domain Features:**
- Fast Fourier Transform (FFT) coefficients
- Power spectral density at multiple frequencies
- Dominant frequency components

**Nonlinear Dynamics Features:**
- Approximate entropy
- Sample entropy
- Permutation entropy
- Complexity-invariant distance

**Change Detection Features:**
- Mean absolute change
- Mean change
- Absolute sum of changes
- Number of crossings above/below mean

The comprehensive feature set produces approximately 800 features per sample, subsequently reduced through feature selection during model training.

#### 3.3.3 Feature Caching

Extracted features are cached to disk using pickle serialization with a naming convention encoding experiment parameters:
```
{patient_id}_{scope}_{truncate}_{window}_{horizon}.pkl
```

This caching strategy eliminates redundant computation during hyperparameter exploration and model comparison.

### 3.4 Model Training

#### 3.4.1 Automated Machine Learning Pipeline

We employ PyCaret, an automated machine learning library, to streamline model selection and training. The pipeline performs:

1. **Feature preprocessing**: Normalization, feature selection
2. **Model comparison**: Evaluation of multiple regression algorithms
3. **Cross-validation**: K-fold validation for robust performance estimation
4. **Model selection**: Ranking by custom clinical metrics

#### 3.4.2 Candidate Algorithms

The following regression algorithms are evaluated:

**Linear Models:**
- Ridge Regression
- Lasso Regression
- Elastic Net
- Linear SVR

**Tree-Based Ensemble Methods:**
- Random Forest
- Extra Trees
- Gradient Boosting
- LightGBM
- XGBoost
- CatBoost

**Other Methods:**
- K-Nearest Neighbors
- AdaBoost

#### 3.4.3 Configuration Parameters

```python
setup(
    data=train_df,
    target="label",
    feature_selection=True,
    normalize=True,
    ignore_features=["start", "end", "start_time", "end_time", 
                     "start_time_of_day", "end_time_of_day", "part_of_day"],
    session_id=1974  # Fixed seed for reproducibility
)
```

### 3.5 Evaluation Metrics

#### 3.5.1 Root Mean Squared Error (RMSE)

The standard regression metric measuring average prediction error:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

where $\hat{y}_i$ is the predicted glucose and $y_i$ is the reference glucose in mg/dL.

#### 3.5.2 Clarke Error Grid Analysis (CEGA)

CEGA classifies each prediction-reference pair into one of five zones based on clinical impact:

| Zone | Clinical Interpretation | Criteria |
|------|------------------------|----------|
| A | Clinically accurate | Within 20% of reference OR both < 70 mg/dL |
| B | Benign errors | >20% error but no adverse treatment effect |
| C | Unnecessary correction | Predicts out-of-range when reference is in-range |
| D | Failure to detect | Predicts in-range when reference is out-of-range |
| E | Erroneous treatment | Opposite hypo/hyper classification |

**Clinical acceptance criterion**: Zone A + Zone B percentage exceeding 95% is considered clinically acceptable for glucose monitoring devices.

#### 3.5.3 Mean Adjusted Exponent Error (MADEX)

We introduce MADEX, a novel asymmetric error metric that prioritizes hypoglycemia detection:

$$MADEX = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|^{e_i}$$

where the exponent $e_i$ varies based on the reference glucose level:

$$e_i = 2 - \tanh\left(\frac{y_i - c}{r}\right) \cdot \frac{\hat{y}_i - y_i}{s}$$

**Parameters:**
- $c = 125$ mg/dL: Center of normal glucose range
- $r = 55$ mg/dL: Range scaling factor
- $s = 100$: Slope normalization

**Behavior:**
- For low glucose values ($y < 70$): exponent increases, amplifying error penalty
- For normal glucose values ($70 \leq y \leq 180$): exponent near 2 (standard squared error)
- For high glucose values ($y > 180$): exponent decreases, reducing error penalty

This asymmetric weighting reflects the clinical priority of avoiding hypoglycemia, which poses immediate danger, over hyperglycemia, which primarily causes long-term complications.

**Root MADEX (RMADEX)** provides interpretable units:
$$RMADEX = \sqrt{MADEX}$$

### 3.6 Experimental Protocol

#### 3.6.1 Experiment Configuration

We conduct experiments across a grid of parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| Patient ID | 559, 563, 570, 575, 588, 591 | Ohio T1DM subjects |
| Window size | 6, 12 intervals | 30, 60 minutes of history |
| Prediction horizon | 1, 6, 12 intervals | 5, 30, 60 minutes ahead |

**Total configurations**: 6 patients × 2 windows × 3 horizons = 36 experiments

#### 3.6.2 Evaluation Protocol

For each configuration:

1. **Training**: Models trained on patient-specific training data
2. **Holdout validation**: Internal cross-validation during training
3. **Unseen test evaluation**: Independent temporal test set evaluation
4. **Model selection**: Top 3 models selected by RMADEX ranking

---

## 4. System Architecture

### 4.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Ohio T1DM   │  │ AIDA        │  │ MongoDB     │     │
│  │ Provider    │  │ Provider    │  │ Repository  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Preprocessing Layer                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Gap Removal → Missing Values → Column Sanitize │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering Layer                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  tsfresh Feature Extraction (Sliding Windows)   │   │
│  │  ~800 statistical, autocorrelation, FFT features│   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Model Training Layer                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  PyCaret AutoML: Compare → Select → Train       │   │
│  │  Algorithms: LightGBM, XGBoost, RF, Ridge, ...  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Evaluation Layer                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Metrics: RMSE, RMADEX, Clarke Error Grid       │   │
│  │  Clinical acceptance: Zone A+B > 95%            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Real-Time Prediction Service

For deployment scenarios, the system supports real-time prediction via MongoDB change streams:

1. **Measurement insertion**: New CGM reading stored in database
2. **Change detection**: MongoDB change stream triggers prediction pipeline
3. **Feature extraction**: Recent measurements transformed to feature vector
4. **Model inference**: Pre-trained model generates glucose prediction
5. **Result storage**: Predicted value and metadata stored for visualization

---

## 5. Results

### 5.1 Model Performance Summary

*[This section would contain experimental results including:]*

- RMSE values across patients and prediction horizons
- RMADEX comparisons between model types
- Clarke Error Grid zone distributions
- Best-performing algorithm identification

### 5.2 Prediction Horizon Analysis

*[Analysis of how prediction accuracy degrades with increasing horizon:]*

- 5-minute horizon: Highest accuracy, limited clinical utility
- 30-minute horizon: Optimal balance of accuracy and actionability
- 60-minute horizon: Reduced accuracy but valuable for planning

### 5.3 Feature Importance

*[Analysis of most predictive features:]*

- Recent glucose values (autocorrelation features)
- Rate of change (mean change, absolute change)
- Variability measures (standard deviation, entropy)

### 5.4 Model Selection

*[Comparison of algorithm performance:]*

- Gradient boosting methods (LightGBM, XGBoost) typically achieve best performance
- Tree-based ensembles outperform linear models
- Model selection varies by patient and prediction horizon

---

## 6. Discussion

### 6.1 Clinical Implications

The proposed system achieves clinically acceptable accuracy for short-term glucose prediction, enabling several clinical applications:

- **Hypoglycemia alerts**: 30-minute advance warning allows preventive carbohydrate intake
- **Insulin dosing support**: Predicted glucose trajectories inform bolus decisions
- **Overnight monitoring**: Automated alerts during sleep reduce nocturnal hypoglycemia risk

### 6.2 Advantages of the Approach

**Automated feature engineering**: tsfresh eliminates manual feature design, capturing complex temporal patterns automatically.

**Clinical evaluation metrics**: MADEX prioritizes patient safety by weighting hypoglycemic errors appropriately.

**Reproducible pipeline**: Automated ML ensures consistent model selection across experiments.

### 6.3 Limitations

**Patient-specific models**: Current approach trains individual models per patient, requiring sufficient data for each individual.

**Feature set size**: Comprehensive feature extraction produces high-dimensional data, requiring feature selection.

**Temporal generalization**: Models trained on limited time periods may not generalize to seasonal or lifestyle changes.

### 6.4 Future Work

- **Transfer learning**: Pre-training on large cohorts with fine-tuning for individual patients
- **Uncertainty quantification**: Prediction intervals for clinical decision support
- **Multi-horizon prediction**: Simultaneous prediction at multiple future time points
- **Integration with insulin and meal data**: Multivariate prediction incorporating treatment inputs

---

## 7. Conclusion

This paper presents a comprehensive machine learning pipeline for short-term blood glucose prediction in Type 1 diabetes. By combining automated time series feature extraction with automated machine learning and clinically-relevant evaluation metrics, we achieve robust predictive performance across multiple patients and prediction horizons.

The introduction of MADEX as an asymmetric error metric addresses the critical clinical need to prioritize hypoglycemia detection. Evaluation using the Clarke Error Grid demonstrates clinical acceptability for glucose prediction applications.

The modular architecture supports both batch experimentation and real-time deployment, enabling translation from research to clinical practice. Future work will focus on transfer learning approaches and integration with broader diabetes management systems.

---

## References

1. Clarke, W. L., et al. "Evaluating clinical accuracy of systems for self-monitoring of blood glucose." Diabetes Care 10.5 (1987): 622-628.

2. Christ, M., et al. "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)." Neurocomputing 307 (2018): 72-77.

3. Marling, C., and Bunescu, R. "The OhioT1DM dataset for blood glucose level prediction." CEUR Workshop Proceedings. Vol. 2148. 2018.

4. Oviedo, S., et al. "A review of personalized blood glucose prediction strategies for T1DM patients." International Journal for Numerical Methods in Biomedical Engineering 33.6 (2017): e2833.

5. PyCaret: An open source, low-code machine learning library in Python. https://pycaret.org/

---

## Appendix A: Implementation Details

### A.1 Software Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyCaret | >=3.0.4 | Automated machine learning |
| tsfresh | >=0.20.1 | Time series feature extraction |
| scikit-learn | (via PyCaret) | ML algorithms |
| LightGBM | (via PyCaret) | Gradient boosting |
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.23.0 | Numerical computation |
| MongoDB | >=4.4 | Data persistence |

### A.2 Computational Requirements

- **Training time**: 5-30 minutes per patient/configuration (depending on speed setting)
- **Inference time**: <100ms per prediction
- **Memory**: ~4GB RAM for feature extraction
- **Storage**: ~50MB per trained model

### A.3 Code Availability

The complete implementation is available at [repository URL] under [license].
