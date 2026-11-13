# Anomaly Detection: Method Overview & Benchmark Comparison

This repository provides a conceptual overview and an empirical benchmark of classical statistical and machine learning methods for tabular anomaly detection. It accompanies a slide deck covering classical statistical approaches, regression diagnostics, supervised/unsupervised ML, deep learning methods, **AnoMATE**, and a deep dive into **Isolation Forests**.

---

## Repository Structure

```text
.
├── Anomalydetection.py           # Main benchmark script (5-fold CV)
├── Results
├───── anomaly_results_summary.csv   # Aggregated evaluation metrics (mean across folds)
├───── plot1_bars.png                # Bar charts comparing AUROC, AP, and F1
├───── plot2_heatmap.png             # AUROC heatmap across datasets and methods
├───── plot3_cm_Speech.png           # Confusion matrices – Speech
├───── plot3_cm_Mammography.png      # Confusion matrices – Mammography
├───── plot3_cm_Optdigits.png        # Confusion matrices – Optdigits
├─── Data
└── README.md
```

*(Note: The benchmark datasets `speech.mat`, `mammography.mat`, and `optdigits.mat` must be placed in the root directory to re-run `Anomalydetection.py`.)*

---

## Methods Benchmarked

The script evaluates four anomaly detection approaches using **5-fold stratified cross-validation** (`random_state=42`):

1. **Interquartile Range (`IQR`)**
   - Computes per-feature $Q_1$, $Q_3$, and $IQR = Q_3 - Q_1$ on the training fold.
   - Scores each observation by the mean normalized distance outside $[Q_1 - 1.5 \cdot IQR,\; Q_3 + 1.5 \cdot IQR]$ across all features.
2. **Isolation Forest (`IForest`)**
   - `StandardScaler` + `IsolationForest` (`n_estimators=200`).
   - Uses the negated `decision_function` as the anomaly score.
3. **Autoencoder (`Autoencoder`)**
   - `MinMaxScaler` + symmetric bottleneck `MLPRegressor` (`relu`, `max_iter=200`, `early_stopping=True`) trained exclusively on normal samples (`y == 0`) to reconstruct its inputs.
   - Latent bottleneck size: $\max(2, \lfloor d / 4 \rfloor)$.
   - Uses per-sample reconstruction MSE as the anomaly score.
4. **Ensemble (`IF+AE`)**
   - Min-max normalizes the Isolation Forest and Autoencoder scores and averages them with equal weights ($0.5 \cdot s_{\text{IF}} + 0.5 \cdot s_{\text{AE}}$).

For thresholding, the contamination rate is estimated from the training fold (`np.clip(y_train.mean(), 0.01, 0.4)`), and the decision threshold is set at the $(1 - \text{contamination})$ quantile of validation scores (or combined scores for the ensemble).

---

## Datasets

The benchmark expects MATLAB v5 `.mat` files containing feature matrix `X` and binary label vector `y`:

| Dataset | File | Samples | Features | Anomaly Rate |
| :--- | :--- | ---: | ---: | ---: |
| **Speech** | `speech.mat` | 3,686 | 400 | 1.65% |
| **Mammography** | `mammography.mat` | 11,183 | 6 | 2.32% |
| **Optdigits** | `optdigits.mat` | 5,216 | 64 | 2.88% |

---

## Benchmark Results

Mean metrics across 5 folds (from `anomaly_results_summary.csv`):

| Dataset | Method | AUROC | AP | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Speech** | IQR | 0.480 | 0.019 | 0.000 | 0.000 | 0.000 |
| **Speech** | IForest | 0.482 | 0.032 | 0.040 | 0.067 | 0.050 |
| **Speech** | Autoencoder | **0.491** | 0.040 | 0.042 | 0.049 | 0.045 |
| **Speech** | IF+AE | 0.485 | **0.041** | **0.062** | **0.067** | **0.064** |
| **Mammography** | IQR | 0.855 | **0.452** | **0.463** | **0.462** | **0.459** |
| **Mammography** | IForest | **0.862** | 0.212 | 0.238 | 0.223 | 0.229 |
| **Mammography** | Autoencoder | 0.841 | 0.166 | 0.240 | 0.231 | 0.234 |
| **Mammography** | IF+AE | 0.861 | 0.203 | 0.238 | 0.238 | 0.238 |
| **Optdigits** | IQR | 0.467 | 0.026 | 0.000 | 0.000 | 0.000 |
| **Optdigits** | IForest | 0.719 | 0.056 | 0.020 | 0.020 | 0.020 |
| **Optdigits** | Autoencoder | **0.983** | **0.676** | **0.679** | **0.680** | **0.677** |
| **Optdigits** | IF+AE | 0.932 | 0.320 | 0.371 | 0.373 | 0.372 |

### Key Takeaways

- **Speech (400 features, 1.65% anomalies):** High dimensionality with relatively few samples causes both univariate rules (IQR) and tree/reconstruction-based models to perform near random (~0.48–0.49 AUROC).
- **Mammography (6 features, 2.32% anomalies):** Low-dimensional feature space with distinct univariate point outliers. While Isolation Forest achieves the highest AUROC (`0.862`), the simple IQR rule significantly outperforms all ML models in AP (`0.452`) and F1 (`0.459`).
- **Optdigits (64 features, 2.88% anomalies):** Rich multidimensional dependencies across features cannot be captured by univariate thresholds (`IQR` F1 = `0.000`). The Autoencoder models these non-linear correlations directly and achieves strong detection performance (`0.983` AUROC, `0.677` F1).

---

## Getting Started

### Requirements

- Python 3.9+
- Dependencies:
  ```bash
  pip install numpy pandas matplotlib seaborn scipy scikit-learn
  ```

### Run the Benchmark

Place `speech.mat`, `mammography.mat`, and `optdigits.mat` in the working directory and run:

```bash
python Anomalydetection.py
```

This will output `anomaly_results_summary.csv`, `plot1_bars.png`, `plot2_heatmap.png`, and `plot3_cm_<Dataset>.png`.

---

## Presentation Topics Included

Beyond the experimental benchmark, the accompanying presentation covers:
- **Taxonomy of Anomaly Detection Methods:** Classical statistical tests (Z-Score, IQR, Grubbs's Test, MAD, GESD, ARIMA), Regression diagnostics (Cook's Distance, Leverage, DFBETA), Supervised ML (Logistic Regression, SVM, KNN), Unsupervised ML (K-Means, Isolation Forest, DBSCAN, LOF), and Deep Learning (Autoencoders, RNNs/LSTMs, GANs).
- **AnoMATE (Mixed-type Tabular Embeddings):** Architectural comparison against standard Autoencoders for mixed numerical, categorical, boolean, and temporal data.
- **Deep Dive: Isolation Forest:** Intuition, random axis-aligned partitioning, algorithmic details (`iForest`, `iTree`, `PathLength`), time complexity, limitations, and Python/R implementations (`scikit-learn`, `isotree`).
