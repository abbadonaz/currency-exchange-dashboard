***

# Isolation Forest Algorithm (iForest)

Isolation Forest (iForest) is an unsupervised **ensemble machine learning algorithm** specifically designed for **anomaly detection (outlier detection)**. It distinguishes itself from other methods by focusing on directly *isolating* observations rather than modeling the profile of normal data.

## I. Core Mechanism and Architecture: The Ensemble Approach

### A. The Principle of Isolation (The Tree Mechanism)

The fundamental concept rests on the observation that anomalies are rare and structurally different, making them **easier to isolate**.

1.  **Random Partitioning:** The core process involves recursively partitioning the data space. In each step, a feature is randomly selected, and a split value is randomly chosen between that feature's maximum and minimum values.
2.  **Path Length:** This recursive partitioning is visualized as a tree structure. The number of splits required to isolate a sample—the **path length** from the root node to the leaf—is the measure of isolation.
3.  **Individual Isolation:** Random partitioning results in **noticeably shorter paths** for anomalies compared to normal points.

### B. The Ensemble Aspect (The Forest Mechanism)

The final decision relies on the collective output of many randomized trees, hence the term "Forest."

1.  **Base Estimators:** The implementation uses an **ensemble of `ExtraTreeRegressor` instances**. The number of trees is controlled by the `n_estimators` parameter (default 100).
2.  **Collective Scoring:** The final anomaly score is determined by the **path length averaged over a forest of such random trees**. When a **forest of random trees collectively produce shorter path lengths** for specific samples, they are deemed highly likely to be anomalies.
3.  **Fitted Collection:** The actual collection of fitted sub-estimators is stored in the **`estimators_`** attribute.

### C. Scikit-learn Implementation Details

*   **Depth Constraint:** The maximum depth of each tree is explicitly limited to $\text{ceil}(\log_2(n))$, where $n$ is the number of samples used to build that tree. This depth constraint ensures efficiency because deep trees are unnecessary for achieving quick isolation.

## II. Data Input and Optimization

For maximum efficiency and performance:

| Usage | Recommended Data Type/Format | Source |
| :--- | :--- | :--- |
| **Input Samples (General)** | Use `dtype=np.float32` | |
| **Sparse Matrix Input for `fit`** | Use sparse `csc_matrix` | |
| **Sparse Matrix Input for `predict`/Scoring** | Internally converted to sparse `csr_matrix` | |

## III. Training, Scoring, and Classification Methods

### A. Training (`fit` and `n_jobs`)

The `fit(*X, y=None, sample_weight=None)` method trains the estimator ensemble.

*   **Unsupervised:** The `y` parameter is ignored, maintaining API consistency.
*   **Parallelization during `fit`:** The `n_jobs` parameter (default `None`; use `-1` for all processors) controls the number of jobs run in parallel during the training phase.

### B. Scoring (`decision_function` and `score_samples`)

The score indicates the degree of isolation:

*   **`decision_function(X)`:** Returns the average anomaly score.
    *   **Negative scores** represent outliers; **positive scores** represent inliers.
    *   The lower the score, the more abnormal the sample.
*   **`score_samples(X)`:** Returns the **opposite** of the anomaly score defined in the original paper.
*   **Parallelization Note:** Scoring methods can be parallelized using a **`joblib` context** (e.g., using threading), but this **does NOT** inherently use the `n_jobs` parameter initialized in the class. This is because calculating the score may be faster without parallelization for small numbers of samples (e.g., 1000 or less).

### C. Classification (`predict` and `fit_predict`)

These methods convert the score into a binary label (+1 or -1):

*   **`predict(X)`:** Returns an array where **+1** signifies an **inlier** and **-1** signifies an **outlier**.
*   **`fit_predict(X)`:** Performs `fit` on $X$ and immediately returns the labels (1 for inliers, -1 for outliers) for those training samples.

## IV. The Classification Threshold: `contamination` and `offset_`

The classification threshold is dynamically set during fitting, based on the expected proportion of anomalies.

1.  **`contamination`:** This parameter specifies the **expected proportion of outliers** in the dataset. It can be `'auto'` or a `float` in the range (0, 0.5]. It defines the threshold on the scores.
2.  **`offset_`:** This float attribute formally defines the classification boundary. It links the scoring functions via the relation:
    $$\text{decision\_function} = \text{score\_samples} - \text{offset}\_ \quad$$
3.  **Threshold Determination:**
    *   **If `contamination='auto'`:** The `offset_` is set to **-0.5**. This default assumes inlier scores are near 0 and outlier scores are near -1.
    *   **If `contamination` is a float:** The $\text{offset}\_$ is calculated during fitting such that the expected number of outliers (samples with $\text{decision\_function} < 0$) is obtained in the training data.
    *   **Classification Rule:** A sample is classified as an outlier (-1) if its `decision_function` score is less than 0.

## V. Key Parameters and Hyperparameter Tuning Hints

| Parameter | Description / Default | Tuning Strategy Hint |
| :--- | :--- | :--- |
| **`contamination`** | Expected outlier proportion ('auto' or float [0, 0.5]). | **Crucial:** Set based on **domain knowledge** regarding anomaly proportion. It defines the cutoff threshold. |
| **`n_estimators`** | Number of trees (default 100). | Increase to stabilize the average path length (anomaly score). |
| **`max_samples`** | Samples drawn per tree (default `'auto'` = $\min(256, n_{\text{samples}})$). | The default is highly efficient. Adjust if needed for very large datasets. |
| **`max_features`** | Number of features drawn per tree (default 1.0). | Using a float < 1.0 enables feature subsampling, increasing robustness in high-dimensional data, but may increase runtime. |
| **`random_state`** | Controls pseudo-randomness. | Always set this to an integer for **reproducible results**. |
| **`n_jobs`** | Number of parallel jobs for `fit` (default `None`). | Use `-1` to leverage all processors during the training phase. |
| **`bootstrap`** | Sampling with replacement (`True`) or without (`False`). | Default is `False`. Only change to test sampling stability. |

## VI. Comparison to Other Outlier Detection Methods

Isolation Forest is often compared to other unsupervised outlier detection estimators in scikit-learn:

1.  **`EllipticEnvelope`:** Detects outliers in datasets assumed to be **Gaussian distributed** (covariance-based).
2.  **`OneClassSVM` (OCSVM):** Estimates the **support of a high-dimensional distribution** by defining a boundary around the normal data (kernel method).
3.  **`LocalOutlierFactor` (LOF):** Uses **local density deviation** relative to neighbors to define outliers (density-based).

iForest is generally favored for its computational efficiency and its ability to handle high-dimensional data without relying on density estimation or distance metrics.

## 🎥 Video Summary

[![Watch the summary video](IsolationForestExplaination)](https://notebooklm.google.com/notebook/f0178d81-5ee1-4ad3-a82c-b6b6425151e0?artifactId=75a76800-e7ee-4a53-b884-132f61b7a5e3)  

## ⚠️ Disclaimer

These learning notes were **generated using my personal notes combined with publicly available resources with the assistance of NotebookLM**. 
They are included here to **showcase my personal learning process** and provide a little introduction to the methods used in the project.  
***