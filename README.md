# Wine Classification Using KNN, PCA, Rule-Based Classification, and Linear Regression

## 📌 Project Overview

This project investigates different approaches for classifying wines from the **Wine dataset** provided by `scikit-learn`.

The following methods are implemented and compared:

* K-Nearest Neighbors (KNN)
* Principal Component Analysis (PCA)
* Rule-Based Classification
* KNN with different distance metrics
* Linear Regression used as a classification approach
* Visualization of decision boundaries in a 2D PCA space

The main goal is to examine how feature scaling, dimensionality reduction, distance metrics, and different classification strategies affect the classification accuracy.

---

## 📊 Dataset

The project uses the built-in Wine dataset from `scikit-learn`.

```python
from sklearn.datasets import load_wine

data = load_wine()
X = data.data
Y = data.target
```

The dataset contains:

* **178 samples**
* **13 numerical features**
* **3 wine classes**

### Features

The 13 input features are:

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

The target variable represents three wine classes:

```text
0
1
2
```

---

# 🔹 Step 1 — Data Preparation and KNN

The dataset is divided into training and testing sets using a 70/30 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
```

Because KNN is distance-based, the features are standardized using `StandardScaler`.

```python
data_standard = StandardScaler()

X_train_scaled = data_standard.fit_transform(X_train)
X_test_scaled = data_standard.transform(X_test)
```

A KNN classifier with `k=3` is then trained:

```python
nn = KNeighborsClassifier(n_neighbors=3)
nn.fit(X_train_scaled, y_train)
```

### Result

```text
KNN accuracy: 0.9630
```

The model correctly classifies approximately **96.30%** of the test samples.

---

# 🔹 Step 2 — PCA + KNN

Principal Component Analysis (PCA) is used to reduce the original 13-dimensional feature space to 2 dimensions.

```python
pca = PCA(n_components=2)

X_pca_train = pca.fit_transform(X_train_scaled)
X_pca_test = pca.transform(X_test_scaled)
```

The explained variance ratio is calculated:

```text
Cumulative explained variance:
[0.36196226 0.54960087]
```

Therefore, the first two principal components explain approximately:

```text
54.96% of the total variance
```

KNN is then trained using the two PCA components.

### Result

```text
KNN with PCA (2D) accuracy: 0.9815
```

The 2D PCA representation produced a test accuracy of approximately **98.15%**.

---

# 🔹 Step 3 — Rule-Based Classification

A simple rule-based classifier is implemented using only the `alcohol` feature.

The rules are:

```text
if alcohol < 12:
    class = 0

if 12 <= alcohol <= 13:
    class = 1

if alcohol > 13:
    class = 2
```

The classifier is applied to the test set.

### Result

```text
Rule-Based System Accuracy: 0.4815
```

The rule-based approach achieves approximately **48.15% accuracy**.

This demonstrates that using only alcohol concentration is not sufficient to accurately distinguish the three wine classes.

---

# 🔹 Step 4 — Comparing KNN Distance Metrics

Several distance metrics are tested with KNN:

* Euclidean
* Manhattan
* Cosine
* Chebyshev
* Mahalanobis

Each classifier uses:

```python
n_neighbors=3
```

The resulting test accuracies are:

| Distance Metric | Test Accuracy |
| --------------- | ------------: |
| Euclidean       |        0.9630 |
| Manhattan       |        0.9630 |
| Cosine          |        0.9815 |
| Chebyshev       |        0.9444 |
| Mahalanobis     |        0.9074 |

### Interpretation

The results show that the choice of distance metric affects KNN performance.

The cosine metric obtained the highest test accuracy among the tested metrics:

```text
Cosine: 98.15%
```

Euclidean and Manhattan distance produced:

```text
96.30%
```

Chebyshev produced:

```text
94.44%
```

Mahalanobis produced:

```text
90.74%
```

---

# 🔹 Training vs Testing Accuracy

Training and testing accuracy were also calculated.

| Metric      | Training Accuracy | Test Accuracy |
| ----------- | ----------------: | ------------: |
| Euclidean   |            0.9516 |        0.9630 |
| Manhattan   |            0.9919 |        0.9630 |
| Cosine      |            0.9435 |        0.9815 |
| Chebyshev   |            0.9597 |        0.9444 |
| Mahalanobis |            0.9677 |        0.9074 |

The difference between training and testing accuracy illustrates how the selected distance metric affects the model's generalization on unseen samples.

---

# 🔹 Step 5 — Linear Regression as a Classification Approach

Although Linear Regression is normally used for predicting continuous values, it is also tested here as an alternative classification approach.

The class labels are transformed:

```text
Class 0 → 0.0
Class 1 → 0.5
Class 2 → 1.0
```

This transformation is performed using:

```python
tf = FunctionTransformer(lambda x: x / 2.0)
```

A Linear Regression model is then trained:

```python
lin_reg = LinearRegression()

lin_reg.fit(X_train_scaled, y_train_reg)
```

The continuous predictions are converted back to classes by assigning each prediction to the nearest class level:

```python
levels = np.array([0.0, 0.5, 1.0])
```

### Result

```text
Linear Regression accuracy: 0.9444
```

The resulting classification accuracy is approximately **94.44%**.

This shows that regression can provide a reasonable approximation for this dataset, but it is not naturally designed for multiclass classification.

---

# 📈 Decision Boundary Visualization

The project also visualizes classification decision boundaries in the 2D PCA space.

PCA is used to project the original standardized 13-dimensional data into two dimensions:

```python
X_pca_full = pca.transform(X_full_scaled)
```

A grid is generated across the PCA space, and the trained model predicts the class of each point in the grid.

The resulting predictions are displayed using `contourf`, while the original samples are shown using a scatter plot.

This makes it possible to visually inspect how different models separate the three wine classes.

---

# 📋 Overall Results

The main results can be summarized as follows:

| Method            |   Accuracy |
| ----------------- | ---------: |
| KNN — Euclidean   | **96.30%** |
| KNN — PCA (2D)    | **98.15%** |
| KNN — Manhattan   | **96.30%** |
| KNN — Cosine      | **98.15%** |
| KNN — Chebyshev   | **94.44%** |
| KNN — Mahalanobis | **90.74%** |
| Rule-Based        | **48.15%** |
| Linear Regression | **94.44%** |

---

# 🧠 Conclusion

This project demonstrates several important concepts in machine learning.

First, **feature scaling is important for KNN**, because KNN relies on distances between observations.

Second, **PCA can reduce the dimensionality of the data while maintaining useful information**. In this experiment, two principal components explained approximately 54.96% of the variance, while KNN using these two components achieved an accuracy of 98.15% on the selected test split.

Third, the simple **rule-based classifier performed considerably worse**, showing that a single feature such as alcohol concentration does not contain enough information to reliably classify all three wine categories.

The experiment with different KNN distance metrics also demonstrates that the distance function can significantly affect model performance.

Finally, Linear Regression achieved 94.44% accuracy after converting its continuous predictions into discrete classes. However, since the original problem is a multiclass classification problem, classification-specific models such as KNN are more directly suited to the task.

Overall, the experiments illustrate the effects of:

* Feature scaling
* Dimensionality reduction
* Distance metrics
* Rule-based classification
* Regression-based classification
* Decision-boundary visualization

---

# 🛠️ Requirements

The project requires Python and the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

# ▶️ How to Run

Run the Python code sequentially in a Jupyter Notebook or Python environment.

The main workflow is:

```text
Load Wine Dataset
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
KNN Classification
       ↓
PCA Dimensionality Reduction
       ↓
KNN + PCA
       ↓
Rule-Based Classification
       ↓
Compare Distance Metrics
       ↓
Linear Regression
       ↓
Decision Boundary Visualization
```

---

# 📁 Suggested Project Structure

```text
wine-classification/
│
├── README.md
├── wine_classification.py
└── requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
```

---

## ⚠️ Note About Reproducibility

The reported accuracies are based on a single train/test split with:

```python
random_state=42
```

Therefore, the reported values describe the performance on this particular split. For a more robust evaluation, cross-validation could be used in a future version of the project.
