# Scikit-learn Library

## Overview
Scikit-learn is a comprehensive machine learning library for Python that provides simple and efficient tools for data mining and data analysis. It features various classification, regression, and clustering algorithms, including support vector machines, random forests, gradient boosting, k-means, and DBSCAN. Scikit-learn is built on NumPy, SciPy, and Matplotlib, making it a cornerstone of the Python data science ecosystem.

## Installation
```bash
pip install scikit-learn
```

## Key Features
- **Supervised Learning**: Classification and regression algorithms
- **Unsupervised Learning**: Clustering, dimensionality reduction, and density estimation
- **Model Selection**: Cross-validation, hyperparameter tuning, and model evaluation
- **Preprocessing**: Feature scaling, encoding, and transformation
- **Feature Extraction**: Text and image feature extraction
- **Pipeline**: Combine preprocessing and models into workflows
- **Integration**: Works seamlessly with NumPy, Pandas, and Matplotlib

## Core Components

### Supervised Learning

#### Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

#### Regression
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate model
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
```

### Unsupervised Learning

#### Clustering
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=200, c='red', marker='x', linewidths=3)
plt.title('K-means Clustering')
plt.show()
```

#### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
ax1.set_title('PCA')
ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
ax2.set_title('t-SNE')
plt.show()
```

## Data Preprocessing

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (to [0,1] range)
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)

# Robust scaling (handles outliers better)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### Feature Encoding
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Sample data with categorical variables
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'large', 'medium', 'small', 'large'],
    'price': [10, 20, 15, 12, 25]
})

# Label encoding
le = LabelEncoder()
data['color_encoded'] = le.fit_transform(data['color'])

# One-hot encoding
ohe = OneHotEncoder(sparse=False)
color_encoded = ohe.fit_transform(data[['color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['color']))

# Column transformer for mixed data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['price']),
        ('cat', OneHotEncoder(), ['color', 'size'])
    ]
)
```

### Handling Missing Values
```python
from sklearn.impute import SimpleImputer

# Simple imputation strategies
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_most_frequent = SimpleImputer(strategy='most_frequent')

# Apply imputation
X_imputed = imputer_mean.fit_transform(X)
```

## Model Selection and Evaluation

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# K-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Stratified K-fold (for classification)
strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(clf, X, y, cv=strat_cv, scoring='accuracy')
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# Grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Random search (faster for large parameter spaces)
from scipy.stats import uniform, randint
param_dist = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.001, 0.1),
    'kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)
```

### Model Evaluation Metrics
```python
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve
import seaborn as sns

# Classification metrics
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
y_pred_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.3f}")

# Regression metrics
from sklearn.metrics import mean_absolute_error, explained_variance_score
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"Explained Variance: {explained_variance_score(y_test, y_pred):.3f}")
```

## Pipelines

### Creating Pipelines
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Cross-validation with pipeline
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Pipeline CV Score: {cv_scores.mean():.3f}")
```

### Feature Union
```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Combine different feature extraction methods
feature_union = FeatureUnion([
    ('pca', PCA(n_components=3)),
    ('select', SelectKBest(k=3))
])

# Use in pipeline
pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', RandomForestClassifier())
])
```

## Popular Algorithms

### Ensemble Methods
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svc', SVC(probability=True))],
    voting='soft'
)
```

### Support Vector Machines
```python
from sklearn.svm import SVC, SVR

# Classification
svc = SVC(kernel='rbf', C=1.0, gamma='scale')

# Regression
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
```

### Neural Networks
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Multi-layer perceptron regressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
```

## Use Cases
- **Classification**: Email spam detection, image classification, medical diagnosis
- **Regression**: House price prediction, stock price forecasting, demand prediction
- **Clustering**: Customer segmentation, document clustering, image segmentation
- **Dimensionality Reduction**: Feature selection, data visualization, noise reduction
- **Anomaly Detection**: Fraud detection, network security, quality control
- **Recommendation Systems**: Product recommendations, content filtering

## Best Practices
1. **Data Preprocessing**: Always scale features and handle missing values
2. **Train-Test Split**: Use proper train/test splits and cross-validation
3. **Feature Engineering**: Create meaningful features from raw data
4. **Model Selection**: Try multiple algorithms and compare performance
5. **Hyperparameter Tuning**: Use grid search or random search for optimization
6. **Evaluation**: Use appropriate metrics for your problem type
7. **Overfitting**: Monitor for overfitting and use regularization techniques

## Advantages
- **Comprehensive**: Wide range of algorithms and tools
- **Easy to Use**: Consistent API across all algorithms
- **Well Documented**: Extensive documentation and examples
- **Community Support**: Large, active community
- **Integration**: Works well with other data science libraries
- **Production Ready**: Stable and reliable for production use

## Limitations
- **Deep Learning**: Limited support for deep neural networks
- **Scalability**: May not handle very large datasets efficiently
- **Customization**: Less flexible than lower-level frameworks
- **GPU Support**: Limited GPU acceleration compared to TensorFlow/PyTorch

## Related Libraries
- **NumPy**: Numerical computing foundation
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **TensorFlow/PyTorch**: Deep learning frameworks
- **XGBoost/LightGBM**: Gradient boosting libraries 