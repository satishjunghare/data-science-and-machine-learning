# XGBoost Library

## Overview
XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework, providing parallel tree boosting that solves many data science problems in a fast and accurate way. XGBoost is widely used in competitions and production systems due to its superior performance and scalability.

## Installation
```bash
# Basic installation
pip install xgboost

# With GPU support
pip install xgboost[gpu]

# Latest version
pip install xgboost==2.0.0

# From conda
conda install -c conda-forge xgboost
```

## Key Features
- **Gradient Boosting**: Advanced gradient boosting algorithm implementation
- **High Performance**: Optimized C++ backend with parallel processing
- **GPU Support**: CUDA acceleration for faster training
- **Regularization**: Built-in L1 and L2 regularization
- **Cross-Validation**: Built-in cross-validation support
- **Feature Importance**: Automatic feature importance calculation
- **Early Stopping**: Prevent overfitting with early stopping
- **Missing Values**: Native handling of missing values

## Core Concepts

### Basic Usage
```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create sample data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions
predictions = model.predict(dtest)
predictions_binary = (predictions > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy:.4f}")
```

### Scikit-learn API
```python
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train
xgb_clf.fit(X_train, y_train)

# Predict
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Regression
xgb_reg = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# For regression, create continuous target
y_reg = np.random.randn(1000)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

xgb_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = xgb_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"MSE: {mse:.4f}")
```

## Parameter Tuning

### Key Parameters
```python
# Core parameters
params = {
    # Booster parameters
    'max_depth': 3,              # Maximum depth of trees
    'learning_rate': 0.1,        # Step size shrinkage
    'n_estimators': 100,         # Number of boosting rounds
    'subsample': 0.8,            # Subsample ratio of training instances
    'colsample_bytree': 0.8,     # Subsample ratio of columns per tree
    'colsample_bylevel': 0.8,    # Subsample ratio of columns per level
    
    # Regularization
    'reg_alpha': 0,              # L1 regularization
    'reg_lambda': 1,             # L2 regularization
    'gamma': 0,                  # Minimum loss reduction for split
    
    # Tree construction
    'min_child_weight': 1,       # Minimum sum of instance weight in child
    'max_delta_step': 0,         # Maximum delta step for each tree
    
    # Randomness
    'random_state': 42,          # Random seed
    'scale_pos_weight': 1        # Control balance of positive and negative weights
}
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Grid search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Random search (faster for large parameter spaces)
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(50, 300),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

## Advanced Features

### Cross-Validation
```python
# Built-in cross-validation
params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Cross-validation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    metrics='logloss',
    early_stopping_rounds=10,
    seed=42
)

print("CV Results:")
print(cv_results.tail())

# Plot CV results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cv_results['train-logloss-mean'], label='Train')
plt.plot(cv_results['test-logloss-mean'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Cross-Validation')
plt.legend()
plt.show()
```

### Early Stopping
```python
# Training with early stopping
evals = [(dtrain, 'train'), (dtest, 'test')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=10
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")
```

### Feature Importance
```python
# Get feature importance
importance = model.get_score(importance_type='gain')
print("Feature Importance:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")

# Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.show()

# Using scikit-learn API
feature_importance = xgb_clf.feature_importances_
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

## GPU Acceleration

### GPU Training
```python
# GPU parameters
gpu_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',  # Use GPU
    'gpu_id': 0
}

# Train on GPU
gpu_model = xgb.train(gpu_params, dtrain, num_boost_round=100)

# Scikit-learn API with GPU
gpu_clf = XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)

gpu_clf.fit(X_train, y_train)
```

## Model Persistence

### Save and Load Models
```python
# Save model
model.save_model('xgboost_model.json')

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')

# Save as pickle (scikit-learn API)
import pickle

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

# Load pickle
with open('xgboost_model.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
```

### Model Interpretation
```python
# SHAP values for model interpretation
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_clf)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test, feature_names=[f'feature_{i}' for i in range(X_test.shape[1])])

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
```

## Custom Objectives and Metrics

### Custom Objective Function
```python
def custom_objective(y_true, y_pred):
    """
    Custom objective function
    """
    grad = 2 * (y_pred - y_true)
    hess = 2 * np.ones_like(y_pred)
    return grad, hess

# Use custom objective
custom_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': custom_objective
}

custom_model = xgb.train(custom_params, dtrain, num_boost_round=100)
```

### Custom Evaluation Metric
```python
def custom_metric(y_true, y_pred):
    """
    Custom evaluation metric
    """
    return 'custom_metric', np.mean((y_true - y_pred) ** 2)

# Use custom metric
metric_params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': custom_metric
}

metric_model = xgb.train(metric_params, dtrain, num_boost_round=100)
```

## Use Cases
- **Classification**: Binary and multi-class classification problems
- **Regression**: Continuous target prediction
- **Ranking**: Learning to rank problems
- **Anomaly Detection**: Outlier detection using isolation forests
- **Feature Selection**: Automatic feature importance and selection
- **Competitions**: Kaggle and other ML competitions
- **Production Systems**: High-performance production deployments

## Best Practices
1. **Start with Defaults**: Begin with default parameters and tune gradually
2. **Use Cross-Validation**: Always use cross-validation for parameter tuning
3. **Early Stopping**: Use early stopping to prevent overfitting
4. **Feature Engineering**: Invest in good feature engineering
5. **Regularization**: Use L1/L2 regularization to prevent overfitting
6. **Subsampling**: Use subsample and colsample parameters for large datasets
7. **GPU Acceleration**: Use GPU for large datasets when available
8. **Model Interpretation**: Use SHAP or other tools for model interpretation

## Advantages
- **Performance**: Superior performance on structured/tabular data
- **Scalability**: Handles large datasets efficiently
- **Flexibility**: Supports custom objectives and metrics
- **Regularization**: Built-in regularization to prevent overfitting
- **Missing Values**: Native handling of missing values
- **GPU Support**: GPU acceleration for faster training
- **Interpretability**: Feature importance and model interpretation tools

## Limitations
- **Black Box**: Less interpretable than linear models
- **Overfitting**: Can overfit if not properly tuned
- **Computational Cost**: Training can be computationally expensive
- **Feature Engineering**: Still requires good feature engineering
- **Hyperparameter Tuning**: Many parameters to tune

## Related Libraries
- **LightGBM**: Alternative gradient boosting library
- **CatBoost**: Gradient boosting with categorical features
- **Scikit-learn**: Machine learning framework
- **SHAP**: Model interpretation library
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing 