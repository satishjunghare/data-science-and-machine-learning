# LightGBM Library

## Overview
LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, support for parallel and GPU learning, and capable of handling large-scale data. LightGBM is particularly popular in machine learning competitions and production systems due to its speed and performance.

## Installation
```bash
# Basic installation
pip install lightgbm

# With GPU support
pip install lightgbm[gpu]

# Latest version
pip install lightgbm==4.1.0

# From conda
conda install -c conda-forge lightgbm
```

## Key Features
- **High Performance**: Optimized for speed and memory efficiency
- **Leaf-wise Growth**: More efficient tree growth strategy
- **GPU Support**: GPU acceleration for faster training
- **Categorical Features**: Native support for categorical variables
- **Parallel Learning**: Multi-threading and distributed training
- **Memory Efficient**: Optimized memory usage for large datasets
- **Early Stopping**: Built-in early stopping to prevent overfitting
- **Cross-Validation**: Integrated cross-validation support

## Core Concepts

### Basic Usage
```python
import lightgbm as lgb
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

# Create Dataset (LightGBM's optimized data structure)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100)

# Make predictions
predictions = model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy:.4f}")
```

### Scikit-learn API
```python
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
lgb_clf = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)

# Train
lgb_clf.fit(X_train, y_train)

# Predict
y_pred = lgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Regression
lgb_reg = LGBMRegressor(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)

# For regression, create continuous target
y_reg = np.random.randn(1000)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

lgb_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lgb_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"MSE: {mse:.4f}")
```

## Parameter Tuning

### Key Parameters
```python
# Core parameters
params = {
    # Boosting parameters
    'boosting_type': 'gbdt',      # Gradient boosting decision tree
    'objective': 'binary',         # Objective function
    'metric': 'binary_logloss',    # Evaluation metric
    'num_leaves': 31,             # Number of leaves in one tree
    'learning_rate': 0.05,        # Learning rate
    'feature_fraction': 0.9,      # Feature fraction for bagging
    'bagging_fraction': 0.8,      # Bagging fraction
    'bagging_freq': 5,            # Bagging frequency
    'min_data_in_leaf': 20,       # Minimum data in leaf
    'min_sum_hessian_in_leaf': 1e-3,  # Minimum sum hessian in leaf
    
    # Regularization
    'lambda_l1': 0,               # L1 regularization
    'lambda_l2': 0,               # L2 regularization
    'min_gain_to_split': 0,       # Minimum gain to split
    
    # Randomness
    'feature_fraction_seed': 2,   # Random seed for feature fraction
    'bagging_seed': 3,            # Random seed for bagging
    'drop_seed': 4,               # Random seed for drop
    'data_random_seed': 5,        # Random seed for data
    'verbose': -1                 # Verbosity level
}
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# Grid search
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Random search
param_dist = {
    'num_leaves': randint(10, 100),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 300),
    'feature_fraction': uniform(0.6, 0.4),
    'bagging_fraction': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
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
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Cross-validation
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=100,
    nfold=5,
    stratified=True,
    shuffle=True,
    seed=42,
    return_cvbooster=True
)

print("CV Results:")
print(f"Best iteration: {len(cv_results['valid binary_logloss-mean'])}")
print(f"Best CV score: {min(cv_results['valid binary_logloss-mean']):.4f}")

# Plot CV results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cv_results['train binary_logloss-mean'], label='Train')
plt.plot(cv_results['valid binary_logloss-mean'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('Binary Log Loss')
plt.title('LightGBM Cross-Validation')
plt.legend()
plt.show()
```

### Early Stopping
```python
# Training with early stopping
evals_result = {}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=10),
               lgb.record_evaluation(evals_result)]
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")

# Plot training history
lgb.plot_metric(evals_result, metric='binary_logloss')
plt.show()
```

### Feature Importance
```python
# Get feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance_df.head(10))

# Plot feature importance
lgb.plot_importance(model, max_num_features=10)
plt.show()

# Using scikit-learn API
feature_importance = lgb_clf.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

## Categorical Features

### Handling Categorical Variables
```python
# Create data with categorical features
categorical_features = ['category_1', 'category_2']
numerical_features = ['feature_1', 'feature_2', 'feature_3']

# Create sample data
df = pd.DataFrame({
    'category_1': np.random.choice(['A', 'B', 'C'], 1000),
    'category_2': np.random.choice(['X', 'Y', 'Z'], 1000),
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Convert categorical to integer
for col in categorical_features:
    df[col] = df[col].astype('category').cat.codes

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset with categorical features
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_indices)

# Train model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

model = lgb.train(params, train_data, num_boost_round=100)
```

## GPU Acceleration

### GPU Training
```python
# GPU parameters
gpu_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'device': 'gpu',              # Use GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1
}

# Train on GPU
gpu_model = lgb.train(gpu_params, train_data, num_boost_round=100)

# Scikit-learn API with GPU
gpu_clf = LGBMClassifier(
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0,
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    verbose=-1
)

gpu_clf.fit(X_train, y_train)
```

## Model Persistence

### Save and Load Models
```python
# Save model
model.save_model('lightgbm_model.txt')

# Load model
loaded_model = lgb.Booster(model_file='lightgbm_model.txt')

# Save as pickle (scikit-learn API)
import pickle

with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgb_clf, f)

# Load pickle
with open('lightgbm_model.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
```

### Model Interpretation
```python
# SHAP values for model interpretation
import shap

# Create explainer
explainer = shap.TreeExplainer(lgb_clf)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
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
    'objective': custom_objective,
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

custom_model = lgb.train(custom_params, train_data, num_boost_round=100)
```

### Custom Evaluation Metric
```python
def custom_metric(y_true, y_pred):
    """
    Custom evaluation metric
    """
    return 'custom_metric', np.mean((y_true - y_pred) ** 2), False

# Use custom metric
metric_params = {
    'objective': 'regression',
    'metric': custom_metric,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

metric_model = lgb.train(metric_params, train_data, num_boost_round=100)
```

## Use Cases
- **Classification**: Binary and multi-class classification problems
- **Regression**: Continuous target prediction
- **Ranking**: Learning to rank problems
- **Feature Selection**: Automatic feature importance and selection
- **Competitions**: Kaggle and other ML competitions
- **Production Systems**: High-performance production deployments
- **Large-scale Data**: Efficient handling of big datasets

## Best Practices
1. **Start with Defaults**: Begin with default parameters and tune gradually
2. **Use Cross-Validation**: Always use cross-validation for parameter tuning
3. **Early Stopping**: Use early stopping to prevent overfitting
4. **Feature Engineering**: Invest in good feature engineering
5. **Categorical Features**: Use native categorical feature support
6. **Memory Optimization**: Use appropriate parameters for memory constraints
7. **GPU Acceleration**: Use GPU for large datasets when available
8. **Model Interpretation**: Use SHAP or other tools for model interpretation

## Advantages
- **Speed**: Faster training compared to other gradient boosting libraries
- **Memory Efficiency**: Lower memory usage for large datasets
- **Accuracy**: Often achieves better accuracy with proper tuning
- **Categorical Features**: Native support for categorical variables
- **GPU Support**: GPU acceleration for faster training
- **Parallel Learning**: Efficient parallel and distributed training
- **Leaf-wise Growth**: More efficient tree growth strategy

## Limitations
- **Black Box**: Less interpretable than linear models
- **Overfitting**: Can overfit if not properly tuned
- **Parameter Sensitivity**: Many parameters to tune
- **Feature Engineering**: Still requires good feature engineering
- **Memory Usage**: Can be memory-intensive for very large datasets

## Related Libraries
- **XGBoost**: Alternative gradient boosting library
- **CatBoost**: Gradient boosting with categorical features
- **Scikit-learn**: Machine learning framework
- **SHAP**: Model interpretation library
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing 