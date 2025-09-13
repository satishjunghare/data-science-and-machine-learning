# CatBoost Library

## Overview
CatBoost is a high-performance, open-source gradient boosting library developed by Yandex. It's designed to handle categorical features natively without requiring preprocessing, making it particularly effective for datasets with many categorical variables. CatBoost provides excellent performance, handles missing values automatically, and includes built-in overfitting detection and regularization techniques.

## Installation
```bash
# Basic installation
pip install catboost

# With GPU support
pip install catboost[gpu]

# Latest version
pip install catboost==1.2.2

# From conda
conda install -c conda-forge catboost

# Install with all dependencies
pip install catboost[all]
```

## Key Features
- **Native Categorical Support**: Handles categorical features without preprocessing
- **High Performance**: Optimized for speed and memory efficiency
- **Automatic Missing Value Handling**: Built-in missing value processing
- **Overfitting Detection**: Early stopping and regularization
- **GPU Acceleration**: Support for GPU training
- **Cross-Validation**: Built-in cross-validation capabilities
- **Feature Importance**: Automatic feature importance calculation
- **Model Interpretability**: SHAP values and feature effects

## Core Concepts

### Basic Classification
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create sample data with categorical features
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
    'categorical_3': np.random.choice(['Low', 'Medium', 'High'], n_samples)
}

# Create target with some relationship to features
data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A') |
    (data['numerical_2'] < -0.5) & 
    (data['categorical_2'] == 'Y')
).astype(int)

df = pd.DataFrame(data)

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2', 'categorical_3']

# Create CatBoost datasets
train_pool = cb.Pool(X_train, y_train, cat_features=categorical_features)
test_pool = cb.Pool(X_test, y_test, cat_features=categorical_features)

# Train model
model = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)

model.fit(train_pool)

# Make predictions
predictions = model.predict(test_pool)
probabilities = model.predict_proba(test_pool)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature importance
feature_importance = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
```

### Regression with CatBoost
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create sample regression data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

# Create target with some relationship
data['target'] = (
    2 * data['numerical_1'] + 
    1.5 * data['numerical_2'] + 
    (data['categorical_1'] == 'A') * 0.5 +
    (data['categorical_2'] == 'Y') * 0.3 +
    np.random.normal(0, 0.1, n_samples)
)

df = pd.DataFrame(data)

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Create CatBoost datasets
train_pool = cb.Pool(X_train, y_train, cat_features=categorical_features)
test_pool = cb.Pool(X_test, y_test, cat_features=categorical_features)

# Train regression model
model = cb.CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    verbose=False
)

model.fit(train_pool)

# Make predictions
predictions = model.predict(test_pool)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
```

## Advanced Features

### Cross-Validation
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Create CatBoost dataset
pool = cb.Pool(X, y, cat_features=categorical_features)

# Cross-validation
cv_results = cb.cv(
    pool=pool,
    params={
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'loss_function': 'Logloss',
        'verbose': False
    },
    fold_count=5,
    shuffle=True,
    seed=42,
    return_models=True
)

print("Cross-validation results:")
print(f"Best iteration: {cv_results['best_iteration']}")
print(f"Best score: {cv_results['test-Logloss-mean'].min():.4f}")

# Plot cross-validation results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cv_results['iterations'], cv_results['train-Logloss-mean'], label='Train')
plt.plot(cv_results['iterations'], cv_results['test-Logloss-mean'], label='Test')
plt.xlabel('Iterations')
plt.ylabel('Logloss')
plt.title('Cross-validation Results')
plt.legend()
plt.grid(True)
plt.show()
```

### Early Stopping and Overfitting Detection
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Create CatBoost datasets
train_pool = cb.Pool(X_train, y_train, cat_features=categorical_features)
test_pool = cb.Pool(X_test, y_test, cat_features=categorical_features)

# Train model with early stopping
model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    train_pool,
    eval_set=test_pool,
    plot=True
)

print(f"Best iteration: {model.get_best_iteration()}")
print(f"Best score: {model.get_best_score()}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(model.get_evals_result()['validation']['Logloss'], label='Validation')
plt.plot(model.get_evals_result()['learn']['Logloss'], label='Training')
plt.xlabel('Iterations')
plt.ylabel('Logloss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.show()
```

### Hyperparameter Tuning
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Define parameter grid
param_grid = {
    'iterations': [50, 100],
    'learning_rate': [0.05, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

# Create base model
base_model = cb.CatBoostClassifier(
    loss_function='Logloss',
    verbose=False,
    random_seed=42
)

# Grid search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train, cat_features=categorical_features)

# Best parameters
print("Best parameters:")
print(grid_search.best_params_)

# Best score
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {test_accuracy:.4f}")

# Results summary
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nTop 5 parameter combinations:")
top_results = results_df.nlargest(5, 'mean_test_score')
print(top_results[['param_iterations', 'param_learning_rate', 'param_depth', 'param_l2_leaf_reg', 'mean_test_score']])
```

## Model Interpretation

### SHAP Values
```python
import catboost as cb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Train model
model = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)

model.fit(X_train, y_train, cat_features=categorical_features)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Force plot for a specific prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_test.iloc[0, :],
    feature_names=X.columns
)

# Dependence plot
shap.dependence_plot('numerical_1', shap_values, X_test, feature_names=X.columns)
```

### Feature Effects
```python
import catboost as cb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Train model
model = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)

model.fit(X_train, y_train, cat_features=categorical_features)

# Get feature effects
feature_effects = model.get_feature_effect()

# Plot feature effects
plt.figure(figsize=(10, 6))
feature_names = X.columns
effects = [feature_effects[feature] for feature in feature_names]

plt.barh(feature_names, effects)
plt.xlabel('Feature Effect')
plt.title('Feature Effects')
plt.grid(True, alpha=0.3)
plt.show()

# Get feature interactions
interactions = model.get_feature_effect('Interaction')

# Plot top interactions
if interactions:
    plt.figure(figsize=(10, 8))
    interaction_matrix = np.zeros((len(feature_names), len(feature_names)))
    
    for interaction in interactions:
        i = feature_names.index(interaction[0])
        j = feature_names.index(interaction[1])
        interaction_matrix[i, j] = interaction[2]
        interaction_matrix[j, i] = interaction[2]
    
    plt.imshow(interaction_matrix, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Interactions')
    plt.tight_layout()
    plt.show()
```

## GPU Acceleration

### GPU Training
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 10000  # Larger dataset for GPU demonstration

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Create CatBoost datasets
train_pool = cb.Pool(X_train, y_train, cat_features=categorical_features)
test_pool = cb.Pool(X_test, y_test, cat_features=categorical_features)

# Train model with GPU
model_gpu = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    task_type='GPU',
    devices='0:1',  # Use GPU devices 0 and 1
    verbose=False
)

# Train model with CPU for comparison
model_cpu = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    task_type='CPU',
    verbose=False
)

import time

# Time GPU training
start_time = time.time()
model_gpu.fit(train_pool)
gpu_time = time.time() - start_time

# Time CPU training
start_time = time.time()
model_cpu.fit(train_pool)
cpu_time = time.time() - start_time

print(f"GPU training time: {gpu_time:.2f} seconds")
print(f"CPU training time: {cpu_time:.2f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")

# Compare predictions
gpu_predictions = model_gpu.predict(test_pool)
cpu_predictions = model_cpu.predict(test_pool)

# Check if predictions are the same
predictions_match = np.all(gpu_predictions == cpu_predictions)
print(f"Predictions match: {predictions_match}")
```

## Model Persistence

### Saving and Loading Models
```python
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'numerical_1': np.random.normal(0, 1, n_samples),
    'numerical_2': np.random.normal(0, 1, n_samples),
    'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
}

data['target'] = (
    (data['numerical_1'] > 0) & 
    (data['categorical_1'] == 'A')
).astype(int)

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = ['categorical_1', 'categorical_2']

# Train model
model = cb.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)

model.fit(X_train, y_train, cat_features=categorical_features)

# Save model
model.save_model('catboost_model.cbm')

# Load model
loaded_model = cb.CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')

# Compare predictions
original_predictions = model.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

predictions_match = np.all(original_predictions == loaded_predictions)
print(f"Predictions match after loading: {predictions_match}")

# Save model in different formats
model.save_model('catboost_model.json', format='json')
model.save_model('catboost_model.pkl', format='pickle')

# Export to C++ code
model.save_model('catboost_model.cpp', format='cpp')
```

## Use Cases
- **Tabular Data**: Structured data with categorical features
- **Click Prediction**: Online advertising and recommendation systems
- **Fraud Detection**: Financial fraud and anomaly detection
- **Risk Assessment**: Credit scoring and insurance
- **Customer Segmentation**: Marketing and customer analytics
- **Medical Diagnosis**: Healthcare and medical applications
- **E-commerce**: Product recommendation and pricing
- **Time Series**: Temporal data with categorical variables

## Best Practices
1. **Categorical Features**: Let CatBoost handle categorical features natively
2. **Missing Values**: Use CatBoost's built-in missing value handling
3. **Early Stopping**: Use early stopping to prevent overfitting
4. **Cross-Validation**: Use cross-validation for robust evaluation
5. **Feature Engineering**: Focus on domain-specific features
6. **Hyperparameter Tuning**: Tune learning rate, depth, and regularization
7. **GPU Acceleration**: Use GPU for large datasets
8. **Model Interpretation**: Use SHAP values for feature importance

## Advantages
- **Native Categorical Support**: No preprocessing needed for categorical features
- **High Performance**: Optimized for speed and memory efficiency
- **Automatic Missing Value Handling**: Built-in missing value processing
- **Overfitting Prevention**: Early stopping and regularization
- **GPU Acceleration**: Fast training on GPU
- **Interpretability**: SHAP values and feature effects
- **Production Ready**: Robust and scalable
- **Easy to Use**: Simple API with good defaults

## Limitations
- **Memory Usage**: Can be memory-intensive for large datasets
- **Black Box**: Less interpretable than linear models
- **Hyperparameter Tuning**: Requires careful tuning
- **Computational Cost**: Training can be expensive
- **Feature Engineering**: May still benefit from domain-specific features
- **Overfitting**: Can overfit without proper regularization

## Related Libraries
- **XGBoost**: Alternative gradient boosting library
- **LightGBM**: Microsoft's gradient boosting framework
- **Scikit-learn**: Machine learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SHAP**: Model interpretation
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization 