# Optuna

## Overview

**Optuna** is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API that allows users to define the search space dynamically. Optuna provides efficient sampling algorithms and pruning strategies, making it easy to optimize hyperparameters for machine learning models.

## Installation

```bash
# Install Optuna
pip install optuna

# Install with additional features
pip install optuna[visualization]  # For visualization features
pip install optuna[mysql]          # For MySQL storage
pip install optuna[postgresql]     # For PostgreSQL storage
pip install optuna[redis]          # For Redis storage

# Install with all optional dependencies
pip install optuna[all]
```

## Key Features

- **Define-by-run API**: Dynamic search space definition
- **Efficient sampling**: Advanced sampling algorithms (TPE, CmaEs, etc.)
- **Pruning**: Automatic pruning of unpromising trials
- **Visualization**: Built-in visualization tools
- **Distributed optimization**: Support for distributed computing
- **Storage backends**: Multiple storage options (SQLite, MySQL, PostgreSQL, Redis)
- **Integration**: Easy integration with popular ML frameworks
- **Multi-objective optimization**: Support for multiple objectives

## Core Concepts

### 1. Study
A study represents one optimization session and contains all trials.

```python
import optuna

# Create a study
study = optuna.create_study(
    direction='maximize',  # or 'minimize'
    study_name='my_optimization'
)

# Load existing study
study = optuna.load_study(
    study_name='my_optimization',
    storage='sqlite:///optuna.db'
)
```

### 2. Trial
A trial represents one evaluation of the objective function.

```python
def objective(trial):
    # Define hyperparameters
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_int('y', 0, 10)
    
    # Evaluate objective
    return (x - 2) ** 2 + (y - 5) ** 2

# Run optimization
study.optimize(objective, n_trials=100)
```

### 3. Sampler
Samplers define how hyperparameters are sampled from the search space.

```python
import optuna.samplers

# Use TPE sampler
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='minimize')

# Use Random sampler
sampler = optuna.samplers.RandomSampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='minimize')
```

## Basic Usage

### Simple Optimization
```python
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def objective(trial):
    # Define hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Evaluate model
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get best parameters
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
```

### Advanced Hyperparameter Search
```python
def objective(trial):
    # Categorical parameters
    model_type = trial.suggest_categorical('model_type', ['rf', 'svm', 'knn'])
    
    if model_type == 'rf':
        # Random Forest parameters
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    elif model_type == 'svm':
        # SVM parameters
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e2, log=True)
        
        from sklearn.svm import SVC
        model = SVC(C=C, gamma=gamma, random_state=42)
    
    else:  # knn
        # KNN parameters
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    
    # Evaluate model
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Optimize with pruning
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=100)
```

## Hyperparameter Types

### Numeric Parameters
```python
def objective(trial):
    # Integer parameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10, step=1)
    
    # Float parameters
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
    # Log-uniform parameters
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    gamma = trial.suggest_float('gamma', 1e-4, 1e2, log=True)
    
    # Uniform parameters
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    
    return objective_value
```

### Categorical Parameters
```python
def objective(trial):
    # Categorical parameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    
    # Conditional parameters
    if optimizer == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
    
    return objective_value
```

### Conditional Parameters
```python
def objective(trial):
    # Define model type
    model_type = trial.suggest_categorical('model_type', ['linear', 'polynomial', 'rbf'])
    
    if model_type == 'linear':
        # Linear model parameters
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        
    elif model_type == 'polynomial':
        # Polynomial model parameters
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        degree = trial.suggest_int('degree', 2, 5)
        
    else:  # rbf
        # RBF model parameters
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e2, log=True)
    
    return objective_value
```

## Pruning

### Automatic Pruning
```python
import optuna.pruners

def objective(trial):
    # Get parameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    
    # Create model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Train and evaluate incrementally
    for step in range(10):
        # Train for a few iterations
        model.fit(X_train, y_train)
        
        # Evaluate intermediate result
        score = model.score(X_val, y_val)
        
        # Report intermediate value
        trial.report(score, step)
        
        # Prune if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

# Create study with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50)
```

### Custom Pruning
```python
def objective(trial):
    # Get parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Training loop
    for epoch in range(100):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader)
        val_loss = validate_epoch(model, val_loader)
        
        # Report intermediate values
        trial.report(val_loss, epoch)
        
        # Custom pruning logic
        if epoch > 10 and val_loss > 1.0:
            raise optuna.TrialPruned()
    
    return val_loss
```

## Visualization

### Study Visualization
```python
import optuna.visualization as vis

# Plot optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Plot parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Plot parameter relationships
fig = vis.plot_param_importances(study)
fig.show()

# Plot parallel coordinate
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Plot contour
fig = vis.plot_contour(study, params=['param1', 'param2'])
fig.show()
```

### Interactive Dashboard
```python
# Start Optuna dashboard
# optuna-dashboard sqlite:///optuna.db

# Or programmatically
import optuna.visualization as vis

# Create dashboard
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
vis.plot_parallel_coordinate(study)
```

## Advanced Features

### Multi-Objective Optimization
```python
def objective(trial):
    # Define hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Multiple objectives
    accuracy = model.score(X_test, y_test)
    training_time = measure_training_time(model, X_train, y_train)
    
    return accuracy, training_time

# Multi-objective study
study = optuna.create_study(
    directions=['maximize', 'minimize'],
    sampler=optuna.samplers.NSGAIISampler()
)
study.optimize(objective, n_trials=100)
```

### Distributed Optimization
```python
# Study with database storage
study = optuna.create_study(
    study_name='distributed_optimization',
    storage='mysql://user:password@localhost/optuna',
    direction='maximize'
)

# Multiple processes can optimize the same study
study.optimize(objective, n_trials=100)
```

### Custom Samplers
```python
import optuna.samplers

# Custom sampler
class CustomSampler(optuna.samplers.BaseSampler):
    def sample_independent(self, study, trial, param_name, param_distribution):
        # Custom sampling logic
        return custom_value
    
    def sample_relative(self, study, trial, search_space):
        # Custom relative sampling logic
        return custom_values

# Use custom sampler
study = optuna.create_study(sampler=CustomSampler())
```

## Integration with ML Frameworks

### Scikit-learn Integration
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Define hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### XGBoost Integration
```python
import xgboost as xgb

def objective(trial):
    # Define hyperparameters
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    # Create model
    model = xgb.XGBClassifier(**param, random_state=42)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### PyTorch Integration
```python
import torch
import torch.nn as nn

def objective(trial):
    # Define hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_int('n_units', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Create model
    model = create_model(n_layers, n_units, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(100):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        val_loss = validate_epoch(model, criterion, val_loader)
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Prune if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

# Optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## Use Cases

### 1. Model Hyperparameter Tuning
```python
# Comprehensive hyperparameter optimization
def objective(trial):
    # Model selection
    model_type = trial.suggest_categorical('model_type', ['rf', 'svm', 'knn', 'mlp'])
    
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 10, 200),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            random_state=42
        )
    elif model_type == 'svm':
        model = SVC(
            C=trial.suggest_float('C', 1e-4, 1e2, log=True),
            gamma=trial.suggest_float('gamma', 1e-4, 1e2, log=True),
            random_state=42
        )
    # ... other models
    
    # Evaluate
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()
```

### 2. Neural Network Architecture Search
```python
def objective(trial):
    # Architecture parameters
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_units = trial.suggest_int('n_units', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
    # Training parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Create and train model
    model = create_nn(n_layers, n_units, dropout_rate)
    # ... training code
    
    return validation_accuracy
```

### 3. Feature Engineering Optimization
```python
def objective(trial):
    # Feature engineering parameters
    n_components = trial.suggest_int('n_components', 5, 50)
    use_polynomial = trial.suggest_categorical('use_polynomial', [True, False])
    polynomial_degree = trial.suggest_int('polynomial_degree', 2, 4)
    
    # Apply feature engineering
    X_processed = apply_feature_engineering(
        X, n_components, use_polynomial, polynomial_degree
    )
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X_processed, y, cv=5)
    
    return scores.mean()
```

## Best Practices

### 1. Objective Function Design
- Keep objective functions simple and focused
- Use appropriate evaluation metrics
- Implement proper error handling
- Consider computational cost

### 2. Search Space Definition
- Define realistic parameter ranges
- Use appropriate parameter types
- Consider parameter dependencies
- Start with broad ranges and refine

### 3. Optimization Strategy
- Choose appropriate samplers for your problem
- Use pruning for expensive evaluations
- Monitor optimization progress
- Set reasonable time limits

### 4. Storage and Reproducibility
- Use persistent storage for long-running optimizations
- Save study objects for later analysis
- Document optimization setup
- Use fixed random seeds for reproducibility

## Advantages

1. **Easy to use**: Simple and intuitive API
2. **Efficient**: Advanced sampling algorithms
3. **Flexible**: Dynamic search space definition
4. **Visualization**: Built-in visualization tools
5. **Distributed**: Support for distributed optimization
6. **Integration**: Easy integration with ML frameworks
7. **Pruning**: Automatic pruning of unpromising trials
8. **Multi-objective**: Support for multiple objectives

## Limitations

1. **Learning curve**: Requires understanding of optimization concepts
2. **Computational cost**: Can be expensive for complex problems
3. **Parameter tuning**: Requires tuning of optimization parameters
4. **Black-box nature**: Limited interpretability of optimization process
5. **Local optima**: May get stuck in local optima

## Related Libraries

- **Hyperopt**: Alternative hyperparameter optimization library
- **Scikit-optimize**: Bayesian optimization library
- **GPyOpt**: Gaussian process optimization
- **Ray Tune**: Distributed hyperparameter tuning
- **Weights & Biases**: Experiment tracking with hyperparameter tuning
- **MLflow**: Machine learning lifecycle management
- **Keras Tuner**: Hyperparameter tuning for Keras models 