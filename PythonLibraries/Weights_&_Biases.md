# Weights & Biases Library

## Overview
Weights & Biases (W&B) is a machine learning platform that helps data scientists and ML engineers track experiments, manage models, and collaborate on projects. It provides tools for experiment tracking, model versioning, dataset versioning, and model deployment. W&B is particularly popular in the deep learning community and integrates seamlessly with popular frameworks like PyTorch, TensorFlow, and scikit-learn.

## Installation
```bash
# Basic installation
pip install wandb

# Latest version
pip install wandb==0.16.1

# With additional dependencies
pip install wandb[media]

# From conda
conda install -c conda-forge wandb

# Login to W&B
wandb login
```

## Key Features
- **Experiment Tracking**: Log metrics, parameters, and artifacts
- **Model Versioning**: Track and manage model versions
- **Dataset Versioning**: Version control for datasets
- **Visualization**: Interactive plots and dashboards
- **Collaboration**: Team collaboration and sharing
- **Model Registry**: Centralized model management
- **Sweeps**: Hyperparameter optimization
- **Reports**: Create and share reports

## Core Concepts

### Basic Experiment Tracking
```python
import wandb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Initialize W&B run
wandb.init(
    project="my-ml-project",
    name="random-forest-experiment",
    config={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
)

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    random_state=wandb.config.random_state
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({
    "accuracy": accuracy,
    "training_samples": len(X_train),
    "test_samples": len(X_test)
})

# Log classification report
wandb.log({
    "classification_report": wandb.Table(
        columns=["precision", "recall", "f1-score", "support"],
        data=[[0.85, 0.82, 0.83, 200]]
    )
})

# Log feature importance
feature_importance = model.feature_importances_
wandb.log({
    "feature_importance": wandb.Table(
        columns=["feature", "importance"],
        data=[[f"feature_{i}", importance] for i, importance in enumerate(feature_importance)]
    )
})

# Finish the run
wandb.finish()
```

### Advanced Experiment Tracking
```python
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize W&B run with more detailed config
wandb.init(
    project="advanced-ml-project",
    name="comprehensive-experiment",
    config={
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5
    },
    tags=["classification", "random-forest", "experiment"]
)

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=wandb.config.test_size, 
    random_state=wandb.config.random_state
)

# Train model
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    min_samples_split=wandb.config.min_samples_split,
    min_samples_leaf=wandb.config.min_samples_leaf,
    random_state=wandb.config.random_state
)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=wandb.config.cv_folds)
wandb.log({
    "cv_mean": cv_scores.mean(),
    "cv_std": cv_scores.std(),
    "cv_scores": cv_scores.tolist()
})

# Train final model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log metrics
wandb.log({
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1": f1,
    "training_samples": len(X_train),
    "test_samples": len(X_test)
})

# Log confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close()

# Log feature importance plot
feature_importance = model.feature_importances_
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(feature_importance)), feature_importance)
ax.set_title('Feature Importance')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Importance')
wandb.log({"feature_importance_plot": wandb.Image(fig)})
plt.close()

# Log model file
import joblib
model_path = "random_forest_model.joblib"
joblib.dump(model, model_path)
wandb.save(model_path)

# Log model summary
wandb.log({
    "model_summary": wandb.Table(
        columns=["Parameter", "Value"],
        data=[
            ["n_estimators", wandb.config.n_estimators],
            ["max_depth", wandb.config.max_depth],
            ["min_samples_split", wandb.config.min_samples_split],
            ["min_samples_leaf", wandb.config.min_samples_leaf]
        ]
    )
})

# Finish the run
wandb.finish()
```

## Hyperparameter Optimization with Sweeps

### Basic Sweep Configuration
```python
import wandb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define sweep configuration
sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "name": "random-forest-sweep",
    "metric": {
        "name": "accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {
            "values": [50, 100, 200, 300]
        },
        "max_depth": {
            "values": [5, 10, 15, None]
        },
        "min_samples_split": {
            "values": [2, 5, 10]
        },
        "min_samples_leaf": {
            "values": [1, 2, 4]
        },
        "learning_rate": {
            "min": 0.01,
            "max": 0.3
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-sweep-project")

# Define training function
def train_model():
    # Initialize run
    wandb.init()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with sweep parameters
    model = RandomForestClassifier(
        n_estimators=wandb.config.n_estimators,
        max_depth=wandb.config.max_depth,
        min_samples_split=wandb.config.min_samples_split,
        min_samples_leaf=wandb.config.min_samples_leaf,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metric
    wandb.log({"accuracy": accuracy})

# Run sweep
wandb.agent(sweep_id, train_model, count=20)
```

### Advanced Sweep with Bayesian Optimization
```python
import wandb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Define advanced sweep configuration
sweep_config = {
    "method": "bayes",
    "name": "bayesian-optimization-sweep",
    "metric": {
        "name": "cv_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {
            "min": 50,
            "max": 500,
            "distribution": "int_uniform"
        },
        "max_depth": {
            "min": 3,
            "max": 20,
            "distribution": "int_uniform"
        },
        "min_samples_split": {
            "min": 2,
            "max": 20,
            "distribution": "int_uniform"
        },
        "min_samples_leaf": {
            "min": 1,
            "max": 10,
            "distribution": "int_uniform"
        },
        "max_features": {
            "values": ["sqrt", "log2", None]
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="advanced-sweep-project")

def train_with_cv():
    wandb.init()
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=wandb.config.n_estimators,
        max_depth=wandb.config.max_depth,
        min_samples_split=wandb.config.min_samples_split,
        min_samples_leaf=wandb.config.min_samples_leaf,
        max_features=wandb.config.max_features,
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Log metrics
    wandb.log({
        "cv_accuracy": cv_accuracy,
        "cv_std": cv_std,
        "cv_scores": cv_scores.tolist()
    })

# Run sweep
wandb.agent(sweep_id, train_with_cv, count=50)
```

## Model Versioning and Registry

### Model Versioning
```python
import wandb
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize run
wandb.init(project="model-versioning")

# Generate and train model
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model locally
model_path = "model.joblib"
joblib.dump(model, model_path)

# Log model to W&B
model_artifact = wandb.Artifact(
    name="random-forest-model",
    type="model",
    description="Random Forest classifier for binary classification"
)

model_artifact.add_file(model_path)
wandb.log_artifact(model_artifact)

# Log model metadata
wandb.log({
    "model_info": wandb.Table(
        columns=["Parameter", "Value"],
        data=[
            ["n_estimators", 100],
            ["max_depth", "default"],
            ["random_state", 42],
            ["training_samples", len(X_train)],
            ["test_samples", len(X_test)]
        ]
    )
})

wandb.finish()
```

### Model Registry Usage
```python
import wandb
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize API
api = wandb.Api()

# Get the best model from a sweep
sweep = api.sweep("username/project/sweep_id")
best_run = sweep.best_run

# Download the model
model_artifact = best_run.logged_artifacts()[0]
model_path = model_artifact.download()

# Load the model
model = joblib.load(f"{model_path}/model.joblib")

# Use the model for predictions
X_new = np.random.randn(10, 10)
predictions = model.predict(X_new)
print(f"Predictions: {predictions}")

# Register model in model registry
model_registry = wandb.init(project="model-registry")

registered_model = wandb.Artifact(
    name="production-model",
    type="model",
    description="Production-ready Random Forest model"
)

registered_model.add_file("model.joblib")
model_registry.log_artifact(registered_model, aliases=["latest", "v1.0"])

model_registry.finish()
```

## Dataset Versioning

### Dataset Logging
```python
import wandb
import pandas as pd
import numpy as np

# Generate sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.normal(0, 1, n_samples),
    'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# Save dataset locally
df.to_csv("dataset.csv", index=False)

# Initialize W&B
wandb.init(project="dataset-versioning")

# Create dataset artifact
dataset_artifact = wandb.Artifact(
    name="classification-dataset",
    type="dataset",
    description="Sample dataset for binary classification"
)

dataset_artifact.add_file("dataset.csv")

# Log dataset statistics
wandb.log({
    "dataset_info": wandb.Table(
        columns=["Statistic", "Value"],
        data=[
            ["samples", len(df)],
            ["features", len(df.columns) - 1],
            ["target_distribution", df['target'].value_counts().to_dict()],
            ["missing_values", df.isnull().sum().sum()]
        ]
    )
})

# Log dataset preview
wandb.log({
    "dataset_preview": wandb.Table(
        columns=df.columns.tolist(),
        data=df.head(10).values.tolist()
    )
})

# Log the artifact
wandb.log_artifact(dataset_artifact)

wandb.finish()
```

## Visualization and Reporting

### Custom Plots and Tables
```python
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize W&B
wandb.init(project="visualization-demo")

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create custom plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Feature importance
feature_importance = model.feature_importances_
axes[0, 0].bar(range(len(feature_importance)), feature_importance)
axes[0, 0].set_title('Feature Importance')
axes[0, 0].set_xlabel('Feature Index')
axes[0, 0].set_ylabel('Importance')

# Training history (simulated)
epochs = range(1, 101)
train_loss = [1.0 - 0.01 * i + np.random.normal(0, 0.01) for i in epochs]
val_loss = [1.0 - 0.008 * i + np.random.normal(0, 0.02) for i in epochs]

axes[0, 1].plot(epochs, train_loss, label='Training Loss')
axes[0, 1].plot(epochs, val_loss, label='Validation Loss')
axes[0, 1].set_title('Training History')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()

# Distribution of predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
axes[1, 0].hist(y_pred_proba, bins=20, alpha=0.7)
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Count')

# Correlation matrix
correlation_matrix = np.corrcoef(X_test.T)
sns.heatmap(correlation_matrix, ax=axes[1, 1], cmap='coolwarm', center=0)
axes[1, 1].set_title('Feature Correlation Matrix')

plt.tight_layout()

# Log the plot
wandb.log({"custom_plots": wandb.Image(fig)})
plt.close()

# Create custom table
results_table = wandb.Table(
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
    data=[
        ["Random Forest", 0.85, 0.83, 0.87, 0.85],
        ["Baseline", 0.50, 0.50, 0.50, 0.50]
    ]
)

wandb.log({"model_comparison": results_table})

# Log metrics over time
for epoch in range(1, 101):
    wandb.log({
        "epoch": epoch,
        "train_loss": 1.0 - 0.01 * epoch + np.random.normal(0, 0.01),
        "val_loss": 1.0 - 0.008 * epoch + np.random.normal(0, 0.02),
        "learning_rate": 0.1 * (0.95 ** epoch)
    })

wandb.finish()
```

## Integration with Deep Learning Frameworks

### PyTorch Integration
```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Initialize W&B
wandb.init(project="pytorch-integration")

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, 1000).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create data loader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = SimpleNN(input_size=10, hidden_size=50, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Watch model
wandb.watch(model, log="all")

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Save model
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

wandb.finish()
```

### TensorFlow Integration
```python
import wandb
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Initialize W&B
wandb.init(project="tensorflow-integration")

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, 1000).astype(np.float32)

# Define model
model = keras.Sequential([
    keras.layers.Dense(50, activation='relu', input_shape=(10,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Create W&B callback
wandb_callback = wandb.keras.WandbCallback(
    monitor='val_loss',
    mode='min',
    save_model=True
)

# Train model
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[wandb_callback],
    verbose=1
)

# Log additional metrics
wandb.log({
    "final_accuracy": history.history['accuracy'][-1],
    "final_val_accuracy": history.history['val_accuracy'][-1]
})

wandb.finish()
```

## Use Cases
- **Experiment Tracking**: Track ML experiments and compare results
- **Hyperparameter Optimization**: Automated hyperparameter tuning
- **Model Versioning**: Version control for ML models
- **Dataset Versioning**: Track dataset changes and versions
- **Collaboration**: Team collaboration on ML projects
- **Model Deployment**: Model registry and deployment tracking
- **Research**: Academic and research applications
- **Production ML**: Production machine learning pipelines

## Best Practices
1. **Consistent Naming**: Use consistent naming conventions for runs and projects
2. **Comprehensive Logging**: Log all relevant metrics, parameters, and artifacts
3. **Sweep Configuration**: Use appropriate sweep methods for hyperparameter optimization
4. **Model Registry**: Use model registry for production models
5. **Dataset Versioning**: Version datasets to ensure reproducibility
6. **Collaboration**: Use W&B for team collaboration
7. **Monitoring**: Monitor model performance in production
8. **Documentation**: Document experiments and model decisions

## Advantages
- **Easy Integration**: Simple integration with popular ML frameworks
- **Comprehensive Tracking**: Track metrics, parameters, and artifacts
- **Visualization**: Rich visualization and dashboard capabilities
- **Collaboration**: Excellent team collaboration features
- **Hyperparameter Optimization**: Built-in sweep functionality
- **Model Registry**: Centralized model management
- **Production Ready**: Supports production ML workflows
- **Cloud Platform**: Managed cloud platform with good performance

## Limitations
- **Cost**: Can be expensive for large-scale usage
- **Internet Dependency**: Requires internet connection
- **Data Privacy**: Data is stored on W&B servers
- **Learning Curve**: Requires learning W&B-specific concepts
- **Vendor Lock-in**: Dependency on W&B platform
- **Customization**: Limited customization compared to self-hosted solutions

## Related Libraries
- **MLflow**: Alternative experiment tracking platform
- **DVC**: Data version control
- **Optuna**: Hyperparameter optimization
- **TensorBoard**: TensorFlow visualization
- **Weights & Biases**: Model monitoring
- **Kubeflow**: Kubernetes-based ML platform
- **Apache Airflow**: Workflow orchestration
- **Hugging Face**: Model sharing and deployment 