# DVC (Data Version Control)

## Overview

**DVC** is an open-source version control system for machine learning projects. It's designed to handle large files, data sets, machine learning models, and metrics, while keeping them in sync with your source code using Git. DVC works alongside Git to provide version control for data and models, making ML projects reproducible and collaborative.

## Installation

```bash
# Install DVC
pip install dvc

# Install with additional features
pip install dvc[all]  # Includes all optional dependencies

# Install specific backends
pip install dvc[s3]   # For AWS S3
pip install dvc[gdrive]  # For Google Drive
pip install dvc[ssh]  # For SSH/SFTP
```

## Key Features

- **Git-like workflow**: Familiar commands for data versioning
- **Multiple storage backends**: Local, S3, GCS, Azure, SSH, HTTP, etc.
- **Pipeline management**: Define and run ML pipelines
- **Experiment tracking**: Track metrics and parameters
- **Data lineage**: Track data dependencies and transformations
- **Collaboration**: Share data and models across teams
- **Reproducibility**: Ensure experiments can be reproduced
- **Large file handling**: Efficiently handle large datasets

## Core Concepts

### 1. Data Versioning
DVC tracks data files and directories similar to how Git tracks source code.

```bash
# Initialize DVC in your project
dvc init

# Add data directory to version control
dvc add data/raw/
dvc add data/processed/
dvc add models/

# Commit the .dvc files to Git
git add .dvc .gitignore
git commit -m "Add data files to version control"
```

### 2. Remote Storage
Configure remote storage for your data.

```bash
# Add remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Or use local storage
dvc remote add -d myremote /path/to/local/storage

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

### 3. Data Pipelines
Define reproducible data processing and ML pipelines.

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/
    outs:
      - data/processed/:
          persist: true
    metrics:
      - metrics/accuracy.json:
          cache: false
  
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/
    outs:
      - models/model.pkl:
          persist: true
    metrics:
      - metrics/accuracy.json:
          cache: false
```

## Basic Usage

### Setting Up a Project
```bash
# Initialize a new project
mkdir ml-project
cd ml-project
git init
dvc init

# Create project structure
mkdir -p data/{raw,processed}
mkdir -p src
mkdir -p models
mkdir -p metrics
```

### Adding Data Files
```python
# Example: Adding a large dataset
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'feature1': np.random.randn(10000),
    'feature2': np.random.randn(10000),
    'target': np.random.randint(0, 2, 10000)
})

# Save data
data.to_csv('data/raw/dataset.csv', index=False)
```

```bash
# Add data to DVC
dvc add data/raw/dataset.csv

# Check status
dvc status

# Commit to Git
git add .dvc .gitignore
git commit -m "Add dataset to version control"
```

### Working with Remote Storage
```bash
# Configure remote storage (AWS S3 example)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push data to remote
dvc push

# Pull data from remote (on another machine)
dvc pull

# Check remote status
dvc status --remote myremote
```

## Pipeline Management

### Creating Pipelines
```python
# src/prepare.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/raw/dataset.csv')

# Preprocess data
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.drop('target', axis=1)),
    columns=df.drop('target', axis=1).columns
)
df_scaled['target'] = df['target']

# Save processed data
df_scaled.to_csv('data/processed/dataset_processed.csv', index=False)
```

```python
# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import pickle

# Load processed data
df = pd.read_csv('data/processed/dataset_processed.csv')

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save metrics
metrics = {'accuracy': accuracy}
with open('metrics/accuracy.json', 'w') as f:
    json.dump(metrics, f)
```

### Running Pipelines
```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro prepare

# Run with force (ignore cache)
dvc repro --force

# Show pipeline graph
dvc dag

# Show pipeline status
dvc status
```

### Pipeline Dependencies
```yaml
# dvc.yaml with dependencies
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw/dataset.csv
    outs:
      - data/processed/dataset_processed.csv:
          persist: true
  
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/dataset_processed.csv
    outs:
      - models/model.pkl:
          persist: true
    metrics:
      - metrics/accuracy.json:
          cache: false
```

## Experiment Tracking

### Tracking Parameters
```python
# src/train.py with parameters
import yaml
import argparse

def train_model(params):
    # Load parameters
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    
    # Train model with parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    # ... training code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    
    train_model(params)
```

```yaml
# params.yaml
train:
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

```yaml
# dvc.yaml with parameters
stages:
  train:
    cmd: python src/train.py --params params.yaml
    deps:
      - src/train.py
      - params.yaml
      - data/processed/dataset_processed.csv
    params:
      - train.n_estimators
      - train.max_depth
    outs:
      - models/model.pkl:
          persist: true
    metrics:
      - metrics/accuracy.json:
          cache: false
```

### Running Experiments
```bash
# Run experiment with different parameters
dvc exp run --set-param train.n_estimators=200

# Run multiple experiments
dvc exp run --set-param train.n_estimators=100 --set-param train.max_depth=5
dvc exp run --set-param train.n_estimators=200 --set-param train.max_depth=10

# List experiments
dvc exp list

# Show experiment results
dvc exp show

# Compare experiments
dvc exp diff HEAD~1
```

## Advanced Features

### Data Lineage
```bash
# Show data lineage
dvc dag

# Show dependencies for specific file
dvc dag --target models/model.pkl

# Show what depends on specific file
dvc dag --target data/raw/dataset.csv
```

### Caching and Optimization
```bash
# Check cache status
dvc status

# Clean cache
dvc gc

# Remove unused cache
dvc gc --workspace

# Show cache info
dvc cache dir
```

### Importing Data
```bash
# Import data from external source
dvc import https://github.com/iterative/dataset-registry get-started/data.xml

# Import with specific revision
dvc import https://github.com/iterative/dataset-registry get-started/data.xml --rev v1.0

# Update imported data
dvc update data.xml.dvc
```

### Working with Large Files
```bash
# Add large file with specific chunk size
dvc add --file large_file.dvc data/large_file.csv

# Configure chunk size
dvc config cache.type reflink,symlink,copy

# Use hard links for better performance
dvc config cache.type reflink
```

## Integration with Git

### Git Workflow
```bash
# Typical workflow
dvc add data/raw/dataset.csv
git add .dvc .gitignore
git commit -m "Add dataset"

# Make changes to data
# ... modify data ...

# Update DVC tracking
dvc add data/raw/dataset.csv
git add .dvc
git commit -m "Update dataset"

# Push to remote
dvc push
git push
```

### Branching and Merging
```bash
# Create feature branch
git checkout -b feature/new-data
dvc checkout

# Make changes
# ... modify data and code ...

# Commit changes
dvc add data/raw/new_dataset.csv
git add .dvc
git commit -m "Add new dataset"

# Merge back to main
git checkout main
git merge feature/new-data
dvc checkout
```

## Use Cases

### 1. Data Science Projects
```bash
# Project structure
project/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── prepare.py
│   └── train.py
├── models/
├── metrics/
├── dvc.yaml
├── params.yaml
└── .dvc/
```

### 2. Collaborative ML Projects
```bash
# Team workflow
# 1. Clone repository
git clone <repo-url>
dvc pull

# 2. Make changes
# ... modify code and data ...

# 3. Commit and push
dvc add data/
git add .dvc
git commit -m "Update data"
dvc push
git push

# 4. Other team members
git pull
dvc pull
```

### 3. Experiment Management
```bash
# Run experiments with different parameters
dvc exp run --set-param train.learning_rate=0.01
dvc exp run --set-param train.learning_rate=0.001

# Compare results
dvc exp show

# Apply best experiment
dvc exp apply <experiment-name>
```

## Best Practices

### 1. Project Structure
- Keep data separate from code
- Use consistent naming conventions
- Organize data by type (raw, processed, external)
- Document data sources and transformations

### 2. Pipeline Design
- Make pipelines reproducible
- Use parameters for configurable values
- Include data validation steps
- Add proper error handling

### 3. Storage Management
- Choose appropriate storage backend
- Use compression for large files
- Implement data retention policies
- Monitor storage costs

### 4. Collaboration
- Document data sources and transformations
- Use meaningful commit messages
- Review data changes before merging
- Set up CI/CD for data pipelines

### 5. Performance
- Use appropriate chunk sizes for large files
- Optimize cache settings
- Use parallel processing when possible
- Monitor pipeline execution times

## Advantages

1. **Git integration**: Seamless integration with Git workflow
2. **Multiple backends**: Support for various storage solutions
3. **Pipeline management**: Reproducible data processing pipelines
4. **Experiment tracking**: Built-in experiment management
5. **Large file handling**: Efficient handling of large datasets
6. **Collaboration**: Easy sharing of data and models
7. **Reproducibility**: Ensures experiments can be reproduced
8. **Flexibility**: Works with any programming language

## Limitations

1. **Learning curve**: Requires understanding of Git and DVC concepts
2. **Storage costs**: Remote storage can be expensive for large datasets
3. **Network dependency**: Requires internet for remote operations
4. **Setup complexity**: Initial setup can be complex for teams
5. **Tool integration**: May require additional tools for full ML lifecycle

## Related Libraries

- **Git**: Version control for source code
- **MLflow**: Machine learning lifecycle management
- **Weights & Biases**: Experiment tracking and model management
- **Kubeflow**: Kubernetes-based ML platform
- **Airflow**: Workflow orchestration
- **Prefect**: Modern workflow orchestration
- **Luigi**: Pipeline building framework
- **Snakemake**: Workflow management system 