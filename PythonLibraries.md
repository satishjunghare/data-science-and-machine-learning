# Python Libraries for Data Science, Machine Learning, and AI

There are **dozens of powerful and widely used libraries** in Python for data science, machine learning, and AI. Below is a categorized list of the **most useful and commonly used ones**, especially relevant to projects like your Fronx car trip analysis and ML experimentation.

**📚 Comprehensive Documentation Available:** Each library marked with 🔗 has detailed documentation including installation, features, code examples, use cases, and best practices.

---

## 🧮 **Core Data & Computation Libraries**

| Library    | Purpose                                                         | Documentation |
| ---------- | --------------------------------------------------------------- | ------------- |
| **NumPy**  | Fast numerical computing, arrays, matrices                      | 🔗 [NumPy.md](PythonLibraries/NumPy.md) |
| **Pandas** | Data manipulation and analysis                                  | 🔗 [Pandas.md](PythonLibraries/Pandas.md) |
| **Dask**   | Parallel and distributed Pandas-like operations                 | 🔗 [Dask.md](PythonLibraries/Dask.md) |
| **Polars** | Fast DataFrame library (alternative to Pandas, written in Rust) | 🔗 [Polars.md](PythonLibraries/Polars.md) |

---

## 📊 **Visualization Libraries**

| Library        | Purpose                                      | Documentation |
| -------------- | -------------------------------------------- | ------------- |
| **Matplotlib** | Low-level, customizable plotting             | 🔗 [Matplotlib.md](PythonLibraries/Matplotlib.md) |
| **Seaborn**    | High-level statistical plots                 | 🔗 [Seaborn.md](PythonLibraries/Seaborn.md) |
| **Plotly**     | Interactive plots (web-style dashboards)     | 🔗 [Plotly.md](PythonLibraries/Plotly.md) |
| **Altair**     | Declarative plotting (good for clean charts) | 🔗 [Altair.md](PythonLibraries/Altair.md) |
| **Bokeh**      | Interactive web plots, great for dashboards  | 🔗 [Bokeh.md](PythonLibraries/Bokeh.md) |

---

## 🤖 **Machine Learning Libraries**

| Library                           | Purpose                                                       | Documentation |
| --------------------------------- | ------------------------------------------------------------- | ------------- |
| **Scikit-learn**                  | Classic ML algorithms: regression, classification, clustering | 🔗 [Scikit-learn.md](PythonLibraries/Scikit-learn.md) |
| **XGBoost**                       | Gradient boosting (for structured/tabular data)               | 🔗 [XGBoost.md](PythonLibraries/XGBoost.md) |
| **LightGBM**                      | High-performance gradient boosting                            | 🔗 [LightGBM.md](PythonLibraries/LightGBM.md) |
| **CatBoost**                      | Gradient boosting with categorical features                   | 🔗 [CatBoost.md](PythonLibraries/CatBoost.md) |
| **TensorFlow**                    | Deep learning framework (Google)                              | 🔗 [TensorFlow.md](PythonLibraries/TensorFlow.md) |
| **PyTorch**                       | Deep learning framework (Meta, more Pythonic, very popular)   | 🔗 [PyTorch.md](PythonLibraries/PyTorch.md) |
| **Keras**                         | High-level API for TensorFlow                                 | 🔗 [Keras.md](PythonLibraries/Keras.md) |
| **Hugging Face Transformers**     | Pre-trained LLMs and NLP tools (BERT, GPT, etc.)              | 🔗 [HuggingFace_Transformers.md](PythonLibraries/HuggingFace_Transformers.md) |

---

## 🧹 **Data Cleaning & Feature Engineering**

| Library                                         | Purpose                                                | Documentation |
| ----------------------------------------------- | ------------------------------------------------------ | ------------- |
| **OpenRefine / Pandas-Profiling / Sweetviz**    | Auto data profiling and summary                        | *Documentation coming soon* |
| **Feature-engine / Scikit-learn Preprocessing** | Feature scaling, encoding, transformation              | *Documentation coming soon* |
| **Missingno**                                   | Visualizing missing data                               | *Documentation coming soon* |
| **Category Encoders**                           | Encoding categorical variables (target, one-hot, etc.) | *Documentation coming soon* |

---

## 🌐 **Data Access & Storage**

| Library                           | Purpose                                      | Documentation |
| --------------------------------- | -------------------------------------------- | ------------- |
| **SQLAlchemy**                    | Interface to SQL databases                   | *Documentation coming soon* |
| **Pandas (read_csv, read_sql)**   | Built-in readers                             | 🔗 [Pandas.md](PythonLibraries/Pandas.md) |
| **PyODBC / psycopg2**             | Database connectors (SQL Server, PostgreSQL) | *Documentation coming soon* |
| **HDF5 / Feather / Parquet**      | High-performance file formats                | *Documentation coming soon* |

---

## ⏳ **Time Series & Signal Processing**

| Library               | Purpose                                       | Documentation |
| --------------------- | --------------------------------------------- | ------------- |
| **Statsmodels**       | Statistical tests, ARIMA, regressions         | 🔗 [Statsmodels.md](PythonLibraries/Statsmodels.md) |
| **Prophet (by Meta)** | Time series forecasting                       | 🔗 [Prophet.md](PythonLibraries/Prophet.md) |
| **tsfresh / Kats**    | Automated feature extraction from time series | *Documentation coming soon* |
| **Scipy.signal**      | Signal filtering, FFT, etc.                   | *Documentation coming soon* |

---

## 📦 **Model Deployment & Serving**

| Library                | Purpose                                        | Documentation |
| ---------------------- | ---------------------------------------------- | ------------- |
| **Flask / FastAPI**    | API creation to serve models                   | *Documentation coming soon* |
| **Gradio**             | Rapid prototyping of web apps for ML models    | 🔗 [Gradio.md](PythonLibraries/Gradio.md) |
| **Streamlit**          | Web app framework for data science             | 🔗 [Streamlit.md](PythonLibraries/Streamlit.md) |
| **ONNX / TorchScript** | Model format conversion for cross-platform use | *Documentation coming soon* |

---

## 🧪 **Experiment Tracking & MLOps**

| Library               | Purpose                                        | Documentation |
| --------------------- | ---------------------------------------------- | ------------- |
| **MLflow**            | Track experiments, models, parameters          | 🔗 [MLflow.md](PythonLibraries/MLflow.md) |
| **Weights & Biases**  | Powerful experiment tracking and visualization | 🔗 [Weights_&_Biases.md](PythonLibraries/Weights_&_Biases.md) |
| **DVC**               | Version control for datasets and ML models     | 🔗 [DVC.md](PythonLibraries/DVC.md) |
| **Optuna**            | Hyperparameter tuning and optimization         | 🔗 [Optuna.md](PythonLibraries/Optuna.md) |

---

## 🧾 **NLP and Text Analysis**

| Library      | Purpose                                     | Documentation |
| ------------ | ------------------------------------------- | ------------- |
| **NLTK**     | Traditional NLP (tokenization, POS tagging) | 🔗 [NLTK.md](PythonLibraries/NLTK.md) |
| **spaCy**    | Fast, modern NLP pipeline                   | 🔗 [spaCy.md](PythonLibraries/spaCy.md) |
| **Gensim**   | Topic modeling, word embeddings             | 🔗 [Gensim.md](PythonLibraries/Gensim.md) |
| **TextBlob** | Simple NLP, sentiment analysis              | *Documentation coming soon* |

---

## 🖼️ **Computer Vision**

| Library                            | Purpose                               | Documentation |
| ---------------------------------- | ------------------------------------- | ------------- |
| **OpenCV**                         | Image processing                      | 🔗 [OpenCV.md](PythonLibraries/OpenCV.md) |
| **Pillow (PIL)**                   | Image I/O and basic manipulation      | *Documentation coming soon* |
| **PyTorch/TensorFlow (with CNNs)** | Deep learning for vision tasks        | 🔗 [PyTorch.md](PythonLibraries/PyTorch.md) / 🔗 [TensorFlow.md](PythonLibraries/TensorFlow.md) |
| **Ultralytics YOLO**               | Object detection models (easy to use) | *Documentation coming soon* |

---

## 📁 **Data Annotation & Labeling**

| Library                     | Purpose                                | Documentation |
| --------------------------- | -------------------------------------- | ------------- |
| **LabelImg / Label Studio** | Image and text annotation tools        | *Documentation coming soon* |
| **Roboflow**                | Dataset management for computer vision | *Documentation coming soon* |

---

## 🔍 **Exploratory Data Analysis (EDA)**

| Library              | Purpose                             | Documentation |
| -------------------- | ----------------------------------- | ------------- |
| **Pandas Profiling** | Automated data reports              | *Documentation coming soon* |
| **Sweetviz**         | Comparison reports between datasets | *Documentation coming soon* |
| **Autoviz**          | Automated visualization of datasets | *Documentation coming soon* |

---

## 📚 **Documentation Summary**

### ✅ **Completed Documentation (25 libraries):**

**Core Data Science:**
- [NumPy.md](PythonLibraries/NumPy.md) - Numerical computing foundation
- [Pandas.md](PythonLibraries/Pandas.md) - Data manipulation and analysis
- [Dask.md](PythonLibraries/Dask.md) - Parallel computing framework
- [Polars.md](PythonLibraries/Polars.md) - High-performance DataFrame library

**Visualization:**
- [Matplotlib.md](PythonLibraries/Matplotlib.md) - Foundation plotting library
- [Seaborn.md](PythonLibraries/Seaborn.md) - Statistical visualization
- [Plotly.md](PythonLibraries/Plotly.md) - Interactive plotting library
- [Altair.md](PythonLibraries/Altair.md) - Declarative visualization
- [Bokeh.md](PythonLibraries/Bokeh.md) - Interactive web visualizations

**Machine Learning:**
- [Scikit-learn.md](PythonLibraries/Scikit-learn.md) - Machine learning framework
- [XGBoost.md](PythonLibraries/XGBoost.md) - Gradient boosting library
- [LightGBM.md](PythonLibraries/LightGBM.md) - High-performance gradient boosting
- [CatBoost.md](PythonLibraries/CatBoost.md) - Gradient boosting with categorical features
- [TensorFlow.md](PythonLibraries/TensorFlow.md) - Deep learning framework
- [PyTorch.md](PythonLibraries/PyTorch.md) - Alternative deep learning framework
- [Keras.md](PythonLibraries/Keras.md) - High-level neural network API
- [HuggingFace_Transformers.md](PythonLibraries/HuggingFace_Transformers.md) - NLP and transformer models

**Statistical & Time Series:**
- [Statsmodels.md](PythonLibraries/Statsmodels.md) - Statistical modeling and econometrics
- [Prophet.md](PythonLibraries/Prophet.md) - Time series forecasting

**Natural Language Processing:**
- [spaCy.md](PythonLibraries/spaCy.md) - Industrial-strength NLP processing
- [NLTK.md](PythonLibraries/NLTK.md) - Natural language processing toolkit
- [Gensim.md](PythonLibraries/Gensim.md) - Topic modeling and word embeddings

**Computer Vision:**
- [OpenCV.md](PythonLibraries/OpenCV.md) - Computer vision and image processing

**Web & Deployment:**
- [Streamlit.md](PythonLibraries/Streamlit.md) - Web app framework for data science
- [Gradio.md](PythonLibraries/Gradio.md) - Machine learning model interfaces
- [MLflow.md](PythonLibraries/MLflow.md) - Machine learning lifecycle management
- [Weights_&_Biases.md](PythonLibraries/Weights_&_Biases.md) - Experiment tracking and ML platform
- [DVC.md](PythonLibraries/DVC.md) - Data version control
- [Optuna.md](PythonLibraries/Optuna.md) - Hyperparameter optimization

### 🚧 **Coming Soon:**
- Keras, DVC, Optuna, Gensim, TextBlob, FastAPI, Flask, Panel, Dash, and more!

---

## 🎯 **Getting Started Recommendations**

### **For Beginners:**
1. Start with **NumPy** and **Pandas** for data manipulation
2. Learn **Matplotlib** and **Seaborn** for visualization
3. Use **Scikit-learn** for machine learning
4. Try **Streamlit** or **Gradio** for creating demos

### **For Intermediate Users:**
1. Explore **XGBoost/LightGBM/CatBoost** for advanced ML
2. Learn **TensorFlow** or **PyTorch** for deep learning
3. Use **MLflow** or **Weights & Biases** for experiment tracking
4. Try **spaCy** or **Hugging Face Transformers** for NLP

### **For Advanced Users:**
1. Master **Dask** and **Polars** for big data
2. Use **OpenCV** for computer vision
3. Implement **MLflow** for production ML pipelines
4. Explore **Prophet** and **Statsmodels** for time series

---

*Each documentation file includes: Installation, Key Features, Core Concepts, Code Examples, Use Cases, Best Practices, Advantages, Limitations, and Related Libraries.*

