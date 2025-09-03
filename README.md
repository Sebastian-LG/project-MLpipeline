# Project: Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an **end-to-end machine learning pipeline** using:

- **DVC (Data Version Control)** for data and model versioning  
- **MLflow** for experiment tracking  

The pipeline trains a **Random Forest Classifier** on the *Pima Indians Diabetes Dataset*, with clear stages for **data preprocessing, model training, and evaluation**.

---

## 🚀 Key Features

### 🔹 Data Version Control (DVC)
- Tracks and versions datasets, models, and pipeline stages for reproducibility.
- Pipelines are structured into **stages** (preprocessing, training, evaluation).
- Automatically re-executes stages if dependencies change (data, scripts, parameters).
- Supports **remote storage** (e.g., DagsHub, S3) for large datasets and models.

### 🔹 Experiment Tracking with MLflow
- Tracks **experiment metrics, parameters, and artifacts**.
- Logs:
  - Model hyperparameters (e.g., `n_estimators`, `max_depth`)
  - Performance metrics (e.g., accuracy)
- Enables easy comparison across runs to optimize the ML pipeline.

---

## 📂 Pipeline Stages

### 1. Preprocessing
- Script: `src/preprocess.py`
- Input: `data/raw/data.csv`
- Output: `data/processed/data.csv`
- Tasks:
  - Reads raw dataset
  - Performs preprocessing (e.g., renaming columns)
  - Saves consistent processed data for training

---

### 2. Training
- Script: `src/train.py`
- Input: `data/processed/data.csv`
- Output: `models/random_forest.pkl`
- Tasks:
  - Trains a **Random Forest Classifier**
  - Logs model + hyperparameters to MLflow
  - Ensures results are reproducible and comparable

---

### 3. Evaluation
- Script: `src/evaluate.py`
- Inputs: `models/random_forest.pkl`, `data/processed/data.csv`
- Tasks:
  - Evaluates trained model accuracy
  - Logs evaluation metrics into MLflow

---

## 🎯 Goals

- **Reproducibility** → Same data + code + params = same results  
- **Experimentation** → Track and compare experiments easily with MLflow  
- **Collaboration** → Share datasets, models, and experiments seamlessly across teams  

---

## 📌 Use Cases
- **Data Science Teams** → Collaborative tracking of datasets, models, and experiments  
- **Machine Learning Research** → Rapid experiment iteration with reproducible results  

---

## 🛠️ Technology Stack
- **Python** → Data processing, training, and evaluation  
- **DVC** → Version control for data/models/pipelines  
- **MLflow** → Experiment logging & tracking  
- **Scikit-learn** → Random Forest Classifier  

---

## ⚙️ DVC Commands for Adding Stages

### Preprocess Stage
```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py