# 🛡️ Phishing Website Detection – End-to-End ML Pipeline

---

## 📌 Project Overview

This project is a **production-style End-to-End Machine Learning Pipeline** for detecting whether a website is **Phishing** or **Legitimate**.

The system is built using:

* FastAPI (API Layer)
* MongoDB (Data Source)
* Scikit-learn (ML Models)
* MLflow (Experiment Tracking)
* AWS S3 (Artifact Storage)
* Docker (Containerization)

It supports:

* ✅ Model Training via API
* ✅ Model Prediction via API
* ✅ Modular Pipeline Architecture
* ✅ Cloud Artifact Management

---

# 🎯 Problem Statement

Given 30 engineered URL-based features, predict whether a website is:

* `1 → Legitimate`
* `0 → Phishing`

This is a **Binary Classification Problem**.

---

# 🏗️ High-Level Architecture

```
Client
   ↓
FastAPI
   ↓
Training Pipeline  →  Model + Preprocessor Saved  →  S3 Upload
   ↓
Prediction Pipeline → Load Saved Artifacts → Return Predictions
```

---

# 🚀 API Endpoints

## 1️⃣ Training API

```
GET /train
```

### What It Does:

* Runs full ML pipeline
* Trains multiple models
* Selects best model
* Saves artifacts

---

## 2️⃣ Prediction API

```
POST /predict
```

### What It Does:

* Accepts CSV input
* Loads saved model & preprocessor
* Transforms data
* Returns predictions

---

# 📂 Complete Folder Structure

```
.
├── app.py
├── requirements.txt
├── Dockerfile
├── .env
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── final_model/
│       ├── model.pkl
│       └── preprocessor.pkl
│
├── networksecurity/
│   ├── __init__.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   └── artifact_entity.py
│   │
│   ├── config/
│   │   └── configuration.py
│   │
│   ├── utils/
│   │   ├── main_utils.py
│   │   └── ml_utils.py
│   │
│   ├── exception.py
│   └── logger.py
```

---

# 🧩 Detailed Module Explanation

---

# 🔹 app.py

This is the **entry point** of the application.

Responsibilities:

* Initialize FastAPI
* Define `/train` endpoint
* Define `/predict` endpoint
* Handle file uploads
* Return prediction output

---

# 🔹 networksecurity/components/

These are the **core ML building blocks**.

---

## 1️⃣ data_ingestion.py

### Purpose:

* Connect to MongoDB
* Fetch dataset
* Save raw dataset
* Split into train/test

### Output:

* train.csv
* test.csv

---

## 2️⃣ data_validation.py

### Purpose:

* Validate dataset schema
* Check column count
* Ensure data consistency

If validation fails → training stops.

---

## 3️⃣ data_transformation.py

### Purpose:

* Separate features (X) and target (y)
* Replace -1 with 0 in target
* Handle missing values using KNNImputer
* Save preprocessor object

### Output:

* train.npy
* test.npy
* preprocessor.pkl

---

## 4️⃣ model_trainer.py

### Purpose:

* Train multiple ML models:

  * Random Forest
  * Decision Tree
  * Gradient Boosting
  * Logistic Regression
  * AdaBoost
* Perform hyperparameter tuning
* Evaluate using:

  * F1 Score
  * Precision
  * Recall
* Select best model

### Output:

* model.pkl
* final_model/model.pkl

---

# 🔹 networksecurity/pipeline/

This controls execution flow.

---

## 1️⃣ training_pipeline.py

### Steps Executed:

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Save Artifacts
6. Upload to S3

This file orchestrates the entire training process.

---

## 2️⃣ prediction_pipeline.py

### Steps Executed:

1. Load saved model
2. Load saved preprocessor
3. Transform input data
4. Generate predictions
5. Return results

---

# 🔹 networksecurity/entity/

Contains data classes.

## config_entity.py

Defines configuration structure for:

* Data Ingestion
* Validation
* Transformation
* Model Trainer

## artifact_entity.py

Defines output structure of each pipeline stage.

---

# 🔹 networksecurity/config/

## configuration.py

Handles:

* Reading environment variables
* Defining artifact directories
* Centralized configuration management

---

# 🔹 networksecurity/utils/

## main_utils.py

Helper functions:

* Save object (pickle)
* Load object
* Evaluate models

## ml_utils.py

Model-related helper utilities.

---

# 🔹 exception.py

Custom exception handling.
Wraps errors with detailed traceback.

---

# 🔹 logger.py

Centralized logging configuration.

---

# 🧠 Training Workflow (Deep Explanation)

When `/train` is called:

1. Data fetched from MongoDB
2. Dataset validated
3. Missing values handled via KNN Imputer
4. Multiple models trained
5. Best model selected using F1 Score
6. Model saved as model.pkl
7. Preprocessor saved as preprocessor.pkl
8. Artifacts uploaded to S3

---

# 🔍 Prediction Workflow (Deep Explanation)

When `/predict` is called:

1. CSV file uploaded
2. Load model.pkl
3. Load preprocessor.pkl
4. Transform input
5. Predict labels
6. Return predictions

---

# 📊 Model Performance

Train Metrics:

* F1 Score: 0.991
* Precision: 0.987
* Recall: 0.995

Test Metrics:

* F1 Score: 0.972
* Precision: 0.965
* Recall: 0.979

---

# 🐳 Docker Support

The project is containerized using Docker.

To build image:

```
docker build -t phishing-detector .
```

To run container:

```
docker run -p 8000:8000 phishing-detector
```

---

# ☁️ Cloud Integration

* MongoDB → Data Storage
* AWS S3 → Artifact Storage
* MLflow → Experiment Tracking

---

# 📦 Installation

```
pip install -r requirements.txt
```

Run server:

```
uvicorn app:app --reload
```

---

# 🎤 How To Explain In Interview

“I built a modular end-to-end ML pipeline for phishing detection using FastAPI. The architecture separates components into ingestion, validation, transformation, and model training. The system supports API-based training and prediction, stores artifacts in S3, and follows production-level structure.”

---

# 💡 Key Engineering Concepts Used

* Modular Architecture
* Separation of Concerns
* Pipeline Orchestration
* Experiment Tracking
* Artifact Versioning
* API-based ML Serving
* Cloud Integration
* Containerization

---

# 📌 Future Improvements

* Add confidence score in prediction
* Add JSON-based prediction API
* Add model monitoring
* Add CI/CD pipeline
* Add Swagger documentation enhancements

---

# 👨‍💻 Author

Om Trivedi

---

# ⭐ Final Note

This is not a beginner ML project.
This is a production-style structured ML system demonstrating strong understanding of:

* Machine Learning
* Backend APIs
* System Design
* Cloud Deployment
