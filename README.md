# 🏦 Bank Customer Churn Prediction (MLOps Lab Project)

This machine learning project aims to predict whether a customer will leave the bank (churn) based on demographic and transactional data. It implements training and evaluation of models such as Logistic Regression and Random Forest, with automated experiment tracking using MLflow.


---

## 📌 Overview

Customer churn is a critical issue in the banking industry. Identifying customers likely to leave allows the bank to take proactive retention measures. In this project:

- We train classification models on customer data from a CSV file.
- We evaluate model performance using accuracy.
- We log the model and metrics using MLflow for version control and experiment tracking.
- RandomForestClassifier improved the test accuracy from **0.816** (Logistic Regression) to **0.865**.

---


## 📊 Dataset Information

- **File**: `Churn_Modelling.csv`
- **Target Column**: `Exited` (1 = churned, 0 = stayed)
- **Dropped Columns**: RowNumber, CustomerId, Surname (non-predictive)
- **Encoded Columns**: Categorical features (like Geography, Gender) are one-hot encoded.

---

## 🛠️ Tools and Libraries

- Python 3.x
- pandas
- scikit-learn
- mlflow
- os, warnings

---
### ✔️ Model Accuracy Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.8160   |
| Random Forest       | 0.8650   |

## 📊 Screenshot of MLflow UI

![Capture](https://github.com/user-attachments/assets/6e80029c-e882-41e4-92f0-1be396d47754)

![accuracy](https://github.com/user-attachments/assets/66c87c64-6f1f-4510-8e84-3f6887fef1e5)
