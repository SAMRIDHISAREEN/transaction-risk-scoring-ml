# Transaction Risk Scoring using Machine Learning

## Overview
This is a beginner-friendly machine learning project that builds an end-to-end supervised learning pipeline for **transaction risk scoring** (binary classification).

The workflow includes preprocessing structured data, training a baseline classification model, and evaluating it using **Precision, Recall, F1-score, and ROC-AUC**.

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook

## Workflow
1. Load dataset
2. Train-test split
3. Feature scaling
4. Train Logistic Regression baseline model
5. Evaluate model performance
6. Plot ROC curve

## Results
ROC curve and ROC-AUC:

![ROC Curve](images/roc_curve.png)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
