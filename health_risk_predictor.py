"""
Health Risk Predictor using Logistic Regression

Project Explanation:
This project predicts health risk levels such as diabetes, heart disease, or COVID-19 risk using logistic regression—a widely used classification algorithm.
We use publicly available datasets for demonstration: Pima Indians Diabetes dataset and a Heart Disease Risk dataset.
The project workflow covers:
1. Dataset acquisition: Download and load the health risk datasets automatically.
2. Data preprocessing: Handle missing data, encode features if needed.
3. Feature-target split and train-test split.
4. Train a logistic regression model on training data.
5. Evaluate the model using accuracy, precision, recall, and classification report.
6. Demonstrate sample predictions on test samples.
Logistic regression provides probabilistic prediction ideal for health risk probability estimations.

Datasets: 
- Diabetes: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
- Heart Disease: https://raw.githubusercontent.com/ahmedbesbes/Heart-Disease-UCI/master/heart.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import urllib.request

def load_diabetes_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=col_names)
    return df

def load_heart_disease_dataset():
    # Try multiple URLs for heart disease dataset
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/ahmedbesbes/Heart-Disease-UCI/master/heart.csv",
        "https://raw.githubusercontent.com/datasets/heart-disease/master/data/heart.csv"
    ]
    
    # Column names for the UCI heart disease dataset
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    for i, url in enumerate(urls):
        try:
            if i == 0:  # UCI format
                df = pd.read_csv(url, header=None, names=col_names)
                # Remove rows with missing values (marked as '?')
                df = df.replace('?', np.nan)
                df = df.dropna()
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                # Convert target to binary (0 = no disease, 1 = disease)
                df['target'] = (df['target'] > 0).astype(int)
            else:
                df = pd.read_csv(url)
            print(f"Successfully loaded heart disease dataset from URL {i+1}")
            return df
        except Exception as e:
            print(f"Failed to load from URL {i+1}: {str(e)}")
            continue
    
    raise Exception("Could not load heart disease dataset from any available URL")

def preprocess_data(df, target_col):
    # Create a copy to avoid warnings
    df = df.copy()
    # Fill zeros/invalid values in certain columns with median (for diabetes)
    if target_col == 'Outcome':
        cols_fill_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_fill_zero:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def logistic_regression_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", report)
    return model

# Choose dataset to use: 'diabetes' or 'heart'
dataset_choice = 'diabetes'  # Change to 'heart' for heart disease prediction

if dataset_choice == 'diabetes':
    print("Loading Diabetes dataset...")
    data = load_diabetes_dataset()
    X, y = preprocess_data(data, 'Outcome')
elif dataset_choice == 'heart':
    print("Loading Heart Disease dataset...")
    data = load_heart_disease_dataset()
    X, y = preprocess_data(data, 'target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
print(f"Training Logistic Regression model on {dataset_choice} dataset...")
model = logistic_regression_model(X_train, y_train, X_test, y_test)

# Sample predictions on test data
print("Sample Predictions:")
for i in range(5):
    sample_X = X_test[i].reshape(1, -1)
    pred_prob = model.predict_proba(sample_X)[0][1]
    pred_class = model.predict(sample_X)[0]
    print(f"Input features: {sample_X}")
    print(f"Predicted risk probability: {pred_prob:.2f} -> Class: {pred_class}")

"""
This completes the Health Risk Predictor project.
The prompt covers dataset loading, preprocessing, logistic regression training,
evaluation, and sample predictions.
It can be switched between diabetes or heart disease by changing 'dataset_choice'.
Run fully in Cursor for an end-to-end health risk prediction demonstration.
"""
