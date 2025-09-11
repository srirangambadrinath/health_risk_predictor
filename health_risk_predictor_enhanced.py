"""
Enhanced Health Risk Predictor using Logistic Regression with Visualizations

This enhanced version includes data visualization capabilities for better insights
into the datasets and model performance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    return X_scaled, y, df, scaler

def plot_data_distribution(df, target_col, dataset_name):
    """Plot data distribution and correlations, each in its own window"""
    # 1. Target distribution
    fig1 = plt.figure(figsize=(6, 5))
    target_counts = df[target_col].value_counts()
    plt.pie(target_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%', startangle=90)
    plt.title(f'{dataset_name} - Target Distribution')
    fig1.canvas.manager.set_window_title(f'{dataset_name} - Target Distribution')

    # 2. Feature correlation heatmap
    if dataset_name == 'Diabetes':
        numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        corr_matrix = df[numeric_cols].corr()
    else:
        numeric_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        corr_matrix = df[numeric_cols].corr()

    fig2 = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'{dataset_name} - Feature Correlation')
    fig2.canvas.manager.set_window_title(f'{dataset_name} - Feature Correlation')

    # 3. Key features distribution
    fig3 = plt.figure(figsize=(8, 6))
    if dataset_name == 'Diabetes':
        key_features = ['Glucose', 'BMI', 'Age']
        for feature in key_features:
            plt.hist(df[df[target_col] == 0][feature], alpha=0.5, label=f'No Disease ({feature})', bins=20)
            plt.hist(df[df[target_col] == 1][feature], alpha=0.5, label=f'Disease ({feature})', bins=20)
    else:
        key_features = ['age', 'trestbps', 'chol']
        for feature in key_features:
            plt.hist(df[df[target_col] == 0][feature], alpha=0.5, label=f'No Disease ({feature})', bins=20)
            plt.hist(df[df[target_col] == 1][feature], alpha=0.5, label=f'Disease ({feature})', bins=20)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Key Features Distribution')
    plt.legend()
    fig3.canvas.manager.set_window_title(f'{dataset_name} - Key Features Distribution')

    # 4. Box plots for key features
    if dataset_name == 'Diabetes':
        df_melted = df[['Glucose', 'BMI', 'Age', target_col]].melt(id_vars=[target_col], 
                                                                   value_vars=['Glucose', 'BMI', 'Age'])
    else:
        df_melted = df[['age', 'trestbps', 'chol', target_col]].melt(id_vars=[target_col], 
                                                                     value_vars=['age', 'trestbps', 'chol'])
    fig4 = plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melted, x='variable', y='value', hue=target_col)
    plt.title(f'{dataset_name} - Key Features Box Plot')
    plt.xticks(rotation=45)
    fig4.canvas.manager.set_window_title(f'{dataset_name} - Key Features Box Plot')

    # 5. Age distribution by target
    age_col = 'Age' if dataset_name == 'Diabetes' else 'age'
    fig5 = plt.figure(figsize=(8, 6))
    plt.hist(df[df[target_col] == 0][age_col], alpha=0.7, label='No Disease', bins=20)
    plt.hist(df[df[target_col] == 1][age_col], alpha=0.7, label='Disease', bins=20)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Age Distribution')
    plt.legend()
    fig5.canvas.manager.set_window_title(f'{dataset_name} - Age Distribution')

    # 6. Dataset info
    fig6 = plt.figure(figsize=(6, 4))
    plt.axis('off')
    info_text = f"""
    Dataset: {dataset_name}
    Total Samples: {len(df)}
    Features: {len(df.columns) - 1}
    Disease Cases: {df[target_col].sum()}
    No Disease: {len(df) - df[target_col].sum()}
    Disease Rate: {df[target_col].mean():.1%}
    """
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    fig6.canvas.manager.set_window_title(f'{dataset_name} - Dataset Info')

    plt.show()

def plot_model_performance(y_test, y_pred, dataset_name):
    """Plot model performance metrics in separate windows"""
    # 1. Confusion Matrix
    fig1 = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    fig1.canvas.manager.set_window_title(f'{dataset_name} - Confusion Matrix')

    # 2. Prediction distribution (classes)
    fig2 = plt.figure(figsize=(6, 5))
    plt.hist(y_pred, bins=2, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Class')
    plt.ylabel('Frequency')
    plt.title(f'{dataset_name} - Prediction Distribution')
    fig2.canvas.manager.set_window_title(f'{dataset_name} - Prediction Distribution')

    # 3. Accuracy bar
    fig3 = plt.figure(figsize=(6, 5))
    accuracy = accuracy_score(y_test, y_pred)
    categories = ['Correct', 'Incorrect']
    values = [accuracy, 1 - accuracy]
    colors = ['lightgreen', 'lightcoral']
    plt.bar(categories, values, color=colors)
    plt.ylabel('Proportion')
    plt.title(f'{dataset_name} - Accuracy: {accuracy:.1%}')
    fig3.canvas.manager.set_window_title(f'{dataset_name} - Accuracy')

    plt.show()

def logistic_regression_model(X_train, y_train, X_test, y_test, dataset_name):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", report)
    
    # Plot model performance
    plot_model_performance(y_test, y_pred, dataset_name)
    
    return model, y_pred_proba

def main():
    # Choose dataset to use: 'diabetes' or 'heart'
    dataset_choice = 'diabetes'  # Change to 'heart' for heart disease prediction
    
    if dataset_choice == 'diabetes':
        print("Loading Diabetes dataset...")
        data = load_diabetes_dataset()
        X, y, df, scaler = preprocess_data(data, 'Outcome')
        dataset_name = 'Diabetes'
        target_col = 'Outcome'
    elif dataset_choice == 'heart':
        print("Loading Heart Disease dataset...")
        data = load_heart_disease_dataset()
        X, y, df, scaler = preprocess_data(data, 'target')
        dataset_name = 'Heart Disease'
        target_col = 'target'
    
    # Plot data distribution
    print(f"Generating visualizations for {dataset_name} dataset...")
    plot_data_distribution(df, target_col, dataset_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    print(f"Training Logistic Regression model on {dataset_choice} dataset...")
    model, y_pred_proba = logistic_regression_model(X_train, y_train, X_test, y_test, dataset_name)
    
    # Sample predictions on test data
    print("Sample Predictions:")
    for i in range(5):
        sample_X = X_test[i].reshape(1, -1)
        pred_prob = model.predict_proba(sample_X)[0][1]
        pred_class = model.predict(sample_X)[0]
        print(f"Sample {i+1}:")
        print(f"  Predicted risk probability: {pred_prob:.2f} -> Class: {pred_class}")
        print(f"  Actual class: {y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]}")
        print()

if __name__ == "__main__":
    main()
