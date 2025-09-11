"""
Demo script to run both diabetes and heart disease predictions
"""

from health_risk_predictor_enhanced import *

def run_diabetes_demo():
    print("=" * 60)
    print("DIABETES RISK PREDICTION DEMO")
    print("=" * 60)
    
    # Load and preprocess diabetes data
    print("Loading Diabetes dataset...")
    data = load_diabetes_dataset()
    X, y, df, scaler = preprocess_data(data, 'Outcome')
    
    # Plot data distribution
    print("Generating visualizations...")
    plot_data_distribution(df, 'Outcome', 'Diabetes')
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model...")
    model, y_pred_proba = logistic_regression_model(X_train, y_train, X_test, y_test, 'Diabetes')
    
    print("\nSample Predictions:")
    for i in range(3):
        sample_X = X_test[i].reshape(1, -1)
        pred_prob = model.predict_proba(sample_X)[0][1]
        pred_class = model.predict(sample_X)[0]
        actual_class = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        print(f"Sample {i+1}: Risk Probability = {pred_prob:.2f}, Predicted = {pred_class}, Actual = {actual_class}")

def run_heart_disease_demo():
    print("\n" + "=" * 60)
    print("HEART DISEASE RISK PREDICTION DEMO")
    print("=" * 60)
    
    # Load and preprocess heart disease data
    print("Loading Heart Disease dataset...")
    data = load_heart_disease_dataset()
    X, y, df, scaler = preprocess_data(data, 'target')
    
    # Plot data distribution
    print("Generating visualizations...")
    plot_data_distribution(df, 'target', 'Heart Disease')
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model...")
    model, y_pred_proba = logistic_regression_model(X_train, y_train, X_test, y_test, 'Heart Disease')
    
    print("\nSample Predictions:")
    for i in range(3):
        sample_X = X_test[i].reshape(1, -1)
        pred_prob = model.predict_proba(sample_X)[0][1]
        pred_class = model.predict(sample_X)[0]
        actual_class = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        print(f"Sample {i+1}: Risk Probability = {pred_prob:.2f}, Predicted = {pred_class}, Actual = {actual_class}")

if __name__ == "__main__":
    print("HEALTH RISK PREDICTOR - COMPREHENSIVE DEMO")
    print("This demo will run both diabetes and heart disease predictions")
    print("with visualizations and performance metrics.\n")
    
    # Run diabetes demo
    run_diabetes_demo()
    
    # Run heart disease demo
    run_heart_disease_demo()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("Both models have been trained and evaluated.")
    print("Check the generated plots for detailed insights!")
