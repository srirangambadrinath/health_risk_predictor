"""
Interactive Health Risk Predictor

A simple interactive script that allows users to choose between datasets
and see predictions with explanations.
"""

import sys
from health_risk_predictor_enhanced import *

def get_user_choice():
    """Get user's choice for dataset"""
    print("Health Risk Predictor - Interactive Mode")
    print("=" * 40)
    print("Choose a dataset to analyze:")
    print("1. Diabetes Risk Prediction")
    print("2. Heart Disease Risk Prediction")
    print("3. Both (Comprehensive Demo)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Please enter a valid choice (1-4)")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def explain_predictions(dataset_name, model, X_test, y_test, scaler, feature_names):
    """Explain sample predictions in detail"""
    print(f"\n{dataset_name} - Detailed Sample Predictions:")
    print("-" * 50)
    
    for i in range(3):
        sample_X = X_test[i].reshape(1, -1)
        pred_prob = model.predict_proba(sample_X)[0][1]
        pred_class = model.predict(sample_X)[0]
        actual_class = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        
        print(f"\nSample {i+1}:")
        print(f"  Risk Probability: {pred_prob:.2f} ({pred_prob*100:.1f}%)")
        print(f"  Prediction: {'HIGH RISK' if pred_class == 1 else 'LOW RISK'}")
        print(f"  Actual: {'HIGH RISK' if actual_class == 1 else 'LOW RISK'}")
        print(f"  Correct: {'✓' if pred_class == actual_class else '✗'}")
        
        # Show feature values (original scale)
        print(f"  Key Features:")
        for j, feature in enumerate(feature_names[:5]):  # Show first 5 features
            print(f"    {feature}: {sample_X[0][j]:.2f}")

def run_diabetes_analysis():
    """Run diabetes risk analysis"""
    print("\n" + "=" * 50)
    print("DIABETES RISK ANALYSIS")
    print("=" * 50)
    
    # Load data
    print("Loading Diabetes dataset...")
    data = load_diabetes_dataset()
    X, y, df, scaler = preprocess_data(data, 'Outcome')
    
    # Show basic info
    print(f"Dataset loaded: {len(data)} samples, {len(data.columns)-1} features")
    print(f"Diabetes cases: {data['Outcome'].sum()} ({data['Outcome'].mean():.1%})")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training logistic regression model...")
    model, y_pred_proba = logistic_regression_model(X_train, y_train, X_test, y_test, 'Diabetes')
    
    # Explain predictions
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    explain_predictions("Diabetes", model, X_test, y_test, scaler, feature_names)

def run_heart_disease_analysis():
    """Run heart disease risk analysis"""
    print("\n" + "=" * 50)
    print("HEART DISEASE RISK ANALYSIS")
    print("=" * 50)
    
    # Load data
    print("Loading Heart Disease dataset...")
    data = load_heart_disease_dataset()
    X, y, df, scaler = preprocess_data(data, 'target')
    
    # Show basic info
    print(f"Dataset loaded: {len(data)} samples, {len(data.columns)-1} features")
    print(f"Heart disease cases: {data['target'].sum()} ({data['target'].mean():.1%})")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training logistic regression model...")
    model, y_pred_proba = logistic_regression_model(X_train, y_train, X_test, y_test, 'Heart Disease')
    
    # Explain predictions
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    explain_predictions("Heart Disease", model, X_test, y_test, scaler, feature_names)

def run_comprehensive_demo():
    """Run both analyses"""
    run_diabetes_analysis()
    run_heart_disease_analysis()

def main():
    """Main interactive loop"""
    while True:
        choice = get_user_choice()
        
        if choice == 1:
            run_diabetes_analysis()
        elif choice == 2:
            run_heart_disease_analysis()
        elif choice == 3:
            run_comprehensive_demo()
        elif choice == 4:
            print("Thank you for using Health Risk Predictor!")
            break
        
        if choice != 4:
            try:
                input("\nPress Enter to continue...")
            except EOFError:
                print("\nContinuing...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
