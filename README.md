# Health Risk Predictor using Logistic Regression

A machine learning project that predicts health risk levels such as diabetes and heart disease using logistic regression classification.

## Project Overview

This project demonstrates how to build a health risk prediction system using logistic regression, a widely used classification algorithm. The system can predict:

- **Diabetes Risk**: Using the Pima Indians Diabetes dataset
- **Heart Disease Risk**: Using the Heart Disease UCI dataset

## Features

- **Multiple Scripts**: Basic, enhanced, interactive, and demo versions
- **Automatic Dataset Loading**: Downloads datasets from public sources
- **Data Preprocessing**: Handles missing values and feature scaling
- **Logistic Regression**: Model training and evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Visualizations**: Data distribution, correlations, and performance plots
- **Sample Predictions**: Probability scores with detailed explanations
- **Multiple Datasets**: Diabetes and heart disease prediction
- **Interactive Interface**: User-friendly command-line interface
- **Error Handling**: Robust dataset loading with fallback URLs

## Datasets

### Diabetes Dataset
- **Source**: Pima Indians Diabetes dataset
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Diabetes outcome (0 = No diabetes, 1 = Diabetes)

### Heart Disease Dataset
- **Source**: Heart Disease UCI dataset
- **Features**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, etc.
- **Target**: Heart disease presence (0 = No disease, 1 = Disease)

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the basic script with default settings (diabetes dataset):

```bash
python health_risk_predictor.py
```

### Enhanced Usage with Visualizations

Run the enhanced version with comprehensive visualizations:

```bash
python health_risk_predictor_enhanced.py
```

### Interactive Mode

Run the interactive predictor for a guided experience:

```bash
python interactive_predictor.py
```

### Comprehensive Demo

Run both diabetes and heart disease predictions:

```bash
python demo_both_datasets.py
```

### Switching Datasets

To switch between datasets, modify the `dataset_choice` variable in `health_risk_predictor.py`:

```python
# For diabetes prediction
dataset_choice = 'diabetes'

# For heart disease prediction
dataset_choice = 'heart'
```

### Advanced Usage

You can also use the functions individually:

```python
from health_risk_predictor import load_diabetes_dataset, preprocess_data, logistic_regression_model

# Load and preprocess data
data = load_diabetes_dataset()
X, y = preprocess_data(data, 'Outcome')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = logistic_regression_model(X_train, y_train, X_test, y_test)
```

## Project Structure

```
health_risk_predictor/
├── health_risk_predictor.py           # Basic script
├── health_risk_predictor_enhanced.py  # Enhanced version with visualizations
├── demo_both_datasets.py              # Demo script for both datasets
├── interactive_predictor.py           # Interactive command-line interface
├── requirements.txt                   # Python dependencies
└── README.md                         # Project documentation
```

## Model Performance

The logistic regression model typically achieves:

- **Diabetes Dataset**: ~75-80% accuracy
- **Heart Disease Dataset**: ~80-85% accuracy

Performance may vary based on data preprocessing and model parameters.

## Key Features

### Data Preprocessing
- Handles missing values by replacing zeros with median values
- Standardizes features using StandardScaler
- Automatic feature-target separation

### Model Training
- Logistic regression with 200 max iterations
- 80/20 train-test split
- Random state for reproducibility

### Evaluation Metrics
- Accuracy score
- Classification report (precision, recall, F1-score)
- Sample predictions with probability scores

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Example Output

```
Loading Diabetes dataset...
Training Logistic Regression model on diabetes dataset...
Accuracy: 76.62%
Classification Report:
               precision    recall  f1-score   support

           0       0.79      0.89      0.84        99
           1       0.70      0.52      0.60        54

    accuracy                           0.77       153
   macro avg       0.75      0.71      0.72       153
weighted avg       0.76      0.77      0.76       153

Sample Predictions:
Input features: [[ 0.64027131 -0.84488501 -0.84488501  0.20401277 -0.84488501  0.46849198
  -0.47367351 -1.42575339]]
Predicted risk probability: 0.15 -> Class: 0
...
```

## Contributing

Feel free to contribute to this project by:
- Adding new health risk datasets
- Improving data preprocessing techniques
- Adding visualization capabilities
- Enhancing model performance

## License

This project is open source and available under the MIT License.

## Disclaimer

This project is for educational and demonstration purposes only. It should not be used for actual medical diagnosis or health risk assessment without proper validation and medical supervision.
