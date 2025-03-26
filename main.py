# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Data Loading Functions
def load_data(train_path, test_path):
    print("Loading data...")
    df_train = pd.read_json(train_path)
    df_test = pd.read_json(test_path)
    return df_train, df_test

# Data Summary Functions
def summarize_data(df_train, df_test):
    print("First few rows of training data:")
    print(df_train.head())
    print("\nFirst few rows of testing data:")
    print(df_test.head())
    
    print("\nMissing values in each column (Train):")
    print(df_train.isnull().sum())
    print("\nMissing values in each column (Test):")
    print(df_test.isnull().sum())
    
    print("\nData types of each feature (Train):")
    print(df_train.dtypes)
    print("\nData types of each feature (Test):")
    print(df_test.dtypes)

# Visualization Functions
def visualize_numerical_features(df_train, df_test):
    num_features_train = df_train.select_dtypes(include=[np.number]).columns
    num_features_test = df_test.select_dtypes(include=[np.number]).columns

    # Histograms
    df_train[num_features_train].hist(bins=20, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Histograms of Numerical Features (Train)', fontsize=16)
    plt.tight_layout()
    plt.show()

    df_test[num_features_test].hist(bins=20, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Histograms of Numerical Features (Test)', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    if len(num_features_train) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_train[num_features_train].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap (Train)')
        plt.show()

# Feature Engineering Functions
def add_features(df_train, df_test):
    print("Adding 'num_ing' feature...")
    df_train['num_ing'] = df_train['ingredients'].apply(lambda x: len(x))
    df_test['num_ing'] = df_test['ingredients'].apply(lambda x: len(x))
    return df_train, df_test

# Model Preparation Functions
def prepare_data(df_train, df_test):
    print("Preparing training and testing data...")
    df_train['ingredients_text'] = df_train['ingredients'].apply(lambda x: " ".join(x))
    df_test['ingredients_text'] = df_test['ingredients'].apply(lambda x: " ".join(x))

    X_train = df_train['ingredients_text']
    y_train = df_train['cuisine']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(df_test['ingredients_text'])

    return X_train_vec, X_test_vec, y_train_encoded, label_encoder

# Model Training and Evaluation Functions
def train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, label_encoder):
    print("Training XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

    return model

def train_and_evaluate_knn(X_train, y_train, X_val, y_val, label_encoder):
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("\nClassification Report (KNN):")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

    return model

def train_and_evaluate_logistic_regression(X_train, y_train, X_val, y_val, label_encoder):
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

    return model

# Prediction Functions
def predict_and_save_results(model, X_test, label_encoder, df_test, column_name):
    print(f"Making predictions with {column_name}...")
    y_test_pred = model.predict(X_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    df_test[column_name] = y_test_pred_labels
    return df_test

# Main Workflow
def main():
    train_path = "train.json"
    test_path = "test.json"

    # Load and summarize data
    df_train, df_test = load_data(train_path, test_path)
    summarize_data(df_train, df_test)

    # Add features and visualize data
    df_train, df_test = add_features(df_train, df_test)
    visualize_numerical_features(df_train, df_test)

    # Prepare data for modeling
    X_train_vec, X_test_vec, y_train_encoded, label_encoder = prepare_data(df_train, df_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train_vec, y_train_encoded, test_size=0.2, random_state=42)

    # Train and evaluate models
    xgb_model = train_and_evaluate_xgboost(X_train, y_train, X_val, y_val, label_encoder)
    knn_model = train_and_evaluate_knn(X_train, y_train, X_val, y_val, label_encoder)
    log_reg_model = train_and_evaluate_logistic_regression(X_train, y_train, X_val, y_val, label_encoder)

    # Save predictions
    df_test = predict_and_save_results(xgb_model, X_test_vec, label_encoder, df_test, 'predicted_cuisine_xgb')
    df_test = predict_and_save_results(knn_model, X_test_vec, label_encoder, df_test, 'predicted_cuisine_knn')
    df_test = predict_and_save_results(log_reg_model, X_test_vec, label_encoder, df_test, 'predicted_cuisine_log_reg')

    print("\nSample Test Predictions:")
    print(df_test[['id', 'predicted_cuisine_xgb', 'predicted_cuisine_knn', 'predicted_cuisine_log_reg']].head())

if __name__ == "__main__":
    main()
