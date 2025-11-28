# Task 1: Classical ML with Scikit-learn
# Dataset: Iris Species Dataset
# Goal: Preprocess data, train Decision Tree, evaluate metrics.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def main():
    print("-------------------------------------------------")
    print("Task 1: Iris Species Classification with Scikit-learn")
    print("-------------------------------------------------")

    # 1. Load Dataset
    print("\n[1] Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Convert to DataFrame for better visualization (optional but good practice)
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

    # 2. Preprocessing
    # The Iris dataset from sklearn is already clean (no missing values, numeric features).
    # However, in a real scenario, we would handle missing values here.
    print("\n[2] Preprocessing...")
    if pd.isnull(X).any():
        print("Warning: Missing values detected. Imputing...")
        # Imputation logic would go here
    else:
        print("No missing values found.")

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 3. Train Model
    print("\n[3] Training Decision Tree Classifier...")
    # Initialize the classifier
    clf = DecisionTreeClassifier(random_state=42)
    # Train the classifier
    clf.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate Model
    print("\n[4] Evaluating Model...")
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    # Average='macro' is used for multiclass classification to treat all classes equally
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
