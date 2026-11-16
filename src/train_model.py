"""
Model Training Script for Deployment
Internship Track: Week 4 - Task 2
Author: Sayantani Adhikary
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_and_save_model(output_path: str = "model.joblib"):
    """Train a Random Forest Classifier and export the serialized artifact."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained successfully. Test Accuracy: {acc * 100:.2f}%")

    joblib.dump({"model": model, "target_names": list(iris.target_names)}, output_path)
    print(f"Saved trained model artifact to {output_path}")


if __name__ == "__main__":
    train_and_save_model()
