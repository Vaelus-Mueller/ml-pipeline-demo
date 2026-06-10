import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality

# Configuration
CONFIG = {
    "data_url": "https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/student_dropout_dataset.csv",
    "target": "Dropout",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10,
    "numeric_columns": [
        "Age", "Family_Income", "Study_Hours_per_Day", "Attendance_Rate",
        "Assignment_Delay_Days", "Travel_Time_Minutes", "Stress_Index",
        "GPA", "Semester_GPA", "CGPA"
    ],
    "categorical_columns": [
        "Gender", "Internet_Access", "Part_Time_Job", "Scholarship",
        "Semester", "Department", "Parental_Education"
    ],
    "min_accuracy": 0.35,
    "min_f1": 0.30,
}

def load_data(url):
    """Load dataset from URL."""
    print(f"Loading data from {url}...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def train_model(config=None):
    """Full training pipeline. Returns metrics dictionary."""
    if config is None:
        config = CONFIG

    # Load
    df = load_data(config["data_url"])

    # Drop ID column
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    # Validate
    required = config["numeric_columns"] + config["categorical_columns"] + [config["target"]]
    validate_dataframe(df, required, config["target"])

    # Data quality check
    quality = check_data_quality(df, config["numeric_columns"])
    print(f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates")

    # Clean
    df = clean_data(df, config["numeric_columns"], config["categorical_columns"])

    # Encode
    df = encode_categoricals(df, config["categorical_columns"])

    # Split
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train
    print("Training random forest...")
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"]
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    }

    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")

    # Check thresholds
    meets_accuracy = metrics["accuracy"] >= config.get("min_accuracy", 0)
    meets_f1 = metrics["f1_score"] >= config.get("min_f1", 0)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(repo_root, "models")
    metrics_dir = os.path.join(repo_root, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, "results.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    if meets_accuracy and meets_f1:
        model_path = os.path.join(models_dir, "model.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        print(f"Saved model to {model_path}")
    else:
        print("Model did not meet performance thresholds; model not saved.")

    print(f"Saved metrics to {metrics_path}")
    return metrics

if __name__ == "__main__":
    metrics = train_model()
    # Exit with non-zero code if thresholds not met
    if metrics["accuracy"] < CONFIG.get("min_accuracy", 0) or metrics["f1_score"] < CONFIG.get("min_f1", 0):
        sys.exit(2)
