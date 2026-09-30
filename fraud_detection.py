import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# CREDIT CARD FRAUD DETECTION
# Task 2 - Machine Learning
# Algorithms: Logistic Regression, Decision Tree, Random Forest
# ============================================================

TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"

# Set this to None if your computer has enough RAM/time.
# 250000 is a practical starting point for a student laptop.
MAX_TRAIN_ROWS = 250000
RANDOM_STATE = 42

os.makedirs("results", exist_ok=True)


def load_data():
    print("\n[1/7] Loading dataset...")

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"{TRAIN_FILE} not found. Put fraudTrain.csv in the same folder as this script."
        )

    train = pd.read_csv(TRAIN_FILE)

    if os.path.exists(TEST_FILE):
        test = pd.read_csv(TEST_FILE)
        print(f"Training file: {train.shape}")
        print(f"Test file:     {test.shape}")
    else:
        # If fraudTest.csv is unavailable, make a validation split.
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(
            train,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=train["is_fraud"]
        )
        print("fraudTest.csv not found, so an 80/20 validation split was created.")

    return train, test


def engineer_features(df):
    """Create useful numeric/time features and remove high-cardinality identifiers."""
    df = df.copy()

    # Convert transaction timestamp into useful features.
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["transaction_hour"] = dt.dt.hour
        df["transaction_day"] = dt.dt.day
        df["transaction_month"] = dt.dt.month
        df["transaction_dayofweek"] = dt.dt.dayofweek
        df.drop(columns=["trans_date_trans_time"], inplace=True)

    # These are identifiers / personal-information-like fields and are not
    # useful as direct model inputs for this student project.
    drop_cols = [
        "Unnamed: 0",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "dob",
        "trans_num",
        "unix_time",
        "merchant",
        "job"
    ]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def prepare_xy(train, test):
    print("\n[2/7] Preparing features...")

    if "is_fraud" not in train.columns:
        raise ValueError("The training data must contain the 'is_fraud' column.")

    if "is_fraud" not in test.columns:
        raise ValueError(
            "The supplied fraudTest.csv does not contain 'is_fraud'. "
            "For this assignment, use a labeled test/validation file."
        )

    # Sample the training data while preserving the fraud/non-fraud ratio.
    if MAX_TRAIN_ROWS and len(train) > MAX_TRAIN_ROWS:
        from sklearn.model_selection import train_test_split
        train, _ = train_test_split(
            train,
            train_size=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
            stratify=train["is_fraud"]
        )
        print(f"Using stratified sample of {len(train):,} training rows for faster execution.")

    train = engineer_features(train)
    test = engineer_features(test)

    y_train = train.pop("is_fraud").astype(int)
    y_test = test.pop("is_fraud").astype(int)

    # Keep only columns common to both datasets.
    common_cols = [c for c in train.columns if c in test.columns]
    X_train = train[common_cols]
    X_test = test[common_cols]

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    print(f"Fraud cases in training sample: {y_train.sum():,}")
    print(f"Legitimate cases in training sample: {(y_train == 0).sum():,}")

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n[4/7] Training {name}...")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    # Save confusion matrix.
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"results/{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def main():
    print("=" * 65)
    print("       CREDIT CARD FRAUD DETECTION - TASK 2")
    print("=" * 65)

    train, test = load_data()

    # EDA: fraud distribution
    print("\n[3/7] Creating fraud distribution chart...")
    fraud_counts = train["is_fraud"].value_counts().sort_index()
    print(fraud_counts)

    plt.figure(figsize=(7, 5))
    plt.bar(["Legitimate (0)", "Fraud (1)"], fraud_counts.values)
    plt.title("Transaction Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig("results/fraud_distribution.png", dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_xy(
        train, test
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=150,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    results = []

    for name, classifier in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        result = evaluate_model(
            name, pipe, X_train, X_test, y_train, y_test
        )
        results.append(result)

    print("\n[6/7] Creating model comparison...")
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    # Comparison chart
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150)
    plt.close()

    # Train the Random Forest once more and save it for optional reuse.
    print("\n[7/7] Saving final Random Forest model...")
    import joblib

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", models["Random Forest"])
    ])
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, "results/final_random_forest.joblib")

    print("\n" + "=" * 65)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print("\nCheck the 'results' folder for:")
    print("  - fraud_distribution.png")
    print("  - logistic_regression_confusion_matrix.png")
    print("  - decision_tree_confusion_matrix.png")
    print("  - random_forest_confusion_matrix.png")
    print("  - model_comparison.png")
    print("  - model_comparison.csv")
    print("  - final_random_forest.joblib")
    print("\nUse the generated graphs/screenshots in your PPT and demo video.")


if __name__ == "__main__":
    main()
