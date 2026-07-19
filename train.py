"""
train.py

Trains two regression models (Linear Regression and Random Forest) on
dataset.csv, compares their R^2 scores on a held-out test set, and saves
whichever model performs better to model.pkl using joblib.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["Area", "Bedrooms", "Bathrooms", "LocationScore"]
TARGET_COLUMN = "Price"
DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"


def load_data(path=DATASET_PATH):
    """Load the housing dataset from a CSV file."""
    return pd.read_csv(path)


def split_data(df):
    """Split the dataset into training and testing sets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest Regressor model."""
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return the R^2 score of a model on the test set (higher is better)."""
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Train both candidate models on the same training split.
    linear_model = train_linear_regression(X_train, y_train)
    forest_model = train_random_forest(X_train, y_train)

    # Compare them on the untouched test split.
    linear_r2 = evaluate_model(linear_model, X_test, y_test)
    forest_r2 = evaluate_model(forest_model, X_test, y_test)

    print(f"Linear Regression R2 score: {linear_r2:.4f}")
    print(f"Random Forest R2 score:     {forest_r2:.4f}")

    # Keep whichever model scored higher. app.py inspects the saved model's
    # class name to know which one is in use, so no extra metadata file
    # is needed alongside model.pkl.
    if forest_r2 >= linear_r2:
        best_model, best_name = forest_model, "Random Forest Regressor"
    else:
        best_model, best_name = linear_model, "Linear Regression"

    print(f"Best model: {best_name}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
