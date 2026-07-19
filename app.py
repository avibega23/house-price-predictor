"""
app.py

Streamlit web app for predicting house prices using the model saved in
model.pkl. Users enter house details and click "Predict Price" to see the
estimated price and which model produced it.
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
FEATURE_COLUMNS = ["Area", "Bedrooms", "Bathrooms", "LocationScore"]

# Friendly display names for the model classes train.py can produce.
MODEL_DISPLAY_NAMES = {
    "LinearRegression": "Linear Regression",
    "RandomForestRegressor": "Random Forest Regressor",
}


@st.cache_resource
def load_model():
    """Load the trained model from disk (cached so it's only loaded once)."""
    return joblib.load(MODEL_PATH)


def predict_price(model, area, bedrooms, bathrooms, location_score):
    """Predict the price for a single house given its features."""
    # Use a DataFrame with the same column names as training data so
    # scikit-learn doesn't warn about missing feature names.
    features = pd.DataFrame([[area, bedrooms, bathrooms, location_score]], columns=FEATURE_COLUMNS)
    return model.predict(features)[0]


def format_inr(amount):
    """Format a number using Indian digit grouping, e.g. 1234567 -> 12,34,567."""
    rupees = str(round(amount))
    last_three, rest = rupees[-3:], rupees[:-3]
    if not rest:
        return last_three
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    groups.insert(0, rest)
    return ",".join(groups) + "," + last_three


def main():
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
    st.title("🏠 House Price Predictor")
    st.write("Enter the details of a house below to get an estimated price.")

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("model.pkl not found. Run `python train.py` first to train and save a model.")
        return

    # --- User inputs ---
    area = st.number_input("Area (sq ft)", min_value=800, max_value=5000, value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
    location_score = st.slider("Location Score", min_value=1, max_value=10, value=5)

    if st.button("Predict Price"):
        price = predict_price(model, area, bedrooms, bathrooms, location_score)
        model_name = MODEL_DISPLAY_NAMES.get(type(model).__name__, type(model).__name__)

        st.success(f"Estimated House Price: ₹{format_inr(price)}")
        st.caption(f"Prediction made using: {model_name}")


if __name__ == "__main__":
    main()
