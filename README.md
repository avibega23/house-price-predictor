# House Price Predictor

## Project Overview

A simple machine learning web app that estimates a house's price from its
area, number of bedrooms, number of bathrooms, and a location score.

- `dataset.csv` — a synthetic dataset of 1000 houses with realistic
  relationships between size/location and price.
- `train.py` — trains a Linear Regression model and a Random Forest
  Regressor, compares their R² scores, and saves the better one to
  `model.pkl`.
- `app.py` — a Streamlit app that loads `model.pkl` and lets you enter house
  details to get an estimated price.

## Installation

Clone or download this project, then follow the steps below.

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

Run the training script to train both models and save the better one as
`model.pkl`:

```bash
python train.py
```

This prints the R² score for each model and tells you which one was saved.

## Running the Streamlit App

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).
Enter the house's area, bedrooms, bathrooms, and location score, then click
**Predict Price** to see the estimated price (in ₹, Indian Rupees) and which
model produced it.

## Example Screenshots

_Add screenshots of the running app here, e.g.:_

```text
screenshots/
├── app-form.png       # The input form before predicting
└── app-prediction.png # The estimated price after clicking Predict
```

## Project Structure

```text
house-price-predictor/
│
├── app.py
├── train.py
├── dataset.csv
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```
