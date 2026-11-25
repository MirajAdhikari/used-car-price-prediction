"""Used car price predictor (Streamlit app).

Run from the project folder with:
    streamlit run app/streamlit_app.py

The app loads the saved model (models/car_price_model.joblib), asks for the car's
details, and shows the predicted selling price in Indian rupees.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# File locations, worked out from this file's position so the app runs from any folder
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
DATA_PATH = ROOT / "data" / "processed" / "cardekho_cleaned.csv"

# When to show a warning next to the price
EXPENSIVE_LAKH = 50         # predicted price above this many lakh rupees
FEW_CARS = 10               # the chosen car model has fewer cars than this in the data
MAX_AGE_YEARS = 20          # a car older than this is unusual
MAX_KM_PER_YEAR = 100_000   # more kilometers per year than this is unusual

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def format_inr(amount):
    """Write a number with Indian digit grouping, for example 745000 -> 7,45,000."""
    digits = str(int(round(amount)))
    if len(digits) <= 3:
        return digits
    last_three, rest = digits[-3:], digits[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last_three])


@st.cache_resource
def load_model():
    """Load the saved pipeline (preprocessing + model). Cached, so it loads only once."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_info():
    """Load the facts about the model (settings, scores, features, library versions)."""
    with open(INFO_PATH) as file:
        return json.load(file)


@st.cache_data
def load_data():
    """Load the cleaned car data. It is used to fill in typical values for each car model."""
    return pd.read_csv(DATA_PATH)


def typical_specs(data, brand, model, fuel):
    """The typical engine, power, mileage, and seats for a car model.

    Uses the cars of the same model and fuel type. If there are fewer than 3 of those,
    it uses all cars of that model.
    """
    same_model = data[(data["brand"] == brand) & (data["model"] == model)]
    same_fuel = same_model[same_model["fuel_type"] == fuel]
    source = same_fuel if len(same_fuel) >= 3 else same_model
    return {
        "engine": int(round(source["engine"].median())),
        "max_power": round(float(source["max_power"].median()), 1),
        "mileage": round(float(source["mileage"].median()), 1),
        "seats": int(round(source["seats"].median())),
    }


# ---------------------------------------------------------------------------
# Check that everything needed is in place
# ---------------------------------------------------------------------------

missing = [str(path) for path in (MODEL_PATH, INFO_PATH, DATA_PATH) if not path.exists()]
if missing:
    st.error("These files are missing, so the app cannot start:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

info = load_info()
data = load_data()

# A model saved with one scikit-learn version can fail with another, so check first
trained_with = info["library_versions"]["scikit-learn"]
if sklearn.__version__ != trained_with:
    st.error(
        f"This model was saved with scikit-learn {trained_with}, but this computer has "
        f"scikit-learn {sklearn.__version__}. Install the right version with "
        f"`pip install -r requirements.txt` and start the app again."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"The model file could not be loaded: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Used Car Price Predictor")
st.write(
    "Enter the details of a used car to get an estimated selling price in Indian rupees. "
    "The model was trained on used car listings from the Indian market (CarDekho data)."
)

st.subheader("Car details")
left, right = st.columns(2)

with left:
    brand = st.selectbox("Brand", sorted(data["brand"].unique()), key="brand")
    models_of_brand = sorted(data.loc[data["brand"] == brand, "model"].unique())
    # The keys include the choices made so far, so a widget starts fresh when they change
    model_name = st.selectbox("Model", models_of_brand, key=f"model|{brand}")

    # Only offer fuel types and transmissions that exist in the data for this car model
    cars_of_model = data[(data["brand"] == brand) & (data["model"] == model_name)]
    fuel_options = sorted(cars_of_model["fuel_type"].unique())
    fuel = st.selectbox("Fuel type", fuel_options, key=f"fuel|{brand}|{model_name}")

    cars_with_fuel = cars_of_model[cars_of_model["fuel_type"] == fuel]
    transmission_options = sorted(cars_with_fuel["transmission_type"].unique())
    transmission = st.selectbox("Transmission", transmission_options,
                                key=f"transmission|{brand}|{model_name}|{fuel}")

with right:
    seller_type = st.selectbox("Seller type", sorted(data["seller_type"].unique()), key="seller")
    vehicle_age = st.number_input("Vehicle age (years)", min_value=0, max_value=40,
                                  value=5, step=1, key="age")
    km_driven = st.number_input("Kilometers driven", min_value=0, max_value=1_000_000,
                                value=50_000, step=1_000, key="km")

st.subheader("Specifications")
st.caption(
    "These start with the typical values for this model and fuel type. "
    "Change them if you know the exact values for the car."
)
typical = typical_specs(data, brand, model_name, fuel)
spec_key = f"{brand}|{model_name}|{fuel}"      # new defaults whenever the car model or fuel changes

col1, col2, col3, col4 = st.columns(4)
with col1:
    engine = st.number_input("Engine (cc)", min_value=600, max_value=7000,
                             value=typical["engine"], step=1, key=f"engine|{spec_key}")
with col2:
    max_power = st.number_input("Max power (bhp)", min_value=30.0, max_value=700.0,
                                value=typical["max_power"], step=0.1, key=f"power|{spec_key}")
with col3:
    mileage = st.number_input("Mileage (km/l)", min_value=3.0, max_value=40.0,
                              value=typical["mileage"], step=0.1, key=f"mileage|{spec_key}")
with col4:
    seats = st.number_input("Seats", min_value=2, max_value=10,
                            value=typical["seats"], step=1, key=f"seats|{spec_key}")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

# One row with the same column names the model was trained on
columns = info["numeric_features"] + info["categorical_features"]
values = {
    "vehicle_age": vehicle_age,
    "km_driven": km_driven,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats,
    "brand": brand,
    "model": model_name,
    "seller_type": seller_type,
    "fuel_type": fuel,
    "transmission_type": transmission,
}
car = pd.DataFrame([values])[columns]

predicted_price = max(float(model.predict(car)[0]), 0.0)       # in rupees

st.subheader("Estimated price")
st.metric("Estimated selling price", f"₹ {predicted_price / 100_000:.2f} lakh")
st.write(f"About ₹ {format_inr(round(predicted_price, -3))}")
st.caption(
    f"In testing, half of the predictions were within about "
    f"{info['test_all_cars']['Median % error']:.0f}% of the real price."
)

# Warnings: the price is always shown, but the person should know how much to trust it
if predicted_price > EXPENSIVE_LAKH * 100_000:
    top_lakh = info["highest_training_price_rupees"] / 100_000
    st.warning(
        f"This is a very expensive car. The model has seen few cars like this and cannot "
        f"predict prices above the most expensive car it was trained on (₹ {top_lakh:.0f} lakh). "
        f"Treat this price as a rough guide only."
    )

cars_in_data = len(cars_of_model)
if cars_in_data < FEW_CARS:
    st.warning(
        f"The data has only {cars_in_data} {model_name} car(s), so the model has learned little "
        f"about this model. Treat this price as a rough guide."
    )

km_per_year = km_driven / max(vehicle_age, 1)
if vehicle_age > MAX_AGE_YEARS:
    st.warning(f"A {vehicle_age}-year-old car is unusual in the data, so the price may be unreliable.")
if km_per_year > MAX_KM_PER_YEAR:
    st.warning(
        f"{km_per_year:,.0f} km per year is far more than normal use, so the price may be "
        f"unreliable. Check the age and kilometers."
    )


# ---------------------------------------------------------------------------
# About the model
# ---------------------------------------------------------------------------

with st.expander("About this model"):
    cv = info["cross_validation"]
    test = info["test_all_cars"]
    test_in_range = info["test_without_cars_above_training_range"]

    st.write(
        f"**Model:** {info['model']}, trained on {info['training_rows']:,} used cars. "
        f"It predicts the price in rupees from the car's age, kilometers, specifications, brand, "
        f"model, fuel type, transmission, and seller type."
    )

    scores = pd.DataFrame(
        {
            "Cross-validation": cv,
            "Test set": test,
            "Test set (without the 2 most expensive cars)": test_in_range,
        }
    ).round(2)
    st.table(scores)

    st.write(
        "**How to read this:** the typical error is about 10%. The model works well for "
        "ordinary cars. It is unreliable for very expensive luxury cars, because it cannot predict "
        f"prices above the most expensive car in its training data (₹ "
        f"{info['highest_training_price_rupees'] / 100_000:.0f} lakh). Two such cars in the test set "
        "cause the large RMSE and low R2 in the test column."
    )
    st.write("Errors (RMSE and MAE) are in lakh rupees. 1 lakh = 100,000 rupees.")

st.caption("This is an estimate from a statistical model, not an official valuation.")
