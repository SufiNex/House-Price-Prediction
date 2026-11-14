
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="wide"
)


# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("🏡 California House Price Prediction")

st.caption(
    "Machine Learning Based House Price Prediction Application"
)


# --------------------------------------------------
# LOAD DATASET AND SCALER
# --------------------------------------------------

data = fetch_california_housing()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["MedHouseVal"] = data.target


# Using the same seven features as the existing app

feature_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude"
]

X = df[feature_columns]

scaler = MinMaxScaler()
scaler.fit(X)


# --------------------------------------------------
# SIDEBAR - HOUSE FEATURES
# --------------------------------------------------

st.sidebar.title("Select House Features 🏡")

st.sidebar.image(
    "https://i.pinimg.com/originals/b1/cf/53/b1cf530756b23930aab499c1a11b443c.gif",
    use_container_width=True
)

feature_config = {
    "MedInc": (0.49, 15.01, 4.0, 0.1),
    "HouseAge": (1, 52, 30, 1),
    "AveRooms": (0.84, 141.91, 5.5, 0.1),
    "AveBedrms": (0.33, 34.07, 1.1, 0.1),
    "Population": (3, 35682, 1200, 1),
    "AveOccup": (0.69, 1243.34, 3.0, 0.1),
    "Latitude": (32.54, 41.95, 37.5, 0.1)
}

all_values = []

for feature in feature_columns:

    min_val, max_val, default, step = feature_config[feature]

    value = st.sidebar.slider(
        f"Select {feature} value",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step
    )

    all_values.append(value)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

try:
    with open("chatgpt_for_house.pkl", "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    st.error("Model file chatgpt_for_house.pkl was not found.")
    st.stop()


# --------------------------------------------------
# HOUSE PRICE PREDICTION
# --------------------------------------------------

input_data = pd.DataFrame(
    [all_values],
    columns=feature_columns
)

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)[0]

predicted_price = float(prediction) * 100000


# --------------------------------------------------
# MAIN PAGE DESIGN
# --------------------------------------------------

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# LEFT SIDE - HOUSE IMAGE

with left_col:

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg",
        use_container_width=True
    )

    st.caption("California Residential Housing")


# RIGHT SIDE - PROJECT DESCRIPTION

with right_col:

    st.subheader("📌 About This Project")

    st.write(
        "This application uses Machine Learning to predict "
        "estimated house prices in California based on "
        "various housing-related features."
    )

    st.write(
        "Users can enter details such as median income, "
        "house age, average rooms, average bedrooms, "
        "population, average occupancy, and latitude "
        "to get an estimated house price."
    )

    with st.expander("View Project Details"):

        st.markdown("""
        ### Key Features

        - Predicts estimated California house prices.
        - Interactive input sliders for housing features.
        - Machine Learning-based price prediction.
        - Simple and user-friendly interface.
        - Displays the predicted house value in USD.

        **Technologies Used:**

        Python, Pandas, NumPy, Scikit-learn,
        Streamlit, and Pickle.
        """)

    st.divider()

    st.subheader("💰 House Price Prediction")

    st.metric(
        label="Estimated House Price",
        value=f"${predicted_price:,.2f}"
    )

    st.success("House price prediction completed successfully!")
