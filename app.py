import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pickle
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

st.title('California House Price Prediction')

st.image('https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg')

st.write('''This application uses Machine Learning to predict estimated house prices in California based on various housing-related features.

Users can enter details such as median income, house age, average rooms, average bedrooms, population, average occupancy, and latitude to get an estimated house price.

The application is built using Python and Scikit-learn, with a Linear Regression model for prediction and Streamlit for the interactive user interface.

### Key Features

- Predicts estimated California house prices.
- Interactive input sliders for housing features.
- Machine Learning-based price prediction.
- Simple and user-friendly interface.
- Displays the predicted house value in USD.

**Technologies Used:** Python, Pandas, NumPy, Scikit-learn, Streamlit, and Pickle.''')

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['MedHouseVal'] = data.target

X = df.iloc[:, :-2]
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

st.sidebar.title('Select House Features🏡')
st.sidebar.image('https://i.pinimg.com/originals/b1/cf/53/b1cf530756b23930aab499c1a11b443c.gif')
all_value = []

feature_config = {
    "MedInc": (0.49, 15.01, 4.0, 0.1),
    "HouseAge": (1, 52, 30, 1),
    "AveRooms": (0.84, 141.91, 5.5, 0.1),
    "AveBedrms": (0.33, 34.07, 1.1, 0.1),
    "Population": (3, 35682, 1200, 1),
    "AveOccup": (0.69, 1243.34, 3.0, 0.1),
    "Latitude": (32.54, 41.95, 37.5, 0.1)
}

for i in X.columns:
    min_val, max_val, default, step = feature_config[i]

    value = st.sidebar.slider(
        f'Select {i} value',
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step
    )

    all_value.append(value)


final_data = [all_value]
final_data = scaler.transform(final_data)

with open('chatgpt_for_house.pkl', 'rb') as f:
    chatgpt = pickle.load(f)
print('Model Loaded Successfully!!!')

price = chatgpt.predict(final_data)[0]

if price:
    with st.spinner('Wait for price prediction....'):
        time.sleep(5)
    st.success(f'Predicted price is: $ {np.abs(round(price*100000,2))}')
