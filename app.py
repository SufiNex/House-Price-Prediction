import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pickle
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

st.title('House Price Prediction Using Machine Learning')

st.image('https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg')

st.write('''Our project, House Price Prediction Using Machine Learning, aims to predict house prices based on various factors such as location, area, number of bedrooms, bathrooms, and other property features. We use machine learning algorithms to analyze housing data, identify patterns, and generate accurate price predictions. This project helps users make better decisions when buying or selling properties.''')

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['MedHouseVal'] = data.target

X = df.iloc[:, :-2]
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

st.sidebar.title('Select House Features🏡')
st.sidebar.image('https://i.pinimg.com/originals/b1/cf/53/b1cf530756b23930aab499c1a11b443c.gif')
all_value = []
for i in X.columns:
    value = st.sidebar.slider(f'Select {i} value')
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






