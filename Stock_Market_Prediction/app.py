import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from tensorflow.keras.models import load_model

# Import the code from the converted Python script
import stock_predict

model = load_model('C:\\Users\\ranes\\Stock Predictions Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')

# Split data into training and testing sets
data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

st.write(data)

from sklearn.preprocessing import MinMaxScaler

# Assuming data_train is already defined somewhere above
scaler = MinMaxScaler(feature_range=(0, 1))
pass_100_days = data_train.tail(100)
data_test = pd.concat([pass_100_days, data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

#run the command
#streamlit run :filepath