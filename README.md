# Stock Price Prediction Using LSTM Neural Networks

This project predicts stock prices using LSTM neural networks. It is implemented using TensorFlow, Keras, and Streamlit.

## Run the project

streamlit run app.py

# Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Load and Preprocess the Stock Data

stock = 'GOOG'  # Google stock
start = '2012-01-01'
end = '2022-12-21'
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Plot Moving Averages

ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='100-day MA')
plt.plot(ma_200_days, 'b', label='200-day MA')
plt.plot(data['Close'], 'g', label='Stock Price')
plt.legend()
plt.show()

# Prepare Training and Testing Data
Split the dataset: 80% for training, 20% for testing.
Normalize the data using MinMaxScaler (scales values between 0 and 1)

data.dropna(inplace=True)
train_size = int(len(data) * 0.80)
data_train = data['Close'][:train_size]
data_test = data['Close'][train_size:]

scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(np.array(data_train).reshape(-1, 1))

# Create Training Data for LSTM Model
The model uses 100 previous stock prices to predict the next one.
Convert data into feature (X) and target (Y) variables.

x_train, y_train = [], []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i])  
    y_train.append(data_train_scaled[i, 0])  

x_train, y_train = np.array(x_train), np.array(y_train)

#  Build the LSTM Model
The model consists of:
5 LSTM layers with different units (50, 60, 80, 80, 120).
Dropout layers to prevent overfitting.
Dense output layer to predict stock price.

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(units=120, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Prepare Testing Data
Take the last 100 days from training data and combine it with test data.
Transform it using the same MinMaxScaler.

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(np.array(data_test).reshape(-1, 1))

x_test, y_test = [], []
for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#  Predict Stock Prices
Use the trained model to predict stock prices.
Reverse the scaling to get actual prices.

y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#  Visualize the Results
Plot actual vs. predicted prices.

plt.figure(figsize=(10, 8))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Save the Model 
model.save('Stock_Predictions_Model.keras')


#  Conclusion
This project shows how LSTM (a type of neural network) can be used to predict stock prices. The model studies past stock prices to find patterns and then tries to guess future prices based on those patterns.




