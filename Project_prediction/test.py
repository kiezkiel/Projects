import streamlit as st
from datetime import date 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 
from plotly.subplots import make_subplots
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools
import numpy as np
import plotly.tools as tls
import neptune
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

start = "2012-01-01"
end = date.today().strftime("%Y-%m-%d")
st.title("Stock Market Prediction")

stocks = ("AAPL","NVDA", 'MSFT', "GME", "AMD","MSTR")
selected_stocks = st.selectbox("Select The Stocks for prediction", stocks)

n_years = st.slider("Years of Prediction:", 1 , 4)
num_future = 365 * n_years  # e.g., predict for n_years

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data....")
data = load_data(selected_stocks)
data_load_state.text("Loading data ..... done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Close'], name = 'stock_close'))
    fig.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# Define the parameter grid
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df_train['y'].values.reshape(-1, 1))
lookback = 10 
X, Y = [], []
for i in range(len(data)-lookback-1):
    X.append(data[i:(i+lookback), 0])
    Y.append(data[(i+lookback), 0])
X, Y = np.array(X), np.array(Y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)  # Unnormalize the predicted price

# Number of future predictions

# Last `lookback` days from the training set
last_sequence = np.copy(X_train[-1])

# Initialize array for predictions
predictions = []

# Loop for number of future predictions
for _ in range(num_future):
    # Reshape last sequence
    last_sequence_reshaped = last_sequence.reshape((1, lookback, 1))
    
    # Predict next step (next day)
    next_step = model.predict(last_sequence_reshaped)
    
    # Append prediction to predictions array
    predictions.append(next_step[0])
    
    # Update last sequence (remove first value and append the prediction)
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_step

# Unnormalize the predicted price
predicted_price_future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create a date range for the future predictions
future_dates = pd.date_range(start=df_train.ds.iloc[-1], periods=num_future)
# Convert predictions to DataFrame
predicted_data_future = pd.DataFrame({
    'ds': future_dates,
    'Predicted Stock Price': predicted_price_future.flatten()
})

# Plot the predictions
st.subheader('Forecast data')
predicted_data = pd.DataFrame({
    'Actual Stock Price': y_test.flatten(),
    'Predicted Stock Price': predicted_price.flatten()
}, index=df_train.ds.tail(len(y_test)))
st.line_chart(predicted_data)

st.subheader('Future Forecast data')
st.line_chart(predicted_data_future.set_index('ds'))