import datetime
import streamlit as st
from datetime import date 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt 
from prophet import Prophet
from prophet.plot import plot_plotly 
from plotly import graph_objs as go 
from plotly.subplots import make_subplots
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
import neptune
import ta

# Initialize Neptune run
run = neptune.init_run(
    project="yuki-pikazo/Jotaro",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTg0NWVmOC03YTdiLTRhMmMtYmQ5Zi04OGQxMzJjNGJkYWMifQ=="
)

# Define date range for data
start = "1980-09-01"
end = date.today().strftime("%Y-%m-%d")
st.title("Stock Market Prediction")

# Select stock
stocks = ("TSLA", "NVDA", 'MSFT', "GME", "AMD", "MSRT", "META", "GOOG", "JPM", "AAPL", "MA", "CAT", "YMM", "LYV", "COCO", "NFLX", "AMZN", "ETSY", "YELP", "3LMI", "SOUNW", "SOUN","FFIE")
selected_stocks = st.selectbox("Select The Stocks for prediction", stocks)

# Select prediction period
n_years = st.slider("Years of Prediction:", 1 , 4)
period = n_years * 365 

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text("Load Data....")
data = load_data(selected_stocks)
data_load_state.text("Loading data ..... done")

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Preprocess data for Prophet
df_train = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
df_train = df_train.rename(columns={"Date": "ds", "Open": "open", "High": "high", "Low": "low", "Close": "y", "Adj Close": "adj_close", "Volume": "volume"})

# Fill NaN values
df_train['high'] = df_train['high'].interpolate(method='linear')
df_train['low'] = df_train['low'].interpolate(method='linear')
df_train['volume'] = df_train['volume'].interpolate(method='linear')
df_train['open'] = df_train['open'].interpolate(method='linear')
df_train['adj_close'] = df_train['adj_close'].interpolate(method='linear')
df_train.fillna(0.0, inplace=True)

@st.cache_data
def tune_model(df_train, param_grid):
    best_rmse = None
    best_params = None
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    for params in all_params:
        m = Prophet(**params)
        m.add_regressor('high')
        m.add_regressor('low')
        m.add_regressor('volume')
        m.add_regressor('open')
        m.add_regressor('adj_close')
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        future = future.merge(df_train, on='ds', how='left')
        future['high'] = future['high'].interpolate(method='linear')
        future['low'] = future['low'].interpolate(method='linear')
        future['volume'] = future['volume'].interpolate(method='linear')
        future['open'] = future['open'].interpolate(method='linear')
        future['adj_close'] = future['adj_close'].interpolate(method='linear')
        future.fillna(0.0, inplace=True)
        forecast = m.predict(future)
        rmse = np.sqrt(np.mean((forecast['yhat'] - df_train['y'])**2))
        
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params

# Define hyperparameter grid
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1.0],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 20.0],
    'growth': ['linear'],
    'changepoint_range': [0.8, 0.9]
}

# Tune model and get best parameters
best_params = tune_model(df_train, param_grid)
m = Prophet(**best_params)
m.add_regressor('high')
m.add_regressor('low')
m.add_regressor('volume')
m.add_regressor('open')
m.add_regressor('adj_close')
m.add_seasonality(name='daily', period=1.5, fourier_order=7)
m.add_seasonality(name='weekly', period=7.5, fourier_order=5)
m.add_seasonality(name='monthly', period=30.5, fourier_order=15)
m.add_seasonality(name='yearly', period=365.5, fourier_order=12)
m.fit(df_train)

# Forecasting
future = m.make_future_dataframe(periods=period)
future = future.merge(df_train, on='ds', how='left')
future['high'] = future['high'].interpolate(method='linear')
future['low'] = future['low'].interpolate(method='linear')
future['volume'] = future['volume'].interpolate(method='linear')
future['open'] = future['open'].interpolate(method='linear')
future['adj_close'] = future['adj_close'].interpolate(method='linear')
future.fillna(0.0, inplace=True)
forecast = m.predict(future)

# Anomaly detection
df_train['fact'] = df_train['y'].copy()
df_train.loc[(df_train['ds'] > '2023-01-01'), 'fact'] = None
forecast = pd.merge(forecast, df_train, on='ds', how='outer')
forecast['residuals'] = forecast['fact'] - forecast['yhat']
forecast['anomaly'] = forecast.apply(lambda x: 'Yes' if (x['residuals'] > 0.5) or (x['residuals'] < -0.5) else 'No', axis=1)
st.write(forecast[forecast['anomaly'] == 'Yes'])

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

run.stop()
