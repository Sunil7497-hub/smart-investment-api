from fastapi import FastAPI, HTTPException, Depends
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import firebase_admin
from firebase_admin import credentials, auth

app = FastAPI()

# Load Firebase credentials
cred = credentials.Certificate("firebase_credentials.json")  # Replace with your actual file
firebase_admin.initialize_app(cred)

# Function to fetch stock data
def fetch_stock_data(stock, period="1y"):
    data = yf.download(stock, period=period)
    if data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {stock}")
    return data

# Function to preprocess data for LSTM
def prepare_lstm_data(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(x), np.array(y), scaler

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# API to predict stock prices using LSTM
@app.get("/predict/{stock}")
def predict_stock(stock: str):
    data = fetch_stock_data(stock)
    x, y, scaler = prepare_lstm_data(data)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model_file = f"models/{stock}_lstm.h5"
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = build_lstm_model(x.shape)
        model.fit(x, y, epochs=5, batch_size=16, verbose=0)
        model.save(model_file)

    last_sequence = x[-1].reshape(1, x.shape[1], 1)
    predicted_price_scaled = model.predict(last_sequence)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    return {"stock": stock, "predicted_price": predicted_price}

# API Health Check
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
