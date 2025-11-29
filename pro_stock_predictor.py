import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ------------------------------------------------------------
# Helper: Download data
# ------------------------------------------------------------
def load_stock(ticker, years=10):
    df = yf.download(ticker, period=f"{years}y")
    df = df.dropna()
    return df


# ------------------------------------------------------------
# Helper: Add technical indicators
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import yfinance as yf
import ta

def load_stock(ticker, years=10):
    # Download historical data
    df = yf.download(ticker, period=f"{years}y")
    
    # Ensure 'Close' is 1D float array
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns multi-indexed columns
        if 'Close' in df.columns.get_level_values(0):
            df['Close'] = df['Close'].iloc[:, 0]
        else:
            # fallback: usually Close is 4th column
            df['Close'] = df.iloc[:, 3]
    
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]

    # Convert to numeric
    df['Close'] = pd.to_numeric(np.array(df['Close']), errors='coerce')
    
    # Fill missing values
    df['Close'] = df['Close'].fillna(method='ffill')
    
    df = df.dropna()
    return df

def add_indicators(df):
    df = df.copy()
    close_prices = df['Close']
    
    # RSI
    rsi_indicator = ta.momentum.RSIIndicator(close=close_prices, window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(close=close_prices)
    df['macd'] = macd_indicator.macd_diff()
    
    # Volatility & returns
    df['volatility'] = close_prices.pct_change().rolling(20).std()
    df['return'] = close_prices.pct_change()
    
    df = df.dropna()
    return df





# ------------------------------------------------------------
# Helper: Scale dataset
# ------------------------------------------------------------
def scale(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


# ------------------------------------------------------------
# Helper: Create supervised sequence data for LSTM
# ------------------------------------------------------------
def create_supervised(data, lookback=60, horizon=7):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon, 0])
    return np.array(X), np.array(y)


# ------------------------------------------------------------
# Build LSTM Model
# ------------------------------------------------------------
def build_lstm(input_shape, horizon=7):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        LSTM(64),
        Dropout(0.25),
        Dense(horizon)
    ])

    model.compile(loss="mse", optimizer="adam")
    return model


# ------------------------------------------------------------
# Training pipeline (everything combined)
# ------------------------------------------------------------
def train_lstm_model(ticker):
    # 1. Load data
    df = load_stock(ticker)
    df = add_indicators(df)

    # 2. Use features
    features = ["Close", "rsi", "macd", "volatility", "return"]
    data = df[features].values

    # 3. Scale
    scaled, scaler = scale(data)

    # 4. Create sequences
    X, y = create_supervised(scaled, lookback=60, horizon=7)

    # 5. Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 6. Build model
    model = build_lstm((X.shape[1], X.shape[2]), horizon=7)

    # 7. Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # 8. Predict
    predictions = model.predict(X_test)

    # -------------------------------------------------------
    # Inverse transform ONLY the Close column
    # -------------------------------------------------------
    def inverse_transform(pred):
        dummy = np.zeros((7, data.shape[1]))
        dummy[:, 0] = pred
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]

    preds_inv = [inverse_transform(p) for p in predictions]
    actual_inv = [inverse_transform(a) for a in y_test]

    return df, preds_inv, actual_inv


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.set_page_config(page_title="Pro Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Professional Stock Predictor (LSTM + Indicators)")
st.write("High-accuracy deep learning forecast using RSI, MACD, volatility, and more.")

ticker = st.text_input("Enter a stock ticker:", "AAPL")

if st.button("Train & Predict"):
    with st.spinner("Training advanced LSTM model… this takes 20–40 seconds"):
        df, preds, actual = train_lstm_model(ticker)

    st.success("Model training complete!")

    st.subheader("📅 7-Day Forecast vs Actual (last window)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual[-1], label="Actual", marker="o")
    ax.plot(preds[-1], label="Predicted", marker="o")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Forecast Values (next 7 days)")
    st.write(preds[-1])

    st.subheader("📉 Last 200 Days Close Price")
    st.line_chart(df["Close"].tail(200))

