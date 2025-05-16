import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# === Assume LSTM model and XGBoost model already trained and loaded ===
# model: trained LSTM
# lstm_feature_model: Model to extract LSTM embeddings
# xgb: trained XGBoost model
# scaler: trained feature scaler

# === Rolling buffer of the last 50 minutes ===
rolling_buffer = []

def get_latest_price():
    df = yf.download("SOXL", period="1d", interval="1m")
    return df.tail(50)  # keep last 50 rows

# === Feature engineering function ===
def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['RollingMean'] = df['Close'].rolling(window=10).mean()
    df['RollingStd'] = df['Close'].rolling(window=10).std()
    df['VolumeChange'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    return df

# === Main live loop ===
while True:
    try:
        raw_data = get_latest_price()
        df = create_features(raw_data.copy())

        # Most recent 10-minute window
        seq = df[['Close', 'Volume', 'Return', 'Volatility', 'RollingMean', 'RollingStd', 'VolumeChange']].values[-10:]

        # Scale features
        seq_scaled = scaler.transform(seq)
        seq_input = np.expand_dims(seq_scaled, axis=0)  # shape: (1, 10, features)

        # Get LSTM embedding
        lstm_vector = lstm_feature_model.predict(seq_input)

        # Get current (last) feature row
        current_feats = seq_scaled[-1].reshape(1, -1)

        # Combine
        combined = np.hstack((current_feats, lstm_vector))

        # Predict with XGBoost
        pred = xgb.predict(combined)[0]
        confidence = xgb.predict_proba(combined)[0][pred]

        # Log prediction
        direction = "UP" if pred == 1 else "DOWN"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Prediction: {direction} | Confidence: {confidence:.2f}")

        # Optional: Execute trade here (placeholder)
        # if pred == 1 and confidence > 0.7:
        #     execute_buy()
        # elif pred == 0 and confidence > 0.7:
        #     execute_sell()

        time.sleep(60)  # wait 1 minute before next loop

    except Exception as e:
        print("Error:", e)
        time.sleep(60)
