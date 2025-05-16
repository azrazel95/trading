import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Download SOXL data
df = yf.download("SOXL", period="7d", interval="1m")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Step 2: Feature Engineering
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(window=10).std()
df['RollingMean'] = df['Close'].rolling(window=10).mean()
df['RollingStd'] = df['Close'].rolling(window=10).std()
df['VolumeChange'] = df['Volume'].pct_change()
df.dropna(inplace=True)

#  Step 3: Create labels — will price rise in the next 5 minutes?
if len(df) > 5:  # Ensure there are enough rows for the shift operation
    print("DataFrame shape after feature engineering:", df.shape)
    df[('FutureClose', '')] = df[('Close', 'SOXL')].shift(-5)
    print("FutureClose column created. DataFrame shape:", df.shape)
    print("Columns in DataFrame:", df.columns)

    if ('FutureClose', '') in df.columns:  # Ensure the column exists
        df.dropna(subset=[('FutureClose', '')], inplace=True)  # Ensure FutureClose has no NaN values
        df[('Target', '')] = (df[('FutureClose', '')] > df[('Close', 'SOXL')]).astype(int)
    else:
        raise ValueError("The 'FutureClose' column was not created. Check your data.")
else:
    raise ValueError("Not enough data to create 'FutureClose'. Ensure the DataFrame has more than 5 rows.")

# Step 4: Prepare features
feature_cols = ['Close', 'Volume', 'Return', 'Volatility', 'RollingMean', 'RollingStd', 'VolumeChange']
X = df[feature_cols]
y = df['Target']

# Step 5: Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
import joblib
joblib.dump({
  'X_train': X_train, 'X_test': X_test,
  'y_train': y_train, 'y_test': y_test
}, 'data_splits.pkl')

# Optional: Save scaler if you'll use it in live trading
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("✅ Data ready. X_train shape:", X_train.shape)






