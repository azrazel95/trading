import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score

# Load your preprocessed SOXL data here
# This is from the previous data prep step
# Make sure `df` includes features + 'Target'
# ... (Insert your feature engineering from earlier)

# Define features and target
feature_cols = ['Close', 'Volume', 'Return', 'Volatility', 'RollingMean', 'RollingStd', 'VolumeChange']
X = df[feature_cols]
y = df['Target']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

# Reshape for LSTM (samples, timesteps, features)
timesteps = 10
X_lstm, y_lstm = [], []
for i in range(timesteps, len(X_train)):
    X_lstm.append(X_train[i-timesteps:i])
    y_lstm.append(y_train.values[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Build LSTM model
model = Sequential([
    LSTM(32, input_shape=(timesteps, X_lstm.shape[2])),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=1)

# Prepare test data
X_test_lstm, y_test_lstm = [], []
for i in range(timesteps, len(X_test)):
    X_test_lstm.append(X_test[i-timesteps:i])
    y_test_lstm.append(y_test.values[i])
X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)

# Predict and evaluate
y_pred = (model.predict(X_test_lstm) > 0.5).astype(int)
print("LSTM Accuracy:", accuracy_score(y_test_lstm, y_pred))
