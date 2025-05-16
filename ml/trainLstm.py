# === Step 2: Build & Train LSTM ===
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

import joblib
data = joblib.load('data_splits.pkl')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 1. Create rolling sequences for LSTM
timesteps = 10

def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i-timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_lstm_train, y_lstm_train = make_sequences(X_train, y_train.to_numpy(), timesteps)
X_lstm_test, y_lstm_test   = make_sequences(X_test,  y_test.to_numpy(),  timesteps)

print("LSTM train shape:", X_lstm_train.shape)  # (samples, 10, 7)
print("LSTM test  shape:", X_lstm_test.shape)   # (samples, 10, 7)

# 2. Build the model (Functional API)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense

inp = Input(shape=(timesteps, X_lstm_train.shape[2]), name="lstm_input")
lstm_out = LSTM(32, name="lstm_layer")(inp)
out = Dense(1, activation="sigmoid", name="output")(lstm_out)

model = Model(inputs=inp, outputs=out, name="lstm_model")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# 3. Train with early stopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_lstm_train, y_lstm_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


import numpy as np
from tensorflow.keras.models import Model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model
lstm_feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer("lstm_layer").output
)

# 3.2 Get LSTM embeddings for train & test
X_lstm_train_feats = lstm_feature_model.predict(X_lstm_train)
X_lstm_test_feats  = lstm_feature_model.predict(X_lstm_test)

# 3.3 Get the “current” engineered features (last timestep of each window)
X_current_train = X_train[timesteps:]
X_current_test  = X_test[timesteps:]

# 3.4 Concatenate [current_features | LSTM_features]
X_combined_train = np.hstack([X_current_train, X_lstm_train_feats])
X_combined_test  = np.hstack([X_current_test,  X_lstm_test_feats ])

# 3.5 Train XGBoost on the fused features
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_combined_train, y_lstm_train)

# 3.6 Evaluate
y_hybrid_pred = xgb.predict(X_combined_test)
hybrid_acc   = accuracy_score(y_lstm_test, y_hybrid_pred)
print(f"\n✅ Hybrid LSTM+XGBoost accuracy: {hybrid_acc:.4f}")

# 4. Evaluate on test set
y_pred_prob = model.predict(X_lstm_test)
y_pred      = (y_pred_prob > 0.5).astype(int)
acc = accuracy_score(y_lstm_test, y_pred)

print(f"\n✅ LSTM test accuracy: {acc:.4f}")
