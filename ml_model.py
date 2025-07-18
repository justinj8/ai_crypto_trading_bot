import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

LSTM_MODEL_FILE = "lstm_model.h5"
XGB_MODEL_FILE = "xgb_model.json"

def build_lstm_model(input_shape):
    """Builds an LSTM model for time-series prediction"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_models(df, feature_columns):
    """Train LSTM & XGBoost models."""
    X = df[feature_columns].values
    y = df["close"].shift(-1).dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:-1])

    # Train LSTM
    # Reshape for LSTM which expects 3D input: (samples, timesteps, features)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_model = build_lstm_model((1, X_scaled.shape[1]))
    lstm_model.fit(X_lstm, y, epochs=50, batch_size=32)
    lstm_model.save(LSTM_MODEL_FILE)

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100)
    xgb_model.fit(X_scaled, y)
    xgb_model.save_model(XGB_MODEL_FILE)

    joblib.dump(scaler, "scaler.pkl")

def predict_next_price(df, feature_columns):
    """Ensemble prediction from LSTM and XGBoost"""
    scaler = joblib.load("scaler.pkl")
    X = df[feature_columns].values[-1:]
    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    lstm_model = load_model(LSTM_MODEL_FILE)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_FILE)

    lstm_pred = lstm_model.predict(X_lstm)[0][0]
    xgb_pred = xgb_model.predict(X_scaled)[0]

    return (lstm_pred + xgb_pred) / 2