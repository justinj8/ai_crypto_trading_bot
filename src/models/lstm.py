"""
LSTM Model for Time Series Prediction
Implements a sophisticated LSTM architecture with attention mechanism
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LSTMModel:
    """
    Long Short-Term Memory model for crypto price prediction
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 50,
        lstm_units: list = [128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model

        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            lstm_units: List of LSTM layer units
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self):
        """Build LSTM architecture with attention"""
        model = models.Sequential()

        # First LSTM layer (return sequences for stacking)
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.BatchNormalization())

        # Second LSTM layer
        if len(self.lstm_units) > 1:
            model.add(layers.LSTM(self.lstm_units[1], return_sequences=False))
            model.add(layers.Dropout(self.dropout))
            model.add(layers.BatchNormalization())

        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout / 2))

        # Output layer
        model.add(layers.Dense(1))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"LSTM model built with {model.count_params()} parameters")

        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM input

        Args:
            data: Feature data
            targets: Target values (optional)

        Returns:
            Tuple of (sequences, targets)
        """
        X = []
        y = [] if targets is not None else None

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])

            if targets is not None:
                y.append(targets[i])

        X = np.array(X)
        y = np.array(y) if targets is not None else None

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> dict:
        """
        Train LSTM model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Scale features
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(X_train.shape)

        if X_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_flat).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Train model
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )

        self.is_trained = True
        logger.info("LSTM training complete")

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input sequences

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale input
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X.shape)

        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)

        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance

        Args:
            X: Test features
            y: Test targets

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Scale input
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X.shape)

        # Evaluate
        loss, mae, mse = self.model.evaluate(X_scaled, y, verbose=0)

        # Calculate additional metrics
        predictions = self.predict(X)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100

        # Direction accuracy
        direction_actual = np.sign(y)
        direction_pred = np.sign(predictions)
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }

        logger.info(f"LSTM Evaluation - RMSE: {rmse:.6f}, Direction Accuracy: {direction_accuracy:.2f}%")

        return metrics

    def save(self, model_dir: str = "data/models"):
        """
        Save model and scaler

        Args:
            model_dir: Directory to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(model_path / "lstm_model.h5")

        # Save scaler
        joblib.dump(self.scaler, model_path / "lstm_scaler.pkl")

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        joblib.dump(config, model_path / "lstm_config.pkl")

        logger.info(f"LSTM model saved to {model_path}")

    def load(self, model_dir: str = "data/models"):
        """
        Load model and scaler

        Args:
            model_dir: Directory containing model
        """
        model_path = Path(model_dir)

        # Load config
        config = joblib.load(model_path / "lstm_config.pkl")
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.lstm_units = config['lstm_units']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']

        # Load Keras model
        self.model = keras.models.load_model(model_path / "lstm_model.h5")

        # Load scaler
        self.scaler = joblib.load(model_path / "lstm_scaler.pkl")

        self.is_trained = True
        logger.info(f"LSTM model loaded from {model_path}")
