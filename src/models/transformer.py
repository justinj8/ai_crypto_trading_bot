"""
Transformer Model for Time Series Prediction
Implements a Transformer architecture with multi-head attention
"""
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        Initialize Transformer block

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """Forward pass"""
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """Positional encoding layer"""

    def __init__(self, sequence_length, d_model):
        """
        Initialize positional encoding

        Args:
            sequence_length: Length of input sequence
            d_model: Model dimension
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, position, i, d_model):
        """Calculate angles for positional encoding"""
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, sequence_length, d_model):
        """Generate positional encoding"""
        angle_rads = self.get_angles(
            position=tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        """Add positional encoding to inputs"""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerModel:
    """
    Transformer model for crypto price prediction
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 50,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.0001
    ):
        """
        Initialize Transformer model

        Args:
            sequence_length: Number of time steps
            n_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def build_model(self):
        """Build Transformer architecture"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))

        # Project to d_model dimension
        x = layers.Dense(self.d_model)(inputs)

        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)

        # Stack transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout
            )(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        logger.info(f"Transformer model built with {model.count_params()} parameters")

        return model

    def prepare_sequences(
        self,
        data: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for Transformer input"""
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
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> dict:
        """Train Transformer model"""
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
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Train model
        logger.info("Training Transformer model...")
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
        logger.info("Transformer training complete")

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale input
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X.shape)

        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)

        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance"""
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

        logger.info(f"Transformer Evaluation - RMSE: {rmse:.6f}, Direction Accuracy: {direction_accuracy:.2f}%")

        return metrics

    def save(self, model_dir: str = "data/models"):
        """Save model and scaler"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(model_path / "transformer_model.h5")

        # Save scaler
        joblib.dump(self.scaler, model_path / "transformer_scaler.pkl")

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        joblib.dump(config, model_path / "transformer_config.pkl")

        logger.info(f"Transformer model saved to {model_path}")

    def load(self, model_dir: str = "data/models"):
        """Load model and scaler"""
        model_path = Path(model_dir)

        # Load config
        config = joblib.load(model_path / "transformer_config.pkl")
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.ff_dim = config['ff_dim']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']

        # Load Keras model (with custom objects)
        self.model = keras.models.load_model(
            model_path / "transformer_model.h5",
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
        )

        # Load scaler
        self.scaler = joblib.load(model_path / "transformer_scaler.pkl")

        self.is_trained = True
        logger.info(f"Transformer model loaded from {model_path}")
