"""
Ensemble Model Manager
Combines predictions from multiple models with intelligent weighting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """
    Ensemble model combining LSTM, Transformer, and XGBoost
    Supports weighted averaging, voting, and stacking
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        ensemble_method: str = "weighted_average"
    ):
        """
        Initialize ensemble model

        Args:
            model_weights: Weights for each model (lstm, transformer, xgboost)
            ensemble_method: Method for combining predictions
                - "weighted_average": Weighted average of predictions
                - "median": Median of predictions
                - "best_of": Use prediction from best performing model
        """
        self.ensemble_method = ensemble_method

        # Default weights
        if model_weights is None:
            self.model_weights = {
                'lstm': 0.35,
                'transformer': 0.35,
                'xgboost': 0.30
            }
        else:
            self.model_weights = model_weights

        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}

        # Initialize models
        self.lstm_model = LSTMModel()
        self.transformer_model = TransformerModel()
        self.xgboost_model = XGBoostModel()

        self.models = {
            'lstm': self.lstm_model,
            'transformer': self.transformer_model,
            'xgboost': self.xgboost_model
        }

        self.model_performance = {}
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """
        Train all models in the ensemble

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            models_to_train: List of models to train (default: all)

        Returns:
            Training results for each model
        """
        if models_to_train is None:
            models_to_train = ['lstm', 'transformer', 'xgboost']

        logger.info(f"Training ensemble models: {models_to_train}")

        results = {}

        # Train LSTM
        if 'lstm' in models_to_train:
            logger.info("=" * 50)
            logger.info("Training LSTM Model")
            logger.info("=" * 50)

            # Prepare sequences for LSTM
            X_train_seq, y_train_seq = self.lstm_model.prepare_sequences(X_train, y_train)

            if X_val is not None:
                X_val_seq, y_val_seq = self.lstm_model.prepare_sequences(X_val, y_val)
            else:
                X_val_seq, y_val_seq = None, None

            results['lstm'] = self.lstm_model.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                epochs=100,
                batch_size=32,
                verbose=1
            )

        # Train Transformer
        if 'transformer' in models_to_train:
            logger.info("=" * 50)
            logger.info("Training Transformer Model")
            logger.info("=" * 50)

            # Prepare sequences for Transformer
            X_train_seq, y_train_seq = self.transformer_model.prepare_sequences(X_train, y_train)

            if X_val is not None:
                X_val_seq, y_val_seq = self.transformer_model.prepare_sequences(X_val, y_val)
            else:
                X_val_seq, y_val_seq = None, None

            results['transformer'] = self.transformer_model.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                epochs=50,
                batch_size=32,
                verbose=1
            )

        # Train XGBoost
        if 'xgboost' in models_to_train:
            logger.info("=" * 50)
            logger.info("Training XGBoost Model")
            logger.info("=" * 50)

            results['xgboost'] = self.xgboost_model.train(
                X_train, y_train,
                X_val, y_val,
                early_stopping_rounds=50,
                verbose=True
            )

        self.is_trained = True
        logger.info("Ensemble training complete")

        return results

    def predict(
        self,
        X: np.ndarray,
        return_individual: bool = False
    ) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Input features
            return_individual: Return individual model predictions

        Returns:
            Ensemble predictions (and individual predictions if requested)
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        predictions = {}

        # Get predictions from each model
        if self.lstm_model.is_trained:
            X_seq, _ = self.lstm_model.prepare_sequences(X)
            if len(X_seq) > 0:
                predictions['lstm'] = self.lstm_model.predict(X_seq)

        if self.transformer_model.is_trained:
            X_seq, _ = self.transformer_model.prepare_sequences(X)
            if len(X_seq) > 0:
                predictions['transformer'] = self.transformer_model.predict(X_seq)

        if self.xgboost_model.is_trained:
            predictions['xgboost'] = self.xgboost_model.predict(X)

        # Align predictions to same length (handle sequence models)
        min_length = min(len(p) for p in predictions.values())
        aligned_predictions = {k: v[-min_length:] for k, v in predictions.items()}

        # Combine predictions based on ensemble method
        if self.ensemble_method == "weighted_average":
            ensemble_pred = self._weighted_average(aligned_predictions)

        elif self.ensemble_method == "median":
            ensemble_pred = self._median_ensemble(aligned_predictions)

        elif self.ensemble_method == "best_of":
            ensemble_pred = self._best_model_ensemble(aligned_predictions)

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        if return_individual:
            return ensemble_pred, aligned_predictions
        else:
            return ensemble_pred

    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average ensemble"""
        ensemble = np.zeros(len(next(iter(predictions.values()))))

        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble += weight * pred

        return ensemble

    def _median_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Median ensemble"""
        pred_array = np.array(list(predictions.values()))
        return np.median(pred_array, axis=0)

    def _best_model_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Use prediction from best performing model"""
        if not self.model_performance:
            # Default to weighted average if no performance data
            return self._weighted_average(predictions)

        # Find best model
        best_model = min(self.model_performance.items(), key=lambda x: x[1]['rmse'])[0]

        return predictions[best_model]

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, dict]:
        """
        Evaluate all models and ensemble

        Args:
            X: Test features
            y: Test targets

        Returns:
            Evaluation metrics for each model and ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")

        results = {}

        # Evaluate individual models
        if self.lstm_model.is_trained:
            X_seq, y_seq = self.lstm_model.prepare_sequences(X, y)
            if len(X_seq) > 0:
                results['lstm'] = self.lstm_model.evaluate(X_seq, y_seq)
                self.model_performance['lstm'] = results['lstm']

        if self.transformer_model.is_trained:
            X_seq, y_seq = self.transformer_model.prepare_sequences(X, y)
            if len(X_seq) > 0:
                results['transformer'] = self.transformer_model.evaluate(X_seq, y_seq)
                self.model_performance['transformer'] = results['transformer']

        if self.xgboost_model.is_trained:
            results['xgboost'] = self.xgboost_model.evaluate(X, y)
            self.model_performance['xgboost'] = results['xgboost']

        # Evaluate ensemble
        ensemble_pred, individual_preds = self.predict(X, return_individual=True)

        # Align y to ensemble predictions
        min_length = len(ensemble_pred)
        y_aligned = y[-min_length:]

        # Calculate ensemble metrics
        mse = np.mean((y_aligned - ensemble_pred) ** 2)
        mae = np.mean(np.abs(y_aligned - ensemble_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_aligned - ensemble_pred) / (y_aligned + 1e-10))) * 100

        direction_actual = np.sign(y_aligned)
        direction_pred = np.sign(ensemble_pred)
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        results['ensemble'] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }

        # Log results
        logger.info("=" * 50)
        logger.info("Ensemble Evaluation Results")
        logger.info("=" * 50)

        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

        return results

    def get_prediction_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble prediction with confidence measure

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        ensemble_pred, individual_preds = self.predict(X, return_individual=True)

        # Calculate confidence as inverse of prediction variance
        pred_array = np.array(list(individual_preds.values()))
        pred_std = np.std(pred_array, axis=0)

        # Normalize confidence to 0-1 (lower std = higher confidence)
        confidence = 1 / (1 + pred_std)

        return ensemble_pred, confidence

    def save(self, model_dir: str = "data/models"):
        """Save all models"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save individual models
        if self.lstm_model.is_trained:
            self.lstm_model.save(model_dir)

        if self.transformer_model.is_trained:
            self.transformer_model.save(model_dir)

        if self.xgboost_model.is_trained:
            self.xgboost_model.save(model_dir)

        # Save ensemble config
        config = {
            'model_weights': self.model_weights,
            'ensemble_method': self.ensemble_method,
            'model_performance': self.model_performance
        }
        joblib.dump(config, model_path / "ensemble_config.pkl")

        logger.info(f"Ensemble models saved to {model_path}")

    def load(self, model_dir: str = "data/models"):
        """Load all models"""
        model_path = Path(model_dir)

        # Load ensemble config
        config = joblib.load(model_path / "ensemble_config.pkl")
        self.model_weights = config['model_weights']
        self.ensemble_method = config['ensemble_method']
        self.model_performance = config['model_performance']

        # Load individual models if they exist
        try:
            self.lstm_model.load(model_dir)
            logger.info("LSTM model loaded")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")

        try:
            self.transformer_model.load(model_dir)
            logger.info("Transformer model loaded")
        except Exception as e:
            logger.warning(f"Could not load Transformer model: {e}")

        try:
            self.xgboost_model.load(model_dir)
            logger.info("XGBoost model loaded")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}")

        self.is_trained = True
        logger.info("Ensemble models loaded successfully")
