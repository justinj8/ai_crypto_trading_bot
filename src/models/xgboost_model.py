"""
XGBoost Model for Price Prediction
Gradient boosting model with advanced features
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    """
    XGBoost model for crypto price prediction
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        min_child_weight: int = 1,
        reg_alpha: float = 0,
        reg_lambda: float = 1
    ):
        """
        Initialize XGBoost model

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            min_child_weight: Minimum sum of instance weight
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None

    def build_model(self):
        """Build XGBoost model"""
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )

        logger.info("XGBoost model initialized")
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Early stopping rounds
            verbose: Verbosity

        Returns:
            Training results
        """
        if self.model is None:
            self.build_model()

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        else:
            eval_set = [(X_train_scaled, y_train)]

        # Train model
        logger.info("Training XGBoost model...")

        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )

        self.is_trained = True

        # Get feature importance
        self.feature_importance = self.model.feature_importances_

        logger.info("XGBoost training complete")

        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale input
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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

        # Make predictions
        predictions = self.predict(X)

        # Calculate metrics
        mse = np.mean((y - predictions) ** 2)
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100

        # Direction accuracy
        direction_actual = np.sign(y)
        direction_pred = np.sign(predictions)
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

        logger.info(f"XGBoost Evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}, Direction Accuracy: {direction_accuracy:.2f}%")

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Targets
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        if self.model is None:
            self.build_model()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validate
        scores = cross_val_score(
            self.model,
            X_scaled,
            y,
            cv=cv,
            scoring='neg_mean_squared_error'
        )

        rmse_scores = np.sqrt(-scores)

        results = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'cv_scores': rmse_scores
        }

        logger.info(f"XGBoost Cross-Validation - Mean RMSE: {results['mean_rmse']:.6f} (+/- {results['std_rmse']:.6f})")

        return results

    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df

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

        # Save XGBoost model
        self.model.save_model(str(model_path / "xgboost_model.json"))

        # Save scaler
        joblib.dump(self.scaler, model_path / "xgboost_scaler.pkl")

        # Save config
        config = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda
        }
        joblib.dump(config, model_path / "xgboost_config.pkl")

        logger.info(f"XGBoost model saved to {model_path}")

    def load(self, model_dir: str = "data/models"):
        """
        Load model and scaler

        Args:
            model_dir: Directory containing model
        """
        model_path = Path(model_dir)

        # Load config
        config = joblib.load(model_path / "xgboost_config.pkl")

        # Reinitialize with saved config
        self.__init__(**config)

        # Build model
        self.build_model()

        # Load trained model
        self.model.load_model(str(model_path / "xgboost_model.json"))

        # Load scaler
        self.scaler = joblib.load(model_path / "xgboost_scaler.pkl")

        # Get feature importance
        self.feature_importance = self.model.feature_importances_

        self.is_trained = True
        logger.info(f"XGBoost model loaded from {model_path}")
