"""
ML-Based Trading Strategy
Uses ensemble ML models to generate trading signals
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from src.strategies.base import BaseStrategy
from src.core.events import SignalEvent, SignalType
from src.models.ensemble import EnsembleModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLEnsembleStrategy(BaseStrategy):
    """
    Machine Learning Ensemble Strategy
    Combines predictions from multiple ML models to generate trading signals
    """

    def __init__(
        self,
        name: str = "ML_Ensemble",
        symbols: List[str] = ["BTC/USDT"],
        config: dict = None
    ):
        """
        Initialize ML Ensemble Strategy

        Args:
            name: Strategy name
            symbols: Trading symbols
            config: Strategy configuration
        """
        default_config = {
            'prediction_threshold': 0.002,  # 0.2% price change threshold
            'confidence_threshold': 0.65,
            'lookback_periods': 5,
            'use_trend_filter': True,
            'min_trend_strength': 0.01,
            'stop_loss_pct': 0.03,  # 3%
            'take_profit_pct': 0.06,  # 6%
            'model_weights': {
                'lstm': 0.30,
                'transformer': 0.35,
                'xgboost': 0.35
            }
        }

        if config:
            default_config.update(config)

        super().__init__(name, symbols, default_config)

        # Initialize ensemble model
        self.ensemble_model = EnsembleModel(
            model_weights=self.config['model_weights']
        )

        self.last_predictions = {}

    def load_models(self, model_dir: str = "data/models"):
        """Load pre-trained models"""
        try:
            self.ensemble_model.load(model_dir)
            logger.info(f"ML models loaded for {self.name}")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")

    def generate_signals(self, market_data: pd.DataFrame) -> List[SignalEvent]:
        """
        Generate trading signals using ML predictions

        Args:
            market_data: DataFrame with OHLCV and features

        Returns:
            List of trading signals
        """
        signals = []

        if len(market_data) < 100:
            logger.warning("Insufficient data for ML strategy")
            return signals

        if not self.ensemble_model.is_trained:
            logger.warning("ML models not trained")
            return signals

        for symbol in self.symbols:
            # Filter data for symbol
            symbol_data = market_data[market_data.get('symbol', symbol) == symbol].copy()

            if len(symbol_data) < 60:
                continue

            # Prepare features
            feature_cols = [col for col in symbol_data.columns
                          if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            if not feature_cols:
                logger.warning("No features available for prediction")
                continue

            X = symbol_data[feature_cols].values

            try:
                # Get ensemble prediction with confidence
                predictions, confidence = self.ensemble_model.get_prediction_confidence(X)

                if len(predictions) == 0:
                    continue

                # Use most recent prediction
                predicted_return = predictions[-1]
                pred_confidence = confidence[-1]

                # Store prediction
                self.last_predictions[symbol] = {
                    'predicted_return': predicted_return,
                    'confidence': pred_confidence,
                    'timestamp': datetime.utcnow()
                }

                # Calculate current price and target
                current_price = symbol_data['close'].iloc[-1]
                predicted_price = current_price * (1 + predicted_return)

                # Generate signal based on prediction
                signal = self._create_signal_from_prediction(
                    symbol=symbol,
                    predicted_return=predicted_return,
                    confidence=pred_confidence,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    market_data=symbol_data
                )

                if signal and self.validate_signal(signal, symbol_data):
                    signals.append(signal)
                    logger.info(f"ML Signal: {signal.signal_type.value} {symbol} "
                              f"(confidence: {pred_confidence:.2f}, predicted return: {predicted_return:.4f})")

            except Exception as e:
                logger.error(f"Error generating ML signal for {symbol}: {e}")

        return signals

    def _create_signal_from_prediction(
        self,
        symbol: str,
        predicted_return: float,
        confidence: float,
        current_price: float,
        predicted_price: float,
        market_data: pd.DataFrame
    ) -> Optional[SignalEvent]:
        """Create signal from ML prediction"""

        # Check confidence threshold
        if confidence < self.config['confidence_threshold']:
            return None

        # Check prediction threshold
        threshold = self.config['prediction_threshold']

        # Trend filter
        if self.config['use_trend_filter']:
            if not self._check_trend_alignment(market_data, predicted_return):
                return None

        # Generate signal
        if predicted_return > threshold:
            signal_type = SignalType.LONG
            stop_loss = current_price * (1 - self.config['stop_loss_pct'])
            take_profit = current_price * (1 + self.config['take_profit_pct'])

        elif predicted_return < -threshold:
            signal_type = SignalType.SHORT
            stop_loss = current_price * (1 + self.config['stop_loss_pct'])
            take_profit = current_price * (1 - self.config['take_profit_pct'])

        else:
            return None

        # Create signal
        signal = SignalEvent(
            strategy_name=self.name,
            symbol=symbol,
            signal_type=signal_type,
            strength=abs(predicted_return),
            target_price=predicted_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=f"ML Ensemble predicts {predicted_return:.4f} return with {confidence:.2f} confidence"
        )

        return signal

    def _check_trend_alignment(self, market_data: pd.DataFrame, predicted_return: float) -> bool:
        """Check if prediction aligns with current trend"""
        if len(market_data) < 50:
            return True

        # Check if we have moving averages
        if 'ema_12' not in market_data.columns or 'ema_26' not in market_data.columns:
            return True

        # Get recent trend
        ema_12 = market_data['ema_12'].iloc[-1]
        ema_26 = market_data['ema_26'].iloc[-1]

        current_trend = ema_12 - ema_26
        trend_strength = abs(current_trend / market_data['close'].iloc[-1])

        # Check minimum trend strength
        if trend_strength < self.config['min_trend_strength']:
            return False

        # Check alignment
        if predicted_return > 0 and current_trend < 0:
            return False  # Predicted bullish but trend is bearish

        if predicted_return < 0 and current_trend > 0:
            return False  # Predicted bearish but trend is bullish

        return True

    def calculate_confidence(self, market_data: pd.DataFrame) -> float:
        """Calculate overall strategy confidence"""
        confidences = []

        for symbol in self.symbols:
            if symbol in self.last_predictions:
                confidences.append(self.last_predictions[symbol]['confidence'])

        if confidences:
            return np.mean(confidences)
        else:
            return 0.0

    def get_prediction_summary(self) -> dict:
        """Get summary of recent predictions"""
        return {
            symbol: {
                'predicted_return': pred['predicted_return'],
                'confidence': pred['confidence'],
                'age_seconds': (datetime.utcnow() - pred['timestamp']).total_seconds()
            }
            for symbol, pred in self.last_predictions.items()
        }
