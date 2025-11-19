"""
Model Training Script
Trains LSTM, Transformer, XGBoost, and Ensemble models
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.ensemble import EnsembleModel
from src.core.config import config
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging({'level': 'INFO', 'log_dir': 'logs'})
logger = get_logger('model_training')


def main():
    """Main training function"""
    logger.info("=" * 80)
    logger.info("AI CRYPTO TRADING BOT - MODEL TRAINING")
    logger.info("=" * 80)

    # Configuration
    symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
    timeframe = config.get('timeframes.primary', '1h')
    lookback_days = config.get('data.lookback_days', 365)

    logger.info(f"Training models for: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Lookback period: {lookback_days} days")

    # Step 1: Fetch data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Fetching market data")
    logger.info("=" * 80)

    fetcher = DataFetcher(exchange_name='binance')
    all_data = []

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        df = fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            days=lookback_days
        )

        if not df.empty:
            all_data.append(df)
            logger.info(f"Fetched {len(df)} candles for {symbol}")
        else:
            logger.warning(f"No data fetched for {symbol}")

    if not all_data:
        logger.error("No data available for training. Exiting.")
        return

    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total data points: {len(combined_df)}")

    # Step 2: Feature engineering
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Feature engineering (50+ indicators)")
    logger.info("=" * 80)

    processor = DataProcessor()
    processed_df = processor.process(combined_df, add_all_features=True)

    logger.info(f"Total features created: {len(processor.feature_names)}")
    logger.info(f"Features: {processor.feature_names[:10]}... (showing first 10)")

    # Create target variable (next period return)
    processed_df['target'] = processed_df.groupby('symbol')['close'].shift(-1) / processed_df['close'] - 1

    # Remove rows with NaN
    processed_df = processed_df.dropna()

    logger.info(f"Data after cleaning: {len(processed_df)} rows")

    # Step 3: Prepare training data
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Preparing training data")
    logger.info("=" * 80)

    # Select features
    feature_cols = processor.feature_names
    X = processed_df[feature_cols].values
    y = processed_df['target'].values

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False  # Don't shuffle time series data
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Features: {X_train.shape[1]}")

    # Step 4: Train ensemble models
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training ensemble models")
    logger.info("=" * 80)

    # Initialize ensemble
    model_weights = config.get('ml_models.ensemble_weights', {
        'lstm': 0.30,
        'transformer': 0.35,
        'xgboost': 0.35
    })

    ensemble = EnsembleModel(model_weights=model_weights)

    # Update model configs from config file
    ensemble.lstm_model.sequence_length = config.get('ml_models.lstm.sequence_length', 60)
    ensemble.lstm_model.lstm_units = config.get('ml_models.lstm.layers', [128, 64])
    ensemble.lstm_model.dropout = config.get('ml_models.lstm.dropout', 0.3)
    ensemble.lstm_model.n_features = X_train.shape[1]

    ensemble.transformer_model.sequence_length = config.get('ml_models.transformer.sequence_length', 60)
    ensemble.transformer_model.d_model = config.get('ml_models.transformer.d_model', 128)
    ensemble.transformer_model.num_heads = config.get('ml_models.transformer.nhead', 8)
    ensemble.transformer_model.num_layers = config.get('ml_models.transformer.num_layers', 4)
    ensemble.transformer_model.n_features = X_train.shape[1]

    ensemble.xgboost_model.n_estimators = config.get('ml_models.xgboost.n_estimators', 200)
    ensemble.xgboost_model.max_depth = config.get('ml_models.xgboost.max_depth', 7)
    ensemble.xgboost_model.learning_rate = config.get('ml_models.xgboost.learning_rate', 0.05)

    # Train models
    try:
        results = ensemble.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            models_to_train=['lstm', 'transformer', 'xgboost']
        )

        logger.info("\nTraining results:")
        for model_name, result in results.items():
            logger.info(f"{model_name}: {result}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Evaluate models
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Evaluating models")
    logger.info("=" * 80)

    try:
        eval_results = ensemble.evaluate(X_val, y_val)

        logger.info("\nEvaluation Results:")
        logger.info("-" * 80)

        for model_name, metrics in eval_results.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.6f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

    # Step 6: Save models
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Saving models")
    logger.info("=" * 80)

    model_dir = config.get('paths.models', 'data/models')

    try:
        ensemble.save(model_dir)
        logger.info(f"Models saved successfully to {model_dir}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")

    # Step 7: Test predictions
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Testing predictions")
    logger.info("=" * 80)

    # Get predictions with confidence
    test_predictions, test_confidence = ensemble.get_prediction_confidence(X_val[:100])

    logger.info(f"Sample predictions (first 5):")
    for i in range(min(5, len(test_predictions))):
        logger.info(f"  Prediction: {test_predictions[i]:.6f}, "
                   f"Confidence: {test_confidence[i]:.4f}, "
                   f"Actual: {y_val[i]:.6f}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)

    summary = ensemble.evaluate(X_val, y_val)
    ensemble_metrics = summary.get('ensemble', {})

    logger.info(f"\nEnsemble Performance:")
    logger.info(f"  RMSE: {ensemble_metrics.get('rmse', 0):.6f}")
    logger.info(f"  Direction Accuracy: {ensemble_metrics.get('direction_accuracy', 0):.2f}%")
    logger.info(f"\nModels saved to: {model_dir}")
    logger.info(f"Ready for backtesting and live trading!")


if __name__ == "__main__":
    main()
