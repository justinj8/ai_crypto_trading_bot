# AI Cryptocurrency Trading Bot

This repository provides a starting point for building an algorithmic trading system for digital assets. It focuses on a data pipeline for retrieving market data, feature engineering through technical indicators, and a basic machine-learning ensemble for price prediction. The project is intended as a foundation that can be extended with reinforcement learning, sentiment analysis, and natural-language-driven strategy modules.

## Features

- **Data ingestion**: Uses the [`ccxt`](https://github.com/ccxt/ccxt) library to download OHLCV data from Binance.
- **Technical indicators**: Computes commonly used indicators such as SMA, RSI, MACD, and ATR using the [`ta`](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) library.
- **Machine-learning ensemble**:
  - LSTM network (TensorFlow/Keras) for capturing short-term temporal dynamics.
  - XGBoost regressor for tabular feature modeling.
  - Predictions from both models are averaged to produce the final forecast.
- **Extensibility**: The structure is designed to accommodate additional components such as reinforcement-learning policies (e.g., PPO with Ray RLlib), sentiment analysis of Reddit/Twitter data, GPT-based trading insights, and a Streamlit dashboard for real-time analytics.

## Installation

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Environment variables

`data_utils.fetch_crypto_data` requires Binance API credentials. Supply them via environment variables or a `.env` file:

```bash
export BINANCE_API_KEY=your_api_key
export BINANCE_SECRET=your_api_secret
```

`.env` example:

```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_api_secret
```

## Example usage

```python
from data_utils import fetch_crypto_data, add_technical_indicators
from ml_model import train_models, predict_next_price

# Download data and build features
frame = fetch_crypto_data(symbol="BTC/USDT", timeframe="1h")
frame = add_technical_indicators(frame)
features = ["sma_20", "rsi", "macd", "atr"]

# Train models and make a prediction
train_models(frame, features)
next_price = predict_next_price(frame, features)
print(f"Predicted next close: {next_price:.2f}")
```

## Roadmap

Planned future enhancements include:

- Reinforcement learning for dynamic position sizing and risk management.
- Weighted sentiment analysis drawing from Reddit and Twitter streams.
- Integration with large language models such as GPT‑4 for high-level strategy advice.
- Streamlit dashboard for monitoring P&L and key metrics in real time.

## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Use at your own risk.

