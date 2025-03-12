import ccxt
import pandas as pd
import ta
import os

# Load API keys from environment variables
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
})

def fetch_crypto_data(symbol="BTC/USDT", timeframe="1h", limit=500):
    """Fetch OHLCV data from Binance"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def add_technical_indicators(df):
    """Compute SMA, RSI, MACD, and ATR"""
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df.dropna(inplace=True)
    return df