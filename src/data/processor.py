"""
Advanced Data Processor
Feature engineering with 50+ technical indicators and price features
"""
import pandas as pd
import numpy as np
import ta
from typing import Optional, List, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Advanced data processor with comprehensive feature engineering
    Includes technical indicators, price features, and derived metrics
    """

    def __init__(self):
        """Initialize data processor"""
        self.feature_names = []

    def process(self, df: pd.DataFrame, add_all_features: bool = True) -> pd.DataFrame:
        """
        Process OHLCV data and add all features

        Args:
            df: DataFrame with OHLCV data
            add_all_features: Add all available features

        Returns:
            Processed DataFrame with features
        """
        if df.empty:
            logger.warning("Empty dataframe provided to processor")
            return df

        # Make a copy to avoid modifying original
        data = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        logger.info("Processing data with advanced feature engineering")

        if add_all_features:
            # Add all feature groups
            data = self.add_price_features(data)
            data = self.add_trend_indicators(data)
            data = self.add_momentum_indicators(data)
            data = self.add_volatility_indicators(data)
            data = self.add_volume_indicators(data)
            data = self.add_support_resistance(data)
            data = self.add_pattern_features(data)
            data = self.add_statistical_features(data)

        # Fill NaN values
        data = self._fill_nan_values(data)

        logger.info(f"Feature engineering complete. Total features: {len(data.columns) - 5}")
        self.feature_names = [col for col in data.columns if col not in required_cols]

        return data

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        data = df.copy()

        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Price changes
        data['price_change'] = data['close'] - data['open']
        data['price_change_pct'] = (data['close'] - data['open']) / data['open']

        # High-Low range
        data['high_low_range'] = data['high'] - data['low']
        data['high_low_range_pct'] = data['high_low_range'] / data['close']

        # Close position in range
        data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-10)

        # Gaps
        data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
        data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)

        # Body and wick sizes (candlestick features)
        data['body_size'] = abs(data['close'] - data['open'])
        data['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
        data['body_to_range'] = data['body_size'] / (data['high_low_range'] + 1e-10)

        return data

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        data = df.copy()

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)

        # Exponential Moving Averages
        for period in [9, 12, 21, 26, 50]:
            data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)

        # MACD
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
        data['adx'] = adx.adx()
        data['adx_pos'] = adx.adx_pos()
        data['adx_neg'] = adx.adx_neg()

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(data['high'], data['low'])
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
        data['ichimoku_base'] = ichimoku.ichimoku_base_line()
        data['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(data['high'], data['low'], data['close'])
        data['psar'] = psar.psar()
        data['psar_up'] = psar.psar_up()
        data['psar_down'] = psar.psar_down()

        # Moving Average Crossovers
        data['sma_20_50_cross'] = (data['sma_20'] > data['sma_50']).astype(int)
        data['ema_12_26_cross'] = (data['ema_12'] > data['ema_26']).astype(int)

        return data

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        data = df.copy()

        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            data[f'rsi_{period}'] = ta.momentum.rsi(data['close'], window=period)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])

        # Rate of Change (ROC)
        for period in [9, 12, 25]:
            data[f'roc_{period}'] = ta.momentum.roc(data['close'], window=period)

        # Commodity Channel Index (CCI)
        data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])

        # Ultimate Oscillator
        data['ultimate_osc'] = ta.momentum.ultimate_oscillator(
            data['high'], data['low'], data['close']
        )

        # Awesome Oscillator
        data['awesome_osc'] = ta.momentum.awesome_oscillator(data['high'], data['low'])

        # Money Flow Index (MFI)
        data['mfi'] = ta.volume.money_flow_index(
            data['high'], data['low'], data['close'], data['volume']
        )

        # TSI (True Strength Index)
        data['tsi'] = ta.momentum.tsi(data['close'])

        return data

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        data = df.copy()

        # Average True Range (ATR)
        for period in [7, 14, 21]:
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=period)
            data[f'atr_{period}'] = atr.average_true_range()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['close'])
        data['bb_high'] = bb.bollinger_hband()
        data['bb_mid'] = bb.bollinger_mavg()
        data['bb_low'] = bb.bollinger_lband()
        data['bb_width'] = bb.bollinger_wband()
        data['bb_pct'] = bb.bollinger_pband()

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(data['high'], data['low'], data['close'])
        data['kc_high'] = kc.keltner_channel_hband()
        data['kc_mid'] = kc.keltner_channel_mband()
        data['kc_low'] = kc.keltner_channel_lband()

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(data['high'], data['low'], data['close'])
        data['dc_high'] = dc.donchian_channel_hband()
        data['dc_mid'] = dc.donchian_channel_mband()
        data['dc_low'] = dc.donchian_channel_lband()

        # Historical Volatility
        for period in [10, 20, 30]:
            data[f'volatility_{period}'] = data['returns'].rolling(window=period).std()

        return data

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        data = df.copy()

        # Volume changes
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()

        # On-Balance Volume (OBV)
        data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])

        # Volume Weighted Average Price (VWAP)
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

        # Accumulation/Distribution Index
        data['adi'] = ta.volume.acc_dist_index(data['high'], data['low'], data['close'], data['volume'])

        # Chaikin Money Flow
        data['cmf'] = ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])

        # Force Index
        data['force_index'] = ta.volume.force_index(data['close'], data['volume'])

        # Ease of Movement
        data['eom'] = ta.volume.ease_of_movement(data['high'], data['low'], data['volume'])

        # Volume Price Trend
        data['vpt'] = ta.volume.volume_price_trend(data['close'], data['volume'])

        return data

    def add_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Add support and resistance levels"""
        data = df.copy()

        # Pivot points
        data['pivot'] = (data['high'] + data['low'] + data['close']) / 3
        data['resistance_1'] = 2 * data['pivot'] - data['low']
        data['support_1'] = 2 * data['pivot'] - data['high']
        data['resistance_2'] = data['pivot'] + (data['high'] - data['low'])
        data['support_2'] = data['pivot'] - (data['high'] - data['low'])

        # Distance from support/resistance
        data['dist_to_resistance'] = (data['resistance_1'] - data['close']) / data['close']
        data['dist_to_support'] = (data['close'] - data['support_1']) / data['close']

        # Rolling highs and lows
        data['rolling_high_20'] = data['high'].rolling(window=20).max()
        data['rolling_low_20'] = data['low'].rolling(window=20).min()
        data['rolling_high_50'] = data['high'].rolling(window=50).max()
        data['rolling_low_50'] = data['low'].rolling(window=50).min()

        return data

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        data = df.copy()

        # Doji pattern
        data['doji'] = ((abs(data['close'] - data['open']) / data['high_low_range']) < 0.1).astype(int)

        # Hammer pattern
        data['hammer'] = (
            (data['lower_wick'] > 2 * data['body_size']) &
            (data['upper_wick'] < data['body_size'])
        ).astype(int)

        # Shooting star pattern
        data['shooting_star'] = (
            (data['upper_wick'] > 2 * data['body_size']) &
            (data['lower_wick'] < data['body_size'])
        ).astype(int)

        # Engulfing patterns
        data['bullish_engulfing'] = (
            (data['close'] > data['open']) &
            (data['close'].shift(1) < data['open'].shift(1)) &
            (data['open'] < data['close'].shift(1)) &
            (data['close'] > data['open'].shift(1))
        ).astype(int)

        data['bearish_engulfing'] = (
            (data['close'] < data['open']) &
            (data['close'].shift(1) > data['open'].shift(1)) &
            (data['open'] > data['close'].shift(1)) &
            (data['close'] < data['open'].shift(1))
        ).astype(int)

        return data

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        data = df.copy()

        # Z-scores
        for period in [20, 50]:
            mean = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            data[f'zscore_{period}'] = (data['close'] - mean) / (std + 1e-10)

        # Skewness and Kurtosis
        data['returns_skew_20'] = data['returns'].rolling(window=20).skew()
        data['returns_kurt_20'] = data['returns'].rolling(window=20).kurt()

        # Autocorrelation
        data['autocorr_1'] = data['returns'].rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)

        # Trend strength
        data['trend_strength'] = abs(data['ema_12'] - data['ema_26']) / data['close']

        # Price momentum
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1

        return data

    def _fill_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values intelligently"""
        data = df.copy()

        # Forward fill first, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Fill any remaining NaNs with 0
        data = data.fillna(0)

        return data

    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for feature importance analysis

        Args:
            df: Processed DataFrame

        Returns:
            DataFrame with features and target
        """
        # Create target (next period return)
        data = df.copy()
        data['target'] = data['close'].shift(-1) / data['close'] - 1

        # Remove rows with NaN target
        data = data.dropna(subset=['target'])

        return data

    def normalize_features(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features

        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax')

        Returns:
            Normalized DataFrame
        """
        data = df.copy()

        # Don't normalize OHLCV columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        if method == 'zscore':
            for col in feature_cols:
                mean = data[col].mean()
                std = data[col].std()
                data[col] = (data[col] - mean) / (std + 1e-10)

        elif method == 'minmax':
            for col in feature_cols:
                min_val = data[col].min()
                max_val = data[col].max()
                data[col] = (data[col] - min_val) / (max_val - min_val + 1e-10)

        return data
