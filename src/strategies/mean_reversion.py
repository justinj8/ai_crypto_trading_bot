"""
Mean Reversion Strategy
Trades based on price returning to mean after extreme deviations
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from src.strategies.base import BaseStrategy
from src.core.events import SignalEvent, SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and statistical analysis

    Entry conditions (LONG):
    - Price below lower Bollinger Band
    - RSI < 30 (oversold)
    - Z-score < -2 (statistically oversold)
    - Volume spike (indicating capitulation)

    Entry conditions (SHORT):
    - Price above upper Bollinger Band
    - RSI > 70 (overbought)
    - Z-score > 2 (statistically overbought)
    - Volume spike

    Exit conditions:
    - Price returns to middle Bollinger Band
    - RSI returns to 50
    - Z-score returns to 0
    - Time-based exit (mean reversion trades are typically short-term)
    """

    def __init__(
        self,
        name: str = "MeanReversion",
        symbols: List[str] = ["BTC/USDT"],
        config: dict = None
    ):
        """Initialize mean reversion strategy"""
        default_config = {
            'bb_std_dev': 2.0,  # Bollinger Band standard deviations
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_exit': 50,
            'zscore_threshold': 2.0,
            'zscore_exit': 0.5,
            'volume_spike_multiplier': 1.5,
            'lookback_period': 20,
            'max_hold_bars': 50,  # Maximum bars to hold position
            'stop_loss_pct': 0.05,  # 5%
            'take_profit_pct': 0.03,  # 3% (conservative for mean reversion)
            'min_confidence': 0.6
        }

        if config:
            default_config.update(config)

        super().__init__(name, symbols, default_config)

        self.open_signals = {}  # Track open signals with entry time
        self.entry_bars = {}  # Track how long positions have been open

    def generate_signals(self, market_data: pd.DataFrame) -> List[SignalEvent]:
        """Generate mean reversion signals"""
        signals = []

        if len(market_data) < 50:
            return signals

        # Get latest data
        latest = market_data.iloc[-1]
        symbol = latest.get('symbol', self.symbols[0])

        # Check if we have required indicators
        required_indicators = ['bb_low', 'bb_mid', 'bb_high', 'rsi_14', 'zscore_20', 'volume']

        if not all(ind in market_data.columns for ind in required_indicators):
            logger.warning(f"Missing required indicators for mean reversion strategy")
            return signals

        # Extract indicators
        price = latest['close']
        bb_low = latest['bb_low']
        bb_mid = latest['bb_mid']
        bb_high = latest['bb_high']
        rsi = latest['rsi_14']
        zscore = latest['zscore_20']
        volume = latest['volume']

        # Calculate average volume
        avg_volume = market_data['volume'].tail(20).mean()
        volume_ratio = volume / avg_volume

        # Check for existing position
        has_position = symbol in self.open_signals

        # Update entry bar count
        if has_position:
            self.entry_bars[symbol] = self.entry_bars.get(symbol, 0) + 1

        # ENTRY SIGNALS
        if not has_position:
            # Check for LONG entry (oversold conditions)
            if self._check_long_entry(price, bb_low, bb_mid, rsi, zscore, volume_ratio):
                # Calculate stop-loss and take-profit
                stop_loss = price * (1 - self.config['stop_loss_pct'])
                take_profit = price * (1 + self.config['take_profit_pct'])

                # Calculate confidence
                confidence = self._calculate_long_confidence(
                    price, bb_low, bb_mid, rsi, zscore, volume_ratio
                )

                signal = SignalEvent(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=1.0,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reasoning=f"Mean Reversion LONG: Price={price:.2f}, BB_Low={bb_low:.2f}, RSI={rsi:.1f}, Z-score={zscore:.2f}"
                )

                if self.validate_signal(signal, market_data):
                    signals.append(signal)
                    self.open_signals[symbol] = signal
                    self.entry_bars[symbol] = 0
                    logger.info(f"Mean Reversion LONG: {symbol} @ ${price:.2f}")

            # Check for SHORT entry (overbought conditions)
            elif self._check_short_entry(price, bb_high, bb_mid, rsi, zscore, volume_ratio):
                # Calculate stop-loss and take-profit
                stop_loss = price * (1 + self.config['stop_loss_pct'])
                take_profit = price * (1 - self.config['take_profit_pct'])

                # Calculate confidence
                confidence = self._calculate_short_confidence(
                    price, bb_high, bb_mid, rsi, zscore, volume_ratio
                )

                signal = SignalEvent(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=1.0,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reasoning=f"Mean Reversion SHORT: Price={price:.2f}, BB_High={bb_high:.2f}, RSI={rsi:.1f}, Z-score={zscore:.2f}"
                )

                if self.validate_signal(signal, market_data):
                    signals.append(signal)
                    self.open_signals[symbol] = signal
                    self.entry_bars[symbol] = 0
                    logger.info(f"Mean Reversion SHORT: {symbol} @ ${price:.2f}")

        # EXIT SIGNALS
        else:
            exit_reason = self._check_exit(
                symbol, price, bb_mid, rsi, zscore
            )

            if exit_reason:
                signal = SignalEvent(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    strength=1.0,
                    confidence=0.9,
                    reasoning=exit_reason
                )

                signals.append(signal)
                del self.open_signals[symbol]
                del self.entry_bars[symbol]
                logger.info(f"Mean Reversion EXIT: {symbol} - {exit_reason}")

        return signals

    def _check_long_entry(
        self,
        price: float,
        bb_low: float,
        bb_mid: float,
        rsi: float,
        zscore: float,
        volume_ratio: float
    ) -> bool:
        """Check if LONG entry conditions are met"""

        # Price below lower Bollinger Band
        if price >= bb_low:
            return False

        # RSI oversold
        if rsi >= self.config['rsi_oversold']:
            return False

        # Z-score indicates oversold
        if zscore >= -self.config['zscore_threshold']:
            return False

        # Volume spike (indicates capitulation)
        if volume_ratio < self.config['volume_spike_multiplier']:
            return False

        # Price significantly below middle band (mean)
        distance_from_mean = (bb_mid - price) / bb_mid
        if distance_from_mean < 0.02:  # At least 2% below mean
            return False

        return True

    def _check_short_entry(
        self,
        price: float,
        bb_high: float,
        bb_mid: float,
        rsi: float,
        zscore: float,
        volume_ratio: float
    ) -> bool:
        """Check if SHORT entry conditions are met"""

        # Price above upper Bollinger Band
        if price <= bb_high:
            return False

        # RSI overbought
        if rsi <= self.config['rsi_overbought']:
            return False

        # Z-score indicates overbought
        if zscore <= self.config['zscore_threshold']:
            return False

        # Volume spike
        if volume_ratio < self.config['volume_spike_multiplier']:
            return False

        # Price significantly above middle band (mean)
        distance_from_mean = (price - bb_mid) / bb_mid
        if distance_from_mean < 0.02:  # At least 2% above mean
            return False

        return True

    def _check_exit(
        self,
        symbol: str,
        price: float,
        bb_mid: float,
        rsi: float,
        zscore: float
    ) -> Optional[str]:
        """Check if exit conditions are met"""

        # Time-based exit (mean reversion trades should be quick)
        bars_held = self.entry_bars.get(symbol, 0)
        if bars_held >= self.config['max_hold_bars']:
            return f"Max hold period reached ({bars_held} bars)"

        # Get original signal
        original_signal = self.open_signals.get(symbol)

        if original_signal.signal_type == SignalType.LONG:
            # Exit LONG when price returns to mean
            if price >= bb_mid * 0.99:  # Within 1% of mean
                return "Price returned to mean (LONG)"

            # RSI normalized
            if rsi >= self.config['rsi_exit']:
                return f"RSI normalized ({rsi:.1f})"

            # Z-score normalized
            if zscore >= -self.config['zscore_exit']:
                return f"Z-score normalized ({zscore:.2f})"

        elif original_signal.signal_type == SignalType.SHORT:
            # Exit SHORT when price returns to mean
            if price <= bb_mid * 1.01:  # Within 1% of mean
                return "Price returned to mean (SHORT)"

            # RSI normalized
            if rsi <= self.config['rsi_exit']:
                return f"RSI normalized ({rsi:.1f})"

            # Z-score normalized
            if zscore <= self.config['zscore_exit']:
                return f"Z-score normalized ({zscore:.2f})"

        return None

    def _calculate_long_confidence(
        self,
        price: float,
        bb_low: float,
        bb_mid: float,
        rsi: float,
        zscore: float,
        volume_ratio: float
    ) -> float:
        """Calculate confidence for LONG entry"""
        confidence_factors = []

        # Distance below lower band (more extreme = higher confidence)
        distance = (bb_low - price) / bb_mid
        if distance > 0.05:
            confidence_factors.append(1.0)
        elif distance > 0.03:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # RSI oversold level (lower = higher confidence)
        if rsi < 20:
            confidence_factors.append(1.0)
        elif rsi < 25:
            confidence_factors.append(0.9)
        elif rsi < 30:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Z-score extremity
        if zscore < -2.5:
            confidence_factors.append(1.0)
        elif zscore < -2.0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Volume spike (larger = higher confidence)
        if volume_ratio > 2.0:
            confidence_factors.append(1.0)
        elif volume_ratio > 1.5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        return np.mean(confidence_factors)

    def _calculate_short_confidence(
        self,
        price: float,
        bb_high: float,
        bb_mid: float,
        rsi: float,
        zscore: float,
        volume_ratio: float
    ) -> float:
        """Calculate confidence for SHORT entry"""
        confidence_factors = []

        # Distance above upper band
        distance = (price - bb_high) / bb_mid
        if distance > 0.05:
            confidence_factors.append(1.0)
        elif distance > 0.03:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # RSI overbought level
        if rsi > 80:
            confidence_factors.append(1.0)
        elif rsi > 75:
            confidence_factors.append(0.9)
        elif rsi > 70:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Z-score extremity
        if zscore > 2.5:
            confidence_factors.append(1.0)
        elif zscore > 2.0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Volume spike
        if volume_ratio > 2.0:
            confidence_factors.append(1.0)
        elif volume_ratio > 1.5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        return np.mean(confidence_factors)

    def calculate_confidence(self, market_data: pd.DataFrame) -> float:
        """Calculate overall strategy confidence"""
        latest = market_data.iloc[-1]

        # Check if conditions are extreme
        price = latest['close']
        bb_mid = latest.get('bb_mid', price)
        rsi = latest.get('rsi_14', 50)
        zscore = latest.get('zscore_20', 0)

        distance = abs(price - bb_mid) / bb_mid
        rsi_deviation = abs(rsi - 50)
        zscore_abs = abs(zscore)

        # Higher confidence when conditions are more extreme
        extremity_score = np.mean([
            min(1.0, distance * 20),  # Normalize distance
            min(1.0, rsi_deviation / 30),  # Normalize RSI deviation
            min(1.0, zscore_abs / 3)  # Normalize z-score
        ])

        return extremity_score
