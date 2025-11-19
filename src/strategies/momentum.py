"""
Momentum Trading Strategy
Identifies and trades strong trends using multiple momentum indicators
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from src.strategies.base import BaseStrategy
from src.core.events import SignalEvent, SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Multi-indicator momentum strategy

    Entry conditions (LONG):
    - Price above SMA(50) and SMA(200)
    - RSI between 50 and 70 (strong but not overbought)
    - MACD positive and rising
    - ADX > 25 (strong trend)
    - Volume above average

    Exit conditions:
    - RSI > 75 (overbought)
    - MACD crossover down
    - Price crosses below SMA(50)
    - Stop-loss or take-profit hit
    """

    def __init__(
        self,
        name: str = "Momentum",
        symbols: List[str] = ["BTC/USDT"],
        config: dict = None
    ):
        """Initialize momentum strategy"""
        default_config = {
            'rsi_entry_min': 50,
            'rsi_entry_max': 70,
            'rsi_exit': 75,
            'adx_threshold': 25,
            'volume_multiplier': 1.2,
            'sma_fast': 50,
            'sma_slow': 200,
            'stop_loss_pct': 0.03,  # 3%
            'take_profit_pct': 0.09,  # 9% (3:1 R/R)
            'min_confidence': 0.6
        }

        if config:
            default_config.update(config)

        super().__init__(name, symbols, default_config)

        self.open_signals = {}  # Track open signals per symbol

    def generate_signals(self, market_data: pd.DataFrame) -> List[SignalEvent]:
        """Generate momentum-based signals"""
        signals = []

        if len(market_data) < 200:
            return signals

        # Get latest data
        latest = market_data.iloc[-1]
        symbol = latest.get('symbol', self.symbols[0])

        # Check if we have required indicators
        required_indicators = ['sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal', 'adx', 'volume']

        if not all(ind in market_data.columns for ind in required_indicators):
            logger.warning(f"Missing required indicators for momentum strategy")
            return signals

        # Extract indicators
        price = latest['close']
        sma_50 = latest['sma_50']
        sma_200 = latest['sma_200']
        rsi = latest['rsi_14']
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        adx = latest['adx']
        volume = latest['volume']

        # Calculate average volume
        avg_volume = market_data['volume'].tail(20).mean()

        # Check for existing position
        has_position = symbol in self.open_signals

        # ENTRY SIGNALS
        if not has_position:
            long_signal = self._check_long_entry(
                price, sma_50, sma_200, rsi, macd, macd_signal, adx, volume, avg_volume
            )

            if long_signal:
                # Calculate stop-loss and take-profit
                stop_loss = price * (1 - self.config['stop_loss_pct'])
                take_profit = price * (1 + self.config['take_profit_pct'])

                # Calculate confidence
                confidence = self.calculate_confidence(market_data)

                signal = SignalEvent(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=1.0,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    reasoning=f"Momentum: Price above SMAs, RSI={rsi:.1f}, ADX={adx:.1f}, MACD bullish"
                )

                if self.validate_signal(signal, market_data):
                    signals.append(signal)
                    self.open_signals[symbol] = signal
                    logger.info(f"Momentum LONG signal: {symbol} @ ${price:.2f}")

        # EXIT SIGNALS
        else:
            exit_signal = self._check_exit(
                price, sma_50, rsi, macd, macd_signal, adx
            )

            if exit_signal:
                signal = SignalEvent(
                    strategy_name=self.name,
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    strength=1.0,
                    confidence=0.9,
                    reasoning=exit_signal
                )

                signals.append(signal)
                del self.open_signals[symbol]
                logger.info(f"Momentum EXIT signal: {symbol} - {exit_signal}")

        return signals

    def _check_long_entry(
        self,
        price: float,
        sma_50: float,
        sma_200: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        adx: float,
        volume: float,
        avg_volume: float
    ) -> bool:
        """Check if long entry conditions are met"""

        # Trend confirmation: Price above both SMAs
        if price <= sma_50 or price <= sma_200:
            return False

        # SMA alignment: Fast SMA above slow SMA
        if sma_50 <= sma_200:
            return False

        # RSI in momentum zone (not overbought)
        if rsi < self.config['rsi_entry_min'] or rsi > self.config['rsi_entry_max']:
            return False

        # MACD bullish
        if macd <= 0 or macd <= macd_signal:
            return False

        # Strong trend
        if adx < self.config['adx_threshold']:
            return False

        # Volume confirmation
        if volume < avg_volume * self.config['volume_multiplier']:
            return False

        return True

    def _check_exit(
        self,
        price: float,
        sma_50: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        adx: float
    ) -> Optional[str]:
        """Check if exit conditions are met"""

        # Overbought
        if rsi > self.config['rsi_exit']:
            return f"RSI overbought ({rsi:.1f})"

        # MACD crossover down
        if macd < macd_signal:
            return "MACD bearish crossover"

        # Price below SMA(50)
        if price < sma_50:
            return f"Price below SMA(50)"

        # Trend weakening
        if adx < 20:
            return f"Trend weakening (ADX={adx:.1f})"

        return None

    def calculate_confidence(self, market_data: pd.DataFrame) -> float:
        """Calculate signal confidence based on multiple factors"""
        latest = market_data.iloc[-1]

        confidence_factors = []

        # RSI position (higher when RSI is in ideal range)
        rsi = latest['rsi_14']
        if 55 <= rsi <= 65:
            confidence_factors.append(1.0)
        elif 50 <= rsi <= 70:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # ADX strength (higher when trend is stronger)
        adx = latest['adx']
        if adx > 40:
            confidence_factors.append(1.0)
        elif adx > 30:
            confidence_factors.append(0.8)
        elif adx > 25:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # MACD strength
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        macd_diff = macd - macd_signal

        if macd_diff > 0:
            confidence_factors.append(min(1.0, 0.6 + (macd_diff / abs(macd) * 0.4)))
        else:
            confidence_factors.append(0.5)

        # Volume confirmation
        recent_volume = market_data['volume'].tail(5).mean()
        avg_volume = market_data['volume'].tail(20).mean()
        volume_ratio = recent_volume / avg_volume

        if volume_ratio > 1.5:
            confidence_factors.append(1.0)
        elif volume_ratio > 1.2:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Average all factors
        overall_confidence = np.mean(confidence_factors)

        return overall_confidence
