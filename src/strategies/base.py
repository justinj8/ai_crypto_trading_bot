"""
Base Strategy Class
All trading strategies inherit from this base class
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.events import SignalEvent, SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    """

    def __init__(self, name: str, symbols: List[str], config: Dict):
        """
        Initialize strategy

        Args:
            name: Strategy name
            symbols: List of symbols to trade
            config: Strategy configuration
        """
        self.name = name
        self.symbols = symbols
        self.config = config

        self.signals = []
        self.is_active = True

    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> List[SignalEvent]:
        """
        Generate trading signals from market data

        Args:
            market_data: DataFrame with OHLCV and indicators

        Returns:
            List of SignalEvent objects
        """
        pass

    @abstractmethod
    def calculate_confidence(self, market_data: pd.DataFrame) -> float:
        """
        Calculate confidence in the current signal

        Args:
            market_data: Market data

        Returns:
            Confidence score (0-1)
        """
        pass

    def calculate_position_size(
        self,
        signal: SignalEvent,
        account_value: float,
        current_price: float
    ) -> float:
        """
        Calculate position size for a signal

        Args:
            signal: Trading signal
            account_value: Current account value
            current_price: Current price

        Returns:
            Position size (quantity)
        """
        # Default: use fixed percentage of account
        risk_per_trade = self.config.get('risk_per_trade', 0.02)
        position_value = account_value * risk_per_trade

        quantity = position_value / current_price

        return quantity

    def validate_signal(self, signal: SignalEvent, market_data: pd.DataFrame) -> bool:
        """
        Validate if a signal meets minimum criteria

        Args:
            signal: Signal to validate
            market_data: Market data

        Returns:
            True if signal is valid
        """
        # Check confidence threshold
        min_confidence = self.config.get('min_confidence', 0.6)

        if signal.confidence < min_confidence:
            logger.debug(f"Signal rejected: confidence {signal.confidence} < {min_confidence}")
            return False

        # Check if symbol is tradeable
        if signal.symbol not in self.symbols:
            logger.warning(f"Signal for non-tradeable symbol: {signal.symbol}")
            return False

        return True

    def update_config(self, new_config: Dict):
        """Update strategy configuration"""
        self.config.update(new_config)
        logger.info(f"Strategy {self.name} configuration updated")

    def reset(self):
        """Reset strategy state"""
        self.signals = []
        logger.info(f"Strategy {self.name} reset")

    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'total_signals': len(self.signals),
            'active': self.is_active
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', symbols={self.symbols})"
