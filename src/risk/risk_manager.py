"""
Risk Management System
Implements comprehensive risk controls and position sizing
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

from src.core.events import OrderEvent, SignalEvent, RiskEvent, OrderSide
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Comprehensive risk management system
    - Position sizing (Kelly Criterion, fixed %, volatility-adjusted)
    - Portfolio-level risk limits
    - Correlation management
    - Drawdown protection
    - Circuit breakers
    """

    def __init__(self, config: Dict):
        """
        Initialize risk manager

        Args:
            config: Risk management configuration
        """
        self.config = config

        # Portfolio state
        self.portfolio_value = config.get('initial_capital', 10000)
        self.initial_capital = self.portfolio_value
        self.cash = self.portfolio_value
        self.positions = {}  # symbol -> position info

        # Risk tracking
        self.daily_pnl = []
        self.trades_today = {}
        self.peak_value = self.portfolio_value
        self.current_drawdown = 0.0

        # Circuit breakers
        self.is_halted = False
        self.halt_reason = ""
        self.halt_timestamp = None

        # Limits
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)
        self.max_position_size = config.get('max_position_size', 0.10)
        self.max_total_exposure = config.get('max_total_exposure', 0.50)
        self.max_drawdown = config.get('max_drawdown', 0.15)

        logger.info("Risk Manager initialized")

    def evaluate_signal(
        self,
        signal: SignalEvent,
        current_price: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, str, float]:
        """
        Evaluate if signal passes risk checks and calculate position size

        Args:
            signal: Trading signal
            current_price: Current price
            market_data: Market data for volatility calculation

        Returns:
            Tuple of (approved, reason, position_size)
        """
        # Check if trading is halted
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}", 0.0

        # Check portfolio-level limits
        if not self._check_portfolio_limits():
            return False, "Portfolio limits exceeded", 0.0

        # Check symbol-specific limits
        if not self._check_symbol_limits(signal.symbol):
            return False, "Symbol limits exceeded", 0.0

        # Check position concentration
        if not self._check_concentration_limits(signal.symbol):
            return False, "Position concentration too high", 0.0

        # Check correlation limits (if we have market data)
        if market_data is not None:
            if not self._check_correlation_limits(signal.symbol, signal.signal_type, market_data):
                return False, "Correlation limits exceeded", 0.0

        # Calculate position size
        position_size = self._calculate_position_size(
            signal=signal,
            current_price=current_price,
            market_data=market_data
        )

        if position_size <= 0:
            return False, "Position size too small", 0.0

        # Check if position size exceeds limits
        position_value = position_size * current_price
        if position_value > self.portfolio_value * self.max_position_size:
            # Scale down to max
            position_size = (self.portfolio_value * self.max_position_size) / current_price
            logger.warning(f"Position size reduced to meet limits: {position_size}")

        # Final validation passed
        return True, "Signal approved", position_size

    def _check_portfolio_limits(self) -> bool:
        """Check portfolio-level risk limits"""

        # Check drawdown
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        if self.current_drawdown > self.max_drawdown:
            self._trigger_circuit_breaker("Max drawdown exceeded")
            return False

        # Check total exposure
        total_exposure = sum(pos['value'] for pos in self.positions.values())
        max_exposure = self.portfolio_value * self.max_total_exposure

        if total_exposure >= max_exposure:
            logger.warning(f"Total exposure {total_exposure:.2f} >= max {max_exposure:.2f}")
            return False

        # Check daily loss limit
        if self.config.get('max_daily_loss'):
            daily_loss = self._calculate_daily_pnl()
            max_daily_loss = self.portfolio_value * self.config['max_daily_loss']

            if daily_loss < -max_daily_loss:
                self._trigger_circuit_breaker("Max daily loss exceeded")
                return False

        return True

    def _check_symbol_limits(self, symbol: str) -> bool:
        """Check symbol-specific limits"""

        # Get symbol limits from config
        symbol_limits = self.config.get('symbol_limits', {})
        limits = symbol_limits.get(symbol, symbol_limits.get('default', {}))

        # Check max daily trades
        if 'max_daily_trades' in limits:
            trades_today = self.trades_today.get(symbol, 0)
            if trades_today >= limits['max_daily_trades']:
                logger.warning(f"Max daily trades reached for {symbol}")
                return False

        return True

    def _check_concentration_limits(self, symbol: str) -> bool:
        """Check if adding position would exceed concentration limits"""

        if symbol in self.positions:
            # Already have position, check if we can add to it
            position_value = self.positions[symbol]['value']
            concentration = position_value / self.portfolio_value

            if concentration >= self.max_position_size:
                logger.warning(f"Max position size reached for {symbol}")
                return False

        return True

    def _check_correlation_limits(
        self,
        symbol: str,
        signal_type,
        market_data: pd.DataFrame
    ) -> bool:
        """Check correlation with existing positions"""

        if not self.positions:
            return True  # No existing positions

        # This is a simplified version
        # In production, you'd calculate actual correlations between assets
        max_correlation = self.config.get('max_correlation', 0.7)

        # For now, just check if we're taking similar positions
        # (e.g., multiple long BTC, ETH, etc. which are typically correlated)

        return True  # Placeholder

    def _calculate_position_size(
        self,
        signal: SignalEvent,
        current_price: float,
        market_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate position size using configured method

        Args:
            signal: Trading signal
            current_price: Current price
            market_data: Market data for volatility

        Returns:
            Position size (quantity)
        """
        sizing_method = self.config.get('position_sizing', {}).get('method', 'fixed')

        if sizing_method == 'kelly_criterion':
            return self._kelly_position_size(signal, current_price)

        elif sizing_method == 'volatility_adjusted':
            return self._volatility_adjusted_size(signal, current_price, market_data)

        else:  # fixed or percent
            return self._fixed_percent_size(signal, current_price)

    def _fixed_percent_size(self, signal: SignalEvent, current_price: float) -> float:
        """Fixed percentage position sizing"""
        risk_per_trade = self.max_portfolio_risk
        position_value = self.portfolio_value * risk_per_trade
        return position_value / current_price

    def _kelly_position_size(self, signal: SignalEvent, current_price: float) -> float:
        """Kelly Criterion position sizing"""

        # Kelly formula: f = (bp - q) / b
        # where:
        # f = fraction of capital to bet
        # b = odds (reward/risk ratio)
        # p = probability of winning (confidence)
        # q = probability of losing (1 - p)

        kelly_fraction = self.config.get('position_sizing', {}).get('kelly_fraction', 0.25)

        # Use signal confidence as win probability
        p = signal.confidence

        # Calculate reward/risk ratio from signal
        if signal.take_profit and signal.stop_loss:
            potential_profit = abs(signal.take_profit - current_price)
            potential_loss = abs(signal.stop_loss - current_price)
            b = potential_profit / (potential_loss + 1e-10)
        else:
            b = 2.0  # Default 2:1 reward/risk

        q = 1 - p

        # Kelly percentage
        kelly_pct = (b * p - q) / b

        # Apply Kelly fraction for safety (e.g., half-Kelly)
        kelly_pct = max(0, kelly_pct * kelly_fraction)

        # Calculate position size
        position_value = self.portfolio_value * min(kelly_pct, self.max_position_size)
        return position_value / current_price

    def _volatility_adjusted_size(
        self,
        signal: SignalEvent,
        current_price: float,
        market_data: Optional[pd.DataFrame]
    ) -> float:
        """Volatility-adjusted position sizing"""

        if market_data is None or len(market_data) < 20:
            # Fall back to fixed sizing
            return self._fixed_percent_size(signal, current_price)

        # Calculate historical volatility
        if 'returns' in market_data.columns:
            volatility = market_data['returns'].rolling(window=20).std().iloc[-1]
        else:
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]

        # Target volatility
        target_vol = 0.02  # 2% daily volatility

        # Adjust position size inversely to volatility
        vol_scalar = target_vol / (volatility + 1e-10)
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit adjustment range

        # Base position size
        base_size = self._fixed_percent_size(signal, current_price)

        return base_size * vol_scalar

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker"""
        self.is_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.utcnow()

        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        # Could send alerts here
        self._send_risk_alert(reason, "CRITICAL")

    def _send_risk_alert(self, message: str, severity: str = "WARNING"):
        """Send risk alert (placeholder for notification system)"""
        logger.warning(f"Risk Alert [{severity}]: {message}")

    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        # Simplified version
        return sum(self.daily_pnl)

    def update_portfolio(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0
    ):
        """
        Update portfolio state after trade

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Trade quantity
            price: Execution price
            commission: Trading commission
        """
        # Update positions
        if side == "BUY":
            if symbol in self.positions:
                # Add to existing position
                old_qty = self.positions[symbol]['quantity']
                old_avg_price = self.positions[symbol]['avg_price']

                new_qty = old_qty + quantity
                new_avg_price = ((old_qty * old_avg_price) + (quantity * price)) / new_qty

                self.positions[symbol]['quantity'] = new_qty
                self.positions[symbol]['avg_price'] = new_avg_price
                self.positions[symbol]['value'] = new_qty * price

            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'value': quantity * price,
                    'opened_at': datetime.utcnow()
                }

            self.cash -= (quantity * price + commission)

        else:  # SELL
            if symbol in self.positions:
                # Reduce or close position
                self.positions[symbol]['quantity'] -= quantity
                self.positions[symbol]['value'] = self.positions[symbol]['quantity'] * price

                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]

                self.cash += (quantity * price - commission)

        # Update portfolio value
        positions_value = sum(pos['value'] for pos in self.positions.values())
        self.portfolio_value = self.cash + positions_value

        # Update peak value
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # Track trades
        if symbol not in self.trades_today:
            self.trades_today[symbol] = 0
        self.trades_today[symbol] += 1

        logger.info(f"Portfolio updated: Cash: ${self.cash:.2f}, Total: ${self.portfolio_value:.2f}")

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'total_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': sum(pos['value'] for pos in self.positions.values()),
            'num_positions': len(self.positions),
            'current_drawdown': self.current_drawdown,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'is_halted': self.is_halted,
            'positions': self.positions.copy()
        }

    def reset_daily_limits(self):
        """Reset daily tracking (call at start of each trading day)"""
        self.trades_today = {}
        self.daily_pnl = []
        logger.info("Daily risk limits reset")

    def resume_trading(self):
        """Resume trading after circuit breaker"""
        if self.is_halted:
            cooldown_minutes = self.config.get('recovery', {}).get('cooldown_period_minutes', 30)

            if self.halt_timestamp:
                time_since_halt = (datetime.utcnow() - self.halt_timestamp).total_seconds() / 60

                if time_since_halt >= cooldown_minutes:
                    self.is_halted = False
                    self.halt_reason = ""
                    self.halt_timestamp = None

                    logger.info("Trading resumed after circuit breaker cooldown")
                    return True
                else:
                    remaining = cooldown_minutes - time_since_halt
                    logger.info(f"Cooldown period remaining: {remaining:.1f} minutes")
                    return False

        return False
