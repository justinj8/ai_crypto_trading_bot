"""
Portfolio Manager
Manages positions, tracks performance, and calculates metrics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """
    Portfolio management system
    - Position tracking
    - P&L calculation
    - Performance metrics
    - Trade history
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize portfolio manager

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> position dict
        self.trades = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.utcnow()]

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")

    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos['current_value'] for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def total_return(self) -> float:
        """Calculate total return percentage"""
        return (self.portfolio_value - self.initial_capital) / self.initial_capital

    def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        side: str,
        strategy: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Open a new position

        Args:
            symbol: Trading symbol
            quantity: Position size
            entry_price: Entry price
            side: LONG or SHORT
            strategy: Strategy name
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            True if position opened successfully
        """
        position_value = quantity * entry_price

        if position_value > self.cash:
            logger.warning(f"Insufficient cash for position: ${position_value:.2f} > ${self.cash:.2f}")
            return False

        # Create position
        self.positions[symbol] = {
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'entry_price': entry_price,
            'current_price': entry_price,
            'entry_value': position_value,
            'current_value': position_value,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': strategy,
            'opened_at': datetime.utcnow()
        }

        self.cash -= position_value

        logger.info(f"Opened {side} position: {quantity} {symbol} @ ${entry_price:.2f}")

        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        partial_quantity: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Close position (fully or partially)

        Args:
            symbol: Symbol to close
            exit_price: Exit price
            partial_quantity: Quantity to close (None = full position)

        Returns:
            Trade result dictionary
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self.positions[symbol]

        # Determine quantity to close
        if partial_quantity is None:
            close_quantity = position['quantity']
        else:
            close_quantity = min(partial_quantity, position['quantity'])

        # Calculate P&L
        if position['side'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * close_quantity
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * close_quantity

        # Update position or remove if fully closed
        if close_quantity >= position['quantity']:
            # Full close
            self.cash += position['quantity'] * exit_price
            trade_result = {
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': pnl / position['entry_value'],
                'strategy': position['strategy'],
                'opened_at': position['opened_at'],
                'closed_at': datetime.utcnow(),
                'duration': (datetime.utcnow() - position['opened_at']).total_seconds() / 3600  # hours
            }

            del self.positions[symbol]

        else:
            # Partial close
            self.cash += close_quantity * exit_price
            position['quantity'] -= close_quantity
            position['current_value'] = position['quantity'] * exit_price
            position['realized_pnl'] += pnl

            trade_result = {
                'symbol': symbol,
                'side': position['side'],
                'quantity': close_quantity,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': pnl / (close_quantity * position['entry_price']),
                'strategy': position['strategy'],
                'partial': True
            }

        # Update trade statistics
        self.trades.append(trade_result)
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        logger.info(f"Closed position: {symbol} P&L: ${pnl:.2f} ({trade_result.get('return_pct', 0)*100:.2f}%)")

        return trade_result

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions

        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position['current_price'] = prices[symbol]
                position['current_value'] = position['quantity'] * prices[symbol]

                # Calculate unrealized P&L
                if position['side'] == 'LONG':
                    position['unrealized_pnl'] = (prices[symbol] - position['entry_price']) * position['quantity']
                else:  # SHORT
                    position['unrealized_pnl'] = (position['entry_price'] - prices[symbol]) * position['quantity']

        # Update equity curve
        self.equity_curve.append(self.portfolio_value)
        self.timestamps.append(datetime.utcnow())

    def check_stop_loss_take_profit(self, prices: Dict[str, float]) -> List[str]:
        """
        Check if any positions hit stop loss or take profit

        Args:
            prices: Current prices

        Returns:
            List of symbols to close
        """
        to_close = []

        for symbol, position in self.positions.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Check stop loss
            if position['stop_loss']:
                if position['side'] == 'LONG' and current_price <= position['stop_loss']:
                    logger.warning(f"Stop loss hit for {symbol}: ${current_price:.2f} <= ${position['stop_loss']:.2f}")
                    to_close.append(symbol)
                    continue

                elif position['side'] == 'SHORT' and current_price >= position['stop_loss']:
                    logger.warning(f"Stop loss hit for {symbol}: ${current_price:.2f} >= ${position['stop_loss']:.2f}")
                    to_close.append(symbol)
                    continue

            # Check take profit
            if position['take_profit']:
                if position['side'] == 'LONG' and current_price >= position['take_profit']:
                    logger.info(f"Take profit hit for {symbol}: ${current_price:.2f} >= ${position['take_profit']:.2f}")
                    to_close.append(symbol)
                    continue

                elif position['side'] == 'SHORT' and current_price <= position['take_profit']:
                    logger.info(f"Take profit hit for {symbol}: ${current_price:.2f} <= ${position['take_profit']:.2f}")
                    to_close.append(symbol)

        return to_close

    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if len(self.equity_curve) < 2:
            return self._get_empty_metrics()

        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Basic metrics
        total_return = self.total_return
        total_trades = self.total_trades

        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Profit factor
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')

        # Average win/loss
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0

        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Drawdown
        max_drawdown = self._calculate_max_drawdown(equity_series)

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'current_capital': self.portfolio_value,
            'cash': self.cash,
            'positions_value': self.portfolio_value - self.cash
        }

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        # Annualized Sharpe
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return sortino

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dict"""
        return {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve
        })
