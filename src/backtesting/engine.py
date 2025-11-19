"""
Event-Driven Backtesting Engine
Realistic simulation of trading with proper order execution modeling
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

from src.core.events import (
    EventQueue, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    OrderType, OrderSide, SignalType
)
from src.portfolio.manager import PortfolioManager
from src.risk.risk_manager import RiskManager
from src.strategies.base import BaseStrategy
from src.data.processor import DataProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine with realistic simulation

    Features:
    - Event-driven architecture
    - Realistic slippage modeling
    - Commission costs
    - Market impact estimation
    - Partial fills
    - Position management
    - Performance tracking
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,  # 0.1%
        slippage_model: str = "fixed",
        slippage_percent: float = 0.001,  # 0.1%
        config: Optional[Dict] = None
    ):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital
            commission: Commission rate (as decimal)
            slippage_model: Slippage model ("fixed", "volume_based")
            slippage_percent: Slippage percentage
            config: Additional configuration
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_model = slippage_model
        self.slippage_percent = slippage_percent
        self.config = config or {}

        # Components
        self.event_queue = EventQueue()
        self.portfolio = PortfolioManager(initial_capital)

        # Risk manager with backtest-specific config
        risk_config = self.config.get('risk_management', {})
        risk_config['initial_capital'] = initial_capital
        self.risk_manager = RiskManager(risk_config)

        # State
        self.strategies: List[BaseStrategy] = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_time = None
        self.data_iterator = None

        # Results
        self.trades = []
        self.equity_curve = []
        self.timestamps = []

    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def set_data(self, data: Dict[str, pd.DataFrame]):
        """
        Set market data for backtesting

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
        """
        self.market_data = data
        logger.info(f"Loaded data for symbols: {list(data.keys())}")

        # Validate data
        for symbol, df in data.items():
            if df.empty:
                logger.warning(f"Empty dataframe for {symbol}")
            else:
                logger.info(f"{symbol}: {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    def _create_data_iterator(self):
        """Create iterator over market data"""
        # Combine all dataframes and sort by timestamp
        all_bars = []

        for symbol, df in self.market_data.items():
            for idx, row in df.iterrows():
                all_bars.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'data': row.to_dict()
                })

        # Sort by timestamp
        all_bars.sort(key=lambda x: x['timestamp'])

        return iter(all_bars)

    def _generate_market_events(self):
        """Generate market events from data"""
        try:
            bar = next(self.data_iterator)
            self.current_time = bar['timestamp']

            # Create market event
            market_event = MarketEvent(
                symbol=bar['symbol'],
                timeframe=bar['data'].get('timeframe', '1h'),
                open=bar['data']['open'],
                high=bar['data']['high'],
                low=bar['data']['low'],
                close=bar['data']['close'],
                volume=bar['data']['volume'],
                data=bar['data']
            )

            self.event_queue.put(market_event)
            return True

        except StopIteration:
            return False

    def _handle_market_event(self, event: MarketEvent):
        """Handle market data event"""
        # Update portfolio prices
        self.portfolio.update_prices({event.symbol: event.close})
        self.risk_manager.update_portfolio(
            event.symbol, "UPDATE", 0, event.close
        )

        # Check stop-loss and take-profit
        symbols_to_close = self.portfolio.check_stop_loss_take_profit({event.symbol: event.close})

        for symbol in symbols_to_close:
            self._close_position(symbol, event.close, reason="SL/TP")

        # Generate signals from strategies
        for strategy in self.strategies:
            if event.symbol in strategy.symbols:
                # Get historical data for this symbol
                symbol_data = self.market_data[event.symbol]

                # Get data up to current time
                historical = symbol_data[symbol_data['timestamp'] <= self.current_time].copy()

                if len(historical) >= 100:  # Minimum data for signal generation
                    signals = strategy.generate_signals(historical)

                    for signal in signals:
                        self.event_queue.put(signal)

    def _handle_signal_event(self, event: SignalEvent):
        """Handle trading signal"""
        # Get current price
        symbol_data = self.market_data[event.symbol]
        current_bar = symbol_data[symbol_data['timestamp'] <= self.current_time].iloc[-1]
        current_price = current_bar['close']

        # Risk management approval
        approved, reason, position_size = self.risk_manager.evaluate_signal(
            signal=event,
            current_price=current_price,
            market_data=symbol_data[symbol_data['timestamp'] <= self.current_time]
        )

        if not approved:
            logger.debug(f"Signal rejected by risk manager: {reason}")
            return

        # Create order
        if event.signal_type == SignalType.LONG:
            order_side = OrderSide.BUY
        elif event.signal_type == SignalType.SHORT:
            order_side = OrderSide.SELL
        elif event.signal_type == SignalType.EXIT or event.signal_type == SignalType.CLOSE:
            # Close existing position
            if event.symbol in self.portfolio.positions:
                position = self.portfolio.positions[event.symbol]
                order_side = OrderSide.SELL if position['side'] == 'LONG' else OrderSide.BUY
                position_size = position['quantity']
            else:
                return
        else:
            return

        # Create order event
        order = OrderEvent(
            symbol=event.symbol,
            order_type=OrderType.MARKET,
            side=order_side,
            quantity=position_size,
            strategy_name=event.strategy_name,
            signal_id=str(id(event))
        )

        self.event_queue.put(order)

    def _handle_order_event(self, event: OrderEvent):
        """Handle order execution"""
        # Get current price
        symbol_data = self.market_data[event.symbol]
        current_bar = symbol_data[symbol_data['timestamp'] <= self.current_time].iloc[-1]

        # Base price
        if event.side == OrderSide.BUY:
            base_price = current_bar['high']  # Conservative: buy at high
        else:
            base_price = current_bar['low']   # Conservative: sell at low

        # Apply slippage
        fill_price = self._calculate_fill_price(
            base_price=base_price,
            side=event.side,
            quantity=event.quantity,
            volume=current_bar['volume']
        )

        # Calculate commission
        commission = event.quantity * fill_price * self.commission

        # Create fill event
        fill = FillEvent(
            order_id=str(id(event)),
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
            exchange="backtest",
            timestamp=self.current_time
        )

        self.event_queue.put(fill)

    def _handle_fill_event(self, event: FillEvent):
        """Handle order fill"""
        # Update portfolio
        side_str = "BUY" if event.side == OrderSide.BUY else "SELL"

        if side_str == "BUY":
            # Open or add to position
            success = self.portfolio.open_position(
                symbol=event.symbol,
                quantity=event.quantity,
                entry_price=event.fill_price,
                side="LONG",
                strategy="backtest"
            )

            if success:
                self.risk_manager.update_portfolio(
                    event.symbol, side_str, event.quantity, event.fill_price, event.commission
                )

                logger.info(f"LONG: {event.quantity} {event.symbol} @ ${event.fill_price:.2f}")

        else:  # SELL
            # Close or reduce position
            if event.symbol in self.portfolio.positions:
                trade = self.portfolio.close_position(
                    symbol=event.symbol,
                    exit_price=event.fill_price
                )

                if trade:
                    self.trades.append(trade)
                    self.risk_manager.update_portfolio(
                        event.symbol, side_str, event.quantity, event.fill_price, event.commission
                    )

                    logger.info(f"CLOSE: {event.symbol} P&L: ${trade['pnl']:.2f} ({trade['return_pct']*100:.2f}%)")

        # Record equity
        self.equity_curve.append(self.portfolio.portfolio_value)
        self.timestamps.append(self.current_time)

    def _calculate_fill_price(
        self,
        base_price: float,
        side: OrderSide,
        quantity: float,
        volume: float
    ) -> float:
        """Calculate realistic fill price with slippage"""
        if self.slippage_model == "fixed":
            # Fixed percentage slippage
            if side == OrderSide.BUY:
                fill_price = base_price * (1 + self.slippage_percent)
            else:
                fill_price = base_price * (1 - self.slippage_percent)

        elif self.slippage_model == "volume_based":
            # Slippage based on order size relative to volume
            volume_ratio = (quantity * base_price) / (volume * base_price + 1)
            slippage = self.slippage_percent * (1 + volume_ratio * 10)  # Higher slippage for larger orders

            if side == OrderSide.BUY:
                fill_price = base_price * (1 + slippage)
            else:
                fill_price = base_price * (1 - slippage)

        else:
            fill_price = base_price

        return fill_price

    def _close_position(self, symbol: str, price: float, reason: str = ""):
        """Close a position"""
        if symbol in self.portfolio.positions:
            logger.info(f"Closing {symbol} position: {reason}")

            order = OrderEvent(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=self.portfolio.positions[symbol]['quantity'],
                strategy_name="risk_management"
            )

            self.event_queue.put(order)

    def run(self) -> Dict:
        """
        Run backtest

        Returns:
            Backtest results dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)

        if not self.market_data:
            raise ValueError("No market data loaded")

        if not self.strategies:
            raise ValueError("No strategies added")

        # Initialize
        self.data_iterator = self._create_data_iterator()
        start_time = datetime.now()

        # Main event loop
        bars_processed = 0

        while True:
            # Generate new market events
            has_more_data = self._generate_market_events()

            if not has_more_data:
                break

            # Process all events in queue
            while not self.event_queue.empty():
                event = self.event_queue.get()

                if event is None:
                    break

                # Route event to appropriate handler
                if isinstance(event, MarketEvent):
                    self._handle_market_event(event)
                elif isinstance(event, SignalEvent):
                    self._handle_signal_event(event)
                elif isinstance(event, OrderEvent):
                    self._handle_order_event(event)
                elif isinstance(event, FillEvent):
                    self._handle_fill_event(event)

            bars_processed += 1

            # Progress logging
            if bars_processed % 1000 == 0:
                logger.info(f"Processed {bars_processed} bars, Portfolio: ${self.portfolio.portfolio_value:.2f}")

        # Close any open positions at end
        final_prices = {}
        for symbol, df in self.market_data.items():
            final_prices[symbol] = df['close'].iloc[-1]

        for symbol in list(self.portfolio.positions.keys()):
            self._close_position(symbol, final_prices[symbol], reason="End of backtest")

        # Process final events
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if isinstance(event, OrderEvent):
                self._handle_order_event(event)
            elif isinstance(event, FillEvent):
                self._handle_fill_event(event)

        # Calculate results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results = self._calculate_results()
        results['bars_processed'] = bars_processed
        results['duration_seconds'] = duration

        # Log summary
        self._log_results(results)

        return results

    def _calculate_results(self) -> Dict:
        """Calculate comprehensive backtest results"""
        metrics = self.portfolio.get_performance_metrics()

        # Additional backtest-specific metrics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)

            # Average trade metrics
            avg_trade_duration = trades_df['duration'].mean() if 'duration' in trades_df else 0

            # Consecutive wins/losses
            returns = trades_df['pnl'].values
            max_consecutive_wins = self._max_consecutive(returns > 0)
            max_consecutive_losses = self._max_consecutive(returns < 0)

            # Expectancy
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            win_rate = metrics['win_rate']
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        else:
            avg_trade_duration = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            expectancy = 0

        results = {
            **metrics,
            'avg_trade_duration_hours': avg_trade_duration,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'expectancy': expectancy,
            'total_commission': sum(t.get('commission', 0) for t in self.trades) if self.trades else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps
        }

        return results

    def _max_consecutive(self, series: np.ndarray) -> int:
        """Calculate maximum consecutive True values"""
        max_count = 0
        current_count = 0

        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _log_results(self, results: Dict):
        """Log backtest results"""
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nCapital:")
        logger.info(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        logger.info(f"  Final Capital:      ${results['current_capital']:,.2f}")
        logger.info(f"  Total Return:       {results['total_return_pct']:.2f}%")

        logger.info(f"\nTrades:")
        logger.info(f"  Total Trades:       {results['total_trades']}")
        logger.info(f"  Winning Trades:     {results['winning_trades']}")
        logger.info(f"  Losing Trades:      {results['losing_trades']}")
        logger.info(f"  Win Rate:           {results['win_rate_pct']:.2f}%")

        logger.info(f"\nPerformance:")
        logger.info(f"  Profit Factor:      {results['profit_factor']:.2f}")
        logger.info(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        logger.info(f"  Sortino Ratio:      {results['sortino_ratio']:.2f}")
        logger.info(f"  Calmar Ratio:       {results['calmar_ratio']:.2f}")
        logger.info(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")

        logger.info(f"\nTrade Statistics:")
        logger.info(f"  Avg Win:            ${results['avg_win']:.2f}")
        logger.info(f"  Avg Loss:           ${results['avg_loss']:.2f}")
        logger.info(f"  Expectancy:         ${results['expectancy']:.2f}")
        logger.info(f"  Max Consecutive Wins:   {results['max_consecutive_wins']}")
        logger.info(f"  Max Consecutive Losses: {results['max_consecutive_losses']}")
        logger.info(f"  Total Commission:   ${results['total_commission']:.2f}")

        logger.info(f"\nExecution:")
        logger.info(f"  Bars Processed:     {results['bars_processed']:,}")
        logger.info(f"  Duration:           {results['duration_seconds']:.2f}s")

        logger.info("=" * 80)

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
