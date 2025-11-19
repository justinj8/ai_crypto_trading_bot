"""
Order Execution Engine
Smart order routing, execution algorithms (TWAP, VWAP), and order management
"""
import ccxt
import time
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

from src.core.events import OrderEvent, FillEvent, OrderType, OrderSide
from src.core.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price
    ICEBERG = "ICEBERG"  # Hide order size
    POV = "POV"  # Percentage of Volume


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderExecutor:
    """
    Sophisticated order execution engine

    Features:
    - Smart order routing
    - Multiple execution algorithms (TWAP, VWAP)
    - Retry logic with exponential backoff
    - Order state tracking
    - Slippage monitoring
    - Market impact estimation
    """

    def __init__(self, exchange_name: str = "binance", mode: str = "paper"):
        """
        Initialize order executor

        Args:
            exchange_name: Exchange name
            mode: Trading mode ("paper" or "live")
        """
        self.exchange_name = exchange_name
        self.mode = mode
        self.exchange = self._initialize_exchange()

        # Order tracking
        self.active_orders = {}  # order_id -> order info
        self.order_history = []

        # Execution parameters
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.max_slippage = config.get('execution.max_slippage', 0.005)  # 0.5%

        logger.info(f"Order Executor initialized in {mode} mode")

    def _initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            # Add credentials for live trading
            if self.mode == "live":
                api_key = config.get('exchange.api_key')
                secret_key = config.get('exchange.secret_key')

                if not api_key or not secret_key:
                    raise ValueError("API credentials required for live trading")

                exchange.apiKey = api_key
                exchange.secret = secret_key

                logger.info("Exchange initialized with API credentials")

            return exchange

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    def execute_order(
        self,
        order: OrderEvent,
        algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    ) -> Optional[FillEvent]:
        """
        Execute an order using specified algorithm

        Args:
            order: Order to execute
            algorithm: Execution algorithm

        Returns:
            FillEvent if successful, None otherwise
        """
        logger.info(f"Executing order: {order.side.value} {order.quantity} {order.symbol}")

        # Check if paper trading
        if self.mode == "paper":
            return self._execute_paper_order(order)

        # Route to appropriate execution algorithm
        if algorithm == ExecutionAlgorithm.MARKET:
            return self._execute_market_order(order)

        elif algorithm == ExecutionAlgorithm.LIMIT:
            return self._execute_limit_order(order)

        elif algorithm == ExecutionAlgorithm.TWAP:
            return self._execute_twap_order(order)

        elif algorithm == ExecutionAlgorithm.VWAP:
            return self._execute_vwap_order(order)

        else:
            logger.error(f"Unsupported execution algorithm: {algorithm}")
            return None

    def _execute_paper_order(self, order: OrderEvent) -> FillEvent:
        """Execute order in paper trading mode"""
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(order.symbol)
            current_price = ticker['last']

            # Simulate slippage
            if order.side == OrderSide.BUY:
                fill_price = current_price * (1 + 0.001)  # 0.1% slippage
            else:
                fill_price = current_price * (1 - 0.001)

            # Calculate commission
            commission = order.quantity * fill_price * 0.001  # 0.1% commission

            # Create fill event
            fill = FillEvent(
                order_id=str(id(order)),
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                exchange=self.exchange_name,
                timestamp=datetime.utcnow()
            )

            logger.info(f"Paper trade executed: {order.side.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}")

            return fill

        except Exception as e:
            logger.error(f"Error executing paper order: {e}")
            return None

    def _execute_market_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """Execute market order with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Get current price for slippage check
                ticker = self.exchange.fetch_ticker(order.symbol)
                expected_price = ticker['last']

                # Place market order
                side = 'buy' if order.side == OrderSide.BUY else 'sell'

                exchange_order = self.exchange.create_market_order(
                    symbol=order.symbol,
                    side=side,
                    amount=order.quantity
                )

                # Wait for fill
                time.sleep(1)

                # Fetch order details
                order_info = self.exchange.fetch_order(
                    id=exchange_order['id'],
                    symbol=order.symbol
                )

                # Check if filled
                if order_info['status'] == 'closed':
                    fill_price = float(order_info['average'])
                    filled_qty = float(order_info['filled'])
                    commission = float(order_info.get('fee', {}).get('cost', 0))

                    # Check slippage
                    slippage = abs(fill_price - expected_price) / expected_price

                    if slippage > self.max_slippage:
                        logger.warning(f"High slippage detected: {slippage*100:.2f}%")

                    # Create fill event
                    fill = FillEvent(
                        order_id=exchange_order['id'],
                        symbol=order.symbol,
                        side=order.side,
                        quantity=filled_qty,
                        fill_price=fill_price,
                        commission=commission,
                        exchange=self.exchange_name,
                        timestamp=datetime.utcnow()
                    )

                    logger.info(f"Order filled: {side.upper()} {filled_qty} {order.symbol} @ ${fill_price:.2f}")

                    return fill

                else:
                    logger.warning(f"Order not filled, status: {order_info['status']}")

            except ccxt.NetworkError as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to execute order after {self.max_retries} attempts")

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        return None

    def _execute_limit_order(
        self,
        order: OrderEvent,
        timeout_seconds: int = 60
    ) -> Optional[FillEvent]:
        """Execute limit order with timeout"""
        try:
            # Determine limit price
            if order.price:
                limit_price = order.price
            else:
                # Use bid/ask for better execution
                orderbook = self.exchange.fetch_order_book(order.symbol, limit=1)

                if order.side == OrderSide.BUY:
                    limit_price = orderbook['bids'][0][0] if orderbook['bids'] else None
                else:
                    limit_price = orderbook['asks'][0][0] if orderbook['asks'] else None

                if not limit_price:
                    logger.error("Could not determine limit price")
                    return None

            # Place limit order
            side = 'buy' if order.side == OrderSide.BUY else 'sell'

            exchange_order = self.exchange.create_limit_order(
                symbol=order.symbol,
                side=side,
                amount=order.quantity,
                price=limit_price
            )

            logger.info(f"Limit order placed: {side.upper()} {order.quantity} {order.symbol} @ ${limit_price:.2f}")

            # Wait for fill with timeout
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                time.sleep(2)

                order_info = self.exchange.fetch_order(
                    id=exchange_order['id'],
                    symbol=order.symbol
                )

                if order_info['status'] == 'closed':
                    # Order filled
                    fill_price = float(order_info['average'])
                    filled_qty = float(order_info['filled'])
                    commission = float(order_info.get('fee', {}).get('cost', 0))

                    fill = FillEvent(
                        order_id=exchange_order['id'],
                        symbol=order.symbol,
                        side=order.side,
                        quantity=filled_qty,
                        fill_price=fill_price,
                        commission=commission,
                        exchange=self.exchange_name,
                        timestamp=datetime.utcnow()
                    )

                    logger.info(f"Limit order filled: {side.upper()} {filled_qty} {order.symbol} @ ${fill_price:.2f}")

                    return fill

            # Timeout - cancel order
            logger.warning(f"Limit order timeout, cancelling...")
            self.exchange.cancel_order(id=exchange_order['id'], symbol=order.symbol)

        except Exception as e:
            logger.error(f"Error executing limit order: {e}")

        return None

    def _execute_twap_order(
        self,
        order: OrderEvent,
        duration_minutes: int = 30,
        num_orders: int = 10
    ) -> Optional[FillEvent]:
        """
        Execute TWAP (Time-Weighted Average Price) order

        Splits order into equal parts over time

        Args:
            order: Order to execute
            duration_minutes: Duration to spread order over
            num_orders: Number of child orders

        Returns:
            Aggregated fill event
        """
        logger.info(f"Executing TWAP order: {num_orders} orders over {duration_minutes} minutes")

        interval_seconds = (duration_minutes * 60) / num_orders
        child_quantity = order.quantity / num_orders

        fills = []
        total_quantity = 0
        weighted_price = 0
        total_commission = 0

        for i in range(num_orders):
            # Create child order
            child_order = OrderEvent(
                symbol=order.symbol,
                order_type=OrderType.MARKET,
                side=order.side,
                quantity=child_quantity,
                strategy_name=order.strategy_name
            )

            # Execute
            fill = self._execute_market_order(child_order)

            if fill:
                fills.append(fill)
                total_quantity += fill.quantity
                weighted_price += fill.fill_price * fill.quantity
                total_commission += fill.commission

                logger.info(f"TWAP child {i+1}/{num_orders} filled: {fill.quantity} @ ${fill.fill_price:.2f}")

            # Wait for next interval (except on last iteration)
            if i < num_orders - 1:
                time.sleep(interval_seconds)

        # Create aggregated fill
        if fills:
            avg_price = weighted_price / total_quantity

            aggregated_fill = FillEvent(
                order_id=f"TWAP_{id(order)}",
                symbol=order.symbol,
                side=order.side,
                quantity=total_quantity,
                fill_price=avg_price,
                commission=total_commission,
                exchange=self.exchange_name,
                timestamp=datetime.utcnow()
            )

            logger.info(f"TWAP order completed: {total_quantity} @ ${avg_price:.2f} (avg)")

            return aggregated_fill

        return None

    def _execute_vwap_order(
        self,
        order: OrderEvent,
        lookback_bars: int = 20
    ) -> Optional[FillEvent]:
        """
        Execute VWAP (Volume-Weighted Average Price) order

        Adjusts order size based on recent volume patterns

        Args:
            order: Order to execute
            lookback_bars: Number of bars to analyze

        Returns:
            Fill event
        """
        logger.info(f"Executing VWAP order with {lookback_bars} bar lookback")

        # Fetch recent candles to analyze volume
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=order.symbol,
                timeframe='5m',
                limit=lookback_bars
            )

            # Calculate volume profile
            volumes = [c[5] for c in candles]
            avg_volume = np.mean(volumes)
            current_volume = volumes[-1]

            # Adjust execution based on current volume
            volume_ratio = current_volume / avg_volume

            if volume_ratio > 1.5:
                # High volume - execute more aggressively
                logger.info(f"High volume detected ({volume_ratio:.2f}x), executing full order")
                return self._execute_market_order(order)

            elif volume_ratio < 0.5:
                # Low volume - execute conservatively with limit orders
                logger.info(f"Low volume detected ({volume_ratio:.2f}x), using limit order")
                return self._execute_limit_order(order)

            else:
                # Normal volume - standard execution
                return self._execute_market_order(order)

        except Exception as e:
            logger.error(f"Error in VWAP execution: {e}")
            # Fallback to market order
            return self._execute_market_order(order)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            self.exchange.cancel_order(id=order_id, symbol=symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get order status"""
        try:
            order_info = self.exchange.fetch_order(id=order_id, symbol=symbol)
            return order_info

        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        try:
            if symbol:
                orders = self.exchange.fetch_open_orders(symbol=symbol)
            else:
                orders = self.exchange.fetch_open_orders()

            return orders

        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    def estimate_market_impact(
        self,
        symbol: str,
        quantity: float,
        side: str
    ) -> float:
        """
        Estimate market impact of an order

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: 'buy' or 'sell'

        Returns:
            Estimated price impact as percentage
        """
        try:
            # Fetch order book
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)

            # Calculate order value
            ticker = self.exchange.fetch_ticker(symbol)
            order_value = quantity * ticker['last']

            # Analyze order book depth
            if side == 'buy':
                levels = orderbook['asks']
            else:
                levels = orderbook['bids']

            # Calculate cumulative volume and weighted price
            cumulative_volume = 0
            cumulative_value = 0

            for price, volume in levels:
                if cumulative_volume >= quantity:
                    break

                qty_at_level = min(volume, quantity - cumulative_volume)
                cumulative_volume += qty_at_level
                cumulative_value += qty_at_level * price

            if cumulative_volume > 0:
                avg_fill_price = cumulative_value / cumulative_volume
                current_price = ticker['last']
                impact = abs(avg_fill_price - current_price) / current_price

                return impact
            else:
                # Not enough liquidity
                return 0.1  # 10% estimated impact

        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return 0.01  # Default 1% impact

    def check_sufficient_balance(self, symbol: str, quantity: float, side: str) -> bool:
        """Check if account has sufficient balance"""
        if self.mode == "paper":
            return True  # Paper trading always has balance

        try:
            balance = self.exchange.fetch_balance()

            if side == 'buy':
                # Need quote currency (e.g., USDT for BTC/USDT)
                quote = symbol.split('/')[1]
                ticker = self.exchange.fetch_ticker(symbol)
                required = quantity * ticker['last']
                available = balance[quote]['free']

                return available >= required

            else:
                # Need base currency (e.g., BTC for BTC/USDT)
                base = symbol.split('/')[0]
                available = balance[base]['free']

                return available >= quantity

        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return False
