"""
Market Data Fetcher
Fetches historical and real-time market data using CCXT
"""
import ccxt
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import time

from src.core.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """
    Market data fetcher using CCXT
    Supports multiple exchanges and real-time data
    """

    def __init__(self, exchange_name: str = "binance"):
        """
        Initialize data fetcher

        Args:
            exchange_name: Name of exchange (binance, coinbase, etc.)
        """
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize CCXT exchange"""
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_name)

            # Initialize exchange
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })

            # Add API keys if available (for private endpoints)
            api_key = config.get(f'exchange.api_key', '')
            secret_key = config.get(f'exchange.secret_key', '')

            if api_key and secret_key:
                exchange.apiKey = api_key
                exchange.secret = secret_key
                logger.info(f"Exchange initialized with API credentials")
            else:
                logger.info(f"Exchange initialized without credentials (public data only)")

            logger.info(f"Connected to {self.exchange_name}")

            return exchange

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")

            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.info(f"Fetched {len(df)} candles for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data over a date range

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to fetch (if start_date not specified)

        Returns:
            DataFrame with historical data
        """
        try:
            # Calculate date range
            if start_date:
                start_dt = pd.to_datetime(start_date)
            else:
                start_dt = datetime.utcnow() - timedelta(days=days)

            if end_date:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = datetime.utcnow()

            logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")

            # Calculate number of candles needed
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = (end_dt - start_dt).total_seconds() / 60
            total_candles = int(total_minutes / timeframe_minutes)

            # Fetch in batches (exchanges limit candles per request)
            max_candles_per_request = 1000
            all_data = []

            current_since = int(start_dt.timestamp() * 1000)
            end_timestamp = int(end_dt.timestamp() * 1000)

            while current_since < end_timestamp:
                # Fetch batch
                batch = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=max_candles_per_request,
                    since=current_since
                )

                if not batch:
                    break

                all_data.extend(batch)

                # Update timestamp for next batch
                current_since = batch[-1][0] + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

                logger.info(f"Fetched {len(all_data)} candles so far...")

            # Convert to DataFrame
            if all_data:
                df = pd.DataFrame(
                    all_data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe

                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

                logger.info(f"Total {len(df)} candles fetched for {symbol}")

                return df
            else:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of trading pairs
            timeframe: Timeframe
            limit: Number of candles

        Returns:
            Dict of symbol -> DataFrame
        """
        data = {}

        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                data[symbol] = df

            # Rate limiting
            time.sleep(self.exchange.rateLimit / 1000)

        return data

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data

        Args:
            symbol: Trading pair

        Returns:
            Ticker data dictionary
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book

        Args:
            symbol: Trading pair
            limit: Depth limit

        Returns:
            Order book data
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        multipliers = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080,
            'M': 43200
        }

        unit = timeframe[-1]
        value = int(timeframe[:-1]) if len(timeframe) > 1 else 1

        return value * multipliers.get(unit, 60)

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            markets = self.exchange.load_markets()
            symbols = [market['symbol'] for market in markets.values() if market['active']]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        try:
            markets = self.exchange.load_markets()
            return symbol in markets and markets[symbol]['active']
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
