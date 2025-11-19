"""
Database Layer
Handles persistence of trades, positions, market data, and performance metrics
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class TradingDatabase:
    """
    Database manager for the trading bot
    Uses SQLite for simplicity, can be extended to PostgreSQL
    """

    def __init__(self, db_path: str = "data/databases/trading.db"):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    strategy TEXT,
                    signal_strength REAL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'executed',
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    strategy TEXT,
                    metadata TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    stop_price REAL,
                    status TEXT DEFAULT 'pending',
                    filled_quantity REAL DEFAULT 0,
                    avg_fill_price REAL,
                    commission REAL DEFAULT 0,
                    strategy TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    filled_at TEXT,
                    metadata TEXT
                )
            """)

            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    indicators TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, timeframe)
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    profitable_trades INTEGER,
                    current_capital REAL,
                    metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    confidence REAL,
                    target_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    reasoning TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Backtests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    config TEXT,
                    results TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")

            conn.commit()

    # ===== TRADE METHODS =====

    def insert_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Insert a trade record

        Args:
            trade_data: Trade data dictionary

        Returns:
            Trade ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (timestamp, symbol, side, quantity, price, commission,
                                  strategy, signal_strength, pnl, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('timestamp', datetime.utcnow().isoformat()),
                trade_data['symbol'],
                trade_data['side'],
                trade_data['quantity'],
                trade_data['price'],
                trade_data.get('commission', 0),
                trade_data.get('strategy', ''),
                trade_data.get('signal_strength', 0),
                trade_data.get('pnl', 0),
                trade_data.get('status', 'executed'),
                trade_data.get('metadata', '{}')
            ))
            return cursor.lastrowid

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Get trade history

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of trades to return

        Returns:
            DataFrame of trades
        """
        with self.get_connection() as conn:
            if symbol:
                query = f"SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
                df = pd.read_sql_query(query, conn, params=(symbol, limit))
            else:
                query = f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
                df = pd.read_sql_query(query, conn, params=(limit,))

            return df

    # ===== POSITION METHODS =====

    def insert_position(self, position_data: Dict[str, Any]) -> int:
        """Insert a position record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO positions (symbol, quantity, avg_entry_price, current_price,
                                     unrealized_pnl, realized_pnl, status, opened_at, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data['symbol'],
                position_data['quantity'],
                position_data['avg_entry_price'],
                position_data.get('current_price', position_data['avg_entry_price']),
                position_data.get('unrealized_pnl', 0),
                position_data.get('realized_pnl', 0),
                position_data.get('status', 'open'),
                position_data.get('opened_at', datetime.utcnow().isoformat()),
                position_data.get('strategy', ''),
                position_data.get('metadata', '{}')
            ))
            return cursor.lastrowid

    def update_position(self, position_id: int, update_data: Dict[str, Any]):
        """Update a position record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
            values = list(update_data.values()) + [position_id]

            cursor.execute(f"""
                UPDATE positions
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, values)

    def get_open_positions(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get all open positions"""
        with self.get_connection() as conn:
            if symbol:
                query = "SELECT * FROM positions WHERE status = 'open' AND symbol = ?"
                df = pd.read_sql_query(query, conn, params=(symbol,))
            else:
                query = "SELECT * FROM positions WHERE status = 'open'"
                df = pd.read_sql_query(query, conn)

            return df

    # ===== ORDER METHODS =====

    def insert_order(self, order_data: Dict[str, Any]) -> int:
        """Insert an order record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO orders (order_id, symbol, side, order_type, quantity, price,
                                  stop_price, status, strategy, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_data.get('order_id', ''),
                order_data['symbol'],
                order_data['side'],
                order_data['order_type'],
                order_data['quantity'],
                order_data.get('price'),
                order_data.get('stop_price'),
                order_data.get('status', 'pending'),
                order_data.get('strategy', ''),
                order_data.get('created_at', datetime.utcnow().isoformat()),
                order_data.get('metadata', '{}')
            ))
            return cursor.lastrowid

    def update_order(self, order_id: str, update_data: Dict[str, Any]):
        """Update an order record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
            values = list(update_data.values()) + [order_id]

            cursor.execute(f"""
                UPDATE orders
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE order_id = ?
            """, values)

    # ===== MARKET DATA METHODS =====

    def insert_market_data(self, market_data: Dict[str, Any]):
        """Insert market data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_data (timestamp, symbol, timeframe, open, high, low, close, volume, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_data['timestamp'],
                market_data['symbol'],
                market_data['timeframe'],
                market_data['open'],
                market_data['high'],
                market_data['low'],
                market_data['close'],
                market_data['volume'],
                market_data.get('indicators', '{}')
            ))

    def get_market_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Get market data"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM market_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            return df

    # ===== PERFORMANCE METHODS =====

    def insert_performance(self, perf_data: Dict[str, Any]) -> int:
        """Insert performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance (timestamp, total_return, sharpe_ratio, sortino_ratio,
                                       max_drawdown, win_rate, profit_factor, total_trades,
                                       profitable_trades, current_capital, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                perf_data.get('timestamp', datetime.utcnow().isoformat()),
                perf_data.get('total_return', 0),
                perf_data.get('sharpe_ratio', 0),
                perf_data.get('sortino_ratio', 0),
                perf_data.get('max_drawdown', 0),
                perf_data.get('win_rate', 0),
                perf_data.get('profit_factor', 0),
                perf_data.get('total_trades', 0),
                perf_data.get('profitable_trades', 0),
                perf_data.get('current_capital', 0),
                perf_data.get('metrics', '{}')
            ))
            return cursor.lastrowid

    def get_performance_history(self, limit: int = 100) -> pd.DataFrame:
        """Get performance history"""
        with self.get_connection() as conn:
            query = "SELECT * FROM performance ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df

    # ===== SIGNAL METHODS =====

    def insert_signal(self, signal_data: Dict[str, Any]) -> int:
        """Insert a trading signal"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (timestamp, symbol, strategy, signal_type, strength,
                                   confidence, target_price, stop_loss, take_profit, reasoning, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('timestamp', datetime.utcnow().isoformat()),
                signal_data['symbol'],
                signal_data['strategy'],
                signal_data['signal_type'],
                signal_data.get('strength', 0),
                signal_data.get('confidence', 0),
                signal_data.get('target_price'),
                signal_data.get('stop_loss'),
                signal_data.get('take_profit'),
                signal_data.get('reasoning', ''),
                signal_data.get('metadata', '{}')
            ))
            return cursor.lastrowid

    # ===== BACKTEST METHODS =====

    def insert_backtest(self, backtest_data: Dict[str, Any]) -> int:
        """Insert backtest results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtests (name, strategy, start_date, end_date, initial_capital,
                                     final_capital, total_return, sharpe_ratio, max_drawdown,
                                     win_rate, total_trades, config, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_data['name'],
                backtest_data['strategy'],
                backtest_data['start_date'],
                backtest_data['end_date'],
                backtest_data['initial_capital'],
                backtest_data.get('final_capital'),
                backtest_data.get('total_return'),
                backtest_data.get('sharpe_ratio'),
                backtest_data.get('max_drawdown'),
                backtest_data.get('win_rate'),
                backtest_data.get('total_trades'),
                backtest_data.get('config', '{}'),
                backtest_data.get('results', '{}')
            ))
            return cursor.lastrowid

    def get_backtest_results(self, limit: int = 10) -> pd.DataFrame:
        """Get backtest results"""
        with self.get_connection() as conn:
            query = "SELECT * FROM backtests ORDER BY created_at DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
