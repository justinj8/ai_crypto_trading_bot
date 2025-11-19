"""
Logging System
Provides comprehensive logging for the trading bot
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        record.name = f"{self.COLORS['RESET']}{record.name}{self.COLORS['RESET']}"
        return super().format(record)


class TradingLogger:
    """
    Central logging system for the trading bot
    Supports console and file logging with rotation
    """

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_dir: str = "logs",
        rotation: str = "daily"
    ) -> logging.Logger:
        """
        Get or create a logger

        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Enable file logging
            log_to_console: Enable console logging
            log_dir: Directory for log files
            rotation: Log rotation strategy ("daily", "size")

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers = []  # Clear any existing handlers

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        colored_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(colored_formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            # Create log directory
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Choose rotation strategy
            if rotation == "daily":
                file_handler = TimedRotatingFileHandler(
                    filename=log_path / f"{name}.log",
                    when='midnight',
                    interval=1,
                    backupCount=30,
                    encoding='utf-8'
                )
            else:  # size-based rotation
                file_handler = RotatingFileHandler(
                    filename=log_path / f"{name}.log",
                    maxBytes=10 * 1024 * 1024,  # 10 MB
                    backupCount=10,
                    encoding='utf-8'
                )

            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

            # Create separate error log
            error_handler = RotatingFileHandler(
                filename=log_path / f"{name}_errors.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        cls._loggers[name] = logger
        return logger


class TradeLogger:
    """
    Specialized logger for trade execution
    Logs all trade-related activities to a separate file
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize trade logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create trade log file with timestamp
        self.trade_log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"

        # Create CSV header if file doesn't exist
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w') as f:
                f.write("timestamp,symbol,side,quantity,price,commission,strategy,signal_strength,pnl,status\n")

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        strategy: str = "",
        signal_strength: float = 0.0,
        pnl: float = 0.0,
        status: str = "executed"
    ):
        """
        Log a trade to CSV file

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Execution price
            commission: Trading commission
            strategy: Strategy name
            signal_strength: Signal strength (0-1)
            pnl: Profit/Loss
            status: Trade status
        """
        timestamp = datetime.utcnow().isoformat()

        with open(self.trade_log_file, 'a') as f:
            f.write(f"{timestamp},{symbol},{side},{quantity},{price},{commission},"
                   f"{strategy},{signal_strength},{pnl},{status}\n")


class PerformanceLogger:
    """
    Specialized logger for performance metrics
    Logs performance snapshots at regular intervals
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize performance logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create performance log file
        self.perf_log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.csv"

        # Create CSV header if file doesn't exist
        if not self.perf_log_file.exists():
            with open(self.perf_log_file, 'w') as f:
                f.write("timestamp,total_return,sharpe_ratio,sortino_ratio,max_drawdown,"
                       "win_rate,profit_factor,total_trades,current_capital\n")

    def log_performance(
        self,
        total_return: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        total_trades: int,
        current_capital: float
    ):
        """
        Log performance metrics

        Args:
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio
            sortino_ratio: Sortino ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate percentage
            profit_factor: Profit factor
            total_trades: Total number of trades
            current_capital: Current capital
        """
        timestamp = datetime.utcnow().isoformat()

        with open(self.perf_log_file, 'a') as f:
            f.write(f"{timestamp},{total_return},{sharpe_ratio},{sortino_ratio},"
                   f"{max_drawdown},{win_rate},{profit_factor},{total_trades},{current_capital}\n")


# Global logger instances
def setup_logging(config_dict: Optional[dict] = None):
    """
    Setup logging system with configuration

    Args:
        config_dict: Configuration dictionary
    """
    if config_dict is None:
        config_dict = {}

    log_level = config_dict.get('level', 'INFO')
    log_dir = config_dict.get('log_dir', 'logs')
    rotation = config_dict.get('rotation', 'daily')

    # Create main logger
    main_logger = TradingLogger.get_logger(
        'trading_bot',
        log_level=log_level,
        log_dir=log_dir,
        rotation=rotation
    )

    main_logger.info("=" * 80)
    main_logger.info("AI Crypto Trading Bot - Logging System Initialized")
    main_logger.info("=" * 80)

    return main_logger


# Convenience function to get logger
def get_logger(name: str = "trading_bot") -> logging.Logger:
    """Get logger by name"""
    return TradingLogger.get_logger(name)
