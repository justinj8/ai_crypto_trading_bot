"""
Configuration Management System
Loads and manages configuration from YAML files and environment variables
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """Central configuration manager for the trading bot"""

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern to ensure single config instance"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration"""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from YAML files"""
        # Load environment variables
        load_dotenv()

        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"

        # Load main configuration
        config_file = config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # Load risk limits
        risk_file = config_dir / "risk_limits.yaml"
        if risk_file.exists():
            with open(risk_file, 'r') as f:
                self._config['risk_limits'] = yaml.safe_load(f)

        # Process environment variable references
        self._process_env_vars()

        # Set project paths
        self._config['paths'] = {
            'root': str(project_root),
            'config': str(config_dir),
            'data': str(project_root / 'data'),
            'models': str(project_root / 'data' / 'models'),
            'databases': str(project_root / 'data' / 'databases'),
            'logs': str(project_root / 'logs'),
        }

    def _process_env_vars(self):
        """Process configuration values that reference environment variables"""
        def process_dict(d: Dict) -> Dict:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = process_dict(value)
                elif isinstance(value, str) and value.endswith('_env'):
                    # This is an environment variable reference
                    env_var = value.replace('_env', '')
                    d[key.replace('_env', '')] = os.getenv(env_var, '')
            return d

        self._config = process_dict(self._config)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., 'exchange.name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def get_all(self) -> Dict:
        """Get entire configuration dictionary"""
        return self._config.copy()

    def reload(self):
        """Reload configuration from files"""
        self._config = None
        self._load_config()

    # Convenience methods for common config values
    @property
    def mode(self) -> str:
        """Get trading mode (paper/live/backtest)"""
        return self.get('mode', 'paper')

    @property
    def exchange_name(self) -> str:
        """Get exchange name"""
        return self.get('exchange.name', 'binance')

    @property
    def symbols(self) -> list:
        """Get list of trading symbols"""
        return self.get('symbols', ['BTC/USDT'])

    @property
    def primary_timeframe(self) -> str:
        """Get primary timeframe"""
        return self.get('timeframes.primary', '1h')

    @property
    def database_path(self) -> str:
        """Get database path"""
        return self.get('database.path', 'data/databases/trading.db')

    @property
    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode"""
        return self.mode == 'paper'

    @property
    def is_live_trading(self) -> bool:
        """Check if in live trading mode"""
        return self.mode == 'live'

    @property
    def is_backtesting(self) -> bool:
        """Check if in backtesting mode"""
        return self.mode == 'backtest'

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if configuration is valid
        """
        errors = []

        # Check required API keys for live trading
        if self.is_live_trading:
            api_key = self.get('exchange.api_key')
            secret_key = self.get('exchange.secret_key')

            if not api_key or not secret_key:
                errors.append("API keys required for live trading")

        # Check symbols
        if not self.symbols:
            errors.append("At least one symbol must be configured")

        # Check risk limits
        max_risk = self.get('risk_management.max_portfolio_risk', 0)
        if max_risk <= 0 or max_risk > 1:
            errors.append("Invalid max_portfolio_risk value")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return True


# Global config instance
config = Config()
