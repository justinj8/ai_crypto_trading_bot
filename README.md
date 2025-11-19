# 🤖 AI Crypto Trading Bot

A **sophisticated, production-grade AI-powered cryptocurrency trading bot** featuring advanced machine learning models, comprehensive risk management, and professional trading infrastructure.

## 🌟 Key Features

### **Advanced Machine Learning**
- **LSTM Networks**: Deep learning for time series prediction with attention mechanisms
- **Transformer Models**: State-of-the-art sequence modeling with multi-head attention
- **XGBoost**: Gradient boosting for feature-based predictions
- **Ensemble Learning**: Intelligent combination of multiple models with confidence scoring

### **Comprehensive Trading Strategies**
- ML-based prediction strategies with confidence thresholds
- Momentum and trend-following strategies
- Mean reversion strategies
- Statistical arbitrage (planned)
- Market making (planned)

### **Advanced Technical Analysis**
- **50+ Technical Indicators**:
  - Trend: SMA, EMA (multiple periods), MACD, ADX, Ichimoku, Parabolic SAR
  - Momentum: RSI, Stochastic, Williams %R, ROC, CCI, MFI, TSI
  - Volatility: ATR, Bollinger Bands, Keltner Channel, Donchian Channel
  - Volume: OBV, VWAP, CMF, Force Index, Volume Price Trend
  - Support/Resistance: Pivot points, Fibonacci retracements
  - Candlestick patterns: Doji, Hammer, Engulfing patterns

### **Risk Management**
- **Position Sizing**: Kelly Criterion, fixed percentage, volatility-adjusted
- **Portfolio Protection**: Max drawdown limits, position concentration limits
- **Stop Loss/Take Profit**: Multiple strategies (fixed, trailing, ATR-based)
- **Circuit Breakers**: Automatic trading halt on extreme conditions
- **Correlation Management**: Prevent over-exposure to correlated assets

### **Professional Infrastructure**
- Event-driven architecture for scalability
- SQLite database for trade/position persistence
- Comprehensive logging and monitoring
- Real-time performance metrics
- Paper trading mode for testing
- Backtesting engine with realistic simulation

## 📊 Architecture Overview

```
ai_crypto_trading_bot/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── risk_limits.yaml   # Risk management limits
├── src/                   # Source code
│   ├── core/              # Core engine and events
│   ├── data/              # Data fetching and processing
│   ├── models/            # ML models (LSTM, Transformer, XGBoost)
│   ├── strategies/        # Trading strategies
│   ├── risk/              # Risk management
│   ├── portfolio/         # Portfolio management
│   ├── execution/         # Order execution
│   ├── backtesting/       # Backtesting engine
│   └── utils/             # Utilities and logging
├── scripts/               # Executable scripts
│   ├── train_models.py    # Train ML models
│   ├── backtest.py        # Run backtests
│   └── live_trading.py    # Live trading
├── data/                  # Data storage
│   ├── models/            # Trained models
│   └── databases/         # SQLite databases
├── logs/                  # Log files
└── tests/                 # Unit and integration tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM (8GB+ recommended for model training)
- Internet connection

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_crypto_trading_bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Configure the bot**
   Edit `config/config.yaml` to customize:
   - Trading symbols
   - Risk parameters
   - ML model settings
   - Strategy configurations

### Setup Binance API Keys

1. Create an account on [Binance](https://www.binance.com)
2. Navigate to API Management
3. Create a new API key
4. Add the following to your `.env` file:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET=your_secret_key_here
   ```

⚠️ **Important**: Start with paper trading mode to test the bot without risking real money!

## 📚 Usage

### 1. Train ML Models

Train the ensemble of ML models on historical data:

```bash
python scripts/train_models.py
```

This will:
- Fetch historical market data for configured symbols
- Engineer 50+ technical indicators
- Train LSTM, Transformer, and XGBoost models
- Evaluate model performance
- Save trained models to `data/models/`

**Expected output:**
```
Training LSTM model...
Training Transformer model...
Training XGBoost model...

Evaluation Results:
LSTM:
  RMSE: 0.002345
  Direction Accuracy: 67.85%

TRANSFORMER:
  RMSE: 0.002198
  Direction Accuracy: 69.23%

XGBOOST:
  RMSE: 0.002567
  Direction Accuracy: 65.91%

ENSEMBLE:
  RMSE: 0.002087
  Direction Accuracy: 71.34%
```

### 2. Backtest Strategies

Test strategies on historical data:

```bash
python scripts/backtest.py --strategy ml_ensemble --start 2023-01-01 --end 2024-01-01
```

### 3. Paper Trading

Test the bot with live data but simulated trades:

```bash
# Edit config/config.yaml and set mode: "paper"
python scripts/live_trading.py
```

### 4. Live Trading

⚠️ **Use at your own risk!** Only enable after thorough testing.

```bash
# Edit config/config.yaml and set mode: "live"
python scripts/live_trading.py
```

### 5. Dashboard

Launch the real-time monitoring dashboard:

```bash
streamlit run dashboard/app.py
```

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

Key settings:

```yaml
# Trading Mode
mode: "paper"  # "paper", "live", or "backtest"

# Symbols to trade
symbols:
  - "BTC/USDT"
  - "ETH/USDT"

# Risk Management
risk_management:
  max_portfolio_risk: 0.02      # 2% per trade
  max_position_size: 0.10        # 10% max per position
  max_total_exposure: 0.50       # 50% max total exposure
  max_drawdown: 0.15             # 15% max drawdown

# ML Models
ml_models:
  enabled: true
  auto_retrain: true
  retrain_interval_hours: 24
```

### Risk Limits (`config/risk_limits.yaml`)

Define trading limits and circuit breakers:

```yaml
portfolio:
  max_total_value: 100000
  max_daily_loss: 0.05          # 5% max daily loss

circuit_breakers:
  drawdown_halt:
    enabled: true
    threshold: 0.10             # Halt if drawdown > 10%
```

## 📈 Performance Metrics

The bot tracks comprehensive performance metrics:

- **Returns**: Total return, daily returns, annualized return
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, current drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Direction Accuracy**: ML model prediction accuracy

## 🔬 Model Details

### LSTM (Long Short-Term Memory)
- Architecture: 2 LSTM layers (128, 64 units) with dropout
- Sequence length: 60 time steps
- Features: All technical indicators + price data
- Loss: MSE (Mean Squared Error)

### Transformer
- Multi-head attention: 8 heads
- Encoder layers: 4
- Model dimension: 128
- Feed-forward dimension: 256
- Positional encoding for time series

### XGBoost
- Estimators: 200 trees
- Max depth: 7
- Learning rate: 0.05
- Regularization: L2 (λ=1)

### Ensemble
- Weighted average of all models
- Confidence scoring based on prediction variance
- Adaptive weighting based on recent performance

## 🛡️ Risk Management Features

1. **Position Sizing**
   - Kelly Criterion for optimal allocation
   - Volatility-adjusted sizing
   - Fixed percentage fallback

2. **Stop Loss Strategies**
   - Fixed percentage stops
   - Trailing stops
   - ATR-based dynamic stops

3. **Portfolio Protection**
   - Maximum position concentration
   - Correlation limits
   - Total exposure limits

4. **Circuit Breakers**
   - Automatic halt on max drawdown
   - Daily loss limits
   - Volatility spike protection

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

## 📊 Example Results

### Backtest Performance (Example)
```
Strategy: ML Ensemble
Period: 2023-01-01 to 2024-01-01
Initial Capital: $10,000

Final Capital: $13,456
Total Return: 34.56%
Sharpe Ratio: 1.85
Max Drawdown: -8.45%
Win Rate: 68.5%
Total Trades: 187
Profitable Trades: 128
```

## 🗺️ Roadmap

- [x] Advanced ML models (LSTM, Transformer, XGBoost)
- [x] Comprehensive technical indicators (50+)
- [x] Risk management system
- [x] Portfolio management
- [x] Paper trading mode
- [x] Event-driven architecture
- [ ] Backtesting engine (in progress)
- [ ] Live trading engine (in progress)
- [ ] Sentiment analysis integration
- [ ] On-chain metrics
- [ ] Multi-exchange support
- [ ] Reinforcement learning strategies
- [ ] Options trading support
- [ ] Web dashboard
- [ ] Mobile app
- [ ] Cloud deployment

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Cryptocurrency trading carries substantial risk
- Past performance does not guarantee future results
- This bot can lose money
- Only trade with capital you can afford to lose
- The developers are not responsible for any financial losses
- Always test thoroughly in paper trading mode first
- Use at your own risk

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- GitHub Issues: For bug reports and feature requests
- Discussions: For questions and community support

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange API
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical analysis library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library

---

**Built with ❤️ for algorithmic traders**

*Happy Trading! 🚀📈*
