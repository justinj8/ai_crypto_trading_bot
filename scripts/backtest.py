"""
Backtesting Script
Run backtests on historical data with various strategies
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.backtesting.engine import BacktestEngine
from src.strategies.ml_strategy import MLEnsembleStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.portfolio.analytics import PerformanceAnalytics
from src.core.config import config
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging({'level': 'INFO', 'log_dir': 'logs'})
logger = get_logger('backtest')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest trading strategies')

    parser.add_argument(
        '--strategy',
        type=str,
        default='ml_ensemble',
        choices=['ml_ensemble', 'momentum', 'mean_reversion', 'all'],
        help='Strategy to backtest'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTC/USDT'],
        help='Symbols to trade'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2024-01-01',
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d)'
    )

    return parser.parse_args()


def fetch_data(symbols, timeframe, start_date, end_date):
    """Fetch historical data for backtesting"""
    logger.info(f"Fetching data for {symbols}")

    fetcher = DataFetcher(exchange_name='binance')
    all_data = {}

    for symbol in symbols:
        logger.info(f"Fetching {symbol}...")

        df = fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if not df.empty:
            # Process data (add technical indicators)
            processor = DataProcessor()
            df_processed = processor.process(df, add_all_features=True)

            all_data[symbol] = df_processed
            logger.info(f"Loaded {len(df_processed)} bars for {symbol}")
        else:
            logger.warning(f"No data fetched for {symbol}")

    return all_data


def run_backtest(strategy_name, data, initial_capital, commission):
    """Run backtest for a specific strategy"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Backtesting: {strategy_name.upper()}")
    logger.info(f"{'='*80}")

    # Initialize strategy
    symbols = list(data.keys())

    if strategy_name == 'ml_ensemble':
        strategy = MLEnsembleStrategy(symbols=symbols)

        # Load pre-trained models
        try:
            strategy.load_models('data/models')
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
            logger.warning("ML strategy requires trained models. Run train_models.py first.")
            return None

    elif strategy_name == 'momentum':
        strategy = MomentumStrategy(symbols=symbols)

    elif strategy_name == 'mean_reversion':
        strategy = MeanReversionStrategy(symbols=symbols)

    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return None

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage_model='fixed',
        slippage_percent=0.001
    )

    # Add strategy and data
    engine.add_strategy(strategy)
    engine.set_data(data)

    # Run backtest
    start_time = datetime.now()
    results = engine.run()
    end_time = datetime.now()

    logger.info(f"\nBacktest completed in {(end_time - start_time).total_seconds():.2f}s")

    return results, engine


def analyze_results(results, engine):
    """Analyze and visualize backtest results"""
    # Get equity curve
    equity_df = engine.get_equity_curve()

    if equity_df.empty:
        logger.warning("No equity data to analyze")
        return

    # Advanced analytics
    analytics = PerformanceAnalytics(risk_free_rate=0.02)

    if not engine.trades:
        logger.warning("No trades executed during backtest")
        trades_df = pd.DataFrame()
    else:
        trades_df = pd.DataFrame(engine.trades)

    # Calculate comprehensive metrics
    metrics = analytics.calculate_comprehensive_metrics(
        equity_curve=equity_df['equity'],
        trades=trades_df
    )

    # Generate report
    report = analytics.generate_performance_report(metrics, format='text')
    print("\n" + report)

    # Plot equity curve
    plot_equity_curve(equity_df, engine.initial_capital)

    # Plot drawdown
    plot_drawdown(equity_df['equity'])

    # Plot trades if available
    if not trades_df.empty:
        plot_trade_distribution(trades_df)

    return metrics


def plot_equity_curve(equity_df, initial_capital):
    """Plot equity curve"""
    plt.figure(figsize=(12, 6))

    plt.plot(equity_df['timestamp'], equity_df['equity'], label='Portfolio Value', linewidth=2)
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital', alpha=0.7)

    plt.title('Equity Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('logs/equity_curve.png', dpi=150)
    logger.info("Equity curve saved to logs/equity_curve.png")

    plt.close()


def plot_drawdown(equity_series):
    """Plot drawdown chart"""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max

    plt.figure(figsize=(12, 6))

    plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    plt.plot(drawdown, color='red', linewidth=1)

    plt.title('Drawdown', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('logs/drawdown.png', dpi=150)
    logger.info("Drawdown chart saved to logs/drawdown.png")

    plt.close()


def plot_trade_distribution(trades_df):
    """Plot trade P&L distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # P&L histogram
    axes[0].hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('P&L Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('P&L ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Cumulative P&L
    cumulative_pnl = trades_df['pnl'].cumsum()
    axes[1].plot(cumulative_pnl, linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Trade Number')
    axes[1].set_ylabel('Cumulative P&L ($)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/trade_distribution.png', dpi=150)
    logger.info("Trade distribution saved to logs/trade_distribution.png")

    plt.close()


def main():
    """Main backtest function"""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("AI CRYPTO TRADING BOT - BACKTESTING")
    logger.info("=" * 80)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Strategy:      {args.strategy}")
    logger.info(f"  Symbols:       {args.symbols}")
    logger.info(f"  Period:        {args.start} to {args.end}")
    logger.info(f"  Timeframe:     {args.timeframe}")
    logger.info(f"  Capital:       ${args.capital:,.2f}")
    logger.info(f"  Commission:    {args.commission*100:.2f}%")

    # Fetch data
    data = fetch_data(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end
    )

    if not data:
        logger.error("No data available for backtesting")
        return

    # Run backtests
    if args.strategy == 'all':
        strategies = ['momentum', 'mean_reversion', 'ml_ensemble']
        all_results = {}

        for strat_name in strategies:
            result = run_backtest(strat_name, data, args.capital, args.commission)

            if result:
                results, engine = result
                metrics = analyze_results(results, engine)
                all_results[strat_name] = metrics

        # Compare strategies
        if all_results:
            logger.info("\n" + "=" * 80)
            logger.info("STRATEGY COMPARISON")
            logger.info("=" * 80)

            comparison_df = pd.DataFrame(all_results).T
            logger.info("\n" + comparison_df.to_string())

    else:
        result = run_backtest(args.strategy, data, args.capital, args.commission)

        if result:
            results, engine = result
            analyze_results(results, engine)

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
