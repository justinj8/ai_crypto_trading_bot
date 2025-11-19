"""
Advanced Performance Analytics
Comprehensive analysis and metrics for trading performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceAnalytics:
    """
    Advanced performance analytics for trading systems

    Provides:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega)
    - Drawdown analysis
    - Trade analysis
    - Risk metrics (VaR, CVaR)
    - Benchmark comparison
    - Rolling performance metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analytics

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_curve: Series of portfolio values over time
            trades: DataFrame of trade history
            benchmark: Optional benchmark returns

        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = self.calculate_cagr(equity_curve)

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(equity_curve)
        omega = self.calculate_omega_ratio(returns)

        # Drawdown metrics
        max_dd, max_dd_duration, avg_dd = self.calculate_drawdown_metrics(equity_curve)

        # Risk measures
        var_95 = self.calculate_var(returns, confidence=0.95)
        cvar_95 = self.calculate_cvar(returns, confidence=0.95)

        # Trade metrics
        if not trades.empty:
            trade_metrics = self.analyze_trades(trades)
        else:
            trade_metrics = {}

        # Benchmark comparison
        if benchmark is not None:
            benchmark_metrics = self.compare_to_benchmark(returns, benchmark)
        else:
            benchmark_metrics = {}

        # Compile all metrics
        metrics = {
            # Returns
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cagr': cagr,
            'cagr_pct': cagr * 100,

            # Risk
            'volatility': volatility,
            'volatility_pct': volatility * 100,

            # Risk-adjusted returns
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,

            # Drawdown
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_duration_days': max_dd_duration,
            'avg_drawdown': avg_dd,
            'avg_drawdown_pct': avg_dd * 100,

            # Risk measures
            'var_95': var_95,
            'cvar_95': cvar_95,

            # Trade metrics
            **trade_metrics,

            # Benchmark comparison
            **benchmark_metrics
        }

        return metrics

    def calculate_cagr(self, equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(equity_curve) < 2:
            return 0.0

        start_value = equity_curve.iloc[0]
        end_value = equity_curve.iloc[-1]

        # Estimate number of years
        num_years = len(equity_curve) / 252  # Assuming daily data, 252 trading days/year

        if num_years == 0 or start_value == 0:
            return 0.0

        cagr = (end_value / start_value) ** (1 / num_years) - 1

        return cagr

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / periods_per_year

        # Excess returns
        excess_returns = returns - daily_rf

        if excess_returns.std() == 0:
            return 0.0

        # Annualized Sharpe
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / periods_per_year

        # Excess returns
        excess_returns = returns - daily_rf

        # Downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        # Annualized Sortino
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)

        return sortino

    def calculate_calmar_ratio(self, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)"""
        cagr = self.calculate_cagr(equity_curve)
        max_dd, _, _ = self.calculate_drawdown_metrics(equity_curve)

        if max_dd == 0:
            return 0.0

        calmar = cagr / abs(max_dd)

        return calmar

    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio

        Ratio of probability-weighted gains to losses
        """
        if len(returns) < 2:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return float('inf')

        omega = gains.sum() / losses.sum()

        return omega

    def calculate_drawdown_metrics(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, int, float]:
        """
        Calculate drawdown metrics

        Returns:
            (max_drawdown, max_drawdown_duration, avg_drawdown)
        """
        # Running maximum
        running_max = equity_curve.expanding().max()

        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # Maximum drawdown duration
        dd_duration = 0
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        # Average drawdown (when in drawdown)
        drawdowns_only = drawdown[drawdown < 0]
        avg_dd = drawdowns_only.mean() if len(drawdowns_only) > 0 else 0.0

        return max_dd, dd_duration, avg_dd

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Return series
            confidence: Confidence level (default 95%)

        Returns:
            VaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Expected value of losses beyond VaR

        Args:
            returns: Return series
            confidence: Confidence level (default 95%)

        Returns:
            CVaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        var = self.calculate_var(returns, confidence)

        # Average of returns worse than VaR
        cvar = returns[returns <= var].mean()

        return cvar

    def analyze_trades(self, trades: pd.DataFrame) -> Dict:
        """
        Analyze trade statistics

        Args:
            trades: DataFrame with trade history

        Returns:
            Dictionary of trade metrics
        """
        if trades.empty:
            return {}

        # Basic stats
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L stats
        total_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0

        # Best and worst trades
        best_trade = trades['pnl'].max()
        worst_trade = trades['pnl'].min()

        # Win/loss streaks
        win_streak, loss_streak = self._calculate_streaks(trades['pnl'])

        # Average trade duration
        if 'duration' in trades.columns:
            avg_duration = trades['duration'].mean()
        else:
            avg_duration = 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Return stats
        if 'return_pct' in trades.columns:
            avg_return = trades['return_pct'].mean()
            return_std = trades['return_pct'].std()
            return_skew = trades['return_pct'].skew()
            return_kurt = trades['return_pct'].kurt()
        else:
            avg_return = return_std = return_skew = return_kurt = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,

            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,

            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,

            'max_win_streak': win_streak,
            'max_loss_streak': loss_streak,

            'avg_trade_duration': avg_duration,
            'expectancy': expectancy,

            'avg_return_pct': avg_return * 100,
            'return_std_pct': return_std * 100,
            'return_skewness': return_skew,
            'return_kurtosis': return_kurt
        }

    def _calculate_streaks(self, pnl_series: pd.Series) -> Tuple[int, int]:
        """Calculate maximum winning and losing streaks"""
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for pnl in pnl_series:
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0

        return max_win_streak, max_loss_streak

    def compare_to_benchmark(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """
        Compare strategy returns to benchmark

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Comparison metrics
        """
        # Align indices
        common_idx = returns.index.intersection(benchmark_returns.index)

        if len(common_idx) < 2:
            return {}

        strategy_ret = returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]

        # Cumulative returns
        strategy_cumret = (1 + strategy_ret).prod() - 1
        bench_cumret = (1 + bench_ret).prod() - 1

        # Excess return
        excess_return = strategy_cumret - bench_cumret

        # Beta (market sensitivity)
        covariance = np.cov(strategy_ret, bench_ret)[0][1]
        benchmark_var = bench_ret.var()

        beta = covariance / benchmark_var if benchmark_var != 0 else 0

        # Alpha (excess return adjusted for risk)
        alpha = strategy_ret.mean() - (beta * bench_ret.mean())

        # Information ratio
        excess_returns = strategy_ret - bench_ret
        tracking_error = excess_returns.std()

        information_ratio = excess_returns.mean() / tracking_error if tracking_error != 0 else 0

        # Correlation
        correlation = strategy_ret.corr(bench_ret)

        return {
            'benchmark_return': bench_cumret,
            'benchmark_return_pct': bench_cumret * 100,
            'excess_return': excess_return,
            'excess_return_pct': excess_return * 100,
            'beta': beta,
            'alpha': alpha,
            'alpha_annualized': alpha * 252,
            'information_ratio': information_ratio,
            'correlation': correlation
        }

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            equity_curve: Portfolio values
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling returns
        rolling_metrics['rolling_return'] = (1 + returns).rolling(window).apply(lambda x: x.prod() - 1, raw=True)

        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling Sharpe
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf

        rolling_metrics['rolling_sharpe'] = (
            excess_returns.rolling(window).mean() /
            excess_returns.rolling(window).std() * np.sqrt(252)
        )

        # Rolling max drawdown
        def calc_rolling_dd(series):
            cummax = series.expanding().max()
            dd = (series - cummax) / cummax
            return dd.min()

        rolling_metrics['rolling_max_dd'] = (
            equity_curve.rolling(window).apply(calc_rolling_dd, raw=False)
        )

        return rolling_metrics

    def generate_performance_report(
        self,
        metrics: Dict,
        format: str = "text"
    ) -> str:
        """
        Generate formatted performance report

        Args:
            metrics: Performance metrics dictionary
            format: Output format ("text", "markdown")

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._format_markdown_report(metrics)
        else:
            return self._format_text_report(metrics)

    def _format_text_report(self, metrics: Dict) -> str:
        """Format performance report as text"""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 80)

        report.append("\nRETURNS:")
        report.append(f"  Total Return:        {metrics.get('total_return_pct', 0):.2f}%")
        report.append(f"  CAGR:                {metrics.get('cagr_pct', 0):.2f}%")

        report.append("\nRISK:")
        report.append(f"  Volatility:          {metrics.get('volatility_pct', 0):.2f}%")
        report.append(f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"  VaR (95%):          {metrics.get('var_95', 0):.4f}")
        report.append(f"  CVaR (95%):         {metrics.get('cvar_95', 0):.4f}")

        report.append("\nRISK-ADJUSTED RETURNS:")
        report.append(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"  Omega Ratio:         {metrics.get('omega_ratio', 0):.2f}")

        if 'total_trades' in metrics:
            report.append("\nTRADES:")
            report.append(f"  Total Trades:        {metrics.get('total_trades', 0)}")
            report.append(f"  Win Rate:            {metrics.get('win_rate_pct', 0):.2f}%")
            report.append(f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}")
            report.append(f"  Expectancy:          ${metrics.get('expectancy', 0):.2f}")

        report.append("=" * 80)

        return "\n".join(report)

    def _format_markdown_report(self, metrics: Dict) -> str:
        """Format performance report as markdown"""
        report = []
        report.append("# Performance Report\n")

        report.append("## Returns")
        report.append(f"- **Total Return**: {metrics.get('total_return_pct', 0):.2f}%")
        report.append(f"- **CAGR**: {metrics.get('cagr_pct', 0):.2f}%\n")

        report.append("## Risk")
        report.append(f"- **Volatility**: {metrics.get('volatility_pct', 0):.2f}%")
        report.append(f"- **Max Drawdown**: {metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"- **VaR (95%)**: {metrics.get('var_95', 0):.4f}")
        report.append(f"- **CVaR (95%)**: {metrics.get('cvar_95', 0):.4f}\n")

        report.append("## Risk-Adjusted Returns")
        report.append(f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"- **Sortino Ratio**: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"- **Calmar Ratio**: {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"- **Omega Ratio**: {metrics.get('omega_ratio', 0):.2f}\n")

        if 'total_trades' in metrics:
            report.append("## Trades")
            report.append(f"- **Total Trades**: {metrics.get('total_trades', 0)}")
            report.append(f"- **Win Rate**: {metrics.get('win_rate_pct', 0):.2f}%")
            report.append(f"- **Profit Factor**: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"- **Expectancy**: ${metrics.get('expectancy', 0):.2f}\n")

        return "\n".join(report)
