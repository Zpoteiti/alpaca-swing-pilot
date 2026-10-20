"""Fast, Realistic Portfolio-Level Historical Backtesting Engine.

Simulates day-by-day execution with zero look-ahead bias, realistic bid-ask
slippage (default 5 bps per side), and full RiskManager stop-loss enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from swing_pilot.config import Settings
from swing_pilot.risk_manager import PositionInfo, RiskManager
from swing_pilot.strategies import BaseStrategy, ensure_indicators


@dataclass
class ClosedTrade:
    """Record of a completed round-trip trade (BUY -> SELL)."""

    symbol: str
    entry_date: str
    exit_date: str
    qty: int
    entry_price: float
    exit_price: float
    pnl_dollar: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestReport:
    """Comprehensive performance metrics from a historical backtest run."""

    strategy_name: str
    display_name: str
    start_date: str
    end_date: str
    trading_days: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    benchmark_symbol: str
    benchmark_return_pct: float
    benchmark_max_dd_pct: float
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


class PortfolioBacktester:
    """Multi-asset daily swing trading backtester."""

    def __init__(
        self,
        settings: Settings,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.settings = settings
        self.initial_capital = initial_capital
        self.slippage_rate = slippage_bps / 10_000.0  # 5 bps = 0.05%

    def run(
        self,
        strategy: BaseStrategy,
        bars_by_symbol: Dict[str, pd.DataFrame],
        warmup_bars: int = 205,
    ) -> BacktestReport:
        """Run historical simulation across the symbol universe."""
        symbols = [s for s in self.settings.symbols if s in bars_by_symbol]
        if not symbols:
            raise ValueError("没有可用于回测的标的日线数据。")

        bars_by_symbol = {
            s: ensure_indicators(df) for s, df in bars_by_symbol.items()
        }

        # Build sorted union of trading dates across symbols
        all_dates = sorted(
            set().union(*(bars_by_symbol[s].index for s in symbols))
        )
        if len(all_dates) <= warmup_bars + 5:
            warmup_bars = max(30, len(all_dates) // 4)

        eval_dates = all_dates[warmup_bars:]
        risk_mgr = RiskManager(self.settings)

        cash = self.initial_capital
        positions: Dict[str, PositionInfo] = {}
        entry_dates: Dict[str, str] = {}
        closed_trades: List[ClosedTrade] = []
        equity_history: List[float] = []
        valid_dates: List[pd.Timestamp] = []

        for current_dt in eval_dates:
            date_str = str(current_dt)[:10]

            # 1. Update mark-to-market prices for held positions
            for sym, pos in positions.items():
                df_sym = bars_by_symbol[sym]
                if current_dt in df_sym.index:
                    px = float(df_sym.loc[current_dt, "close"])
                    pos.current_price = px
                    if px > pos.highest_price:
                        pos.highest_price = px

            # 2. Compute current total portfolio equity
            market_value = sum(
                p.qty * p.current_price for p in positions.values()
            )
            total_equity = cash + market_value

            # 3. Generate strategy signals using strictly historical slice up to current_dt
            signals = []
            for sym in symbols:
                df_sym = bars_by_symbol[sym]
                df_slice = df_sym.loc[:current_dt]
                if df_slice.empty or df_slice.index[-1] != current_dt:
                    continue
                sig = strategy.evaluate(
                    symbol=sym,
                    df=df_slice,
                    has_position=(sym in positions),
                )
                signals.append(sig)

            # 4. Run RiskManager to get SELLs (including stop-losses) and sized BUYs
            orders, _ = risk_mgr.evaluate_signals(
                signals=signals,
                positions=positions,
                total_equity=total_equity,
                available_cash=cash,
            )

            # 5. Execute orders with realistic slippage penalty
            for order in orders:
                if order.side == "SELL" and order.symbol in positions:
                    pos = positions.pop(order.symbol)
                    fill_price = order.estimated_price * (
                        1.0 - self.slippage_rate
                    )
                    proceeds = pos.qty * fill_price
                    cash += proceeds

                    cost_basis = pos.qty * pos.avg_entry_price
                    pnl_dollar = proceeds - cost_basis
                    pnl_pct = (fill_price / pos.avg_entry_price - 1.0) * 100.0

                    closed_trades.append(
                        ClosedTrade(
                            symbol=order.symbol,
                            entry_date=entry_dates.pop(order.symbol, date_str),
                            exit_date=date_str,
                            qty=pos.qty,
                            entry_price=round(pos.avg_entry_price, 2),
                            exit_price=round(fill_price, 2),
                            pnl_dollar=round(pnl_dollar, 2),
                            pnl_pct=round(pnl_pct, 2),
                            exit_reason=order.reason,
                        )
                    )

                elif order.side == "BUY" and order.symbol not in positions:
                    fill_price = order.estimated_price * (
                        1.0 + self.slippage_rate
                    )
                    cost = order.qty * fill_price
                    if cost > cash and fill_price > 0:
                        order.qty = int(cash // fill_price)
                        cost = order.qty * fill_price
                    if order.qty <= 0:
                        continue

                    cash -= cost
                    positions[order.symbol] = PositionInfo(
                        symbol=order.symbol,
                        qty=order.qty,
                        avg_entry_price=fill_price,
                        current_price=order.estimated_price,
                        highest_price=order.estimated_price,
                    )
                    entry_dates[order.symbol] = date_str

            # Record end-of-day equity
            eod_market_val = sum(
                p.qty * p.current_price for p in positions.values()
            )
            equity_history.append(cash + eod_market_val)
            valid_dates.append(current_dt)

        equity_series = pd.Series(equity_history, index=valid_dates)
        final_equity = float(equity_series.iloc[-1])
        total_return_pct = (final_equity / self.initial_capital - 1.0) * 100.0

        # Annualized return (CAGR)
        years = max(len(valid_dates) / 252.0, 0.1)
        cagr_pct = (
            (final_equity / self.initial_capital) ** (1.0 / years) - 1.0
        ) * 100.0

        # Max Drawdown
        rolling_peak = equity_series.cummax()
        drawdowns = (equity_series / rolling_peak - 1.0) * 100.0
        max_drawdown_pct = float(drawdowns.min())

        # Annualized Sharpe Ratio (assuming 4% risk-free rate -> ~0.04/252 daily)
        daily_rets = equity_series.pct_change().dropna()
        if len(daily_rets) > 2 and daily_rets.std() > 1e-9:
            excess_daily = daily_rets - (0.04 / 252.0)
            sharpe_ratio = float(
                (excess_daily.mean() / daily_rets.std()) * np.sqrt(252.0)
            )
        else:
            sharpe_ratio = 0.0

        # Trade statistics
        total_trades = len(closed_trades)
        if total_trades > 0:
            wins = [t for t in closed_trades if t.pnl_dollar > 0]
            losses = [t for t in closed_trades if t.pnl_dollar <= 0]
            win_rate_pct = (len(wins) / total_trades) * 100.0
            gross_profit = sum(t.pnl_dollar for t in wins)
            gross_loss = abs(sum(t.pnl_dollar for t in losses))
            profit_factor = (
                gross_profit / gross_loss
                if gross_loss > 0
                else (99.9 if gross_profit > 0 else 0.0)
            )
        else:
            win_rate_pct = 0.0
            profit_factor = 0.0

        # Benchmark comparison (Buy & Hold SPY over exact same evaluation period)
        bench_sym = self.settings.benchmark_symbol
        if bench_sym not in bars_by_symbol:
            bench_sym = symbols[0]
        bench_df = bars_by_symbol[bench_sym].loc[
            valid_dates[0] : valid_dates[-1]
        ]
        bench_close = bench_df["close"]
        bench_ret_pct = float(
            (bench_close.iloc[-1] / bench_close.iloc[0] - 1.0) * 100.0
        )
        bench_dd_pct = float(
            ((bench_close / bench_close.cummax()) - 1.0).min() * 100.0
        )

        return BacktestReport(
            strategy_name=strategy.name,
            display_name=strategy.display_name,
            start_date=str(valid_dates[0])[:10],
            end_date=str(valid_dates[-1])[:10],
            trading_days=len(valid_dates),
            initial_capital=self.initial_capital,
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            total_trades=total_trades,
            win_rate_pct=round(win_rate_pct, 1),
            profit_factor=round(profit_factor, 2),
            benchmark_symbol=bench_sym,
            benchmark_return_pct=round(bench_ret_pct, 2),
            benchmark_max_dd_pct=round(bench_dd_pct, 2),
            closed_trades=closed_trades,
            equity_curve=equity_series,
        )
