#!/usr/bin/env python3
"""5-Second Historical Backtest CLI for Alpaca Swing Pilot.

Usage:
    python backtest.py                  # Run 3-year real historical backtest via Alpaca API
    python backtest.py --years 4        # Run 4-year historical backtest
    python backtest.py --offline        # Run offline demo with synthetic market data
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swing_pilot.backtester import BacktestReport, PortfolioBacktester
from swing_pilot.config import load_settings
from swing_pilot.data_client import AlpacaService, generate_synthetic_bars
from swing_pilot.strategies import STRATEGIES, get_strategy

console = Console(width=128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpaca Swing Pilot - 历史日线回测与策略对比引擎"
    )
    parser.add_argument(
        "--years",
        type=float,
        default=3.0,
        help="回测历史年数 (默认: 3.0 年)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="初始虚拟本金 (默认: $100,000)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="指定单跑某个策略 (rsi_pullback / sma_cross / hybrid) 或 all 全部对比",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="使用本地合成行情数据进行离线演示（无需 API Key）",
    )
    return parser.parse_args()


def print_comparison_table(reports: list[BacktestReport]) -> None:
    first = reports[0]
    console.print(
        Panel.fit(
            f"[bold cyan]回测区间:[/bold cyan] {first.start_date} 至 {first.end_date} "
            f"({first.trading_days} 个交易日) | "
            f"[bold cyan]初始本金:[/bold cyan] ${first.initial_capital:,.0f}\n"
            f"[bold yellow]基准对比 ({first.benchmark_symbol} 死拿不动):[/bold yellow] "
            f"总收益 [bold]{first.benchmark_return_pct:+.2f}%[/bold] | "
            f"最大回撤 [bold red]{first.benchmark_max_dd_pct:.2f}%[/bold red]",
            title="📊 Alpaca Swing Pilot 历史回测总览",
            border_style="cyan",
        )
    )

    table = Table(
        title="三大日线策略 vs 标普500基准 成绩单对比",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("策略代码", style="cyan", no_wrap=True)
    table.add_column("策略流派名称", style="white")
    table.add_column("期末净值", justify="right", no_wrap=True)
    table.add_column("总收益率", justify="right", no_wrap=True)
    table.add_column("年化(CAGR)", justify="right", no_wrap=True)
    table.add_column("最大回撤", justify="right", no_wrap=True)
    table.add_column("夏普", justify="right", no_wrap=True)
    table.add_column("交易数", justify="right", no_wrap=True)
    table.add_column("胜率", justify="right", no_wrap=True)
    table.add_column("盈亏比", justify="right", no_wrap=True)

    for r in reports:
        ret_color = "green" if r.total_return_pct >= 0 else "red"
        dd_color = (
            "green"
            if abs(r.max_drawdown_pct) < abs(r.benchmark_max_dd_pct)
            else "yellow"
        )
        table.add_row(
            r.strategy_name,
            r.display_name,
            f"${r.final_equity:,.0f}",
            f"[{ret_color}]{r.total_return_pct:+.2f}%[/{ret_color}]",
            f"[{ret_color}]{r.cagr_pct:+.2f}%[/{ret_color}]",
            f"[{dd_color}]{r.max_drawdown_pct:.2f}%[/{dd_color}]",
            f"{r.sharpe_ratio:.2f}",
            str(r.total_trades),
            f"{r.win_rate_pct:.1f}%",
            f"{r.profit_factor:.2f}",
        )

    # Add Benchmark row for direct visual comparison
    bench_final = first.initial_capital * (
        1.0 + first.benchmark_return_pct / 100.0
    )
    years = max(first.trading_days / 252.0, 0.1)
    bench_cagr = (
        (bench_final / first.initial_capital) ** (1.0 / years) - 1.0
    ) * 100.0
    table.add_section()
    table.add_row(
        f"benchmark ({first.benchmark_symbol})",
        f"基准: 无脑买入并死拿 {first.benchmark_symbol}",
        f"${bench_final:,.0f}",
        f"{first.benchmark_return_pct:+.2f}%",
        f"{bench_cagr:+.2f}%",
        f"[red]{first.benchmark_max_dd_pct:.2f}%[/red]",
        "-",
        "1",
        "100.0%" if first.benchmark_return_pct > 0 else "0.0%",
        "-",
        style="dim",
    )

    console.print(table)


def print_recent_trades(report: BacktestReport, limit: int = 8) -> None:
    if not report.closed_trades:
        return

    t_table = Table(
        title=f"🔍 [{report.display_name}] 最近 {min(limit, len(report.closed_trades))} 笔已平仓交易明细",
        show_header=True,
        header_style="bold blue",
    )
    t_table.add_column("标的", style="bold cyan", no_wrap=True)
    t_table.add_column("买入日期", no_wrap=True)
    t_table.add_column("卖出日期", no_wrap=True)
    t_table.add_column("股数", justify="right", no_wrap=True)
    t_table.add_column("买入价", justify="right", no_wrap=True)
    t_table.add_column("卖出价", justify="right", no_wrap=True)
    t_table.add_column("盈亏金额", justify="right", no_wrap=True)
    t_table.add_column("盈亏%", justify="right", no_wrap=True)
    t_table.add_column("卖出/平仓原因")

    for tr in report.closed_trades[-limit:]:
        color = "green" if tr.pnl_dollar > 0 else "red"
        t_table.add_row(
            tr.symbol,
            tr.entry_date,
            tr.exit_date,
            str(tr.qty),
            f"${tr.entry_price:.2f}",
            f"${tr.exit_price:.2f}",
            f"[{color}]${tr.pnl_dollar:+,.2f}[/{color}]",
            f"[{color}]{tr.pnl_pct:+.2f}%[/{color}]",
            tr.exit_reason,
        )

    console.print(t_table)


def main() -> None:
    args = parse_args()
    settings = load_settings()

    t0 = time.perf_counter()
    symbols = list(
        dict.fromkeys(settings.symbols + [settings.benchmark_symbol])
    )
    # Add ~310 calendar days (~210 trading days) for MA200 warmup
    lookback_days = int(args.years * 365) + 320

    if args.offline or not settings.has_credentials:
        if not args.offline:
            console.print(
                "[yellow]提示: 未检测到 .env 中的 Alpaca API Key，自动切换为离线合成行情演示模式 (--offline)。[/yellow]"
            )
        bars_by_symbol = generate_synthetic_bars(
            symbols, days=int(args.years * 252) + 210
        )
    else:
        console.print(
            f"[cyan]正在从 Alpaca Historical API 拉取 {len(symbols)} 只标的 ({', '.join(symbols)}) 的历史日线数据...[/cyan]"
        )
        service = AlpacaService(settings)
        bars_by_symbol = service.fetch_daily_bars(
            symbols, lookback_days=lookback_days
        )

    backtester = PortfolioBacktester(
        settings=settings, initial_capital=args.capital
    )

    if args.strategy.lower() == "all":
        strat_names = list(STRATEGIES.keys())
    else:
        strat_names = [args.strategy.lower()]

    reports: list[BacktestReport] = []
    for name in strat_names:
        strat = get_strategy(name)
        rep = backtester.run(strat, bars_by_symbol)
        reports.append(rep)

    elapsed = time.perf_counter() - t0
    print_comparison_table(reports)

    for rep in reports:
        print_recent_trades(rep, limit=6)

    console.print(
        f"\n[bold green]✓ 历史回测完成！共耗时 {elapsed:.2f} 秒。[/bold green] "
        f"(标的池: {', '.join(settings.symbols)} | 单票上限: {settings.max_position_pct*100:.0f}% | "
        f"硬止损: -{settings.stop_loss_pct*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
