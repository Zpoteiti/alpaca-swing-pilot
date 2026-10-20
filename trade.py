#!/usr/bin/env python3
"""Daily Paper Trading Scanner & Execution Agent for Alpaca Swing Pilot.

Usage:
    python trade.py                     # Scan today's signals & preview orders (Safe Dry-Run)
    python trade.py --execute           # Scan signals & actually submit orders to Alpaca Paper
    python trade.py --smoke-test        # Place a 1-share SPY test order on Paper account to verify API loop
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swing_pilot.config import load_settings
from swing_pilot.data_client import AlpacaService
from swing_pilot.risk_manager import OrderDecision, RiskManager
from swing_pilot.strategies import get_strategy

console = Console(width=128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpaca Swing Pilot - 每日行情扫描与模拟盘自动交易代理"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="覆盖默认策略 (rsi_pullback / sma_cross / hybrid)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="正式向 Alpaca 模拟盘提交由策略与风控生成的买卖订单 (默认仅做 Dry-Run 预览)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="冒烟测试模式：立刻向 Alpaca Paper 账户发送 1 股 SPY 测试买单，验证下单链路是否通畅",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    if not settings.alpaca_paper:
        console.print(
            "[bold red]⚠️ 警告: 当前 ALPACA_PAPER=false！新手阶段请务必保持 ALPACA_PAPER=true 使用模拟盘。[/bold red]"
        )

    service = AlpacaService(settings)
    acct = service.get_account_summary()
    positions = service.get_open_positions()

    strat_name = args.strategy or settings.default_strategy
    strategy = get_strategy(strat_name)

    # 1. Print Account Status Panel
    mode_badge = (
        "[bold green]PAPER 模拟盘[/bold green]"
        if acct["paper"]
        else "[bold red]LIVE 实盘[/bold red]"
    )
    console.print(
        Panel.fit(
            f"账户编号: [bold]{acct['account_number']}[/bold] ({mode_badge}) | 状态: {acct['status']}\n"
            f"账户总净值 (Equity): [bold cyan]${acct['equity']:,.2f}[/bold cyan] | "
            f"可用现金 (Cash): [bold green]${acct['cash']:,.2f}[/bold green]\n"
            f"近5日内交易次数 (Daytrade Count): [bold]{acct['daytrade_count']}[/bold] / 3 "
            f"(日线隔夜策略不消耗日内交易额度)\n"
            f"当前启用策略: [bold yellow]{strategy.display_name} ({strategy.name})[/bold yellow]",
            title="🤖 Alpaca Swing Pilot 账户与风控状态",
            border_style="green",
        )
    )

    # 2. Show Current Positions if any
    if positions:
        pos_table = Table(
            title="📦 当前持仓列表 (Open Positions)",
            header_style="bold cyan",
        )
        pos_table.add_column("标的", style="bold")
        pos_table.add_column("持仓股数", justify="right")
        pos_table.add_column("买入均价", justify="right")
        pos_table.add_column("当前最新价", justify="right")
        pos_table.add_column("浮动盈亏%", justify="right")
        for sym, pos in positions.items():
            pnl_pct = pos.unrealized_pnl_pct * 100.0
            c = "green" if pnl_pct >= 0 else "red"
            pos_table.add_row(
                sym,
                str(pos.qty),
                f"${pos.avg_entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"[{c}]{pnl_pct:+.2f}%[/{c}]",
            )
        console.print(pos_table)
    else:
        console.print("[dim]当前账户暂无持仓 (空仓状态)。[/dim]\n")

    # 3. Smoke-Test Mode (Optional 1-share SPY test order)
    if args.smoke_test:
        console.print(
            "[bold yellow]🚀 正在执行冒烟测试 (--smoke-test): 向 Alpaca 模拟盘提交 1 股 SPY 买单...[/bold yellow]"
        )
        test_order = OrderDecision(
            symbol="SPY",
            side="BUY",
            qty=1,
            estimated_price=0.0,
            reason="冒烟测试订单 (验证 API 自动下单链路)",
        )
        receipts = service.submit_orders([test_order], positions)
        for rec in receipts:
            console.print(
                f"[bold green]✓ 下单成功！[/bold green] 订单ID: {rec['id']} | "
                f"{rec['side']} {rec['qty']} 股 {rec['symbol']} | 状态: {rec['status']}"
            )
        return

    # 4. Fetch latest daily bars & evaluate strategy across all symbols
    symbols_to_scan = list(
        dict.fromkeys(settings.symbols + list(positions.keys()))
    )
    console.print(
        f"[cyan]正在拉取 {', '.join(symbols_to_scan)} 最新日线行情并计算指标...[/cyan]"
    )
    bars_by_symbol = service.fetch_daily_bars(
        symbols_to_scan, lookback_days=340
    )

    signals = []
    scan_table = Table(
        title=f"📈 今日日线指标与信号扫描 ({strategy.display_name})",
        header_style="bold magenta",
    )
    scan_table.add_column("标的", style="bold cyan")
    scan_table.add_column("最新日期")
    scan_table.add_column("收盘价", justify="right")
    scan_table.add_column("核心指标摘要")
    scan_table.add_column("是否持仓", justify="center")
    scan_table.add_column("今日信号", justify="center")
    scan_table.add_column("决策原因 (白盒解释)")

    for sym in symbols_to_scan:
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        has_pos = sym in positions
        sig = strategy.evaluate(sym, df, has_position=has_pos)
        signals.append(sig)

        ind_str = ", ".join(f"{k}={v}" for k, v in sig.indicators.items() if k != "close")
        sig_badge = {
            "BUY": "[bold green]BUY 买入[/bold green]",
            "SELL": "[bold red]SELL 卖出[/bold red]",
            "HOLD": "[dim]HOLD 观望/持有[/dim]",
        }[sig.action]

        scan_table.add_row(
            sym,
            sig.date,
            f"${sig.price:.2f}",
            ind_str,
            "是" if has_pos else "否",
            sig_badge,
            sig.reason,
        )

    console.print(scan_table)

    # 5. Run RiskManager to get final approved orders
    risk_mgr = RiskManager(settings)
    orders, risk_logs = risk_mgr.evaluate_signals(
        signals=signals,
        positions=positions,
        total_equity=float(acct["equity"]),
        available_cash=float(acct["cash"]),
    )

    for log in risk_logs:
        console.print(f"[yellow]⚠️ 风控日志: {log}[/yellow]")

    if not orders:
        console.print(
            "\n[bold green]✓ 今日扫描完毕：当前无触发买卖条件或风控止损的订单，保持现状即可。[/bold green]"
        )
        return

    # 6. Display proposed orders
    order_table = Table(
        title="🎯 风控审核通过的待执行订单清单",
        header_style="bold yellow",
    )
    order_table.add_column("方向", justify="center")
    order_table.add_column("标的", style="bold cyan")
    order_table.add_column("股数", justify="right")
    order_table.add_column("预估价格", justify="right")
    order_table.add_column("预计金额", justify="right")
    order_table.add_column("触发原因")

    for o in orders:
        side_str = (
            "[bold green]BUY[/bold green]"
            if o.side == "BUY"
            else "[bold red]SELL[/bold red]"
        )
        order_table.add_row(
            side_str,
            o.symbol,
            str(o.qty),
            f"${o.estimated_price:.2f}",
            f"${o.qty * o.estimated_price:,.2f}",
            o.reason,
        )
    console.print(order_table)

    if not args.execute:
        console.print(
            "\n[bold cyan]💡 当前为 Dry-Run 预览模式（未实际发送订单）。[/bold cyan]"
            "\n如需向 Alpaca 模拟盘正式提交上述订单，请加参数运行: [bold]python trade.py --execute[/bold]"
        )
        return

    receipts = service.submit_orders(orders, positions)
    for rec in receipts:
        console.print(
            f"[bold green]✓ 已提交订单:[/bold green] ID={rec['id']} | "
            f"{rec['side']} {rec['qty']} 股 {rec['symbol']} | 状态={rec['status']}"
        )


if __name__ == "__main__":
    main()
