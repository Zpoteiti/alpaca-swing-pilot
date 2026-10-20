#!/usr/bin/env python3
"""Daily Paper Trading Scanner, LLM News Guard & Execution Agent for Alpaca Swing Pilot.

Usage:
    python trade.py                         # Scan today's quant signals & preview orders (Dry-Run)
    python trade.py --llm-guard             # Enable local LLM news risk filter on BUY candidates
    python trade.py --llm-audit             # Run local LLM news & risk audit across all universe symbols
    python trade.py --llm-guard --execute   # Submit quant + LLM-approved orders to Alpaca Paper
    python trade.py --smoke-test            # Place a 1-share SPY test order on Paper account
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swing_pilot.config import load_settings
from swing_pilot.data_client import AlpacaService
from swing_pilot.llm_guard import LocalLLMNewsGuard
from swing_pilot.risk_manager import OrderDecision, RiskManager
from swing_pilot.strategies import get_strategy

console = Console(width=128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpaca Swing Pilot - 每日行情扫描、LLM新闻风控与模拟盘自动交易代理"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="覆盖默认策略 (rsi_pullback / sma_cross / hybrid)",
    )
    parser.add_argument(
        "--llm-guard",
        action="store_true",
        help="启用 Phase 2 本地 LLM 新闻守门员：对触发 BUY 的标的自动拉取新闻并由 LLM 排查暴雷风险",
    )
    parser.add_argument(
        "--llm-audit",
        action="store_true",
        help="盘后全景新闻体检：对标的池内所有股票拉取 Alpaca 最新新闻并调用本地 LLM 输出多空与风险评估表",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="同时管理非本策略标的池(TRADING_SYMBOLS)之外的外部已有持仓（默认忽略外部策略持仓以防误卖）",
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
    all_positions = service.get_open_positions()

    universe_set = set(settings.symbols)
    if args.include_external:
        managed_positions = all_positions
        external_positions = {}
    else:
        managed_positions = {
            k: v for k, v in all_positions.items() if k in universe_set
        }
        external_positions = {
            k: v for k, v in all_positions.items() if k not in universe_set
        }

    strat_name = args.strategy or settings.default_strategy
    strategy = get_strategy(strat_name)

    llm_guard = (
        LocalLLMNewsGuard(settings)
        if (args.llm_guard or args.llm_audit)
        else None
    )
    llm_status = (
        f"[bold green]已启用 ({llm_guard.discover_model()} @ {llm_guard.base_url})[/bold green]"
        if llm_guard
        else "[dim]未启用 (加 --llm-guard 或 --llm-audit 开启)[/dim]"
    )

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
            f"当前启用策略: [bold yellow]{strategy.display_name} ({strategy.name})[/bold yellow]\n"
            f"LLM 新闻守门员: {llm_status}",
            title="🤖 Alpaca Swing Pilot 账户与风控状态",
            border_style="green",
        )
    )

    # 2. Show Current Positions if any
    if managed_positions:
        pos_table = Table(
            title="📦 本策略管理持仓 (Managed Positions)",
            header_style="bold cyan",
        )
        pos_table.add_column("标的", style="bold", no_wrap=True)
        pos_table.add_column("持仓股数", justify="right", no_wrap=True)
        pos_table.add_column("买入均价", justify="right", no_wrap=True)
        pos_table.add_column("当前最新价", justify="right", no_wrap=True)
        pos_table.add_column("浮动盈亏%", justify="right", no_wrap=True)
        for sym, pos in managed_positions.items():
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
        console.print("[dim]当前本策略标的池内暂无持仓。[/dim]")

    if external_positions:
        ext_syms = ", ".join(
            f"{s}({p.qty}股)" for s, p in external_positions.items()
        )
        console.print(
            f"[dim]注: 检测到账户内存在非本策略标的池的外部持仓 [{ext_syms}]，已自动隔离保护（不触发卖出）。[/dim]\n"
        )

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
        receipts = service.submit_orders([test_order], managed_positions)
        for rec in receipts:
            console.print(
                f"[bold green]✓ 下单成功！[/bold green] 订单ID: {rec['id']} | "
                f"{rec['side']} {rec['qty']} 股 {rec['symbol']} | 状态: {rec['status']}"
            )
        return

    # 4. Fetch latest daily bars & evaluate strategy across all symbols
    symbols_to_scan = list(
        dict.fromkeys(settings.symbols + list(managed_positions.keys()))
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
    scan_table.add_column("标的", style="bold cyan", no_wrap=True)
    scan_table.add_column("最新日期", no_wrap=True)
    scan_table.add_column("收盘价", justify="right", no_wrap=True)
    scan_table.add_column("核心指标摘要")
    scan_table.add_column("持仓", justify="center", no_wrap=True)
    scan_table.add_column("今日信号", justify="center", no_wrap=True)
    scan_table.add_column("决策原因 (白盒解释)")

    for sym in symbols_to_scan:
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        has_pos = sym in managed_positions
        sig = strategy.evaluate(sym, df, has_position=has_pos)
        signals.append(sig)

        ind_str = ", ".join(
            f"{k}={v}" for k, v in sig.indicators.items() if k != "close"
        )
        sig_badge = {
            "BUY": "[bold green]BUY 买入[/bold green]",
            "SELL": "[bold red]SELL 卖出[/bold red]",
            "HOLD": "[dim]HOLD 观望[/dim]",
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

    # 5. Optional Full LLM News Audit across all scanned symbols (--llm-audit)
    if args.llm_audit and llm_guard is not None:
        console.print(
            f"\n[bold cyan]🧠 正在调用本地 LLM ({llm_guard.discover_model()}) 对标的池进行新闻面与暴雷风险体检...[/bold cyan]"
        )
        audit_table = Table(
            title="📰 Phase 2: LLM 实时财经新闻与基本面风险体检报告",
            header_style="bold blue",
        )
        audit_table.add_column("标的", style="bold cyan", no_wrap=True)
        audit_table.add_column("量化信号", justify="center", no_wrap=True)
        audit_table.add_column("新闻数", justify="right", no_wrap=True)
        audit_table.add_column("情绪判定", justify="center", no_wrap=True)
        audit_table.add_column("风险等级", justify="center", no_wrap=True)
        audit_table.add_column("风控放行", justify="center", no_wrap=True)
        audit_table.add_column("耗时", justify="right", no_wrap=True)
        audit_table.add_column("LLM 新闻综述与风控结论")

        for sig in signals:
            res = llm_guard.audit_symbol(sig.symbol, sig)
            sent_color = {
                "bullish": "green",
                "bearish": "red",
                "neutral": "yellow",
            }.get(res.sentiment.lower(), "white")
            risk_color = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
            }.get(res.risk_level.lower(), "white")
            pass_badge = (
                "[bold green]✓ 放行[/bold green]"
                if res.approved
                else "[bold red]✗ 否决[/bold red]"
            )
            audit_table.add_row(
                sig.symbol,
                sig.action,
                str(res.news_count),
                f"[{sent_color}]{res.sentiment}[/{sent_color}]",
                f"[{risk_color}]{res.risk_level}[/{risk_color}]",
                pass_badge,
                f"{res.latency_sec:.1f}s",
                res.summary_zh,
            )
        console.print(audit_table)

    # 6. Run RiskManager (with optional LLM Pre-Trade Guard hooked in)
    guard_fn = llm_guard.as_pre_trade_guard if llm_guard is not None else None
    risk_mgr = RiskManager(settings, pre_trade_guard=guard_fn)
    orders, risk_logs = risk_mgr.evaluate_signals(
        signals=signals,
        positions=managed_positions,
        total_equity=float(acct["equity"]),
        available_cash=float(acct["cash"]),
    )

    for log in risk_logs:
        console.print(f"[yellow]⚠️ 风控日志: {log}[/yellow]")

    if not orders:
        console.print(
            "\n[bold green]✓ 今日扫描完毕：当前无触发买卖条件或通过风控审核的订单，保持现状即可。[/bold green]"
        )
        return

    # 7. Display proposed orders
    order_table = Table(
        title="🎯 风控审核通过的待执行订单清单",
        header_style="bold yellow",
    )
    order_table.add_column("方向", justify="center", no_wrap=True)
    order_table.add_column("标的", style="bold cyan", no_wrap=True)
    order_table.add_column("股数", justify="right", no_wrap=True)
    order_table.add_column("预估价格", justify="right", no_wrap=True)
    order_table.add_column("预计金额", justify="right", no_wrap=True)
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

    receipts = service.submit_orders(orders, managed_positions)
    for rec in receipts:
        console.print(
            f"[bold green]✓ 已提交订单:[/bold green] ID={rec['id']} | "
            f"{rec['side']} {rec['qty']} 股 {rec['symbol']} | 状态={rec['status']}"
        )


if __name__ == "__main__":
    main()
