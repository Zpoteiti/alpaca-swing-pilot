"""Unit tests for Alpaca Swing Pilot strategies, risk guardrails, and backtester."""

from __future__ import annotations

import pandas as pd

from swing_pilot.backtester import PortfolioBacktester
from swing_pilot.config import Settings
from swing_pilot.data_client import generate_synthetic_bars
from swing_pilot.risk_manager import PositionInfo, RiskManager
from swing_pilot.strategies import (
    MeanReversionRSIStrategy,
    SignalResult,
    TrendSMAStrategy,
    get_strategy,
)


def test_long_only_and_stop_loss_guardrail() -> None:
    """Verify RiskManager enforces 7% hard stop-loss and never shorts."""
    settings = Settings(stop_loss_pct=0.07, max_position_pct=0.20, max_positions=5)
    risk_mgr = RiskManager(settings)

    # Position bought at $100, dropped to $91 (-9% loss -> exceeds 7% stop-loss)
    positions = {
        "AAPL": PositionInfo(
            symbol="AAPL",
            qty=50,
            avg_entry_price=100.0,
            current_price=91.0,
        )
    }
    # Even if strategy says HOLD, RiskManager must force a SELL of exact 50 shares
    signals = [
        SignalResult(
            symbol="AAPL",
            action="HOLD",
            reason="Strategy says hold",
            price=91.0,
            date="2026-10-20",
        )
    ]

    orders, _ = risk_mgr.evaluate_signals(
        signals=signals,
        positions=positions,
        total_equity=100_000.0,
        available_cash=95_000.0,
    )
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].side == "SELL"
    assert orders[0].qty == 50
    assert orders[0].is_stop_loss is True


def test_pre_trade_guard_veto_hook() -> None:
    """Verify Phase 2 LLM Pre-Trade Guard hook can veto risky BUY signals."""
    settings = Settings()

    def mock_llm_veto(symbol: str, sig: SignalResult) -> tuple[bool, str]:
        if symbol == "NVDA":
            return False, "今晚盘后发布财报，波动风险过高，暂缓买入"
        return True, "OK"

    risk_mgr = RiskManager(settings, pre_trade_guard=mock_llm_veto)
    signals = [
        SignalResult(
            symbol="NVDA",
            action="BUY",
            reason="RSI超卖",
            price=120.0,
            date="2026-10-20",
            score=10.0,
        ),
        SignalResult(
            symbol="MSFT",
            action="BUY",
            reason="RSI超卖",
            price=400.0,
            date="2026-10-20",
            score=8.0,
        ),
    ]

    orders, logs = risk_mgr.evaluate_signals(
        signals=signals,
        positions={},
        total_equity=100_000.0,
        available_cash=100_000.0,
    )
    assert len(orders) == 1
    assert orders[0].symbol == "MSFT"
    assert any("NVDA" in log and "一票否决" in log for log in logs)


def test_strategies_and_backtester_offline() -> None:
    """Verify all strategies run cleanly through the backtester without look-ahead errors."""
    settings = Settings(symbols=["SPY", "QQQ", "AAPL", "MSFT"])
    bars = generate_synthetic_bars(settings.symbols, days=450, seed=123)
    bt = PortfolioBacktester(settings=settings, initial_capital=100_000.0)

    for strat_name in ("rsi_pullback", "sma_cross", "hybrid"):
        strat = get_strategy(strat_name)
        report = bt.run(strat, bars)
        assert report.trading_days > 100
        assert report.final_equity > 0
        assert isinstance(report.equity_curve, pd.Series)


def test_llm_guard_json_parser() -> None:
    """Verify LocalLLMNewsGuard cleanly strips <think> blocks and parses JSON."""
    from swing_pilot.llm_guard import LocalLLMNewsGuard

    raw = (
        "<think>\nLet me analyze the news...\n</think>\n"
        '```json\n{"approved": false, "sentiment": "bearish", "risk_level": "high", '
        '"summary_zh": "突发反垄断调查，暂缓抄底"}\n```'
    )
    parsed = LocalLLMNewsGuard._parse_json_response(raw)
    assert parsed["approved"] is False
    assert parsed["sentiment"] == "bearish"
    assert parsed["risk_level"] == "high"
    assert "反垄断" in parsed["summary_zh"]
