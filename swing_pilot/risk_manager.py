"""Risk Manager & Position Sizing Guardrails (4 道风控铁闸).

Separates signal generation from execution safety:
1. Long-Only Lock: Strictly forbids short selling or selling more shares than held.
2. Hard Stop-Loss & Trailing Stop: Overrides HOLD signals if drawdown exceeds thresholds.
3. Equal-Weight Position Sizing: Caps single-stock exposure (e.g. 20%) & preserves cash buffer.
4. Pluggable Pre-Trade Filter Slot: Ready for Phase 2 LLM News Sentiment Veto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

from swing_pilot.config import Settings
from swing_pilot.strategies import SignalResult


@dataclass
class PositionInfo:
    """Current holding state for a single symbol."""

    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    highest_price: float = 0.0  # Highest close since entry (for trailing stop)

    def __post_init__(self) -> None:
        if self.highest_price <= 0:
            self.highest_price = max(self.avg_entry_price, self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price <= 0:
            return 0.0
        return (self.current_price / self.avg_entry_price) - 1.0

    @property
    def drawdown_from_peak_pct(self) -> float:
        peak = max(self.highest_price, self.avg_entry_price)
        if peak <= 0:
            return 0.0
        return (self.current_price / peak) - 1.0


@dataclass
class OrderDecision:
    """Final risk-checked order instruction ready for execution."""

    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    estimated_price: float
    reason: str
    is_stop_loss: bool = False


# Type alias for Phase 2 LLM Pre-Trade News Guard:
# Takes (symbol, SignalResult) -> returns (approved: bool, veto_reason: str)
PreTradeGuardFn = Callable[[str, SignalResult], tuple[bool, str]]


class RiskManager:
    """Enforces portfolio-level risk controls and position sizing."""

    def __init__(
        self,
        settings: Settings,
        pre_trade_guard: Optional[PreTradeGuardFn] = None,
    ) -> None:
        self.settings = settings
        self.pre_trade_guard = pre_trade_guard

    def evaluate_signals(
        self,
        signals: List[SignalResult],
        positions: Dict[str, PositionInfo],
        total_equity: float,
        available_cash: float,
    ) -> tuple[List[OrderDecision], List[str]]:
        """Convert raw strategy signals into safe, sized OrderDecisions.

        Returns:
            (orders_to_execute, risk_logs)
        """
        orders: List[OrderDecision] = []
        risk_logs: List[str] = []

        # Track simulated cash & open position count as we process SELLs then BUYs
        projected_cash = available_cash
        active_symbols = set(positions.keys())

        # ------------------------------------------------------------------
        # Step 1: Process existing positions (Stop-Loss check + Strategy SELL)
        # ------------------------------------------------------------------
        signal_map = {s.symbol: s for s in signals}

        for symbol, pos in list(positions.items()):
            if pos.qty <= 0:
                active_symbols.discard(symbol)
                continue

            sig = signal_map.get(symbol)
            price = sig.price if sig else pos.current_price

            # Update peak for trailing stop calculation
            peak_price = max(pos.highest_price, pos.avg_entry_price, price)
            pnl_pct = (price / pos.avg_entry_price) - 1.0
            dd_peak_pct = (price / peak_price) - 1.0

            # Rule 1A: Hard Stop-Loss Check (硬止损铁闸)
            if pnl_pct <= -self.settings.stop_loss_pct:
                reason = (
                    f"[风控强制止损] 当前价(${price:.2f})较买入均价(${pos.avg_entry_price:.2f})"
                    f"亏损 {pnl_pct * 100:.2f}%，触发 -{self.settings.stop_loss_pct * 100:.1f}% 硬止损线"
                )
                orders.append(
                    OrderDecision(
                        symbol=symbol,
                        side="SELL",
                        qty=pos.qty,  # Long-only: sell exact held qty
                        estimated_price=price,
                        reason=reason,
                        is_stop_loss=True,
                    )
                )
                projected_cash += pos.qty * price
                active_symbols.discard(symbol)
                risk_logs.append(f"{symbol}: {reason}")
                continue

            # Rule 1B: Trailing Stop Check (移动止损保护利润)
            if dd_peak_pct <= -self.settings.trailing_stop_pct:
                reason = (
                    f"[风控移动止盈/止损] 当前价(${price:.2f})较持仓最高点(${peak_price:.2f})"
                    f"回撤 {dd_peak_pct * 100:.2f}%，触发 -{self.settings.trailing_stop_pct * 100:.1f}% 移动保护线"
                )
                orders.append(
                    OrderDecision(
                        symbol=symbol,
                        side="SELL",
                        qty=pos.qty,
                        estimated_price=price,
                        reason=reason,
                        is_stop_loss=True,
                    )
                )
                projected_cash += pos.qty * price
                active_symbols.discard(symbol)
                risk_logs.append(f"{symbol}: {reason}")
                continue

            # Rule 1C: Normal Strategy SELL signal
            if sig and sig.action == "SELL":
                orders.append(
                    OrderDecision(
                        symbol=symbol,
                        side="SELL",
                        qty=pos.qty,
                        estimated_price=price,
                        reason=sig.reason,
                        is_stop_loss=False,
                    )
                )
                projected_cash += pos.qty * price
                active_symbols.discard(symbol)

        # ------------------------------------------------------------------
        # Step 2: Process BUY candidates (sorted by signal priority score)
        # ------------------------------------------------------------------
        buy_candidates = [
            s
            for s in signals
            if s.action == "BUY" and s.symbol not in active_symbols
        ]
        buy_candidates.sort(key=lambda s: s.score, reverse=True)

        cash_reserve = total_equity * self.settings.cash_buffer_pct
        max_allocation_per_stock = total_equity * self.settings.max_position_pct

        for candidate in buy_candidates:
            if len(active_symbols) >= self.settings.max_positions:
                risk_logs.append(
                    f"{candidate.symbol}: 跳过买入 — 已达到最大同时持仓数限制 ({self.settings.max_positions} 只)"
                )
                continue

            spendable_cash = max(0.0, projected_cash - cash_reserve)
            target_budget = min(max_allocation_per_stock, spendable_cash)

            if target_budget < candidate.price:
                risk_logs.append(
                    f"{candidate.symbol}: 跳过买入 — 可用预算(${target_budget:.2f})不足以购买 1 股(${candidate.price:.2f})"
                )
                continue

            # Phase 2 Hook: Pre-Trade Guard (e.g., LLM News Sentiment Veto)
            if self.pre_trade_guard is not None:
                approved, guard_note = self.pre_trade_guard(
                    candidate.symbol, candidate
                )
                if not approved:
                    risk_logs.append(
                        f"{candidate.symbol}: [LLM/风控拦截] 一票否决买入 — {guard_note}"
                    )
                    continue

            qty = int(target_budget // candidate.price)
            if qty <= 0:
                continue

            cost = qty * candidate.price
            projected_cash -= cost
            active_symbols.add(candidate.symbol)

            orders.append(
                OrderDecision(
                    symbol=candidate.symbol,
                    side="BUY",
                    qty=qty,
                    estimated_price=candidate.price,
                    reason=candidate.reason,
                    is_stop_loss=False,
                )
            )

        return orders, risk_logs
