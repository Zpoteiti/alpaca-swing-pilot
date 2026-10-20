"""Configuration management for Alpaca Swing Pilot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Typed configuration for trading universe, strategies, and risk guardrails."""

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # Default universe: High-liquidity US broad-market ETFs + Mega-cap tech leaders
    symbols: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    )
    benchmark_symbol: str = "SPY"
    default_strategy: str = "rsi_pullback"

    # Risk guardrails (4 Iron Rules)
    allow_short: bool = False  # Strictly Long-Only
    max_positions: int = 5  # Max concurrent holdings
    max_position_pct: float = 0.20  # Max 20% of total equity per single symbol
    stop_loss_pct: float = 0.08  # 8% hard stop-loss from entry price
    trailing_stop_pct: float = 0.18  # 18% trailing stop from peak since entry
    cash_buffer_pct: float = 0.02  # Keep 2% cash reserve to avoid margin usage

    @property
    def has_credentials(self) -> bool:
        """Check whether valid-looking Alpaca API credentials are configured."""
        return bool(
            self.alpaca_api_key
            and self.alpaca_secret_key
            and "your_" not in self.alpaca_api_key
        )


def load_settings(env_path: str | Path | None = None) -> Settings:
    """Load settings from .env file or environment variables."""
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    # Support both ALPACA_SECRET_KEY and ALPACA_API_SECRET naming conventions
    secret_key = (
        os.getenv("ALPACA_SECRET_KEY", "").strip()
        or os.getenv("ALPACA_API_SECRET", "").strip()
    )

    paper_raw = os.getenv("ALPACA_PAPER", "true").strip().lower()
    alpaca_paper = paper_raw not in ("false", "0", "no")

    symbols_raw = os.getenv(
        "TRADING_SYMBOLS", "SPY,QQQ,AAPL,MSFT,NVDA,GOOGL,AMZN"
    ).strip()
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    benchmark_symbol = os.getenv("BENCHMARK_SYMBOL", "SPY").strip().upper()
    default_strategy = os.getenv("DEFAULT_STRATEGY", "rsi_pullback").strip().lower()

    max_positions = int(os.getenv("MAX_POSITIONS", "5"))
    max_position_pct = float(os.getenv("MAX_POSITION_PCT", "0.20"))
    stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.08"))
    trailing_stop_pct = float(os.getenv("TRAILING_STOP_PCT", "0.18"))

    return Settings(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        alpaca_paper=alpaca_paper,
        symbols=symbols,
        benchmark_symbol=benchmark_symbol,
        default_strategy=default_strategy,
        allow_short=False,
        max_positions=max_positions,
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
