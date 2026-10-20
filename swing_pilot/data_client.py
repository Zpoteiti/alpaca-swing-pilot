"""Alpaca Market Data & Paper Trading API Client Wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from swing_pilot.config import Settings
from swing_pilot.risk_manager import OrderDecision, PositionInfo


class AlpacaService:
    """Unified wrapper around Alpaca Historical Data and Paper Trading APIs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.has_credentials:
            raise ValueError(
                "未检测到有效的 Alpaca API Key。请复制 .env.example 为 .env 并填入你的模拟盘密钥。"
            )

        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
        )

    def fetch_daily_bars(
        self, symbols: List[str], lookback_days: int = 1250
    ) -> Dict[str, pd.DataFrame]:
        """Fetch split- and dividend-adjusted daily OHLCV bars for multiple symbols."""
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=16)
        start_dt = end_dt - timedelta(days=lookback_days)

        # Try SIP feed first (full US market volume), fallback to IEX if free tier restricts SIP
        for feed in (DataFeed.SIP, DataFeed.IEX):
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=TimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                    adjustment=Adjustment.ALL,
                    feed=feed,
                )
                bars = self.data_client.get_stock_bars(req)
                df_all = bars.df
                if df_all.empty:
                    continue

                result: Dict[str, pd.DataFrame] = {}
                for sym in symbols:
                    if sym in df_all.index.get_level_values(0):
                        sub = df_all.xs(sym, level=0).copy()
                        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
                        result[sym] = sub
                if result:
                    return result
            except Exception:
                if feed == DataFeed.IEX:
                    raise
                continue

        return {}

    def get_account_summary(self) -> Dict[str, float | str | bool]:
        """Return current account equity, cash, buying power, and PDT status."""
        acct = self.trading_client.get_account()
        return {
            "account_number": str(acct.account_number),
            "status": str(acct.status),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "daytrade_count": int(acct.daytrade_count or 0),
            "pattern_day_trader": bool(acct.pattern_day_trader),
            "paper": self.settings.alpaca_paper,
        }

    def get_open_positions(self) -> Dict[str, PositionInfo]:
        """Fetch current open stock/ETF positions from Alpaca."""
        raw_positions = self.trading_client.get_all_positions()
        positions: Dict[str, PositionInfo] = {}
        for pos in raw_positions:
            qty = int(float(pos.qty))
            if qty <= 0:
                continue
            symbol = str(pos.symbol).upper()
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            positions[symbol] = PositionInfo(
                symbol=symbol,
                qty=qty,
                avg_entry_price=entry_price,
                current_price=current_price,
                highest_price=max(entry_price, current_price),
            )
        return positions

    def submit_orders(
        self, orders: List[OrderDecision], current_positions: Dict[str, PositionInfo]
    ) -> List[Dict[str, str | int | float]]:
        """Submit risk-checked orders to Alpaca with Long-Only hardware safety check."""
        receipts: List[Dict[str, str | int | float]] = []

        for order in orders:
            if order.qty <= 0:
                continue

            # Hard Long-Only verification right before API call
            if order.side == "SELL":
                held = current_positions.get(order.symbol)
                if not held or held.qty <= 0:
                    raise RuntimeError(
                        f"安全拦截：尝试卖出未持有的标的 {order.symbol}（禁止做空）"
                    )
                safe_qty = min(order.qty, held.qty)
                side_enum = OrderSide.SELL
            else:
                safe_qty = order.qty
                side_enum = OrderSide.BUY

            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=safe_qty,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
            )
            res = self.trading_client.submit_order(order_data=req)
            receipts.append(
                {
                    "id": str(res.id),
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": safe_qty,
                    "status": str(res.status),
                    "reason": order.reason,
                }
            )
        return receipts


def generate_synthetic_bars(
    symbols: List[str], days: int = 800, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Generate realistic multi-year daily OHLCV bars for offline demo & unit testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.now().date(), periods=days)
    data: Dict[str, pd.DataFrame] = {}

    base_configs = {
        "SPY": (400.0, 0.00048, 0.0095),
        "QQQ": (330.0, 0.00058, 0.0125),
        "AAPL": (150.0, 0.00055, 0.0155),
        "MSFT": (280.0, 0.00056, 0.0145),
        "NVDA": (45.0, 0.00115, 0.0240),
        "GOOGL": (110.0, 0.00052, 0.0160),
        "AMZN": (120.0, 0.00054, 0.0165),
    }

    # Shared market factor + occasional dip regimes so mean-reversion & trend both trigger
    market_shocks = rng.normal(0.0, 0.007, size=days)
    for idx in range(220, days - 20, 65):
        # Simulate realistic 4-day market pullbacks (-1.2% per day) followed by recovery
        market_shocks[idx : idx + 4] -= 0.014
        market_shocks[idx + 4 : idx + 9] += 0.009

    for sym in symbols:
        start_price, drift, vol = base_configs.get(sym, (100.0, 0.00045, 0.014))
        idio = rng.normal(drift, vol, size=days)
        rets = market_shocks * 0.65 + idio * 0.75
        close = start_price * np.cumprod(1.0 + rets)
        high = close * (1.0 + np.abs(rng.normal(0.0, 0.006, size=days)))
        low = close * (1.0 - np.abs(rng.normal(0.0, 0.006, size=days)))
        open_p = np.roll(close, 1)
        open_p[0] = start_price
        volume = rng.integers(15_000_000, 80_000_000, size=days)

        df = pd.DataFrame(
            {
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        data[sym] = df

    return data
