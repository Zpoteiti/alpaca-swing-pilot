"""Phase 2: Pluggable LLM News Sentiment & Fundamental Risk Guardrail.

Connects to Alpaca's free News API and a local OpenAI-compatible LLM endpoint
(default: http://localhost:11435/v1) to review recent news before approving BUY orders
or auditing daily market conditions after US market close.

Handles concurrent agent contention on the local LLM server via generous timeouts
and automatic backoff retries.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from swing_pilot.config import Settings
from swing_pilot.strategies import SignalResult


@dataclass
class NewsAuditResult:
    """Structured result from LLM analysis of recent symbol news."""

    symbol: str
    approved: bool
    sentiment: str  # "bullish" | "neutral" | "bearish"
    risk_level: str  # "low" | "medium" | "high"
    summary_zh: str
    news_count: int
    model_id: str
    latency_sec: float


class LocalLLMNewsGuard:
    """Fetches Alpaca News and queries local OpenAI-compatible LLM as a Pre-Trade Guard."""

    def __init__(
        self,
        settings: Settings,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_sec: int = 120,
        max_retries: int = 4,
    ) -> None:
        self.settings = settings
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL", "http://localhost:11435/v1")
        ).rstrip("/")
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

        self.news_client: Optional[NewsClient] = None
        if settings.has_credentials:
            self.news_client = NewsClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

    def discover_model(self) -> str:
        """Auto-detect loaded model ID from /v1/models if not explicitly set."""
        if self.model_id:
            return self.model_id
        try:
            resp = requests.get(
                f"{self.base_url}/models", timeout=15
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data and isinstance(data, list):
                self.model_id = str(data[0].get("id", "local-model"))
                return self.model_id
        except Exception:
            pass
        return "local-model"

    def fetch_recent_news(
        self, symbol: str, limit: int = 6, lookback_hours: int = 72
    ) -> List[Dict[str, str]]:
        """Fetch recent headlines & summaries for a symbol from Alpaca News API."""
        if self.news_client is None:
            return []

        start_dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        try:
            req = NewsRequest(
                symbols=symbol,
                start=start_dt,
                limit=limit,
                include_content=False,
            )
            news_set = self.news_client.get_news(req)
            raw_list = news_set.data.get("news", []) if hasattr(news_set, "data") else []
            items: List[Dict[str, str]] = []
            for n in raw_list:
                headline = str(getattr(n, "headline", "") or "").strip()
                summary = str(getattr(n, "summary", "") or "").strip()
                created = str(getattr(n, "created_at", "") or "")[:16]
                source = str(getattr(n, "source", "") or "")
                if headline:
                    items.append(
                        {
                            "time": created,
                            "source": source,
                            "headline": headline,
                            "summary": summary[:280],
                        }
                    )
            return items
        except Exception:
            return []

    def _call_llm_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Call /v1/chat/completions with exponential backoff in case another agent is using the GPU."""
        model = self.discover_model()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in (429, 502, 503, 504):
                    wait_s = min(8 * attempt, 30)
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                body = resp.json()
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return str(content)
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(min(5 * attempt, 20))
        raise RuntimeError(f"本地 LLM 调用失败 (已重试 {self.max_retries} 次): {last_err}")

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Strip <think>...</think> tags or markdown fences and parse JSON."""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "approved": True,
            "sentiment": "neutral",
            "risk_level": "low",
            "summary_zh": cleaned[:120] if cleaned else "模型未返回标准JSON，默认放行",
        }

    def audit_symbol(
        self, symbol: str, signal: SignalResult
    ) -> NewsAuditResult:
        """Fetch latest news for symbol and run LLM risk & sentiment audit."""
        t0 = time.perf_counter()
        news_items = self.fetch_recent_news(symbol, limit=6)
        model = self.discover_model()

        if not news_items:
            return NewsAuditResult(
                symbol=symbol,
                approved=True,
                sentiment="neutral",
                risk_level="low",
                summary_zh="近72小时无重大异动新闻，按量化信号正常执行",
                news_count=0,
                model_id=model,
                latency_sec=round(time.perf_counter() - t0, 2),
            )

        news_text = "\n".join(
            f"{i+1}. [{item['time']}] {item['headline']} — {item['summary']}"
            for i, item in enumerate(news_items)
        )
        ind_text = ", ".join(f"{k}={v}" for k, v in signal.indicators.items())

        system_prompt = (
            "你是一位严谨的美股量化基金风控守门员(Risk Guard)。"
            "你的任务是结合量化指标与最新英文财经新闻，排查标的是否存在【重大基本面暴雷】"
            "（如财务造假、监管重罚、核心产品致命缺陷、业绩指引暴雷等）。"
            "如果只是正常的大盘波动或普通行业新闻，应当放行(approved=true)。"
            "请严格只输出一行合法的 JSON，不要输出任何额外解释，格式如下：\n"
            '{"approved": true, "sentiment": "bullish|neutral|bearish", '
            '"risk_level": "low|medium|high", "summary_zh": "一句话中文总结新闻面与风控结论(40字以内)"}'
        )

        user_prompt = (
            f"标的: {symbol}\n"
            f"当前量化信号: {signal.action} ({signal.reason})\n"
            f"当前核心指标: {ind_text}\n\n"
            f"最近新闻列表:\n{news_text}\n\n"
            "请输出 JSON 风控评估结果 (/no_think)："
        )

        raw = self._call_llm_with_retry(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = self._parse_json_response(raw)

        return NewsAuditResult(
            symbol=symbol,
            approved=bool(parsed.get("approved", True)),
            sentiment=str(parsed.get("sentiment", "neutral")),
            risk_level=str(parsed.get("risk_level", "low")),
            summary_zh=str(parsed.get("summary_zh", "新闻面正常")),
            news_count=len(news_items),
            model_id=model,
            latency_sec=round(time.perf_counter() - t0, 2),
        )

    def as_pre_trade_guard(
        self, symbol: str, signal: SignalResult
    ) -> tuple[bool, str]:
        """Adapter matching PreTradeGuardFn signature for RiskManager."""
        audit = self.audit_symbol(symbol, signal)
        note = (
            f"[{audit.sentiment.upper()}/风险:{audit.risk_level}] "
            f"{audit.summary_zh} (扫描{audit.news_count}条新闻, {audit.latency_sec}s)"
        )
        return audit.approved, note
