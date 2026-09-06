"""
app.data.yfinance_fetcher — yfinance 数据源适配
================================================================
职责：基于 yfinance 拉取美股/港股历史 K 线，内置指数退避重试与日志。
     作为 Phase 2 默认测试数据源（美股高流动性标的）。

用法：
    from app.data.yfinance_fetcher import YFinanceFetcher
    fetcher = YFinanceFetcher()
    df = fetcher.fetch_kline("SPY", period="3y")
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.data.base import DataFetcher, standardize_ohlcv
from app.utils.logger import get_logger

logger = get_logger(__name__)


class YFinanceFetcher(DataFetcher):
    """yfinance 数据获取器（美股/港股）。

    Attributes:
        max_retries: 网络请求最大重试次数，默认 3。
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries

    def fetch_kline(
        self,
        symbol: str,
        period: str = "3y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """拉取并标准化日线数据（含重试与日志）。

        Args:
            symbol: 标的代码，如 ``"SPY"``。
            period: 时间跨度，如 ``"3y"``。
            interval: K 线周期，如 ``"1d"``。

        Returns:
            标准化 OHLCV DataFrame。

        Raises:
            RuntimeError: 多次重试后仍失败或返回空数据。
        """
        logger.info("开始请求数据 symbol=%s period=%s interval=%s", symbol, period, interval)
        try:
            raw = self._fetch_with_retry(symbol, period, interval)
            df = standardize_ohlcv(raw)
        except Exception as exc:  # noqa: BLE001 — 记录后向上抛出
            logger.error("请求失败 symbol=%s 原因=%s", symbol, exc)
            raise
        logger.info("请求成功 symbol=%s rows=%d", symbol, len(df))
        return df

    def _fetch_with_retry(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        """真正发起网络请求，由 tenacity 驱动指数退避重试。"""
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                df = yf.Ticker(symbol).history(
                    period=period, interval=interval, auto_adjust=True
                )
                if df is None or df.empty:
                    raise RuntimeError(f"yfinance 返回空数据 symbol={symbol}")
                return df
        # 理论上不会执行到这里（reraise=True 会重新抛出最后异常），仅作类型完整性兜底
        raise RuntimeError(f"重试耗尽后仍未获取到数据 symbol={symbol}")
