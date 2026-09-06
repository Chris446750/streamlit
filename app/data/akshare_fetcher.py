"""
app.data.akshare_fetcher — AkShare（腾讯后端）A 股数据适配
================================================================
职责：基于 AkShare 的 ``stock_zh_a_hist_tx``（腾讯 ifzq.gtimg.cn 后端）
     拉取 A 股前复权日线，内置指数退避重试与日志。

背景：yfinance（Yahoo）在本网络被限流/墙、东财 push2his 后端反复
     ``RemoteDisconnected``，而腾讯源实测稳定可用，故 Phase 2 默认
     数据源切换为 AkShare（腾讯后端）+ A 股高流动性标的。

用法：
    from app.data.akshare_fetcher import AkShareFetcher
    fetcher = AkShareFetcher()
    df = fetcher.fetch_kline("600519", period="3y")
"""
from __future__ import annotations

import re
from datetime import date, datetime

import akshare as ak
import pandas as pd
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.data.base import DataFetcher, standardize_ohlcv
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: 腾讯后端日期参数格式（无分隔符，如 "20230906"）
_DATE_FMT: str = "%Y%m%d"

#: 支持的自带交易所前缀（上交所 sh / 深交所 sz / 北交所 bj）
_VALID_PREFIX: tuple[str, ...] = ("sh", "sz", "bj")

#: 按代码首位数推断交易所前缀（覆盖 A 股主流板块）
_PREFIX_BY_FIRST_DIGIT: dict[str, str] = {
    "6": "sh",  # 上交所主板
    "5": "sh",  # 上交所 ETF / 基金
    "9": "sh",  # 上交所 B 股
    "0": "sz",  # 深交所主板
    "1": "sz",  # 深交所基金
    "3": "sz",  # 深交所创业板
    "4": "bj",  # 北交所
    "8": "bj",  # 北交所（新三板精选层转板后沿用）
}

#: 日线区间解析正则：数字 + 单位（y=年 / mo|m=月 / d=天）
_PERIOD_RE: re.Pattern[str] = re.compile(r"^(\d+)\s*(y|mo|m|d)$", re.IGNORECASE)


def _normalize_symbol(symbol: str) -> str:
    """把用户输入的 A 股代码规范化为腾讯后端格式（``sh600519``）。

    支持两种输入：
        - 已带前缀：``"sh600519"`` / ``"SZ000001"`` / ``"bj430047"``。
        - 纯 6 位代码：``"600519"``，按首位数自动推断交易所。

    Args:
        symbol: 标的代码。

    Returns:
        形如 ``sh600519`` 的腾讯后端代码。

    Raises:
        ValueError: 代码格式无法识别。
    """
    s: str = symbol.strip().lower()
    if re.match(r"^(sh|sz|bj)\d{6}$", s):
        return s
    if re.match(r"^\d{6}$", s):
        prefix: str | None = _PREFIX_BY_FIRST_DIGIT.get(s[0])
        if prefix is None:
            raise ValueError(f"无法识别 A 股代码首位数 {s[0]!r}，symbol={symbol!r}")
        return prefix + s
    raise ValueError(f"无法识别 A 股代码 symbol={symbol!r}，请使用 '600519' 或 'sh600519' 形式")


def _resolve_date_range(period: str) -> tuple[date, date]:
    """把 ``period``（如 ``"3y"``）换算为 (start_date, end_date) 闭区间。

    Args:
        period: 时间跨度，支持 ``3y`` / ``6m`` / ``30d`` 等。

    Returns:
        ``(start_date, end_date)``，end_date 固定为今天。

    Raises:
        ValueError: period 格式无法解析。
    """
    m: re.Match[str] | None = _PERIOD_RE.match(period.strip())
    if m is None:
        raise ValueError(f"无法解析 period={period!r}，支持格式如 '3y'/'6m'/'30d'")

    n: int = int(m.group(1))
    unit: str = m.group(2).lower()
    end: pd.Timestamp = pd.Timestamp(date.today())
    if unit == "y":
        start: pd.Timestamp = end - pd.DateOffset(years=n)
    elif unit in ("m", "mo"):
        start = end - pd.DateOffset(months=n)
    else:
        start = end - pd.DateOffset(days=n)
    return start.date(), end.date()


class AkShareFetcher(DataFetcher):
    """AkShare A 股数据获取器（腾讯后端，前复权日线）。

    Attributes:
        adjust: 复权方式，默认 ``"qfq"``（前复权），可选 ``"hfq"`` / ``""``（不复权）。
        max_retries: 网络请求最大重试次数，默认 3。
    """

    def __init__(self, adjust: str = "qfq", max_retries: int = 3) -> None:
        self._adjust: str = adjust
        self._max_retries: int = max_retries

    def fetch_kline(
        self,
        symbol: str,
        period: str = "3y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """拉取并标准化 A 股日线数据（含重试与日志）。

        Args:
            symbol: A 股代码，如 ``"600519"`` 或 ``"sh600519"``。
            period: 时间跨度，如 ``"3y"``。
            interval: K 线周期，腾讯后端仅支持日线（``"1d"``）。

        Returns:
            标准化 OHLCV DataFrame（列序 Open/High/Low/Close/Volume）。

        Raises:
            ValueError: interval 非日线，或 symbol/period 无法解析。
            RuntimeError: 多次重试后仍失败或返回空数据。
        """
        if interval not in ("1d", "daily", "day"):
            raise ValueError(f"腾讯后端仅支持日线，收到 interval={interval!r}")

        tx_symbol: str = _normalize_symbol(symbol)
        start, end = _resolve_date_range(period)

        logger.info(
            "开始请求数据 symbol=%s(%s) period=%s interval=%s",
            symbol, tx_symbol, period, interval,
        )
        try:
            raw: pd.DataFrame = self._fetch_with_retry(tx_symbol, start, end)
            df: pd.DataFrame = self._to_ohlcv(raw)
        except Exception as exc:  # noqa: BLE001 — 记录后向上抛出
            logger.error("请求失败 symbol=%s 原因=%s", symbol, exc)
            raise
        logger.info("请求成功 symbol=%s rows=%d", symbol, len(df))
        return df

    def _fetch_with_retry(
        self,
        tx_symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """真正发起网络请求，由 tenacity 驱动指数退避重试。"""
        for attempt in Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                df: pd.DataFrame = ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start.strftime(_DATE_FMT),
                    end_date=end.strftime(_DATE_FMT),
                    adjust=self._adjust,
                )
                if df is None or df.empty:
                    raise RuntimeError(f"腾讯源返回空数据 symbol={tx_symbol}")
                return df
        # 理论上不会执行到这里（reraise=True 会重新抛出最后异常），仅作类型完整性兜底
        raise RuntimeError(f"重试耗尽后仍未获取到数据 symbol={tx_symbol}")

    @staticmethod
    def _to_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
        """把腾讯源原始 DataFrame 转成标准 OHLCV（复用 base.standardize_ohlcv）。"""
        if "date" not in raw.columns:
            raise ValueError(f"腾讯源返回缺少 date 列，实际列={list(raw.columns)}")
        out: pd.DataFrame = raw.copy()
        # date 为字符串（object），先转 datetime 再设为 index
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
        # standardize_ohlcv 会自动把小写列 open/close/high/low/volume 重命名为标准大写列
        return standardize_ohlcv(out)
