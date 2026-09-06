"""
app.data.base — 数据获取抽象基类
================================================================
职责：定义统一的数据获取接口与 OHLCV 标准化逻辑，
     保证 yfinance / AkShare / Tushare / ccxt 等异构数据源
     返回结构一致的 DataFrame，供下游特征计算直接消费。

约定：所有 Fetcher 的 fetch_kline 必须返回列序为
      [Open, High, Low, Close, Volume] 的 DataFrame，
      index 为升序、无时区的 DatetimeIndex。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

#: 标准化行情列（顺序固定，下游依赖此顺序）
OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


class DataFetcher(ABC):
    """数据源抽象基类。子类需实现 fetch_kline。"""

    @abstractmethod
    def fetch_kline(
        self,
        symbol: str,
        period: str = "3y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """拉取指定标的的历史 K 线。

        Args:
            symbol: 标的代码，如 ``"SPY"``、``"AAPL"``。
            period: 时间跨度，如 ``"3y"``、``"1y"``。
            interval: K 线周期，如 ``"1d"``、``"1wk"``。

        Returns:
            标准化 DataFrame（列为 OHLCV_COLUMNS，index 为 DatetimeIndex）。
        """
        raise NotImplementedError


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """将异构数据源返回的行情 DataFrame 标准化为统一结构。

    处理步骤：
        1. 统一列名（兼容全小写变体）。
        2. 校验 OHLCV 五列完整。
        3. 仅保留 OHLCV 五列。
        4. index 转无时区 DatetimeIndex（保留墙钟时间，避免日期漂移）。
        5. 去重、按日期升序。

    Args:
        df: 原始行情 DataFrame（可能含 Dividends / Stock Splits 等额外列）。

    Returns:
        仅含 OHLCV 五列、index 为无时区 DatetimeIndex 的 DataFrame。

    Raises:
        ValueError: 数据为空或缺失必需列。
    """
    if df is None or df.empty:
        raise ValueError("行情数据为空，无法标准化")

    # 1) 列名统一：将全小写等变体重命名为标准大写列
    rename_map: dict[str, str] = {c.lower(): c.title() for c in OHLCV_COLUMNS}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2) 校验必需列
    missing: list[str] = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"行情数据缺失必需列: {missing}")

    # 3) 仅保留 OHLCV
    out: pd.DataFrame = df[OHLCV_COLUMNS].copy()

    # 4) index 归一化：转为无时区 DatetimeIndex（tz_localize(None) 保留墙钟时间）
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    # 5) 去重 + 升序
    out = out[~out.index.duplicated(keep="last")].sort_index()

    return out
