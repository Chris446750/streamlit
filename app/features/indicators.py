"""
app.features.indicators — 自研技术指标计算
================================================================
职责：用 pandas/numpy 手写 MACD、布林带（BOLL）、RSI 三大核心指标。
     公式与 pandas-ta 默认实现一致，但零外部依赖、可完整类型提示与单测。

指标列约定：
    MACD          快线 DIF = EMA(close, fast) - EMA(close, slow)
    MACD_signal   慢线 DEA = EMA(DIF, signal)
    MACD_hist     柱状图  = DIF - DEA
    BOLL_upper / BOLL_mid / BOLL_lower  布林上/中/下轨（20 日，2 倍标准差）
    RSI14         14 日相对强弱（Wilder 平滑）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 指标计算所依赖的收盘价列名
_CLOSE: str = "Close"


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """计算 MACD 快慢线与柱状图。

    Args:
        df: 含 ``Close`` 列的行情 DataFrame。
        fast: 快线 EMA 周期，默认 12。
        slow: 慢线 EMA 周期，默认 26。
        signal: 信号线 EMA 周期，默认 9。

    Returns:
        追加 ``MACD`` / ``MACD_signal`` / ``MACD_hist`` 三列的副本。
    """
    out: pd.DataFrame = df.copy()
    close: pd.Series = out[_CLOSE]
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    out["MACD"] = dif
    out["MACD_signal"] = dea
    out["MACD_hist"] = dif - dea
    return out


def add_boll(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """计算布林带（上/中/下轨）。

    Args:
        df: 含 ``Close`` 列的行情 DataFrame。
        window: 移动平均窗口，默认 20。
        num_std: 标准差倍数，默认 2。

    Returns:
        追加 ``BOLL_upper`` / ``BOLL_mid`` / ``BOLL_lower`` 三列的副本。
    """
    out: pd.DataFrame = df.copy()
    close: pd.Series = out[_CLOSE]
    mid = close.rolling(window).mean()
    # ddof=0（总体标准差），与 pandas-ta / TA-Lib 的 BOLL 默认一致
    std = close.rolling(window).std(ddof=0)
    out["BOLL_mid"] = mid
    out["BOLL_upper"] = mid + num_std * std
    out["BOLL_lower"] = mid - num_std * std
    return out


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """计算 RSI 相对强弱指标（Wilder 平滑）。

    Args:
        df: 含 ``Close`` 列的行情 DataFrame。
        length: 周期，默认 14。

    Returns:
        追加 ``RSI14`` 列的副本。
    """
    out: pd.DataFrame = df.copy()
    close: pd.Series = out[_CLOSE]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    # 抑制除零告警：当 avg_loss=0 时 RS=inf，RSI 取 100（无下跌的极限情形）
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    out["RSI14"] = rsi
    return out


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算全部核心指标。

    Args:
        df: 含 ``Close`` 列的行情 DataFrame。

    Returns:
        依次追加 MACD、BOLL、RSI 的副本。
    """
    df = add_macd(df)
    df = add_boll(df)
    df = add_rsi(df)
    return df
