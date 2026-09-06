"""
app.backtest.engine — 自研向量化回测引擎
================================================================
职责：基于 Phase 2 输出的特征 DataFrame（含 MACD/RSI），用 pandas/numpy
     手写持仓序列与逐笔交易统计，计算核心绩效指标。

策略（MACD + RSI 组合信号）：
    - 买入：MACD 金叉（MACD 上穿 MACD_signal）且 RSI14 < 70（超买不追）。
    - 卖出：MACD 死叉（MACD 下穿 MACD_signal）或 RSI14 > 70（超买离场）。

回测约定：
    - 信号日收盘定仓、次日计收益（pos.shift(1)），避免前视偏差。
    - 进出各计一次手续费 + 滑点。
    - 净值曲线从 1 起归一化，指标皆为比率、与初始本金无关。

用法：
    from app.backtest.engine import run_backtest
    result = run_backtest(df, symbol="600519")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

#: 年化交易日数
_TRADING_DAYS: int = 252


@dataclass
class BacktestResult:
    """回测结果与绩效指标。

    Attributes:
        symbol: 标的代码。
        total_return: 累计收益率（如 0.25 表示 25%）。
        annualized_return: 年化收益率（CAGR）。
        sharpe_ratio: 夏普比率。
        max_drawdown: 最大回撤（负值，如 -0.18 表示 -18%）。
        win_rate: 胜率（0~1）。
        num_trades: 交易笔数。
        equity_curve: 净值曲线（从 1 起）。
        trade_returns: 逐笔交易收益。
        metrics: 以上标量的 dict 汇总，供看板直接取用。
    """

    symbol: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series
    trade_returns: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)


def generate_macd_rsi_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """生成 MACD 金叉/死叉 + RSI 过滤的买卖信号。

    Args:
        df: 含 ``MACD`` / ``MACD_signal`` / ``RSI14`` 列的特征 DataFrame。

    Returns:
        ``(entry, exit)`` 布尔 Series。

    Raises:
        ValueError: 缺失指标列。
    """
    required: set[str] = {"MACD", "MACD_signal", "RSI14"}
    missing: set[str] = required - set(df.columns)
    if missing:
        raise ValueError(f"特征 DataFrame 缺失指标列: {missing}")

    macd: pd.Series = df["MACD"]
    signal: pd.Series = df["MACD_signal"]
    golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))

    entry: pd.Series = golden_cross & (df["RSI14"] < 70.0)
    exit_: pd.Series = death_cross | (df["RSI14"] > 70.0)
    return entry.fillna(False), exit_.fillna(False)


def _build_position(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """由买卖信号构建多头持仓序列（1 持仓 / 0 空仓）。"""
    pos: pd.Series = pd.Series(np.nan, index=entry.index, dtype=float)
    pos[entry] = 1.0
    pos[exit_] = 0.0
    return pos.ffill().fillna(0.0)


def _extract_trades(
    position: pd.Series,
    close: pd.Series,
    cost_per_side: float,
) -> pd.Series:
    """从持仓序列提取逐笔交易收益（含双边成本）。"""
    rets: list[float] = []
    entry_price: float | None = None
    prev: float = 0.0
    for idx, pos_val in position.items():
        if prev == 0.0 and pos_val == 1.0:  # 建仓
            entry_price = float(close.loc[idx])
        elif prev == 1.0 and pos_val == 0.0:  # 平仓
            if entry_price is not None:
                rets.append(float(close.loc[idx]) / entry_price - 1.0 - 2.0 * cost_per_side)
                entry_price = None
        prev = float(pos_val)
    # 末尾仍持仓：按最后收盘价平仓
    if entry_price is not None:
        rets.append(float(close.iloc[-1]) / entry_price - 1.0 - 2.0 * cost_per_side)
    return pd.Series(rets, dtype=float)


def run_backtest(
    df: pd.DataFrame,
    symbol: str = "",
    entry: pd.Series | None = None,
    exit_: pd.Series | None = None,
    commission: float = 0.0003,
    slippage: float = 0.0002,
    risk_free_rate: float = 0.02,
) -> BacktestResult:
    """运行向量化回测并计算绩效指标。

    Args:
        df: 12 列特征 DataFrame（index 为日期，含 ``Close`` 及指标列）。
        symbol: 标的代码（仅用于日志/展示）。
        entry: 自定义买入布尔信号；不传则用 MACD+RSI 默认策略。
        exit_: 自定义卖出布尔信号；不传则用 MACD+RSI 默认策略。
        commission: 手续费率（单边）。
        slippage: 滑点率（单边）。
        risk_free_rate: 年化无风险利率（用于夏普）。

    Returns:
        ``BacktestResult``。

    Raises:
        ValueError: 输入为空或缺失 ``Close`` 列。
    """
    if df is None or df.empty:
        raise ValueError("回测输入 DataFrame 为空")
    if "Close" not in df.columns:
        raise ValueError("回测输入缺失 Close 列")
    if entry is None or exit_ is None:
        entry, exit_ = generate_macd_rsi_signals(df)

    close: pd.Series = df["Close"]
    position: pd.Series = _build_position(entry, exit_)

    daily_ret: pd.Series = close.pct_change().fillna(0.0)
    cost_per_side: float = commission + slippage
    turnover: pd.Series = position.diff().abs().fillna(0.0)
    # 信号日收盘定仓、次日计收益；换手当日扣成本
    strategy_ret: pd.Series = position.shift(1).fillna(0.0) * daily_ret - turnover * cost_per_side

    equity: pd.Series = (1.0 + strategy_ret).cumprod()

    n: int = len(strategy_ret)
    total_return: float = float(equity.iloc[-1] - 1.0)
    annualized: float = float(equity.iloc[-1] ** (_TRADING_DAYS / n) - 1.0) if n > 0 else 0.0

    rf_daily: float = risk_free_rate / _TRADING_DAYS
    excess: pd.Series = strategy_ret - rf_daily
    std: float = float(excess.std(ddof=0))
    sharpe: float = float((excess.mean() / std) * np.sqrt(_TRADING_DAYS)) if std > 1e-12 else 0.0

    drawdown: pd.Series = equity / equity.cummax() - 1.0
    max_dd: float = float(drawdown.min())

    trade_returns: pd.Series = _extract_trades(position, close, cost_per_side)
    num_trades: int = int(trade_returns.size)
    win_rate: float = float((trade_returns > 0).mean()) if num_trades > 0 else 0.0

    metrics: dict[str, float] = {
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": float(num_trades),
    }
    logger.info(
        "回测完成 symbol=%s 笔数=%d 累计收益=%.2f%% 夏普=%.2f 最大回撤=%.2f%%",
        symbol or "-", num_trades, total_return * 100, sharpe, max_dd * 100,
    )
    return BacktestResult(
        symbol=symbol,
        total_return=total_return,
        annualized_return=annualized,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity,
        trade_returns=trade_returns,
        metrics=metrics,
    )


if __name__ == "__main__":
    # 测试入口：拉取 600519 3 年特征 → 回测 → 打印绩效
    from app.features.pipeline import FeaturePipeline
    from app.utils.logger import configure_logging

    configure_logging()
    feature_df: pd.DataFrame = FeaturePipeline().run(symbol="600519", period="3y")
    result: BacktestResult = run_backtest(feature_df, symbol="600519")

    print("\n===== 回测绩效指标（600519 / MACD+RSI 策略） =====")
    print(f"交易笔数   : {result.num_trades}")
    print(f"累计收益率 : {result.total_return:.2%}")
    print(f"年化收益率 : {result.annualized_return:.2%}")
    print(f"夏普比率   : {result.sharpe_ratio:.3f}")
    print(f"最大回撤   : {result.max_drawdown:.2%}")
    print(f"胜率       : {result.win_rate:.2%}")
