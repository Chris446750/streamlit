"""
app.features.pipeline — 数据 + 特征编排流水线
================================================================
职责：把「取数 → 指标计算 → 缺失值处理」串成一条可复用流水线，
     并在模块底部提供 if __name__ == "__main__" 端到端测试入口。

运行方式（务必在项目根目录）：
    .venv/Scripts/python.exe -m app.features.pipeline
"""
from __future__ import annotations

import pandas as pd

from app.data.akshare_fetcher import AkShareFetcher
from app.data.base import DataFetcher
from app.features.indicators import add_all_indicators
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """数据采集 + 特征计算流水线。

    Attributes:
        fetcher: 数据源实例，默认使用 AkShare（腾讯后端）。
    """

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self._fetcher: DataFetcher = fetcher if fetcher is not None else AkShareFetcher()

    def run(
        self,
        symbol: str = "600519",
        period: str = "3y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """执行完整流程：取数 → 指标 → dropna。

        Args:
            symbol: 标的代码，默认 ``"600519"``（贵州茅台）。
            period: 时间跨度，默认 ``"3y"``。
            interval: K 线周期，默认 ``"1d"``。

        Returns:
            合并指标并去除 NaN 后的 DataFrame。
        """
        df = self._fetcher.fetch_kline(symbol, period=period, interval=interval)
        df = add_all_indicators(df)
        df = df.dropna()
        logger.info(
            "特征计算完成 symbol=%s rows=%d columns=%d",
            symbol, len(df), len(df.columns),
        )
        return df


if __name__ == "__main__":
    # 端到端测试入口：拉取 600519（贵州茅台）3 年日线 → 计算指标 → 打印最后 5 行
    configure_logging()
    # 让终端完整显示全部 12 列，便于核对最后 5 行是否逐列对齐
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pipeline = FeaturePipeline()
    result = pipeline.run(symbol="600519", period="3y")
    print("\n===== 合并指标后最后 5 行 =====")
    print(result.tail(5))
    print("\n===== 列名 =====")
    print(list(result.columns))
