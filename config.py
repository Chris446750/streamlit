"""
config.py — 全局配置中心
================================================================
职责：集中管理 DeepSeek API / 数据源 / 推送 / 回测 / 存储等全部环境变量。
特性：支持 .env 读取；敏感信息用 SecretStr 隐藏；Pydantic 严格校验；单例缓存。

用法：
    from config import get_settings
    cfg = get_settings()
    cfg.deepseek.model          # 模型名
    cfg.data.provider           # 数据源
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepSeekSettings(BaseSettings):
    """DeepSeek 大模型接口配置。环境变量前缀 DEEPSEEK_"""
    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_", env_file=".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    api_key: Optional[SecretStr] = Field(None, description="DeepSeek API Key；为空则客户端进入 Mock 模式")
    base_url: str = Field("https://api.deepseek.com", description="API 网关地址")
    model: str = Field("deepseek-chat", description="模型名，可在 .env 用 DEEPSEEK_MODEL 覆盖")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(4096, gt=0, description="单次最大输出 token")
    timeout_sec: float = Field(60.0, gt=0, description="请求超时（秒）")
    max_retries: int = Field(3, ge=0, description="失败重试次数")


class DataSourceSettings(BaseSettings):
    """行情数据源配置。环境变量前缀 DATA_"""
    model_config = SettingsConfigDict(
        env_prefix="DATA_", env_file=".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    provider: Literal["akshare", "tushare", "yfinance", "ccxt"] = Field(
        "akshare", description="默认主力数据源"
    )
    tushare_token: Optional[SecretStr] = Field(None, description="Tushare Token，选 tushare 时必填")
    stock_pool: str = Field("600519,000001,300750", description="股票池代码，逗号分隔")
    start_date: str = Field("2020-01-01", description="历史数据起始日期")
    kline_period: Literal["daily", "weekly", "monthly"] = Field("daily")
    request_interval_sec: float = Field(1.0, gt=0, description="两次请求间隔，防限流")


class NotifySettings(BaseSettings):
    """信号推送配置。环境变量前缀 NOTIFY_"""
    model_config = SettingsConfigDict(
        env_prefix="NOTIFY_", env_file=".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    telegram_bot_token: Optional[SecretStr] = Field(None, description="Telegram Bot Token")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram 目标会话 ID")
    smtp_host: Optional[str] = Field(None)
    smtp_port: int = Field(465)
    smtp_user: Optional[str] = Field(None)
    smtp_password: Optional[SecretStr] = Field(None)
    smtp_to: Optional[str] = Field(None)


class BacktestSettings(BaseSettings):
    """回测配置。环境变量前缀 BT_"""
    model_config = SettingsConfigDict(
        env_prefix="BT_", env_file=".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    initial_capital: float = Field(1_000_000.0, gt=0, description="初始资金")
    commission: float = Field(0.0003, ge=0.0, description="手续费率")
    slippage: float = Field(0.0002, ge=0.0, description="滑点")
    risk_free_rate: float = Field(0.02, ge=0.0, description="无风险利率，用于夏普")
    stop_loss_pct: float = Field(0.05, gt=0.0, description="默认止损比例")
    take_profit_pct: float = Field(0.10, gt=0.0, description="默认止盈比例")


class StorageSettings(BaseSettings):
    """存储配置。环境变量前缀 DB_"""
    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=".env",
        env_file_encoding="utf-8", extra="ignore",
    )
    engine: Literal["sqlite", "postgresql"] = Field("sqlite", description="存储引擎")
    sqlite_path: str = Field("data/market.db", description="SQLite 文件路径")
    postgres_dsn: Optional[str] = Field(
        None, description="PostgreSQL DSN，如 postgresql+psycopg2://user:pass@host:5432/db"
    )


class Settings(BaseSettings):
    """顶层配置聚合（进程内单例）"""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False,
    )
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    data: DataSourceSettings = Field(default_factory=DataSourceSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")


@lru_cache
def get_settings() -> Settings:
    """单例获取配置对象（进程内缓存，避免重复读盘）"""
    return Settings()
