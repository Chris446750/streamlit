"""
app.utils.logger — 日志统一配置
================================================================
职责：基于标准库 logging 提供统一日志格式与命名 logger 获取。
设计：与 config.py 解耦（不触发 .env / 密钥加载），
     保证数据与特征链路在无密钥环境下也能独立运行。

用法：
    from app.utils.logger import configure_logging, get_logger
    configure_logging()
    logger = get_logger(__name__)
    logger.info("hello")
"""
from __future__ import annotations

import logging
import sys

#: 统一日志格式
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: 根 logger 是否已配置（幂等保护，避免重复添加 handler）
_configured: bool = False


def configure_logging(level: int = logging.INFO) -> None:
    """配置根 logger（幂等，仅首次生效）。

    Args:
        level: 日志级别，默认 ``logging.INFO``。
    """
    global _configured
    if _configured:
        return
    # 强制 stdout 使用 UTF-8，避免 Windows 控制台下中文日志乱码
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 — 编码设置失败不应阻断日志
            pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。

    Args:
        name: logger 名称，通常传 ``__name__``。

    Returns:
        命名 logger 实例。
    """
    return logging.getLogger(name)
