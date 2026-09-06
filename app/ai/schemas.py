"""
app.ai.schemas — 大模型返回结构定义
================================================================
职责：用 Pydantic 严格约束 DeepSeek 输出的 JSON 结构，
     作为全系统投资建议的唯一数据契约（回测 / 看板 / 推送共用）。

用法：
    from app.ai.schemas import PredictionResult
    result = PredictionResult.model_validate({...})
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    """单只标的的投资建议结构化结果。

    Attributes:
        signal: 交易信号，严格限定 BUY / SELL / HOLD 三选一。
        confidence: 胜率置信度，取值 0.0~1.0。
        target_price: 预测目标价（>0）。
        stop_loss: 建议止损价（>0）。
        reasoning: 核心判断理由（≤150 字，需结合量价与指标）。
    """

    #: 容忍真实 LLM 偶尔多带字段（如 trend），5 个必需字段仍严格校验
    model_config = ConfigDict(extra="ignore")

    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="胜率置信度 0~1")
    target_price: float = Field(..., gt=0.0, description="预测目标价")
    stop_loss: float = Field(..., gt=0.0, description="建议止损价")
    reasoning: str = Field(..., max_length=150, description="核心判断理由，≤150 字")
