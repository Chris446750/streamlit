"""
app.ai.deepseek_client — DeepSeek 智能预测客户端
================================================================
职责：把 Phase 2 计算好的 12 列特征 DataFrame 喂给 DeepSeek，
     强制其输出严格结构化的 JSON 投资建议，并经 Pydantic 校验。

特性（真实 + Mock 双模式）：
    - 真实模式：用 openai 官方 SDK 调用 DeepSeek（接口兼容 OpenAI）。
    - Mock 模式：无有效 API Key 时自动降级到确定性规则引擎，
      保证「数据 → Prompt → JSON → Pydantic」全管道可离线验证。

用法：
    from app.ai.deepseek_client import DeepSeekClient
    client = DeepSeekClient()
    result = client.generate_prediction(df, "600519")
"""
from __future__ import annotations

import json
import re
from numbers import Number
from typing import Any

import pandas as pd
from config import DeepSeekSettings, get_settings
from openai import (
    APITimeoutError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from app.ai.schemas import PredictionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: 喂给模型的最近交易日数量
_RECENT_DAYS: int = 5

#: Mock 结果前缀，标明非真实模型输出，防止被误当真实信号
_MOCK_PREFIX: str = "【Mock】"

#: 占位符 Key 前缀（用户未填真实值时）
_PLACEHOLDER_PREFIX: str = "sk-xxx"

#: 系统角色设定
_SYSTEM_PROMPT: str = (
    "你是一位资深 A 股量化分析师，精通 MACD、布林带（BOLL）、RSI 等技术指标"
    "与量价关系。你的任务是依据给定行情数据，输出一份结构化的 JSON 投资建议。"
)


class DeepSeekClient:
    """DeepSeek 预测客户端（真实调用 + Mock 降级双模式）。

    Attributes:
        _mock: 是否为 Mock 模式；True 时 `_client` 为 None，不发真实请求。
    """

    def __init__(self, settings: DeepSeekSettings | None = None) -> None:
        cfg: DeepSeekSettings = settings if settings is not None else get_settings().deepseek
        self._cfg: DeepSeekSettings = cfg
        self._model: str = cfg.model
        self._temperature: float = cfg.temperature
        self._max_tokens: int = cfg.max_tokens
        self._mock: bool = self._resolve_mode(cfg)
        self._client: OpenAI | None = None
        if not self._mock:
            # DeepSeek 兼容 OpenAI 协议，仅需替换 base_url 与 api_key
            self._client = OpenAI(
                api_key=cfg.api_key.get_secret_value(),
                base_url=cfg.base_url,
                timeout=cfg.timeout_sec,
                max_retries=cfg.max_retries,
            )

    def _resolve_mode(self, cfg: DeepSeekSettings) -> bool:
        """根据 API Key 是否存在/有效，判定是否进入 Mock 模式。"""
        if cfg.api_key is None:
            logger.warning("未配置 DEEPSEEK_API_KEY，使用 Mock 模式，预测结果仅供测试、非真实信号")
            return True
        value: str = cfg.api_key.get_secret_value().strip()
        if not value or value.startswith(_PLACEHOLDER_PREFIX):
            logger.warning("DEEPSEEK_API_KEY 为空或仍为占位符，使用 Mock 模式，预测结果仅供测试、非真实信号")
            return True
        return False

    def generate_prediction(self, df: pd.DataFrame, symbol: str) -> PredictionResult:
        """生成单只标的的投资建议。

        Args:
            df: Phase 2 输出的 12 列特征 DataFrame（index 为日期）。
            symbol: 标的代码，如 ``"600519"``。

        Returns:
            经 Pydantic 校验的 ``PredictionResult``。

        Raises:
            ValueError: 数据为空 / JSON 解析失败 / Pydantic 校验失败。
            openai 相关异常: 真实模式下的网络、鉴权、限流等错误。
        """
        recent: pd.DataFrame = self._extract_recent(df)
        data_str: str = self._df_to_markdown(recent)
        messages: list[dict[str, str]] = self._build_prompt(symbol, data_str)

        if self._mock:
            # Mock 与真实共用同一解析路径，保证整条管道都被验证
            content: str = self._mock_json(df, symbol)
        else:
            content = self._call_api(messages)
        return self._parse_and_validate(content)

    # ------------------------------------------------------------------
    # 数据截取与格式化
    # ------------------------------------------------------------------
    def _extract_recent(self, df: pd.DataFrame) -> pd.DataFrame:
        """截取最近 N 个交易日数据。"""
        if df is None or df.empty:
            raise ValueError("特征 DataFrame 为空，无法生成预测")
        if len(df) < _RECENT_DAYS:
            logger.warning("数据不足 %d 行（实际 %d），按实际行数截取", _RECENT_DAYS, len(df))
        return df.tail(_RECENT_DAYS)

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        """把 DataFrame 转成 Markdown 表格（不依赖 tabulate）。"""
        cols: list[str] = list(df.columns)
        header: str = "| 日期 | " + " | ".join(cols) + " |"
        sep: str = "|" + "---|" * (len(cols) + 1)
        lines: list[str] = [header, sep]
        for idx, row in df.iterrows():
            # index 为 DatetimeIndex，取日期部分；否则回退字符串
            date_str: str = idx.date().isoformat() if hasattr(idx, "date") else str(idx)
            cells: list[str] = [date_str]
            for c in cols:
                v = row[c]
                cells.append(f"{v:.4f}" if isinstance(v, Number) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt 组装
    # ------------------------------------------------------------------
    def _build_prompt(self, symbol: str, data_str: str) -> list[dict[str, str]]:
        """组装 system + user 消息，强制 JSON 输出。"""
        user: str = (
            f"当前标的：{symbol}（A 股）\n\n"
            f"以下是该标的最近 {_RECENT_DAYS} 个交易日的行情与技术指标数据：\n\n"
            f"{data_str}\n\n"
            "请结合量价关系与 MACD / BOLL / RSI 指标，给出投资建议。\n"
            "必须且只能输出一个符合规范的 json 对象，不要输出任何额外文字、解释或代码块标记。\n"
            "目标结构如下：\n"
            "{\n"
            '  "signal": "BUY" 或 "SELL" 或 "HOLD",\n'
            '  "confidence": 0.0 到 1.0 之间的胜率置信度,\n'
            '  "target_price": 预测目标价（数字）,\n'
            '  "stop_loss": 建议止损价（数字）,\n'
            '  "reasoning": "不超过 150 字的核心判断理由，需结合量价和指标"\n'
            "}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------
    # 真实 API 调用
    # ------------------------------------------------------------------
    def _call_api(self, messages: list[dict[str, str]]) -> str:
        """调用 DeepSeek Chat Completions，逐类捕获 openai 异常。"""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except APITimeoutError as exc:
            logger.error("DeepSeek 请求超时 model=%s", self._model)
            raise
        except AuthenticationError as exc:
            logger.error("DeepSeek 鉴权失败，请检查 DEEPSEEK_API_KEY 是否正确")
            raise
        except RateLimitError as exc:
            logger.error("DeepSeek 触发限流（RateLimit），请稍后重试")
            raise
        except APIError as exc:
            logger.error("DeepSeek API 调用失败: %s", exc)
            raise
        content: str | None = resp.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek 返回内容为空")
        return content

    # ------------------------------------------------------------------
    # JSON 解析与 Pydantic 校验
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """从模型输出中鲁棒地提取 JSON dict（三步容错）。"""
        if not content:
            raise ValueError("模型返回内容为空")
        text: str = content.strip()

        # 1) 剥离 ```json ... ``` 代码围栏
        fence: re.Match[str] | None = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if fence:
            text = fence.group(1).strip()

        # 2) 整体解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3) 正则提取首个 { ... } 块
        brace: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从模型输出中解析 JSON，原文前 200 字: {content[:200]!r}")

    def _parse_and_validate(self, content: str) -> PredictionResult:
        """JSON 解析 + Pydantic 校验，失败时记录清晰日志。"""
        try:
            data: dict[str, Any] = self._extract_json(content)
        except ValueError as exc:
            logger.error("JSON 解析失败: %s", exc)
            raise
        try:
            return PredictionResult.model_validate(data)
        except ValidationError as exc:
            logger.error("Pydantic 校验失败: %s | 原始 JSON=%s", exc, content[:200])
            raise ValueError("模型输出未通过 Pydantic 校验") from exc

    # ------------------------------------------------------------------
    # Mock（确定性规则引擎）
    # ------------------------------------------------------------------
    def _mock_json(self, df: pd.DataFrame, symbol: str) -> str:
        """基于末行 MACD_hist / RSI14 生成确定性 mock 结果（用于离线验证）。"""
        last = df.iloc[-1]
        close: float = float(last["Close"])
        macd_hist: float = float(last["MACD_hist"])
        rsi: float = float(last["RSI14"])
        boll_lower: float = float(last["BOLL_lower"])
        boll_upper: float = float(last["BOLL_upper"])

        if macd_hist > 0 and rsi < 70:
            signal = "BUY"
            target = round(close * 1.08, 2)
            stop = round(min(boll_lower, close * 0.95), 2)
            reason = f"MACD 柱转正（{macd_hist:.3f}）且 RSI14={rsi:.1f} 未超买，量价配合看多。"
        elif macd_hist < 0 and rsi > 70:
            signal = "SELL"
            target = round(close * 0.95, 2)
            stop = round(max(boll_upper, close * 1.03), 2)
            reason = f"MACD 柱为负（{macd_hist:.3f}）且 RSI14={rsi:.1f} 超买，建议止盈离场。"
        else:
            signal = "HOLD"
            target = round(close, 2)
            stop = round(boll_lower, 2)
            reason = f"MACD 柱（{macd_hist:.3f}）与 RSI14={rsi:.1f} 均处中性，方向不明，观望。"

        payload: dict[str, Any] = {
            "signal": signal,
            "confidence": 0.6,
            "target_price": target,
            "stop_loss": stop,
            "reasoning": f"{_MOCK_PREFIX}{reason}",
        }
        # 包一层代码围栏，模拟真实 LLM 输出，完整走 _extract_json 的围栏剥离路径
        return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


if __name__ == "__main__":
    # 测试入口：取 600519 特征 → 预测 → 打印解析后的 Pydantic 对象
    from app.features.pipeline import FeaturePipeline
    from app.utils.logger import configure_logging

    configure_logging()

    # 1. 获取最新特征 DataFrame
    feature_df: pd.DataFrame = FeaturePipeline().run(symbol="600519", period="3y")

    # 2. 构造客户端（自动判定 Mock / 真实模式）
    client: DeepSeekClient = DeepSeekClient()

    # 3. 生成预测
    try:
        result: PredictionResult = client.generate_prediction(feature_df, "600519")
    except Exception as exc:  # noqa: BLE001 — 记录后向上抛出
        logger.error("预测失败: %s", exc)
        raise

    # 4. 打印解析成功后的 Pydantic 对象（ensure_ascii=False 保证中文可读）
    print("\n===== DeepSeek 预测结果（PredictionResult） =====")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
