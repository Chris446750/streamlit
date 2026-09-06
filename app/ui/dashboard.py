"""
app.ui.dashboard — Streamlit 全功能投研看板
================================================================
职责：集成「行情图表 + AI 研报诊断 + 回测报告」于一体的 Web 看板。

运行（项目根目录）：
    .venv/Scripts/python.exe -m streamlit run app/ui/dashboard.py
    或：streamlit run app/ui/dashboard.py

访问：http://localhost:8501
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证项目根目录在 sys.path，使 `from app... import` 与 CWD 无关
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.ai.deepseek_client import DeepSeekClient
from app.backtest.engine import run_backtest
from app.features.pipeline import FeaturePipeline
from app.utils.logger import configure_logging

configure_logging()

#: 信号徽标配色（A 股习惯：红涨绿跌）
_SIGNAL_STYLE: dict[str, dict[str, str]] = {
    "BUY": {"label": "买入", "color": "#e53935", "bg": "#ffebee"},
    "SELL": {"label": "卖出", "color": "#43a047", "bg": "#e8f5e9"},
    "HOLD": {"label": "持有 / 观望", "color": "#fb8c00", "bg": "#fff3e0"},
}

_PERIOD_OPTIONS: list[str] = ["1y", "2y", "3y", "5y"]


@st.cache_data(ttl=600, show_spinner="正在拉取行情与计算指标…")
def load_features(symbol: str, period: str) -> pd.DataFrame:
    """拉取并计算特征（缓存 10 分钟，避免每次交互重复请求）。"""
    return FeaturePipeline().run(symbol=symbol, period=period)


def plot_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """绘制 K 线 + BOLL（主图）、MACD 与 RSI（副图）。"""
    fig: go.Figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(f"{symbol} 日线 + BOLL", "MACD", "RSI(14)"),
    )
    # 行 1：K 线 + BOLL 上/中/下轨
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="K线", increasing_line_color="#e53935", decreasing_line_color="#43a047",
    ), row=1, col=1)
    for col, name in [("BOLL_upper", "上轨"), ("BOLL_mid", "中轨"), ("BOLL_lower", "下轨")]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=name, line=dict(width=1, dash="dot"),
        ), row=1, col=1)
    # 行 2：MACD（柱 + DIF + DEA）
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD柱", marker_color="gray"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="DIF", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="DEA", line=dict(width=1)), row=2, col=1)
    # 行 3：RSI + 超买超卖参考线
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#e53935", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#43a047", row=3, col=1)

    fig.update_layout(
        height=760, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02), margin=dict(t=40),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig


def render_ai_card(symbol: str, df: pd.DataFrame) -> None:
    """AI 研报诊断区：高亮信号卡片 + 目标价/止损/置信度 + 理由。"""
    with st.spinner("DeepSeek 正在生成投资建议…"):
        try:
            result = DeepSeekClient().generate_prediction(df, symbol)
        except Exception as exc:  # noqa: BLE001 — 失败展示错误不崩溃
            st.error(f"AI 预测失败：{exc}")
            return
    style: dict[str, str] = _SIGNAL_STYLE.get(result.signal, _SIGNAL_STYLE["HOLD"])
    st.markdown(
        f'<div style="background:{style["bg"]};padding:14px;border-radius:8px;'
        f'border-left:6px solid {style["color"]};">'
        f'<span style="font-size:20px;font-weight:700;color:{style["color"]};">'
        f'{style["label"]}（{result.signal}）</span>'
        f'<span style="float:right;color:#555;">置信度 {result.confidence:.0%}</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("预测目标价", f"{result.target_price:.2f}")
    c2.metric("建议止损价", f"{result.stop_loss:.2f}")
    st.write("**核心判断理由**：", result.reasoning)


def render_backtest(df: pd.DataFrame, symbol: str) -> None:
    """回测报告区：绩效指标卡片 + 累计收益曲线。"""
    from config import get_settings

    bt = get_settings().backtest
    try:
        result = run_backtest(
            df, symbol=symbol,
            commission=bt.commission, slippage=bt.slippage, risk_free_rate=bt.risk_free_rate,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"回测失败：{exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计收益率", f"{result.total_return:.2%}")
    c2.metric("年化收益率", f"{result.annualized_return:.2%}")
    c3.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    c4.metric("最大回撤", f"{result.max_drawdown:.2%}")
    c5, c6 = st.columns(2)
    c5.metric("胜率", f"{result.win_rate:.2%}")
    c6.metric("交易笔数", f"{result.num_trades}")

    cum_ret: pd.Series = (result.equity_curve - 1.0) * 100
    fig: go.Figure = go.Figure(go.Scatter(
        x=cum_ret.index, y=cum_ret.values, name="累计收益%", line=dict(width=1.5),
    ))
    fig.update_layout(height=320, title="策略累计收益率曲线（%）", margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")


# ================= 主流程 =================
st.set_page_config(page_title="AI 智能投资投研终端", page_icon="📈", layout="wide")

st.title("📈 AI 智能投资投研终端")
st.caption("不接真实资金 API，交易在外部独立完成 —— 行情 → 指标 → DeepSeek 信号 → 回测")

with st.sidebar:
    st.header("⚙️ 控制台")
    symbol: str = st.text_input("股票代码", value="600519", help="如 600519 / 000001 / sh600519")
    period: str = st.selectbox("回看周期", _PERIOD_OPTIONS, index=2)
    enable_ai: bool = st.checkbox("启用 DeepSeek AI 预测", value=False, help="需在 .env 配置 DEEPSEEK_API_KEY")

try:
    df: pd.DataFrame = load_features(symbol, period)
except Exception as exc:  # noqa: BLE001
    st.error(f"数据加载失败（请检查股票代码与网络）：{exc}")
    st.stop()

st.subheader("📊 行情与技术指标")
st.plotly_chart(plot_price_chart(df, symbol), width="stretch")

st.subheader("🤖 AI 研报诊断")
if enable_ai:
    render_ai_card(symbol, df)
else:
    st.info("在左侧开启「DeepSeek AI 预测」以生成结构化投资建议（BUY / SELL / HOLD + 目标价 / 止损）。")

st.subheader("🧪 策略回测报告")
render_backtest(df, symbol)
