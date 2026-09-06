# AI 智能投资投研与策略生成终端

> 工业级 AI 投研系统：抓取行情 → 技术指标 → DeepSeek 结构化信号 → 回测验证 → Streamlit 看板。
> **不接真实资金 API，交易在外部独立完成。**

## 架构分层

| 层 | 目录 | 职责 |
|---|---|---|
| 数据层 | `app/data/` | 多数据源适配（AkShare 腾讯后端 / yfinance），OHLCV 标准化 |
| 特征层 | `app/features/` | 自研 MACD / BOLL / RSI 指标 + 流水线编排 |
| AI 层 | `app/ai/` | DeepSeek 客户端（真实 / Mock 双模式）+ Pydantic 结构化校验 |
| 回测层 | `app/backtest/` | 自研向量化回测引擎（年化 / 夏普 / 回撤 / 胜率） |
| UI 层 | `app/ui/` | Streamlit 看板（K线+BOLL+MACD+RSI + AI 研报 + 回测报告） |
| 推送层 | `app/notify/` | Telegram / SMTP 信号推送（预留） |
| 工具层 | `app/utils/` | 日志 / 重试 |

## 环境要求

- Python 3.12（量化栈对 3.14 兼容性差，勿用系统默认 3.14）
- 依赖见 `requirements.txt`

## 本地运行

```bash
# 1. 创建 Python 3.12 虚拟环境
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. 配置环境变量
cp .env.example .env          # 填入 DEEPSEEK_API_KEY（可选，不填则 AI 用 Mock 模式）

# 3. 启动看板
.venv/Scripts/python.exe -m streamlit run app/ui/dashboard.py
# 浏览器访问 http://localhost:8501
```

## 服务器 Docker 一键部署

```bash
# 服务器需已安装 Docker + Docker Compose
# 1. 配置环境变量（务必填写真实 DEEPSEEK_API_KEY）
cp .env.example .env && vim .env

# 2. 一键构建并后台启动
docker-compose up -d --build

# 3. 访问 http://<服务器IP>:8501
# 查看日志：docker-compose logs -f
# 停止：docker-compose down
```

## 模块测试（可选）

```bash
# 数据 + 特征（打印最后 5 行）
.venv/Scripts/python.exe -m app.features.pipeline

# DeepSeek 预测（无 Key 走 Mock）
.venv/Scripts/python.exe -m app.ai.deepseek_client

# 回测绩效
.venv/Scripts/python.exe -m app.backtest.engine
```

## 开发进度

- [x] Phase 1：架构初始化（目录 / 依赖 / 配置 / 环境模板）
- [x] Phase 2：数据采集 + 特征计算
- [x] Phase 3：DeepSeek 智能预测引擎
- [x] Phase 4：策略回测 + 绩效评估
- [x] Phase 5：Streamlit 看板 + Docker 部署
- [ ] 后续：Telegram / SMTP 推送、数据持久化

## 免责声明

本系统仅用于投研学习与信号研究，不构成投资建议；不接入真实资金 API，不自动下单。
