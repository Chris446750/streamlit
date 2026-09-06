# ============================================================
# AI 智能投资投研终端 — 生产镜像
# 构建：docker build -t investment-terminal .
# ============================================================
FROM python:3.12-slim

# 统一编码 / 关闭缓冲 / 时区
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# numba / llvmlite 运行时依赖（libgomp）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖清单并安装（利用 Docker 层缓存，业务代码变更无需重装依赖）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝业务代码
COPY . .

EXPOSE 8501

# Streamlit 监听 0.0.0.0:8501（无浏览器 headless 模式）
CMD ["streamlit", "run", "app/ui/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
