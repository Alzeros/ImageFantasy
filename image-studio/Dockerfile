FROM python:3.11-slim

WORKDIR /app

# 仅复制后端与前端（图库数据走卷挂载，不进镜像）
COPY server.py /app/server.py
COPY static /app/static
# 缩略图依赖；不装也能跑，但图库会直接加载全尺寸原图
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PORT=19150
# 让启动日志（含首次设置令牌）立即出现在 docker logs 中
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
RUN mkdir -p /data/images

EXPOSE 19150

CMD ["python3", "server.py"]
