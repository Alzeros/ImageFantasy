FROM python:3.11-slim

WORKDIR /app

# 仅复制后端与前端（图库数据走卷挂载，不进镜像）
COPY server.py /app/server.py
COPY static /app/static

ENV PORT=19150
# 让启动日志（含首次设置令牌）立即出现在 docker logs 中
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
RUN mkdir -p /data/images

EXPOSE 19150

CMD ["python3", "server.py"]
