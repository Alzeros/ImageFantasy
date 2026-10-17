# Image Fantasy

自托管的 AI 图像创作工坊。后端使用 Python 标准库，前端为静态页面。

- 支持 OpenAI 兼容渠道和 MiniMax 原生图片 API。
- 支持文生图、参考图生成，以及渠道对应的画幅、画风等参数。
- 随机灵感提供简短的画面构思，AI 扩写按需补充细节。
- 支持渠道与模型管理、生成进度、图片下载和本地图库。

## 本地运行

```bash
git clone https://github.com/Alzeros/ImageFantasy.git
cd ImageFantasy
python3 image-studio/server.py
```

浏览器打开 <http://127.0.0.1:19150>，在「渠道设置」中填写接口地址、API Key 和模型。

MiniMax 渠道选择「MiniMax 原生」，默认地址为 `https://api.minimax.cn/v1`，预设模型为 `image-01` 和 `image-01-live`。

## 服务器部署（Docker）

服务器需要安装 Git 和 Docker。当前应用没有登录鉴权，渠道管理接口会返回已配置的 API Key，因此以下命令将服务端口绑定到服务器本机。通过 SSH 隧道使用，或配置带鉴权的 HTTPS 反向代理。

```bash
git clone https://github.com/Alzeros/ImageFantasy.git
cd ImageFantasy
mkdir -p image-studio/data
docker build -t image-fantasy ./image-studio
docker run -d \
  --name image-fantasy \
  --restart unless-stopped \
  -p 127.0.0.1:19150:19150 \
  -e HOST=0.0.0.0 \
  -v "$(pwd)/image-studio/data:/data" \
  image-fantasy
```

个人使用时，在自己的电脑上建立 SSH 隧道（替换服务器用户名和地址）：

```bash
ssh -N -L 19151:127.0.0.1:19150 user@your-server
```

然后访问 <http://127.0.0.1:19151>。若配置反向代理，上游地址为 `http://127.0.0.1:19150`；生成请求可能持续数分钟，请为代理设置足够的读取超时，并关闭响应缓冲以显示实时进度。

仓库也提供 `image-studio/docker-compose.yml`。其默认端口映射为 `19150:19150`；用于服务器时，请先改为 `127.0.0.1:19150:19150`，再执行 `docker compose up -d --build`。

## 数据与更新

运行数据保存在 `image-studio/data/`：

- `providers.json`：渠道配置和 API Key。
- `gallery.json`：图库索引。
- `images/`：生成的图片。

该目录不提交到 Git，也不进入 Docker 镜像。首次部署需要重新配置渠道；如需迁移现有配置和图库，请单独传输整个 `data/` 目录，并保管好其中的密钥。备份和迁移时先停止服务，避免索引与图片不同步。

更新代码后运行 `git pull --ff-only`，重新构建镜像并重建容器，继续挂载同一个数据目录。

## 开发验证

```bash
python3 -m unittest discover -s image-studio -p 'test_*.py'
```

测试使用模拟接口，不调用外部付费生成服务。`openai-image-tool.py` 是早期的独立调用工具；当前 Web 应用入口是 `image-studio/server.py`。
