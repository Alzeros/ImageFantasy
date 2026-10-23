# Image Fantasy

自托管的 AI 图像创作工坊。后端以 Python 标准库为主，前端为静态页面；图库缩略图可选依赖 Pillow。

- 支持 OpenAI 兼容渠道、MiniMax 原生图片 API，以及 Vertex AI Gemini 出图。
- 支持文生图、参考图生成（视渠道而定），以及渠道对应的画幅、画风等参数。
- 随机灵感提供简短的画面构思，AI 扩写按需补充细节。
- 支持渠道与模型管理、生成进度、图片下载和本地图库。
- 图库只加载缩略图，点开才取原图；装不装 Pillow 都能用。
- 账号密码登录：除登录页外，页面、图片和接口都需要登录后才能访问。

## 本地运行

```bash
git clone https://github.com/Alzeros/ImageFantasy.git
cd ImageFantasy
pip3 install -r image-studio/requirements.txt   # Pillow（缩略图）+ google-auth（Vertex）
python3 image-studio/server.py
```

浏览器打开 <http://127.0.0.1:19150>。第一次打开需要设置登录账号，见下方「登录」。登录后在「设置」中填写接口地址、API Key 和模型。

MiniMax 渠道选择「MiniMax 原生」，默认地址为 `https://api.minimax.cn/v1`，预设模型为 `image-01` 和 `image-01-live`。

Vertex AI 渠道选择「Vertex AI」：

- **Location** 推荐填 `global`（也可用 `us-central1` 等区域）。
- **Project ID** 填 GCP 项目；若粘贴的服务账号 JSON 里已有 `project_id`，可自动带出。
- **服务账号**：在设置里上传 Google 下载的服务账号 JSON。页面会解析并填入 Project ID，只显示邮箱，不展示私钥。点「验证连接」只检查账号、项目和模型是否可用，不会生图。
- 预设模型：`gemini-2.5-flash-image`（第一期仅文生图）。
- 部署到香港等地区时，走的是 Vertex（服务账号），一般不像消费级 Gemini API 那样按客户端 IP 一刀切；上线前建议在目标机器上先试生成一张。

## 登录

- **首次设置**：服务第一次启动时会生成一次性的设置令牌，打印在启动日志里（Docker 下用 `docker logs image-fantasy` 查看），同时保存在 `data/setup-token`。在页面上填写令牌，再设置账号和密码。令牌只有能登录服务器的人才能拿到，别人无法抢先设置账号。设置完成后令牌文件会自动删除。
- **密码**：至少 8 个字符，不能是纯数字或纯字母。服务器只保存加盐哈希，无法找回。
- **忘记密码**：删除服务器上的 `data/auth.json`，刷新登录页，`data/setup-token` 中会生成新的设置令牌（或重启服务后在日志里查看），再重新设置。
- **登录有效期**：7 天不使用或登录满 14 天后需要重新登录。重启服务后也需要重新登录。
- **防暴力破解**：同一来源连续输错 5 次会暂时锁定，从 1 分钟起逐次加倍，最长 15 分钟。登录记录写在 `data/security.log`。

## 服务器部署（Docker + HTTPS）

服务器需要安装 Git 和 Docker，并准备一个域名。应用只监听服务器本机，由 HTTPS 反向代理对外提供，避免密码和登录 Cookie 以明文经过公网。

1. 启动应用（把 `img.example.com` 换成你的域名）：

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
     -e PUBLIC_ORIGIN=https://img.example.com \
     -e TRUSTED_PROXIES=172.16.0.0/12 \
     -e IMAGE_FANTASY_SECRET="一串足够长的随机字符" \
     -v "$(pwd)/image-studio/data:/data" \
     image-fantasy
   docker logs image-fantasy    # 记下其中的设置令牌
   ```

2. 配置 HTTPS 反向代理，任选其一。

   Caddy 会自动申请证书，`/etc/caddy/Caddyfile`：

   ```caddyfile
   img.example.com {
       reverse_proxy 127.0.0.1:19150 {
           flush_interval -1
       }
   }
   ```

   Nginx（证书自行配置）：

   ```nginx
   server {
       listen 443 ssl;
       server_name img.example.com;
       # ssl_certificate ...; ssl_certificate_key ...;

       client_max_body_size 64m;          # 参考图会随请求上传，默认 1m 不够
       location / {
           proxy_pass http://127.0.0.1:19150;
           proxy_set_header Host $host;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_buffering off;           # 实时显示生成进度
           proxy_read_timeout 900s;       # 生成可能持续数分钟
       }
   }
   ```

3. 浏览器打开 `https://img.example.com`，填写设置令牌，设置账号和密码。

环境变量说明：

| 变量 | 作用 |
| --- | --- |
| `PUBLIC_ORIGIN` | 对外访问地址。用于校验跨站请求；以 `https://` 开头时，登录 Cookie 只通过 HTTPS 发送。 |
| `TRUSTED_PROXIES` | 反向代理的地址或网段，逗号分隔。应用只采信来自这些地址的 `X-Forwarded-*` 头。Docker 端口映射时，容器看到的来源是 Docker 网关，通常在 `172.16.0.0/12` 内；直接用 Python 运行时填 `127.0.0.1`。不设置时，代理后面的所有访客会共用一个锁定计数，有人故意输错密码会把你也锁住。如果 `security.log` 里的来源始终是同一个内网地址，把它加进来。 |
| `IMAGE_FANTASY_SECRET` | 渠道密钥（API Key、Vertex 服务账号私钥）的加密主密钥。强烈建议设置，并与 `data/` 分开保管（如服务器的 `.env`）。不设置时会自动生成 `data/secret.key`，此时密钥和密文在同一目录，只能防单独泄露 `providers.json`。主密钥丢失后已保存的渠道密钥无法解密，需要重新添加渠道。 |
| `COOKIE_SECURE` | 设为 `1` 时强制登录 Cookie 只走 HTTPS（`PUBLIC_ORIGIN` 已是 https 时无需设置）。 |

没有域名时，也可以只通过 SSH 隧道使用：省略上面的 `PUBLIC_ORIGIN`，在自己的电脑上运行 `ssh -N -L 19151:127.0.0.1:19150 user@your-server`，然后访问 <http://127.0.0.1:19151>。

仓库也提供 `image-studio/docker-compose.yml`，默认只监听服务器本机。在其中填写 `PUBLIC_ORIGIN` 后执行 `docker compose up -d --build`。

## 数据与更新

运行数据保存在 `image-studio/data/`：

- `auth.json`：登录账号和密码哈希。
- `providers.json`：渠道配置；其中的 API Key 和服务账号私钥已加密保存（见 `IMAGE_FANTASY_SECRET`）。
- `secret.key`：未设置 `IMAGE_FANTASY_SECRET` 时自动生成的加密主密钥。
- `gallery.json`：图库索引。
- `images/`：生成的图片。
- `thumbs/`：图库缩略图，可随时删除，访问时会按需重新生成。
- `security.log`：登录记录。

该目录不提交到 Git，也不进入 Docker 镜像。浏览器接口不会返回渠道密钥，编辑渠道时密钥留空表示保持不变。迁移时，要么同时带上 `secret.key`，要么在新服务器设置相同的 `IMAGE_FANTASY_SECRET`。如需迁移现有配置和图库，请单独传输整个 `data/` 目录，并保管好其中的密钥。备份和迁移时先停止服务，避免索引与图片不同步。

更新代码后运行 `git pull --ff-only`，重新构建镜像并重建容器，继续挂载同一个数据目录。

## 开发验证

```bash
python3 -m unittest discover -s image-studio -p 'test_*.py'
```

测试使用模拟接口，不调用外部付费生成服务。`openai-image-tool.py` 是早期的独立调用工具；当前 Web 应用入口是 `image-studio/server.py`。
