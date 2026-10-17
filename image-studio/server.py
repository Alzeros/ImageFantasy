#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Fantasy —— 自托管图像生成 Web 应用（零依赖，仅用 Python 标准库）

特性：
  - 对接 OpenAI 兼容图像 API，以及 MiniMax 原生图片 API
  - 全参数暴露：prompt / size / n / output_format / output_compression /
    response_format / watermark / prompt_extend(提示词润色) / 参考图编辑(edit)
    （尺寸等取值范围由前端控件约束，服务端不做二次校验，原样透传给上游）
  - GPT-Image 兼容参数(quality/style/background)透传 + 任意额外参数 JSON 透传
  - 供应商与模型管理：供应商（Base URL + Key + 模型清单）持久化在 data/providers.json，
    前端只需选择供应商与模型；生成请求由服务端注入凭据
  - 实时进度：/api/generate 以 NDJSON 分块流式返回阶段事件
    （accepted → upstream → saving → done / error），前端可显示生成进度
  - 服务端图库持久化：生成的图像落盘到 data/images，索引存 data/gallery.json，
    重启不丢；前端可按图库回看、删除
  - 单文件后端 + 静态前端，Docker 一键部署

运行（本地）：
  python3 server.py
  # 浏览器打开 http://127.0.0.1:19150
部署（Docker）：
  docker compose up -d
  # compose 中已显式设置 HOST=0.0.0.0 以便从宿主机/外部访问
环境变量：
  PORT  监听端口，默认 19150
  HOST  监听地址，默认 127.0.0.1（仅本机）；对外提供服务时设为 0.0.0.0
        ⚠ 本服务无鉴权，且 /api/providers 会明文返回上游 API Key，
          仅建议在可信网络或反向代理鉴权之后暴露。
"""
import http.server
import socketserver
import threading
import json
import urllib.request
import urllib.error
import urllib.parse
import base64
import os
import time
import re
import uuid
import tempfile
import ipaddress
import socket

PORT = int(os.environ.get("PORT", "19150"))  # 默认高位端口
# 默认只监听本机回环：服务端无鉴权，且 /api/providers 会明文返回上游 API Key，
# 绑定 0.0.0.0 等于把密钥暴露给同网段任何人。需要对外提供时显式设 HOST=0.0.0.0。
HOST = os.environ.get("HOST", "127.0.0.1")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
GALLERY_FILE = os.path.join(DATA_DIR, "gallery.json")
PROVIDERS_FILE = os.path.join(DATA_DIR, "providers.json")
# 单个请求体上限。图生图会带 base64 参考图，4096px PNG 编码后约 20-30MB，
# 多张很容易上到几十 MB，所以留足余量；但仍要有上限，否则一个 Content-Length
# 就能让服务端把内存吃光。
MAX_BODY = 64 * 1024 * 1024
# 请求日志上限，超过则把旧日志重命名为 server.log.1（只保留一代）
LOG_MAX_BYTES = 8 * 1024 * 1024
os.makedirs(IMAGES_DIR, exist_ok=True)

_lock = threading.RLock()  # 重入锁：load_gallery + save_gallery 可以嵌套调用

# 部分本机代理会把公网域名映射到这些 Fake-IP 段，再由代理转发实际请求。
# 仅为已知的图片存储域名兼容这种解析；绝不放行直接使用这些 IP 的 URL，
# 也不把任意域名解析出的私网地址都当成代理地址。
_PROXY_DNS_NETWORKS = (
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("2001:2::/48"),
)
_PROXY_DNS_TRUSTED_DOMAINS = ("aliyuncs.com",)

API_TYPES = ("openai", "minimax")
MINIMAX_MODELS = ("image-01", "image-01-live")
MINIMAX_DEFAULT_BASE = "https://api.minimax.cn/v1"
MINIMAX_ASPECT_RATIOS = ("1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9")
RANDOM_IDEA_MAX_CHARS = 60


# ---- 安全与原子写辅助 ----

def _is_private_host(host: str, allow_proxy_dns: bool = False) -> bool:
    """检查 IPv4/IPv6；可信存储域名可兼容代理的 Fake-IP 解析。"""
    if not host:
        return True
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        ip = getattr(literal_ip, "ipv4_mapped", None) or literal_ip
        return not ip.is_global or ip.is_multicast

    # 不限定 AF_INET，否则仅有 AAAA 记录的图片域名也会被误拦。
    try:
        addresses = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"图片域名解析失败: {host}，请检查网络或 DNS 后重试") from exc
    for _, _, _, _, sockaddr in addresses:
        ip = ipaddress.ip_address(sockaddr[0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if allow_proxy_dns and any(ip in network for network in _PROXY_DNS_NETWORKS):
            continue
        if not ip.is_global or ip.is_multicast:
            return True
    return not addresses


def _validate_url(url: str) -> None:
    """校验图片 URL；错误信息不包含可能带有签名凭据的完整 URL。"""
    try:
        p = urllib.parse.urlparse(url)
        host = (p.hostname or "").lower().rstrip(".")
        p.port  # 同时校验端口格式与范围
    except (TypeError, ValueError) as exc:
        raise ValueError("无效的图片 URL") from exc
    if p.scheme.lower() not in ("http", "https"):
        raise ValueError("图片地址仅支持 HTTP/HTTPS 协议")
    allow_proxy_dns = p.scheme.lower() == "https" and any(
        host == domain or host.endswith("." + domain)
        for domain in _PROXY_DNS_TRUSTED_DOMAINS
    )
    if not host or _is_private_host(host, allow_proxy_dns):
        raise ValueError(f"目标地址不允许访问: {host or '缺少域名'}（图片地址指向本地、内网或特殊网络）")


class _ImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 图片存储可能跳转至 CDN；每一次跳转都要重新校验目标。
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _atomic_write(filepath: str, content: bytes) -> None:
    """原子写入：先写临时文件，再 os.replace 覆盖。不会破坏旧文件。"""
    filepath_dir = os.path.dirname(filepath) or "."
    fd, tmp = tempfile.mkstemp(dir=filepath_dir, prefix=".tmp_")
    try:
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, filepath)


# ---- 图库持久化（原子写 + 错误不静默） ----

def load_gallery():
    # 文件不存在 = 全新部署/首次启动，是正常状态（空图库）
    if not os.path.isfile(GALLERY_FILE):
        return []
    try:
        with open(GALLERY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # 只有真正的读取/解析失败才抛错，绝不让坏文件被空列表覆盖
        raise OSError(f"图库文件加载失败: {e}") from e


def save_gallery(items):
    data = json.dumps(items, ensure_ascii=False, indent=2).encode()
    with _lock:
        _atomic_write(GALLERY_FILE, data)


def load_providers():
    # 文件不存在 = 全新部署/首次启动，是正常状态（无供应商）
    if not os.path.isfile(PROVIDERS_FILE):
        return []
    try:
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise OSError(f"供应商文件加载失败: {e}") from e


def save_providers(items):
    data = json.dumps(items, ensure_ascii=False, indent=2).encode()
    with _lock:
        _atomic_write(PROVIDERS_FILE, data)


def ext_from_b64(b64):
    # 简单嗅探 png/jpeg/webp 头
    head = base64.b64decode(b64[:32] + "===")
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "png"


def save_image(item, idx, out_fmt):
    """把 upstream 返回的一张图(b64_json 或 url)落盘，返回 (filename, local_url)。"""
    name = "img_%d_%d" % (int(time.time() * 1000), idx)
    if item.get("b64_json"):
        ext = out_fmt or ext_from_b64(item["b64_json"])
        data = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        url = item["url"]
        # SSRF 防护：仅 accept http/https，且 host 不能是本地/私有IP
        _validate_url(url)
        req = urllib.request.Request(url, headers={"User-Agent": "image-fantasy"})
        opener = urllib.request.build_opener(_ImageRedirectHandler())
        with opener.open(req, timeout=60) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
            ext = "jpeg" if "jpeg" in ct else "webp" if "webp" in ct else "png"
    else:
        return None, None
    filename = name + "." + ext
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return filename, "/images/" + filename


def upstream_err_text(err, code):
    """把上游错误体压成一行人类可读的字符串。

    上游常返回嵌套结构 {"error": {"message": ..., "code": ...}}，直接丢给前端
    会渲染成一坨 JSON。这里只取最内层的 message，取不到再退回整体序列化。
    """
    if isinstance(err, dict) and isinstance(err.get("base_resp"), dict):
        status = err["base_resp"]
        return "MiniMax 错误 %s：%s" % (
            status.get("status_code", code), status.get("status_msg") or "请求失败"
        )
    msg = None
    if isinstance(err, dict):
        node = err
        for _ in range(4):                       # 最多剥几层 error 包装
            if not isinstance(node, dict):
                break
            if isinstance(node.get("message"), str):
                msg = node["message"]
                break
            node = node.get("error")
        if msg is None and isinstance(node, str):
            msg = node
    elif isinstance(err, str):
        msg = err
    if not msg:
        msg = "HTTP %s" % code
    return msg.strip() or ("HTTP %s" % code)


def build_image_request(api_type, base, model, mode, prompt, params, extra, image=None, mask=None):
    """把界面参数转换为渠道原生请求，返回 (endpoint, payload)。"""
    if api_type not in API_TYPES:
        raise ValueError("不支持的渠道接口类型")
    if not isinstance(extra, dict):
        raise ValueError("额外参数必须是 JSON 对象")
    payload = {"model": model, "prompt": prompt}
    if api_type == "minimax":
        # 官方协议：https://platform.minimax.cn/docs/api-reference/image-generation-t2i
        # 默认直接返回 Base64，避免临时图片外链过期或下载被本机代理影响。
        response_format = params.get("response_format") or "base64"
        payload["response_format"] = "base64" if response_format == "b64_json" else response_format
        if params.get("aspect_ratio"):
            payload["aspect_ratio"] = params["aspect_ratio"]
        elif params.get("size"):
            match = re.fullmatch(r"(\d+)x(\d+)", params["size"])
            if not match:
                raise ValueError("MiniMax 自定义尺寸应为 宽x高，例如 1024x1024")
            payload["width"], payload["height"] = map(int, match.groups())
        if params.get("watermark") is not None:
            payload["aigc_watermark"] = bool(params["watermark"])
        if params.get("prompt_extend") is not None:
            payload["prompt_optimizer"] = bool(params["prompt_extend"])
        if params.get("minimax_style"):
            payload["style"] = {
                "style_type": params["minimax_style"],
                "style_weight": params.get("minimax_style_weight", 0.8),
            }
        if mode == "edit":
            if mask:
                raise ValueError("MiniMax 支持人物参考图，不支持 Mask 遮罩编辑")
            if image:
                if not isinstance(image, str) or not re.match(r"data:image/(png|jpe?g);base64,", image):
                    raise ValueError("MiniMax 人物参考图仅支持 PNG / JPEG 的 Base64 Data URL")
                try:
                    reference = base64.b64decode(image.split(",", 1)[1], validate=True)
                except ValueError as exc:
                    raise ValueError("人物参考图的 Base64 数据无效") from exc
                if not reference or len(reference) >= 10 * 1024 * 1024:
                    raise ValueError("MiniMax 人物参考图大小须小于 10 MB")
                payload["subject_reference"] = [{"type": "character", "image_file": image}]
        base = base.rstrip("/")
        endpoint = base if base.endswith("/image_generation") else (
            base + "/image_generation" if base.endswith("/v1") else base + "/v1/image_generation"
        )
    else:
        for key in ("size", "output_format", "response_format", "quality", "style", "background"):
            if params.get(key):
                payload[key] = params[key]
        if params.get("watermark") is not None:
            payload["watermark"] = bool(params["watermark"])
        if params.get("prompt_extend") is not None:
            payload["prompt_extend"] = bool(params["prompt_extend"])
        if params.get("output_compression") not in (None, ""):
            try:
                payload["output_compression"] = max(0, min(100, int(params["output_compression"])))
            except (ValueError, TypeError):
                pass
        if mode == "edit":
            if image:
                payload["image"] = image
            if mask:
                payload["mask"] = mask
        endpoint = base.rstrip("/") + ("/images/edits" if mode == "edit" else "/images/generations")
    try:
        payload["n"] = max(1, int(params.get("n", 1) or 1))
    except (ValueError, TypeError):
        payload["n"] = 1
    payload.update({key: value for key, value in extra.items() if key and value not in (None, "")})
    if api_type == "minimax":
        _validate_minimax_payload(payload, mode)
    return endpoint, payload


def _validate_minimax_payload(payload, mode):
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1500:
        raise ValueError("MiniMax 提示词须为 1～1500 个字符")
    if payload.get("response_format") not in ("url", "base64"):
        raise ValueError("MiniMax 返回方式仅支持 url / base64")
    count = payload.get("n")
    if type(count) is not int or not 1 <= count <= 9:
        raise ValueError("MiniMax 单次生成数量须为 1～9")
    ratio = payload.get("aspect_ratio")
    if ratio and ratio not in MINIMAX_ASPECT_RATIOS:
        raise ValueError("MiniMax 不支持此画幅比例")
    live = payload.get("model") == "image-01-live"
    if live and ratio == "21:9":
        raise ValueError("image-01-live 不支持 21:9 画幅")
    if "width" in payload or "height" in payload:
        if live:
            raise ValueError("image-01-live 请使用预设画幅，不支持自定义宽高")
        for key in ("width", "height"):
            value = payload.get(key)
            if type(value) is not int or not 512 <= value <= 2048 or value % 8:
                raise ValueError("MiniMax 自定义宽和高须同时设置，范围 512～2048，且为 8 的倍数")
    if "style" in payload:
        style = payload["style"]
        if not live:
            raise ValueError("MiniMax 画风设置仅适用于 image-01-live")
        if not isinstance(style, dict) or style.get("style_type") not in ("漫画", "元气", "中世纪", "水彩"):
            raise ValueError("MiniMax 画风支持：漫画、元气、中世纪、水彩")
        weight = style.get("style_weight", 0.8)
        if type(weight) not in (int, float) or not 0 < weight <= 1:
            raise ValueError("MiniMax 画风权重须大于 0 且不超过 1")
    if mode == "edit" and not payload.get("subject_reference"):
        raise ValueError("MiniMax 人物参考模式需要上传参考图")


def normalize_image_response(upstream, api_type):
    """将渠道响应归一为现有图库使用的 data 数组。"""
    if api_type != "minimax":
        return upstream
    if not isinstance(upstream, dict):
        raise ValueError("MiniMax 返回了无效的响应")
    status = upstream.get("base_resp") or {}
    if status.get("status_code") not in (None, 0, "0"):
        raise ValueError(upstream_err_text(upstream, status.get("status_code")))
    data = upstream.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError("MiniMax 返回了无效的图片数据")
    items = []
    for key, output_key in (("image_base64", "b64_json"), ("image_urls", "url")):
        values = data.get(key) or []
        if not isinstance(values, list):
            raise ValueError("MiniMax 返回了无效的图片列表")
        for value in values:
            if isinstance(value, str) and value.strip():
                if output_key == "b64_json" and value.startswith("data:image/"):
                    value = value.split(",", 1)[-1]
                items.append({output_key: value})
        if items:
            break  # 优先 Base64，避免同时返回两种格式时重复入库。
    if not items:
        failed = (upstream.get("metadata") or {}).get("failed_count")
        if failed and str(failed) != "0":
            raise ValueError("MiniMax 未返回图片：内容安全检查未通过，请调整提示词后重试")
        raise ValueError("MiniMax 未返回有效图片，请稍后重试")
    return {**upstream, "data": items}


def concise_random_idea(content, truncated=False):
    """随机灵感只保留一个主轴；超长响应在完整句或分句处收束。"""
    sentence = re.split(r"(?<=[。！？!?])", " ".join(content.split()), maxsplit=1)[0].strip()
    if len(sentence) <= RANDOM_IDEA_MAX_CHARS and (
        not truncated or sentence.endswith(("。", "！", "？", "!", "?"))
    ):
        return sentence
    # 不追加第二次模型调用；也不在词语中间硬截断。
    boundaries = list(re.finditer(r"[，,；;：:]", sentence[:RANDOM_IDEA_MAX_CHARS]))
    if boundaries:
        idea = sentence[:boundaries[-1].start()].strip()
        if idea:
            return idea + "。"
    reason = "模型输出达到上限" if truncated else "模型返回的灵感过长"
    raise ValueError(reason + "，未能提取完整构思，请重新随机一次")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # 单个连接的读写超时。HTTP/1.1 keep-alive 下连接会一直留着，
    # 没有超时就等于一个慢客户端/半开连接可以永久占住一个线程。
    timeout = 120

    def _send(self, code, obj=None, ctype="application/json", body=None):
        if body is None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8") if obj is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            self._send(404, {"error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- NDJSON 流式响应辅助（手动实现 chunked 传输） ----

    def _start_stream(self):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

    def _emit(self, obj):
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(("%X\r\n" % len(line)).encode("ascii") + line + b"\r\n")
        self.wfile.flush()

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        if p in ("/", "/index.html"):
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        elif p == "/favicon.ico":
            self._send_file(os.path.join(STATIC_DIR, "favicon.ico"), "image/vnd.microsoft.icon")
        elif p.startswith("/static/"):
            fpath = os.path.normpath(os.path.join(STATIC_DIR, p[len("/static/"):]))
            if not fpath.startswith(STATIC_DIR):
                self._send(403, {"error": "forbidden"}); return
            ctype = "text/html; charset=utf-8"
            if fpath.endswith(".js"): ctype = "text/javascript; charset=utf-8"
            elif fpath.endswith(".css"): ctype = "text/css; charset=utf-8"
            elif fpath.endswith(".svg"): ctype = "image/svg+xml"
            elif fpath.endswith(".ico"): ctype = "image/vnd.microsoft.icon"
            self._send_file(fpath, ctype)
        elif p.startswith("/images/"):
            fname = os.path.basename(p)
            fpath = os.path.join(IMAGES_DIR, fname)
            if not os.path.isfile(fpath):
                self._send(404, {"error": "not found"}); return
            ext = fname.rsplit(".", 1)[-1].lower()
            ctype = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp"}.get(ext, "application/octet-stream")
            self._send_file(fpath, ctype)
        elif p == "/api/gallery":
            try:
                self._send(200, load_gallery())
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif p == "/api/providers":
            try:
                self._send(200, load_providers())
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif p == "/api/ping":
            self._send(200, {"ok": True, "time": int(time.time() * 1000)})
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        m = re.match(r"^/api/providers/([^/]+)$", parsed.path)
        if m:
            self._delete_provider(m.group(1)); return
        m = re.match(r"^/api/gallery/([^/]+)$", parsed.path)
        if not m:
            self._send(404, {"error": "not found"}); return
        gid = m.group(1)
        # 在锁内完成 读 -> 改 -> 写，避免并发新旧不同步
        with _lock:
            try:
                items = load_gallery()
            except Exception as e:
                self._send(500, {"error": f"图库读取失败: {e}"})
                return
            new_items = []
            removed = None
            for it in items:
                if it.get("id") == gid:
                    removed = it
                    fpath = os.path.join(IMAGES_DIR, it.get("filename", ""))
                    try:
                        if os.path.isfile(fpath): os.remove(fpath)
                    except Exception:
                        pass
                else:
                    new_items.append(it)
            if removed is None:
                self._send(404, {"error": "not found"}); return
            # 原子写
            data = json.dumps(new_items, ensure_ascii=False, indent=2).encode()
            _atomic_write(GALLERY_FILE, data)
        self._send(200, {"ok": True})

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        m = re.match(r"^/api/providers/([^/]+)$", parsed.path)
        if not m:
            self._send(404, {"error": "not found"}); return
        self._update_provider(m.group(1))

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/api/providers":
            self._create_provider(); return
        if p == "/api/fetch-models":
            self._fetch_models(); return
        if p == "/api/inspire":
            self._generate_inspiration(); return
        if p != "/api/generate":
            self._send(404, {"error": "not found"}); return
        try:
            data = json.loads(self._read_raw_body() or b"{}")
        except ValueError as e:
            self._send(413, {"error": str(e)}); return
        except Exception as e:
            self._send(400, {"error": "请求解析失败: " + str(e)}); return

        provider_id = data.get("provider_id") or ""
        model = (data.get("model") or "").strip()
        mode = data.get("mode") or "generate"
        prompt = data.get("prompt") or ""
        params = data.get("params") or {}
        extra = data.get("extra") or {}

        prov = None
        for it in load_providers():
            if it.get("id") == provider_id:
                prov = it
                break
        if prov is None:
            self._send(400, {"error": "供应商不存在或已被删除，请回首页重新选择。"}); return
        base = (prov.get("base") or "").strip().rstrip("/")
        key = (prov.get("key") or "").strip()
        api_type = prov.get("api_type") or "openai"
        if not base:
            self._send(400, {"error": "该供应商缺少 Base URL，请到供应商管理里补上。"}); return
        if not model:
            models = prov.get("models") or []
            model = models[0] if models else ""
        if not model:
            self._send(400, {"error": "该供应商还没有模型，请先添加。"}); return

        try:
            if api_type == "minimax" and not key:
                raise ValueError("MiniMax 渠道需要填写 API Key，请到渠道设置中补充")
            endpoint, payload = build_image_request(
                api_type, base, model, mode, prompt, params, extra, data.get("image"), data.get("mask")
            )
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        body = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if key:
            hdrs["Authorization"] = "Bearer " + key
        req = urllib.request.Request(endpoint, data=body, headers=hdrs, method="POST")

        # ---- 以下走 NDJSON 流式响应：阶段事件实时推送给前端 ----
        self._start_stream()
        try:
            try:
                self._emit({"stage": "accepted", "time": int(time.time() * 1000)})
                self._emit({"stage": "upstream", "model": model, "endpoint": endpoint})
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        upstream = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as e:
                    try:
                        err = json.loads(e.read().decode("utf-8"))
                    except Exception:
                        err = None
                    self._emit({"stage": "error", "http": e.code,
                                "message": upstream_err_text(err, e.code)})
                    return
                except Exception as e:
                    self._emit({"stage": "error", "message": str(e)})
                    return

                upstream = normalize_image_response(upstream, api_type)
                items = upstream.get("data") or []
                self._emit({"stage": "saving", "count": len(items)})
                # MiniMax 不接受 output_format；按返回的真实图片内容确定后缀。
                out_fmt = None if api_type == "minimax" else params.get("output_format")
                size = params.get("size", "")
                if api_type == "minimax":
                    size = payload.get("aspect_ratio") or (
                        "%sx%s" % (payload["width"], payload["height"]) if "width" in payload else "1:1"
                    )
                saved = []
                # 在锁内完成 读 -> 改 -> 写，避免并发修正
                with _lock:
                    try:
                        gallery = load_gallery()
                    except Exception as e:
                        # 加载失败，返回错误而不静默吞掉
                        self._emit({"stage": "error", "message": f"图库加载失败: {e}"})
                        raise
                    for i, it in enumerate(items):
                        filename, local_url = save_image(it, i, out_fmt)
                        if not filename:
                            continue
                        gid = filename.rsplit(".", 1)[0]
                        entry = {
                            "id": gid, "filename": filename, "url": local_url,
                            "prompt": prompt, "model": model,
                            "size": size,
                            "output_format": filename.rsplit(".", 1)[-1] if api_type == "minimax" else out_fmt or "",
                            "api_type": api_type,
                            "mode": mode, "created": int(time.time() * 1000),
                        }
                        gallery.append(entry)
                        saved.append(entry)
                    # 原子写
                    data = json.dumps(gallery, ensure_ascii=False, indent=2).encode()
                    _atomic_write(GALLERY_FILE, data)
                self._emit({"stage": "done", "created": upstream.get("created"),
                            "usage": upstream.get("usage"), "items": saved})
            except (BrokenPipeError, ConnectionResetError):
                # 前端断开（例如点了取消等待），安静收场
                pass
            except Exception as e:
                try:
                    self._emit({"stage": "error", "message": str(e)})
                except Exception:
                    pass
        finally:
            try:
                self._end_stream()
            except Exception:
                pass

    # ---- 供应商管理 ----

    def _read_raw_body(self):
        """按 Content-Length 读请求体，带上限保护。超限抛 ValueError。"""
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length < 0:
            raise ValueError("Content-Length 非法")
        if length > MAX_BODY:
            raise ValueError("请求体过大（上限 %d MB）" % (MAX_BODY // 1024 // 1024))
        return self.rfile.read(length) if length else b""

    def _read_json_body(self):
        try:
            return json.loads(self._read_raw_body() or b"{}")
        except Exception:
            return None

    def _provider_fields(self, body):
        name = (body.get("name") or "").strip()
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        api_type = body.get("api_type") or "openai"
        if api_type not in API_TYPES:
            raise ValueError("接口类型仅支持 OpenAI 兼容或 MiniMax 原生")
        if api_type == "minimax" and not base:
            base = MINIMAX_DEFAULT_BASE
        raw = body.get("models") or []
        if not isinstance(raw, list):
            raw = []
        seen, models = set(), []
        for m in raw:
            m = str(m).strip()
            if m and m not in seen:
                seen.add(m); models.append(m)
        return name, base, key, models, api_type

    def _create_provider(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        try:
            name, base, key, models, api_type = self._provider_fields(body)
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        if not name or not base:
            self._send(400, {"error": "名称和 Base URL 必填"}); return
        # 在锁内完成 读 -> 改 -> 写，避免并发修正
        with _lock:
            try:
                items = load_providers()
            except Exception as e:
                self._send(500, {"error": f"供应商读取失败: {e}"}); return
            prov = {
                "id": "p_" + uuid.uuid4().hex[:8],
                "name": name, "base": base, "key": key,
                "api_type": api_type,
                "models": models, "created": int(time.time() * 1000),
            }
            items.append(prov)
            data = json.dumps(items, ensure_ascii=False, indent=2).encode()
            _atomic_write(PROVIDERS_FILE, data)
            self._send(200, prov)

    def _update_provider(self, pid):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        try:
            name, base, key, models, api_type = self._provider_fields(body)
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        # 在锁内完成 读 -> 改 -> 写，避免并发修正
        with _lock:
            try:
                items = load_providers()
            except Exception as e:
                self._send(500, {"error": f"供应商读取失败: {e}"}); return
            found = None
            for it in items:
                if it.get("id") == pid:
                    if name: it["name"] = name
                    if base: it["base"] = base
                    it["key"] = key
                    it["models"] = models
                    if "api_type" in body:
                        it["api_type"] = api_type
                    found = it
                    break
            if found is None:
                self._send(404, {"error": "供应商不存在"}); return
            data = json.dumps(items, ensure_ascii=False, indent=2).encode()
            _atomic_write(PROVIDERS_FILE, data)
            self._send(200, found)

    def _delete_provider(self, pid):
        # 在锁内完成 读 -> 改 -> 写，避免并发修正
        with _lock:
            try:
                items = load_providers()
            except Exception as e:
                self._send(500, {"error": f"供应商读取失败: {e}"}); return
            new_items = [it for it in items if it.get("id") != pid]
            if len(new_items) == len(items):
                self._send(404, {"error": "供应商不存在"}); return
            data = json.dumps(new_items, ensure_ascii=False, indent=2).encode()
            _atomic_write(PROVIDERS_FILE, data)
            self._send(200, {"ok": True})

    def _fetch_models(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        api_type = body.get("api_type") or "openai"
        if api_type not in API_TYPES:
            self._send(400, {"error": "不支持的渠道接口类型"}); return
        if api_type == "minimax":
            # 官方图片接口未提供模型枚举端点，明确返回文档中的预设列表。
            self._send(200, {"models": list(MINIMAX_MODELS), "source": "preset"}); return
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        if not base:
            self._send(400, {"error": "缺少 Base URL"}); return
        # 这里只挡协议，不做 _validate_url 的私网 IP 拦截：
        # provider base 是使用者自己在表单里填的（而非上游返回的不可信数据），
        # 而本地部署的模型服务（Ollama :11434 / LM Studio :1234 / LocalAI :8000）
        # 本来就住在环回地址上，一刀切会让这些场景直接不可用。
        # 真正需要防的是 save_image 里上游返回的图片 URL，那条仍走 _validate_url。
        if urllib.parse.urlparse(base).scheme.lower() not in ("http", "https"):
            self._send(400, {"error": "Base URL 仅支持 HTTP/HTTPS 协议"}); return
        hdrs = {}
        if key:
            hdrs["Authorization"] = "Bearer " + key
        req = urllib.request.Request(base + "/models", headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                up = json.loads(resp.read().decode("utf-8"))
            data = up.get("data") or up.get("models") or []
            names = []
            for m in data:
                if isinstance(m, dict) and m.get("id"):
                    names.append(str(m["id"]))
                elif isinstance(m, str):
                    names.append(m)
            self._send(200, {"models": sorted(set(names))})
        except urllib.error.HTTPError as e:
            self._send(502, {"error": "上游返回 HTTP %s（部分渠道不提供 /models 接口，可手动填写模型列表）" % e.code})
        except Exception as e:
            self._send(502, {"error": "拉取失败：%s" % e})

    def _generate_inspiration(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        provider_id = (body.get("provider_id") or "").strip()
        model = (body.get("model") or "").strip()
        user_prompt = (body.get("prompt") or "").strip()
        # 新界面明确区分随机与扩写；未传 action 的旧客户端维持原有行为。
        action = body.get("action") or ("expand" if user_prompt else "random")
        if action not in ("random", "expand"):
            self._send(400, {"error": "提示词操作仅支持随机灵感或 AI 扩写"}); return
        if action == "expand" and not user_prompt:
            self._send(400, {"error": "请先输入提示词或随机生成灵感，再进行扩写"}); return

        if not model:
            self._send(400, {"error": "未指定文本大模型"}); return

        prov = None
        for it in load_providers():
            if it.get("id") == provider_id:
                prov = it
                break
        if prov is None and load_providers():
            prov = load_providers()[0]
        if prov is None:
            self._send(400, {"error": "未找到可用的供应商渠道"}); return
        if prov.get("api_type") == "minimax":
            self._send(400, {"error": "此渠道用于 MiniMax 图片生成，请为灵感功能选择 OpenAI 兼容的文本模型渠道"}); return

        base = (prov.get("base") or "").strip().rstrip("/")
        key = (prov.get("key") or "").strip()
        if not base:
            self._send(400, {"error": "该供应商缺少 Base URL"}); return

        endpoint = base + "/chat/completions"

        if action == "expand":
            few_shots = [
                "复古咖啡馆门头特写：一块木质招牌上\"深夜食堂\"四个手写体汉字清晰呈现，招牌周围环绕暖黄小灯泡，背景虚化，夜晚氛围，电影级调色",
                "未来主义赛博朋克雨夜街道，霓虹招牌在湿漉漉的柏油路面上反射绚烂倒影，悬浮飞车穿梭于摩天楼宇之间，8K 超高清电影质感",
                "新中式水墨风仙山楼阁，孤舟蓑笠翁独钓寒江雪，云雾缭绕其间，传统青绿山水色调，极简留白，大师意境",
                "微距摄影：清晨草叶上一滴晶莹剔透的水珠，水珠中倒映着初升的微曦朝霞与微小的野花，柔和顶光，景深极浅",
                "宫崎骏动画风格的夏日乡野车站，阳光透过香樟树叶洒下斑驳光影，远处是蔚蓝的积雨云和金黄的向日葵花海"
            ]
            system_msg = (
                "你是一位 AI 图像生成艺术家和 Prompt 提示词工程师。\n"
                "你的任务是保留用户的画面主轴，将简短灵感扩写为完整的生图提示词。\n"
                "要求：\n"
                "1. 细化主体、场景环境、构图视角、光影氛围、材质色彩、镜头景深和艺术风格；\n"
                "2. 保留用户指定的主体、场景和风格，不替换成其他题材；\n"
                "3. 输出凝练的中文画面描述，不输出前缀、解释、标号或问候语；\n"
                "细节描述方式参考（不要照搬示例的题材）：\n- " + "\n- ".join(few_shots)
            )
            user_msg = f"请保留用户构思的主体、场景与风格，在此基础上补充构图、光影和画面细节，扩写为一段生图提示词：\n用户构思：{user_prompt}"
            max_tokens = 400
        else:
            system_msg = (
                "你负责提供绘画灵感，帮用户快速确定要画什么。\n"
                "只给一个画面主轴：一个主体、一个核心场景或动作，可以加一个有趣的创意点。\n"
                "用20～40个中文字符写成一句话，最多60个字符（含标点）。先写清主体和场景。\n"
                "不展开光影、构图、镜头、材质或色彩细节，不堆砌画质词；细节由用户之后手动扩写。\n"
                "只输出构思本身，不要标题、解释、列表或多个方案。题材自由变化。\n"
                "简短程度参考（请创造新的题材）：\n"
                "雨后的街角花店，一只橘猫趴在待售的花束之间。\n"
                "云海上的小村庄，居民乘着巨大的风筝出门。"
            )
            user_msg = "随机想一个新的画面主轴，用20～40字的一句话给我灵感。"
            max_tokens = 128

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.95,
            "max_tokens": max_tokens
        }

        hdrs = {"Content-Type": "application/json"}
        if key:
            hdrs["Authorization"] = "Bearer " + key

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=hdrs,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                up = json.loads(resp.read().decode("utf-8"))
            choices = up.get("choices") or []
            if not choices:
                self._send(502, {"error": "模型未返回有效内容"}); return
            content = choices[0].get("message", {}).get("content", "").strip()
            content = re.sub(r'^(["\'“”]|提示词[:：])\s*', '', content)
            content = re.sub(r'["\'“”]\s*$', '', content).strip()
            if not content:
                self._send(502, {"error": "模型未返回有效提示词"}); return
            if action == "random":
                content = concise_random_idea(content, choices[0].get("finish_reason") == "length")
            self._send(200, {"ok": True, "prompt": content})
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
            except Exception:
                err_data = "HTTP " + str(e.code)
            self._send(502, {"error": "LLM 接口返回异常: " + str(err_data)})
        except Exception as e:
            self._send(502, {"error": "调用 LLM 失败: " + str(e)})

    def log_message(self, fmt, *args):
        # 请求日志写入 data/server.log（静默，不刷终端）。
        # 长跑服务不轮转的话日志会无限膨胀，所以超过上限就整体截断一份。
        try:
            path = os.path.join(DATA_DIR, "server.log")
            try:
                if os.path.getsize(path) > LOG_MAX_BYTES:
                    os.replace(path, path + ".1")
            except OSError:
                pass          # 文件不存在即无需轮转
            with open(path, "a", encoding="utf-8") as f:
                f.write("[%s] %s\n" % (time.strftime("%m-%d %H:%M:%S"), fmt % args))
        except Exception:
            pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    with ThreadingHTTPServer((HOST, PORT), Handler) as httpd:
        shown = "127.0.0.1" if HOST in ("127.0.0.1", "localhost") else HOST
        print("Image Fantasy 已启动 ->  http://%s:%d" % (shown, PORT))
        print("图库目录: %s" % DATA_DIR)
        if HOST not in ("127.0.0.1", "localhost"):
            print("⚠ 警告: 当前监听 %s 且服务无鉴权，/api/providers 会返回明文 API Key。" % HOST)
            print("   仅建议在可信网络或有反向代理鉴权的情况下这样运行。")
        else:
            print("按 Ctrl+C 退出。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
