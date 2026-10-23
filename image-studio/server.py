#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Fantasy —— 自托管图像生成 Web 应用（后端以 Python 标准库为主，缩略图可选依赖 Pillow）

特性：
  - 对接 OpenAI 兼容图像 API、MiniMax 原生图片 API，以及 Vertex AI Gemini 出图
  - 全参数暴露：prompt / size / n / output_format / output_compression /
    response_format / watermark / prompt_extend(提示词润色) / 参考图编辑(edit)
    （尺寸等取值范围由前端控件约束，服务端不做二次校验，原样透传给上游）
  - GPT-Image 兼容参数(quality/style/background)透传 + 任意额外参数 JSON 透传
  - 供应商与模型管理：供应商（Base URL + Key + 模型清单）持久化在 data/providers.json，
    前端只需选择供应商与模型；生成请求由服务端注入凭据
    Vertex 渠道额外保存 project_id；Location 写在 base（默认 global）；
    Key 为服务账号 JSON 全文或 JSON 文件路径（也可用本机 ADC）
  - 实时进度：/api/generate 以 NDJSON 分块流式返回阶段事件
    （accepted → upstream → saving → done / error），前端可显示生成进度
  - 服务端图库持久化：生成的图像落盘到 data/images，索引存 data/gallery.json，
    重启不丢；前端可按图库回看、删除
  - 图库缩略图：生成时另存一张长边 640 的 JPEG（data/thumbs），前端格子只加载缩略图，
    点开才取原图。装了 Pillow 才有；没装则退回原图，功能不受影响
  - 单文件后端 + 静态前端，Docker 一键部署
  - 账号密码登录：首次启动需用日志中的设置令牌设置账号；除登录页外的页面、图片与接口都要求登录

运行（本地）：
  python3 server.py
  # 浏览器打开 http://127.0.0.1:19150，首次使用填写终端里打印的设置令牌
部署（Docker）：
  docker compose up -d
  # compose 中已显式设置 HOST=0.0.0.0 以便从宿主机/外部访问
环境变量：
  PORT             监听端口，默认 19150
  HOST             监听地址，默认 127.0.0.1（仅本机）；对外提供服务时设为 0.0.0.0
  PUBLIC_ORIGIN    对外访问地址，如 https://img.example.com；以 https 开头时会话 Cookie 只走 HTTPS
  TRUSTED_PROXIES  反向代理的地址或网段（逗号分隔），只采信来自这些地址的 X-Forwarded-*
  COOKIE_SECURE    设为 1 时强制会话 Cookie 只走 HTTPS
"""
import http.server
import socketserver
import threading
import json
import urllib.request
import urllib.error
import urllib.parse
import base64
import hashlib
import hmac
import os
import secrets
import sys
import time
import re
import uuid
import tempfile
import ipaddress
import socket

try:                       # 可选依赖：只用于图库缩略图，缺失时自动退回原图
    from PIL import Image, ImageOps
except ImportError:        # 保持「只用标准库也能跑」这个前提
    Image = ImageOps = None

PORT = int(os.environ.get("PORT", "19150"))  # 默认高位端口
# 默认只监听本机回环。对外提供时显式设 HOST=0.0.0.0，并放在 HTTPS 反向代理之后：
# 登录密码与会话 Cookie 不应以明文经过公网。
HOST = os.environ.get("HOST", "127.0.0.1")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
GALLERY_FILE = os.path.join(DATA_DIR, "gallery.json")
PROVIDERS_FILE = os.path.join(DATA_DIR, "providers.json")
AUTH_FILE = os.path.join(DATA_DIR, "auth.json")
SECURITY_LOG = os.path.join(DATA_DIR, "security.log")
SETUP_TOKEN_FILE = os.path.join(DATA_DIR, "setup-token")
# 单个请求体上限。图生图会带 base64 参考图，4096px PNG 编码后约 20-30MB，
# 多张很容易上到几十 MB，所以留足余量；但仍要有上限，否则一个 Content-Length
# 就能让服务端把内存吃光。
MAX_BODY = 64 * 1024 * 1024
# 请求日志上限，超过则把旧日志重命名为 server.log.1（只保留一代）
LOG_MAX_BYTES = 8 * 1024 * 1024
# 生成的图片文件名唯一、内容永不修改，可以长期强缓存：否则每次打开页面，
# 图库里所有全尺寸图片都要重新下载一遍。HTML 与静态资源必须走 no-cache，
# 否则更新代码后浏览器还在用旧页面。
IMAGE_CACHE_CONTROL = "private, max-age=31536000, immutable"
# 缩略图长边：图库格子最大约 320 CSS px，2x 屏 640 足够；4K 原图约 5MB，
# 缩略图约 50KB，首屏与浏览器缓存都小两个数量级。
THUMB_SIZE = 640
THUMB_QUALITY = 82
_LANCZOS = None
if Image is not None:
    _LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
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

API_TYPES = ("openai", "minimax", "vertex")
MINIMAX_MODELS = ("image-01", "image-01-live")
MINIMAX_DEFAULT_BASE = "https://api.minimax.cn/v1"
MINIMAX_ASPECT_RATIOS = ("1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9")
VERTEX_DEFAULT_LOCATION = "global"
VERTEX_MODELS = (
    "gemini-2.5-flash-image",
)
VERTEX_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "21:9")
VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
# 测试可改写为本地假上游；生产保持 None，走官方 aiplatform 域名。
VERTEX_API_ROOT = None
_vertex_token_cache = {}
RANDOM_IDEA_MAX_CHARS = 60

# ---- 登录与会话 ----
# PBKDF2 在 Python 3.9 / 3.11 / 3.13 上均可用（hashlib.scrypt 不是所有版本都有）。
PBKDF2_ITERATIONS = 600000
PBKDF2_ALGO = "pbkdf2_sha256"
SESSION_COOKIE = "if_session"
SESSION_IDLE = 7 * 24 * 3600      # 7 天无操作自动失效（服务端判断）
SESSION_MAX = 14 * 24 * 3600      # 无论是否活跃，14 天后必须重新登录
LOGIN_MAX_FAILS = 5               # 同一来源连续失败次数
LOGIN_LOCK_BASE = 60              # 首次锁定 60 秒，之后翻倍
LOGIN_LOCK_MAX = 15 * 60          # 最长锁定 15 分钟
LOGIN_TRACK_MAX = 10000           # 失败记录条数上限，防止被刷爆内存
USERNAME_MAX_LEN = 64
PASSWORD_MIN_LEN = 8
PASSWORD_MAX_LEN = 200
AUTH_BODY_MAX = 4096              # 登录相关接口的请求体上限


def _parse_networks(value):
    """把逗号分隔的 IP / 网段解析成网段列表；无效项打印提示后忽略。"""
    networks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError:
            print("忽略无效的 TRUSTED_PROXIES 项: %s" % item)
            continue
        # ::ffff:a.b.c.d 形式写成 IPv4 网段，与 _ip_in 对来源地址的归一化保持一致
        mapped = getattr(network.network_address, "ipv4_mapped", None)
        if mapped is not None and network.prefixlen >= 96:
            network = ipaddress.ip_network("%s/%d" % (mapped, network.prefixlen - 96), strict=False)
        networks.append(network)
    return tuple(networks)


# 反向代理的地址或网段（逗号分隔，如 127.0.0.1 或 Docker 网关所在的 172.16.0.0/12）。
# 只有来自这些地址的请求，才采信 X-Forwarded-For / X-Forwarded-Proto / X-Forwarded-Host，
# 否则任何人都能伪造来源 IP 绕开登录锁定。直接对外暴露时留空即可。
TRUSTED_PROXIES = _parse_networks(os.environ.get("TRUSTED_PROXIES", ""))
# 对外访问地址（如 https://img.example.com）：用于校验跨站请求（反向代理改写 Host 时需要）
PUBLIC_ORIGIN = os.environ.get("PUBLIC_ORIGIN", "").strip().rstrip("/")
# 会话 Cookie 是否带 Secure（只通过 HTTPS 发送）。对外地址是 https 时自动开启
COOKIE_SECURE = (os.environ.get("COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")
                 or PUBLIC_ORIGIN.lower().startswith("https://"))

_sessions = {}
_sessions_lock = threading.Lock()
_login_fails = {}                 # key -> {"fails", "first", "until"}
_login_lock = threading.Lock()
_auth_write_lock = threading.Lock()   # 认证文件与设置令牌的读改写


def _ip_in(address, networks):
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return any(ip in network for network in networks)


def _strip_port(hop):
    """X-Forwarded-For 里的地址可能带端口：1.2.3.4:443 或 [2001:db8::1]:443。"""
    if hop.startswith("["):
        return hop[1:hop.find("]")] if "]" in hop else hop
    if hop.count(":") == 1:
        return hop.split(":", 1)[0]
    return hop


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


def _atomic_write_private(filepath, content):
    """原子写入并收紧为仅属主可读写：认证文件含密码哈希，不能随默认 umask 放宽。"""
    filepath_dir = os.path.dirname(filepath) or "."
    fd, tmp = tempfile.mkstemp(dir=filepath_dir, prefix=".tmp_")
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, filepath)


def _create_private_exclusive(filepath, content):
    """仅当文件不存在时创建（原子、0600）；已存在则抛 FileExistsError。

    先写临时文件再 os.link：link 在目标已存在时失败，保证并发的首次设置只有一个能成功，
    且读者永远不会看到写了一半的文件。文件系统不支持硬链接时（部分网络盘），
    退回 O_EXCL 直接创建，仍然只有一个能成功。
    """
    filepath_dir = os.path.dirname(filepath) or "."
    fd, tmp = tempfile.mkstemp(dir=filepath_dir, prefix=".tmp_")
    linked = False
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, content)
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(tmp, filepath)
            linked = True
        except FileExistsError:
            raise
        except OSError:
            pass
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if not linked:
        fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)


# ---- 登录：账号密码哈希、会话与失败锁定 ----

def hash_password(password, iterations=PBKDF2_ITERATIONS, salt=None):
    """返回可持久化的密码哈希记录（只存 salt 与派生值，绝不存明文）。"""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    return {
        "algo": PBKDF2_ALGO,
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(dk).decode("ascii"),
    }


def verify_password(password, record):
    """常数时间比较；算法/字段不合法一律视为不匹配，不抛错。"""
    try:
        if not isinstance(password, str) or not isinstance(record, dict):
            return False
        if record.get("algo") != PBKDF2_ALGO:
            return False
        iterations = int(record.get("iterations") or 0)
        if not 10000 <= iterations <= PBKDF2_ITERATIONS * 2:
            return False
        salt = base64.b64decode(record.get("salt") or "")
        expected = base64.b64decode(record.get("hash") or "")
        if len(expected) != 32 or len(salt) < 8:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def validate_username(username):
    """返回错误信息；合法返回 None。"""
    if not isinstance(username, str) or not username.strip():
        return "请填写账号"
    if len(username) > USERNAME_MAX_LEN:
        return "账号最多 %d 个字符" % USERNAME_MAX_LEN
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in username):
        return "账号不能包含控制字符"
    try:
        username.encode("utf-8")
    except UnicodeEncodeError:
        return "账号包含无效字符"
    return None


def validate_password(password):
    """返回错误信息；合法返回 None。"""
    if not isinstance(password, str):
        return "密码格式不正确"
    if len(password) < PASSWORD_MIN_LEN:
        return "密码至少 %d 个字符" % PASSWORD_MIN_LEN
    if len(password) > PASSWORD_MAX_LEN:
        return "密码最长 %d 个字符" % PASSWORD_MAX_LEN
    if password != password.strip():
        return "密码首尾不能是空白字符"
    if password.isdigit() or password.isalpha():
        return "密码不能是纯数字或纯字母，建议混合字母、数字和符号"
    try:
        password.encode("utf-8")
    except UnicodeEncodeError:
        return "密码包含无效字符"
    return None


def auth_configured():
    return os.path.isfile(AUTH_FILE)


def _load_auth():
    """读取认证记录；文件损坏时抛错，绝不静默当成“未配置”而允许重设密码。"""
    with open(AUTH_FILE, "r", encoding="utf-8") as f:
        record = json.load(f)
    if not isinstance(record, dict) or not record.get("password"):
        raise ValueError("认证文件格式不正确")
    return record


def _save_auth(record):
    _atomic_write_private(AUTH_FILE, json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))


def ensure_setup_token():
    """未设置账号时确保存在一次性设置令牌（data/setup-token，0600），并返回它。

    没有令牌，任何先访问到服务的人都能抢先设置账号。令牌只能从服务器本机文件或
    启动日志中拿到，只有部署者能完成首次设置。账号设置好后令牌文件即被删除。
    令牌以文件为准（不缓存在内存里），删除 auth.json 重设密码时也会重新生成。
    """
    with _auth_write_lock:
        if auth_configured():
            try:
                os.unlink(SETUP_TOKEN_FILE)
            except OSError:
                pass
            return ""
        token = ""
        try:
            with open(SETUP_TOKEN_FILE, "r", encoding="utf-8") as f:
                token = f.read().strip()
        except OSError:
            pass
        if len(token) < 16:
            token = secrets.token_urlsafe(18)
            _atomic_write_private(SETUP_TOKEN_FILE, (token + "\n").encode("utf-8"))
        return token


def _check_setup_token(candidate):
    token = ensure_setup_token()
    if not token or not isinstance(candidate, str) or not candidate.strip():
        return False
    try:
        return hmac.compare_digest(candidate.strip().encode("utf-8"), token.encode("utf-8"))
    except UnicodeEncodeError:
        return False


def create_session():
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _sessions_lock:
        _purge_sessions_locked(now)
        _sessions[token] = {"created": now, "seen": now}
    return token


def check_session(token):
    """校验会话并刷新最近活动时间；过期即删除。

    空闲期限只在服务端判断，Cookie 本身用固定的最长有效期，因此不需要在响应里续发 Cookie。
    （续发 Cookie 曾挂在连接对象上，经反向代理复用连接时会串到别的请求里。）
    """
    if not token or not isinstance(token, str):
        return False
    now = time.time()
    with _sessions_lock:
        sess = _sessions.get(token)
        if sess is None:
            return False
        if now - sess["created"] > SESSION_MAX or now - sess["seen"] > SESSION_IDLE:
            del _sessions[token]
            return False
        sess["seen"] = now
        return True


def destroy_session(token):
    if not token:
        return
    with _sessions_lock:
        _sessions.pop(token, None)


def destroy_all_sessions():
    """改密码后让所有旧会话失效（单账号应用，全部清空即可）。"""
    with _sessions_lock:
        _sessions.clear()


def _purge_sessions_locked(now):
    dead = [k for k, v in _sessions.items()
            if now - v["created"] > SESSION_MAX or now - v["seen"] > SESSION_IDLE]
    for k in dead:
        del _sessions[k]


def _rate_source(ip):
    """限流用的来源：IPv6 按 /64 网段计，一个用户通常就能分到整段地址，逐个换地址绕不开锁定。"""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return ip or "-"
    address = getattr(address, "ipv4_mapped", None) or address
    if address.version == 6:
        return str(ipaddress.ip_network("%s/64" % address, strict=False))
    return str(address)


def login_key(username, ip):
    """失败次数按「账号 + 来源」计，并额外按来源单计，避免换账号绕过锁定。

    账号先截断再入键，避免超长账号撑大内存。
    """
    name = (username or "")[:USERNAME_MAX_LEN].lower()
    source = _rate_source(ip)
    return ["u:%s|%s" % (name, source), "ip:%s" % source]


def login_locked_for(keys):
    """返回剩余锁定秒数（0 表示未锁定）。"""
    now = time.time()
    wait = 0
    with _login_lock:
        for key in keys:
            rec = _login_fails.get(key)
            if rec and rec.get("until", 0) > now:
                wait = max(wait, int(rec["until"] - now) + 1)
    return wait


def record_login_failure(keys):
    now = time.time()
    with _login_lock:
        _purge_login_locked(now)
        for key in keys:
            rec = _login_fails.get(key)
            if rec is None:
                if len(_login_fails) >= LOGIN_TRACK_MAX:
                    # 记录表已满：优先丢掉未处于锁定状态的最早记录，保证内存有上限，
                    # 同时不让攻击者靠刷新记录把正在锁定的来源挤出去
                    unlocked = [k for k, v in _login_fails.items() if v.get("until", 0) <= now]
                    victims = unlocked or list(_login_fails)
                    del _login_fails[min(victims, key=lambda k: _login_fails[k]["first"])]
                rec = _login_fails[key] = {"fails": 0, "first": now, "until": 0}
            # 十分钟内没有新的失败就重新计数
            if now - rec["first"] > 600:
                rec["fails"], rec["first"] = 0, now
            rec["fails"] += 1
            if rec["fails"] >= LOGIN_MAX_FAILS:
                extra = rec["fails"] - LOGIN_MAX_FAILS
                lock = min(LOGIN_LOCK_BASE * (2 ** min(extra, 10)), LOGIN_LOCK_MAX)
                rec["until"] = now + lock


def clear_login_failures(keys):
    with _login_lock:
        for key in keys:
            _login_fails.pop(key, None)


def _purge_login_locked(now):
    dead = [k for k, v in _login_fails.items() if v.get("until", 0) < now and now - v["first"] > 3600]
    for k in dead:
        del _login_fails[k]


def _log_safe(value, limit=64):
    """日志字段：数字原样；文本去掉换行等控制字符、截断并加引号，防止伪造日志行或混淆字段。"""
    if isinstance(value, int):
        return str(value)
    text = "".join(ch if ch.isprintable() else "?" for ch in str(value))
    text = text[:limit] + ("…" if len(text) > limit else "")
    return json.dumps(text, ensure_ascii=False)


def log_security(message, *args):
    """把登录相关事件写进独立的 security.log（仅属主可读；不记录密码或 Cookie）。"""
    try:
        path = SECURITY_LOG
        try:
            if os.path.getsize(path) > LOG_MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        line = message % tuple(_log_safe(a) for a in args) if args else message
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        # 旧版本可能已经以 0644 创建过这个文件：每次写入时收紧权限
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
    except Exception:
        pass


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


SECRET_ENV = "IMAGE_FANTASY_SECRET"
SECRET_FILE = os.path.join(DATA_DIR, "secret.key")
ENC_PREFIX = "enc:v1:"
_fernet_cache = [None]


def _get_fernet():
    """渠道密钥加密用的 Fernet。主密钥优先取环境变量，否则用 data/secret.key（自动生成，0600）。
    未安装 cryptography 时返回 None（退回明文，启动时会有警告）。"""
    if _fernet_cache[0] is not None:
        return _fernet_cache[0]
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None
    secret = os.environ.get(SECRET_ENV, "").strip()
    if secret:
        material = hashlib.sha256(secret.encode("utf-8")).digest()
    else:
        if not os.path.isfile(SECRET_FILE):
            os.makedirs(DATA_DIR, exist_ok=True)
            try:
                _create_private_exclusive(SECRET_FILE, secrets.token_urlsafe(48).encode("ascii"))
            except FileExistsError:
                pass
        with open(SECRET_FILE, "rb") as f:
            material = hashlib.sha256(f.read().strip()).digest()
    _fernet_cache[0] = Fernet(base64.urlsafe_b64encode(material))
    return _fernet_cache[0]


def _encrypt_secret(value):
    if not isinstance(value, str) or not value or value.startswith(ENC_PREFIX):
        return value
    f = _get_fernet()
    if f is None:
        return value
    return ENC_PREFIX + f.encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt_secret(value):
    if not isinstance(value, str) or not value.startswith(ENC_PREFIX):
        return value
    f = _get_fernet()
    if f is None:
        raise OSError("渠道密钥已加密，需要安装 cryptography：pip install cryptography")
    try:
        return f.decrypt(value[len(ENC_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception as e:
        raise OSError(
            "渠道密钥解密失败：主密钥（%s 或 data/secret.key）与加密时不一致" % SECRET_ENV
        ) from e


def load_providers():
    # 文件不存在 = 全新部署/首次启动，是正常状态（无供应商）
    if not os.path.isfile(PROVIDERS_FILE):
        return []
    try:
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception as e:
        raise OSError(f"供应商文件加载失败: {e}") from e
    for it in items:
        if isinstance(it, dict) and "key" in it:
            it["key"] = _decrypt_secret(it["key"])
    return items


def save_providers(items):
    stored = []
    for it in items:
        it = dict(it)
        if "key" in it:
            it["key"] = _encrypt_secret(it["key"])
        stored.append(it)
    data = json.dumps(stored, ensure_ascii=False, indent=2).encode()
    with _lock:
        _atomic_write_private(PROVIDERS_FILE, data)


def public_provider(it):
    """返回给浏览器的渠道视图：永远不含 key（API Key / 服务账号私钥），只带状态与提示。"""
    out = {k: v for k, v in it.items() if k != "key"}
    key = (it.get("key") or "").strip()
    out["has_key"] = bool(key)
    out["key_hint"] = ""
    if it.get("api_type") == "vertex":
        out["key_problem"] = ""
        out["key_email"] = ""
        try:
            info = _vertex_service_account_info(key)
            if info is None:
                out["key_problem"] = "尚未配置服务账号 JSON"
            else:
                out["key_email"] = info.get("client_email") or ""
        except ValueError as exc:
            out["key_problem"] = str(exc)
    elif key:
        out["key_hint"] = ("…" + key[-4:]) if len(key) > 8 else "已保存"
    return out


def migrate_providers_encryption():
    """启动时把旧的明文渠道密钥加密重写（无明文备份）；无事可做则不动文件。"""
    if _get_fernet() is None:
        print("警告：未安装 cryptography，渠道密钥以明文保存。pip install cryptography")
        return
    if not os.path.isfile(PROVIDERS_FILE):
        return
    with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if any(isinstance(it, dict) and it.get("key") and not str(it["key"]).startswith(ENC_PREFIX) for it in raw):
        save_providers(load_providers())
        print("已将渠道密钥加密保存。")


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


def thumbs_dir():
    """缩略图目录：与 images 同级。按 IMAGES_DIR 推导，DATA_DIR 自定义时不跑偏。"""
    path = os.path.join(os.path.dirname(IMAGES_DIR), "thumbs")
    os.makedirs(path, exist_ok=True)
    return path


def thumb_name(filename):
    return filename.rsplit(".", 1)[0] + ".jpg"


def thumb_url(filename):
    """缩略图地址带上原图扩展名：同一张图只有一个入口，服务端决定缩略图格式。"""
    return "/thumbs/" + filename


def make_thumb(filename):
    """为一张图生成 JPEG 缩略图，返回是否成功。

    没有 Pillow 或图片无法解码时返回 False：调用方退回原图即可，
    应用在纯标准库环境下必须照常可用，只是图库加载原图会慢一些。
    """
    if Image is None:
        return False
    source = os.path.join(IMAGES_DIR, filename)
    target = os.path.join(thumbs_dir(), thumb_name(filename))
    if os.path.isfile(target):
        return True
    # 并发请求同一张图时各写各的临时文件再 replace，读者不会看到半张图
    temp = target + "." + uuid.uuid4().hex[:6] + ".tmp"
    try:
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((THUMB_SIZE, THUMB_SIZE), _LANCZOS)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                # 带透明通道的图直接转 JPEG 会变黑底，先压到白底上
                im = im.convert("RGBA")
                canvas = Image.new("RGB", im.size, (255, 255, 255))
                canvas.paste(im, mask=im.getchannel("A"))
                im = canvas
            else:
                im = im.convert("RGB")
            im.save(temp, "JPEG", quality=THUMB_QUALITY, optimize=True, progressive=True)
        os.replace(temp, target)
        return True
    except Exception:
        try:
            os.unlink(temp)
        except OSError:
            pass
        return False


def save_image(item, out_fmt):
    """把 upstream 返回的一张图(b64_json 或 url)落盘，返回 (filename, local_url)。

    只做取图与写文件，不读也不写图库索引，因此调用方不需要（更不应该）持锁：
    按 URL 下载可能耗时数十秒，持锁会把整个应用卡住。
    """
    # 时间戳保留文件名里的时间顺序；随机后缀消除并发同毫秒写入时的撞名覆盖。
    name = "img_%d_%s" % (int(time.time() * 1000), uuid.uuid4().hex[:8])
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
    make_thumb(filename)      # 失败不阻塞生成；访问 /thumbs/ 时会再试一次
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


class _UrllibAuthResponse:
    """google-auth 刷新令牌时期望的最小响应。"""

    def __init__(self, status, data):
        self.status = status
        self.data = data


class _UrllibAuthRequest:
    """用标准库发令牌请求，避免再依赖 requests。"""

    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        req = urllib.request.Request(
            url, data=body, headers=dict(headers or {}), method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _UrllibAuthResponse(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return _UrllibAuthResponse(exc.code, exc.read())


def _import_google_auth():
    try:
        from google.oauth2 import service_account
        import google.auth
    except ImportError as exc:
        raise ValueError(
            "Vertex 渠道需要安装 google-auth：pip install google-auth"
        ) from exc
    return service_account, google.auth


def _normalize_vertex_credential_dict(info):
    """兼容标准 GCP 服务账号 JSON 与 CLIProxyAPI 等工具包装的 Vertex 凭据 JSON。"""
    if not isinstance(info, dict):
        raise ValueError("服务账号 JSON 必须是对象")
    outer = info
    for nested_key in ("service_account", "serviceAccount", "credentials"):
        nested = outer.get(nested_key)
        if isinstance(nested, str) and nested.strip().startswith("{"):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = None
        if isinstance(nested, dict):
            merged = dict(nested)
            for fallback_k in ("project_id", "location"):
                if not merged.get(fallback_k) and outer.get(fallback_k):
                    merged[fallback_k] = outer[fallback_k]
            if not merged.get("client_email"):
                email = outer.get("client_email") or outer.get("email") or outer.get("account")
                if email:
                    merged["client_email"] = email
            info = merged
            break

    kind = info.get("type")
    # 识别 CLIProxyAPI 管理面板导出的「账号状态记录」（仅有统计信息与路径，不含 private_key）
    if not info.get("private_key") and (
        kind == "vertex" or info.get("provider") == "vertex"
        or "auth_index" in info or "recent_requests" in info
    ):
        server_path = info.get("path") or info.get("id") or "/root/.cli-proxy-api/vertex-*.json"
        raise ValueError(
            "这份 JSON 是 CLIProxyAPI 管理面板的「账号状态记录」（仅含邮箱、项目 ID 和请求统计，"
            "不含私钥 private_key，无法用于签名鉴权）。"
            "请从服务器取出实际的密钥文件（如 %s），"
            "或到 Google Cloud Console（IAM 和管理 → 服务账号 → 密钥）下载完整的服务账号 JSON。"
            % server_path
        )

    # 兼容顶层 type="vertex" 但已包含 private_key 的导入文件
    if kind == "vertex" and info.get("private_key"):
        info = dict(info)
        info["type"] = "service_account"
        if not info.get("client_email"):
            email = info.get("email") or info.get("account")
            if email:
                info["client_email"] = email
        if not info.get("token_uri"):
            info["token_uri"] = "https://oauth2.googleapis.com/token"

    if isinstance(info.get("private_key"), str) and "\\n" in info["private_key"] and "\n" not in info["private_key"]:
        info = dict(info)
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return info


def _require_service_account(info):
    """确认是可用的服务账号密钥，并对常见误传文件给出明确的中文原因。"""
    info = _normalize_vertex_credential_dict(info)
    kind = info.get("type")
    if kind == "authorized_user":
        raise ValueError(
            "这是 gcloud 用户登录凭据，不是服务账号密钥。"
            "请到 Cloud Console：IAM 和管理 → 服务账号 → 密钥 → 添加密钥 → JSON。"
        )
    if kind not in (None, "service_account"):
        raise ValueError(
            "这个 JSON 的 type 是 %s，不是服务账号密钥。"
            "请到 Cloud Console：IAM 和管理 → 服务账号 → 密钥 → 添加密钥 → JSON。"
            % kind
        )
    missing = [name for name in ("client_email", "token_uri", "private_key") if not info.get(name)]
    if missing or kind != "service_account":
        detail = "、".join(missing) if missing else "type"
        raise ValueError(
            "这个 JSON 不是完整的服务账号密钥，缺少 %s。"
            "服务账号邮箱应类似 name@project.iam.gserviceaccount.com，"
            "token_uri 一般为 https://oauth2.googleapis.com/token。"
            "OAuth 客户端（client_id / client_secret）、面板状态记录和普通 API Key 都不能直接用于此模式。"
            % detail
        )
    return info


def _vertex_service_account_info(key):
    """把渠道里的 Key 解析成服务账号 dict；支持 JSON 全文或本地文件路径。"""
    raw = (key or "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("服务账号 JSON 无法解析，请检查是否完整粘贴") from exc
        return _require_service_account(info)
    path = os.path.expanduser(raw)
    if not os.path.isfile(path):
        raise ValueError("服务账号文件不存在：%s" % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取服务账号文件：%s" % exc) from exc
    return _require_service_account(info)


def vertex_access_token(key):
    """用服务账号或 ADC 换取 access token，顺带返回凭据里的 project_id（若有）。"""
    service_account, google_auth = _import_google_auth()
    info = _vertex_service_account_info(key)
    cache_key = hashlib.sha256((key or "\0adc").encode("utf-8")).hexdigest()
    cached = _vertex_token_cache.get(cache_key)
    now = time.time()
    if cached and cached["expiry"] - 60 > now:
        return cached["token"], cached.get("project_id")

    project_id = None
    if info is not None:
        try:
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=VERTEX_SCOPES
            )
        except Exception as exc:
            raise ValueError("服务账号私钥或格式无效：%s" % exc) from exc
        project_id = (info.get("project_id") or "").strip() or None
    else:
        credentials, adc_project = google_auth.default(scopes=VERTEX_SCOPES)
        project_id = (adc_project or "").strip() or None

    try:
        credentials.refresh(_UrllibAuthRequest())
    except Exception as exc:
        msg = str(exc)
        if "invalid_grant" in msg:
            raise ValueError("向 Google 换取令牌被拒绝（invalid_grant）：该服务账号密钥可能已被删除、吊销，或服务器时间偏差过大") from exc
        raise ValueError("向 Google 换取访问令牌失败：%s" % msg) from exc
    token = credentials.token
    if not token:
        raise ValueError("未能从 Google 凭据获取 access token")
    expiry = now + 3500
    if getattr(credentials, "expiry", None) is not None:
        try:
            expiry = credentials.expiry.timestamp()
        except Exception:
            pass
    _vertex_token_cache[cache_key] = {
        "token": token, "expiry": expiry, "project_id": project_id,
    }
    return token, project_id


def vertex_model_url(project_id, location, model, action=None):
    """拼 Vertex 发布商模型地址。action 为空时只读模型路径。"""
    location = (location or VERTEX_DEFAULT_LOCATION).strip() or VERTEX_DEFAULT_LOCATION
    project_id = (project_id or "").strip()
    model = (model or "").strip()
    if not project_id:
        raise ValueError("Vertex 渠道需要填写 Project ID")
    if not model:
        raise ValueError("Vertex 渠道需要选择模型")
    if VERTEX_API_ROOT:
        root = VERTEX_API_ROOT.rstrip("/")
    elif location == "global":
        root = "https://aiplatform.googleapis.com"
    else:
        root = "https://%s-aiplatform.googleapis.com" % location
    url = (
        "%s/v1/projects/%s/locations/%s/publishers/google/models/%s"
        % (root, urllib.parse.quote(project_id, safe=""),
           urllib.parse.quote(location, safe=""), urllib.parse.quote(model, safe=".-_"))
    )
    if action:
        url += ":" + action
    return url, location, model


def _is_vertex_imagen(model):
    return (model or "").strip().lower().startswith("imagen")


def vertex_endpoint(project_id, location, model):
    action = "predict" if _is_vertex_imagen(model) else "generateContent"
    url, _, _ = vertex_model_url(project_id, location, model, action)
    return url


def _format_vertex_error(err, code, project_id, location, model):
    raw = upstream_err_text(err, code)
    low = raw.lower()
    if code == 401:
        return "Google 拒绝了访问令牌（HTTP 401）：请检查服务账号密钥是否有效"
    if code == 403:
        if "service_disabled" in low or "has not been used in project" in low or "is disabled" in low:
            return "GCP 项目「%s」尚未启用 Vertex AI API，请先在 Cloud Console 中启用 aiplatform.googleapis.com" % project_id
        if "billing" in low:
            return "GCP 项目「%s」尚未开启结算（Billing），请在 Cloud Console 绑定结算账号" % project_id
        return "服务账号在项目「%s」中权限不足（HTTP 403）：请在 IAM 中为该账号授予「Vertex AI User (roles/aiplatform.user)」角色（详情：%s）" % (project_id, raw)
    if code == 404:
        hint = "Imagen 模型建议将 Location 设为 us-central1" if _is_vertex_imagen(model) else "Gemini 出图建议将 Location 设为 global"
        return "在项目「%s」的「%s」区域未找到模型「%s」（%s）" % (project_id, location, model, hint)
    return raw


def check_vertex_account(key, project_id, location, model):
    """换取 access token 并调用轻量 countTokens 验证项目、区域和模型权限，不调用生图、不产生生图计费。"""
    token, inferred = vertex_access_token(key)
    project_id = (project_id or inferred or "").strip()
    target_model = (model or VERTEX_MODELS[0]).strip() or VERTEX_MODELS[0]
    url, location, target_model = vertex_model_url(
        project_id, location, target_model, "countTokens"
    )
    probe_body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": "ping"}]}]
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=probe_body,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read().decode("utf-8"))
        except Exception:
            err = None
        # 部分纯 Imagen 模型不支持 countTokens，若返回 400 INVALID_ARGUMENT 说明鉴权、项目与模型路由已通过
        if exc.code == 400 and _is_vertex_imagen(target_model) and isinstance(err, dict):
            return {"ok": True, "project_id": project_id, "location": location, "model": target_model}
        raise ValueError(_format_vertex_error(err, exc.code, project_id, location, target_model)) from exc
    except urllib.error.URLError as exc:
        raise ValueError("无法连接 Vertex：%s" % exc.reason) from exc
    if not isinstance(body, dict):
        raise ValueError("Vertex 返回了异常响应，请检查模型名和 Location")
    return {"ok": True, "project_id": project_id, "location": location, "model": target_model}


def build_image_request(api_type, base, model, mode, prompt, params, extra, image=None, mask=None,
                        project_id=None):
    """把界面参数转换为渠道原生请求，返回 (endpoint, payload)。"""
    if api_type not in API_TYPES:
        raise ValueError("不支持的渠道接口类型")
    if not isinstance(extra, dict):
        raise ValueError("额外参数必须是 JSON 对象")
    if api_type == "vertex":
        if mode == "edit" or image or mask:
            raise ValueError("Vertex 出图目前仅支持文生图，参考图稍后再接")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请输入提示词")
        aspect = (params.get("aspect_ratio") or "").strip()
        if aspect and aspect not in VERTEX_ASPECT_RATIOS:
            raise ValueError("Vertex 不支持此画幅比例")
        try:
            count = max(1, int(params.get("n", 1) or 1))
        except (ValueError, TypeError):
            count = 1
        if count != 1:
            raise ValueError("Vertex Gemini 出图当前固定单张")
        if _is_vertex_imagen(model):
            parameters = {"sampleCount": count}
            if aspect:
                parameters["aspectRatio"] = aspect
            payload = {
                "instances": [{"prompt": prompt}],
                "parameters": parameters,
            }
            for key, value in extra.items():
                if not key or value in (None, ""):
                    continue
                if key == "parameters" and isinstance(value, dict):
                    parameters.update(value)
                elif key == "instances":
                    payload[key] = value
                else:
                    parameters[key] = value
        else:
            generation = {
                "responseModalities": ["TEXT", "IMAGE"],
            }
            if aspect:
                generation["imageConfig"] = {"aspectRatio": aspect}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": generation,
            }
            # 额外参数可覆盖 / 补充 generationConfig（例如 candidateCount）
            for key, value in extra.items():
                if not key or value in (None, ""):
                    continue
                if key == "generationConfig" and isinstance(value, dict):
                    generation.update(value)
                elif key in ("contents", "safetySettings", "systemInstruction", "labels"):
                    payload[key] = value
                else:
                    generation[key] = value
        endpoint = vertex_endpoint(project_id, base, model)
        return endpoint, payload

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


def _normalize_vertex_image_response(upstream):
    if not isinstance(upstream, dict):
        raise ValueError("Vertex 返回了无效的响应")
    feedback = upstream.get("promptFeedback") or {}
    block = feedback.get("blockReason") or feedback.get("block_reason")
    if block:
        raise ValueError("Vertex 拒绝了该提示词（%s），请调整后重试" % block)
    error = upstream.get("error")
    if isinstance(error, dict) and (error.get("message") or error.get("status")):
        raise ValueError(upstream_err_text(upstream, error.get("code")))
    predictions = upstream.get("predictions")
    if isinstance(predictions, list):
        items = []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            data = pred.get("bytesBase64Encoded") or pred.get("b64_json")
            if isinstance(data, str) and data.strip():
                if data.startswith("data:image/"):
                    data = data.split(",", 1)[-1]
                items.append({"b64_json": data})
        if not items:
            raise ValueError("Vertex Imagen 未返回图片：可能触发了内容安全过滤，请调整提示词后重试")
        return {**upstream, "data": items}
    candidates = upstream.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Vertex 未返回候选结果，请稍后重试")
    items = []
    finish_reasons = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        reason = candidate.get("finishReason") or candidate.get("finish_reason")
        if reason:
            finish_reasons.append(str(reason))
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if not isinstance(inline, dict):
                continue
            data = inline.get("data")
            if isinstance(data, str) and data.strip():
                if data.startswith("data:image/"):
                    data = data.split(",", 1)[-1]
                items.append({"b64_json": data})
    if not items:
        if any(reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII")
               for reason in finish_reasons):
            raise ValueError("Vertex 未返回图片：内容安全检查未通过，请调整提示词后重试")
        if finish_reasons:
            raise ValueError("Vertex 未返回图片（%s）" % "/".join(finish_reasons))
        raise ValueError("Vertex 未返回有效图片，请稍后重试")
    return {**upstream, "data": items}


def normalize_image_response(upstream, api_type):
    """将渠道响应归一为现有图库使用的 data 数组。"""
    if api_type == "vertex":
        return _normalize_vertex_image_response(upstream)
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

    def _send(self, code, obj=None, ctype="application/json", body=None, headers=None):
        if body is None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8") if obj is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype, cache="no-cache"):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            self._send(404, {"error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", cache)
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

    # ---- 登录接口 ----

    def _read_auth_body(self):
        """登录相关接口只接受小体积的 JSON 对象；其余一律视为格式错误。"""
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            return None
        if length < 0 or length > AUTH_BODY_MAX:
            # 不读超大请求体，直接关闭连接
            self.close_connection = True
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    @staticmethod
    def _str_field(body, name):
        value = body.get(name)
        return value if isinstance(value, str) else ""

    def _auth_status(self):
        configured = auth_configured()
        if not configured:
            # 删除 auth.json 重设账号时无需重启：打开登录页即在服务器上重新生成设置令牌
            try:
                ensure_setup_token()
            except Exception:
                pass
        self._send(200, {
            "configured": configured,
            "logged_in": configured and self._session_valid(),
        }, headers={"Cache-Control": "no-store"})

    def _auth_setup(self):
        """首次设置账号需要一次性设置令牌；已设置时只能在登录状态下用当前密码修改。"""
        if not self._csrf_ok():
            self._send(403, {"error": "跨站请求被拒绝"}); return
        body = self._read_auth_body()
        if body is None:
            self._send(400, {"error": "请求格式不正确"}); return
        username = self._str_field(body, "username").strip()
        password = self._str_field(body, "password")
        err = validate_username(username) or validate_password(password)
        if err:
            self._send(400, {"error": err}); return

        ip = self._client_ip()
        keys = login_key(username, ip)
        wait = login_locked_for(keys)
        if wait:
            self._send(429, {"error": "尝试次数过多，请 %d 秒后再试" % wait, "retry_after": wait},
                       headers={"Retry-After": str(wait)})
            return

        try:
            configured = auth_configured()
            old_record = _load_auth() if configured else None
        except Exception:
            self._send(500, {"error": "认证文件损坏，请检查服务器上的 data/auth.json"}); return

        if configured:
            if not self._session_valid():
                self._send(401, {"error": "未登录，不能修改密码"}); return
            # 改密码需要提供当前密码，并计入失败次数，防止会话被借用后暴力猜测
            if not verify_password(self._str_field(body, "current_password"), old_record["password"]):
                record_login_failure(keys)
                log_security("修改密码失败（当前密码错误） 来源=%s", ip)
                self._send(400, {"error": "当前密码不正确"}); return
        else:
            try:
                token_ok = _check_setup_token(self._str_field(body, "setup_token"))
            except OSError as e:
                log_security("读取设置令牌失败: %s", e)
                self._send(500, {"error": "无法读取设置令牌，请检查服务器 data 目录的写入权限"}); return
            if not token_ok:
                record_login_failure(keys)
                log_security("首次设置失败（设置令牌错误） 来源=%s", ip)
                self._send(403, {"error": "设置令牌不正确。请在服务器上查看启动日志或 data/setup-token 文件"}); return

        now_ms = int(time.time() * 1000)
        record = {
            "version": 1,
            "username": username,
            "password": hash_password(password),
            "created": (old_record or {}).get("created", now_ms),
            "updated": now_ms,
        }
        data = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            with _auth_write_lock:
                if configured:
                    _atomic_write_private(AUTH_FILE, data)
                else:
                    # 原子地「仅当不存在时创建」：并发的首次设置只会有一个成功
                    _create_private_exclusive(AUTH_FILE, data)
        except FileExistsError:
            self._send(409, {"error": "账号已被设置，请刷新页面后登录"}); return
        except Exception as e:
            log_security("保存认证信息失败: %s", e)
            self._send(500, {"error": "保存认证信息失败，请检查服务器 data 目录的写入权限"}); return

        ensure_setup_token()          # 已设置账号：删除设置令牌文件
        clear_login_failures(keys)
        destroy_all_sessions()
        token = create_session()
        log_security(("修改密码" if configured else "设置登录账号") + " 账号=%s 来源=%s", username, ip)
        self._send(200, {"ok": True, "username": username},
                   headers={"Set-Cookie": self._session_cookie(token)})

    def _auth_login(self):
        if not self._csrf_ok():
            self._send(403, {"error": "跨站请求被拒绝"}); return
        body = self._read_auth_body()
        if body is None:
            self._send(400, {"error": "请求格式不正确"}); return
        if not auth_configured():
            self._send(400, {"error": "还没有设置登录账号", "setup": True}); return
        username = self._str_field(body, "username").strip()[:USERNAME_MAX_LEN]
        password = self._str_field(body, "password")[:PASSWORD_MAX_LEN]
        ip = self._client_ip()
        keys = login_key(username, ip)
        wait = login_locked_for(keys)
        if wait:
            log_security("登录被锁定 %ss 来源=%s 账号=%s", wait, ip, username)
            self._send(429, {"error": "尝试次数过多，请 %d 秒后再试" % wait,
                             "retry_after": wait}, headers={"Retry-After": str(wait)})
            return
        try:
            record = _load_auth()
        except Exception:
            self._send(500, {"error": "认证文件损坏，请检查服务器上的 data/auth.json"}); return

        try:
            username_ok = hmac.compare_digest(
                (record.get("username") or "").encode("utf-8"), username.encode("utf-8"))
        except UnicodeEncodeError:
            username_ok = False
        # 账号不对时同样计算一次哈希，响应时间不暴露账号是否存在
        password_ok = verify_password(password, record.get("password"))
        if not (username_ok and password_ok):
            record_login_failure(keys)
            log_security("登录失败 来源=%s 账号=%s", ip, username)
            # 不区分「账号不存在」和「密码错误」，避免被用来枚举账号
            self._send(401, {"error": "账号或密码不正确"}); return

        clear_login_failures(keys)
        token = create_session()
        log_security("登录成功 来源=%s 账号=%s", ip, username)
        self._send(200, {"ok": True, "username": record.get("username")},
                   headers={"Set-Cookie": self._session_cookie(token)})

    def _auth_logout(self):
        if not self._csrf_ok():
            self._send(403, {"error": "跨站请求被拒绝"}); return
        destroy_session(self._request_cookie())
        parts = ["%s=" % SESSION_COOKIE, "Path=/", "HttpOnly", "SameSite=Lax", "Max-Age=0"]
        if self._is_https():
            parts.append("Secure")
        # 图片按文件名长期强缓存，登出时必须让浏览器清掉这些副本，否则知道 URL 的人
        # 仍能从缓存里看到已登出账号的作品。只清 HTTP 缓存：主题与表单配置存在
        # localStorage 里，不能用 "storage" 一起清掉。
        self._send(200, {"ok": True}, headers={
            "Set-Cookie": "; ".join(parts),
            "Clear-Site-Data": '"cache"',
        })

    # ---- 安全响应头、登录校验 ----

    def end_headers(self):
        # 所有响应统一带上这些头（图片、静态文件、接口和流式响应都一样）。
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data: blob:; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                         "connect-src 'self'; object-src 'none'; base-uri 'none'; "
                         "form-action 'self'; frame-ancestors 'none'")
        if self._is_https():
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        super().end_headers()

    def _request_cookie(self):
        """自行解析 Cookie 头：标准库 SimpleCookie 遇到别的站点写入的特殊 Cookie 会整串放弃。"""
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, sep, value = part.strip().partition("=")
            if sep and name == SESSION_COOKIE:
                return value.strip().strip('"')
        return ""

    def _from_trusted_proxy(self):
        # 请求行解析失败时（如 414）还没有 headers，此时按非代理处理
        if not TRUSTED_PROXIES or getattr(self, "headers", None) is None or not self.client_address:
            return False
        return _ip_in(self.client_address[0], TRUSTED_PROXIES)

    def _is_https(self):
        if COOKIE_SECURE:
            return True
        if self._from_trusted_proxy():
            return self.headers.get("X-Forwarded-Proto", "").split(",")[0].strip().lower() == "https"
        return False

    def _client_ip(self):
        """限流用的来源 IP。只有请求来自配置的反向代理时，才采信代理转发的头；
        否则这些头可被任意伪造，用来绕过锁定。"""
        direct = self.client_address[0] if self.client_address else "-"
        if not self._from_trusted_proxy():
            return direct
        forwarded = [h.strip() for h in (self.headers.get("X-Forwarded-For") or "").split(",") if h.strip()]
        real = (self.headers.get("X-Real-IP") or "").strip()
        # 从右往左取第一个不属于可信代理的地址：最左侧由客户端自己填写，可伪造
        candidates = list(reversed(forwarded)) or ([real] if real else [])
        for hop in candidates:
            address = _strip_port(hop)
            if _ip_in(address, TRUSTED_PROXIES):
                continue
            try:
                return str(ipaddress.ip_address(address))
            except ValueError:
                break
        return direct

    def _session_valid(self):
        return check_session(self._request_cookie())

    def _session_cookie(self, token):
        parts = ["%s=%s" % (SESSION_COOKIE, token), "Path=/", "HttpOnly", "SameSite=Lax",
                 "Max-Age=%d" % SESSION_MAX]
        if self._is_https():
            parts.append("Secure")
        return "; ".join(parts)

    def _csrf_ok(self):
        """写操作校验来源：跨站页面伪造的请求在这里被挡掉。

        SameSite=Lax Cookie 已经能挡住绝大多数跨站 POST，这一层是兜底。
        浏览器发出的请求带 Origin（或 Sec-Fetch-Site）；没有这两个头的是 curl 等非浏览器客户端，
        它们拿不到受害者的 Cookie，由会话本身保护。
        """
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site == "cross-site":
            return False         # 浏览器明确标记为跨站，不管 Origin 写的是什么
        if site in ("same-origin", "none"):
            return True
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True          # 浏览器的跨站请求一定带 Origin 或 Sec-Fetch-Site
        if origin == "null":
            return False
        try:
            o = urllib.parse.urlparse(origin.lower())
        except ValueError:
            return False
        if o.scheme not in ("http", "https") or not o.netloc:
            return False
        if PUBLIC_ORIGIN and origin.lower().rstrip("/") == PUBLIC_ORIGIN.lower():
            return True
        # 已知本站走 HTTPS 时，同一域名的 http 页面也不能代为提交
        if self._is_https() and o.scheme != "https":
            return False
        hosts = {(self.headers.get("Host") or "").lower()}
        if self._from_trusted_proxy():
            forwarded = (self.headers.get("X-Forwarded-Host") or "").split(",")[0].strip().lower()
            if forwarded:
                hosts.add(forwarded)
        return o.netloc in hosts

    def _require_login(self, write=False):
        """未登录时返回 401；页面与登录接口不走这里。"""
        if not self._session_valid():
            self._send(401, {"error": "未登录或登录已过期，请重新登录", "login": "/login"},
                       headers={"Cache-Control": "no-store"})
            return False
        if write and not self._csrf_ok():
            self._send(403, {"error": "跨站请求被拒绝"}); return False
        return True

    def _serve_image(self, p):
        fname = os.path.basename(p)
        fpath = os.path.join(IMAGES_DIR, fname)
        if not os.path.isfile(fpath):
            self._send(404, {"error": "not found"}); return
        ext = fname.rsplit(".", 1)[-1].lower()
        ctype = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
                 "webp": "image/webp"}.get(ext, "application/octet-stream")
        self._send_file(fpath, ctype, cache=IMAGE_CACHE_CONTROL)

    def _serve_thumb(self, p):
        """缩略图。缺失时懒生成（老图库第一次访问时补上）；不能生成就退回原图。"""
        fname = os.path.basename(p)                 # 原图文件名，如 img_1_ab.png
        if not os.path.isfile(os.path.join(IMAGES_DIR, fname)):
            self._send(404, {"error": "not found"}); return
        target = os.path.join(thumbs_dir(), thumb_name(fname))
        if not os.path.isfile(target):
            make_thumb(fname)
        if os.path.isfile(target):
            self._send_file(target, "image/jpeg", cache=IMAGE_CACHE_CONTROL)
            return
        # 没装 Pillow 或解码失败：重定向到原图，前端不需要任何兜底逻辑
        self._send(302, headers={"Location": "/images/" + urllib.parse.quote(fname)})

    def _serve_static(self, p):
        fpath = os.path.normpath(os.path.join(STATIC_DIR, p[len("/static/"):]))
        if not fpath.startswith(STATIC_DIR):
            self._send(403, {"error": "forbidden"}); return
        ctype = "text/html; charset=utf-8"
        if fpath.endswith(".js"): ctype = "text/javascript; charset=utf-8"
        elif fpath.endswith(".css"): ctype = "text/css; charset=utf-8"
        elif fpath.endswith(".svg"): ctype = "image/svg+xml"
        elif fpath.endswith(".ico"): ctype = "image/vnd.microsoft.icon"
        self._send_file(fpath, ctype)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        # 未登录也要能打开登录页、取登录页资源、查询登录状态
        if p == "/api/auth/status":
            self._auth_status(); return
        if p == "/login":
            self._send_file(os.path.join(STATIC_DIR, "login.html"), "text/html; charset=utf-8"); return
        if p in ("/", "/index.html"):
            # 已登录直接进应用；未登录（或会话过期）先到登录页
            page = "index.html" if self._session_valid() else "login.html"
            self._send_file(os.path.join(STATIC_DIR, page), "text/html; charset=utf-8"); return
        if p == "/favicon.ico":
            self._send_file(os.path.join(STATIC_DIR, "favicon.ico"), "image/vnd.microsoft.icon"); return
        if p in ("/static/login.css", "/static/login.js", "/static/favicon.svg"):
            self._serve_static(p); return

        # 其余接口与图片一律要求登录；前端页面本身也只在已登录时下发
        if p.startswith("/api/") or p.startswith("/images/") or p.startswith("/thumbs/"):
            if not self._require_login():
                return
            no_store = {"Cache-Control": "no-store"}
            if p == "/api/auth/me":
                self._send(200, {"ok": True}, headers=no_store); return
            if p == "/api/gallery":
                try:
                    items = load_gallery()
                    # 没装 Pillow 就直接给原图地址，省掉每张图一次无意义的 302；
                    # 装了则给旧条目补上 thumb，前端不用做兼容判断。
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if Image is None:
                            it.pop("thumb", None)
                        elif it.get("filename") and not it.get("thumb"):
                            it["thumb"] = thumb_url(it["filename"])
                    self._send(200, items, headers=no_store)
                except Exception as e:
                    self._send(500, {"error": str(e)})
                return
            if p == "/api/providers":
                # 不含密钥，但仍禁止缓存
                try:
                    self._send(200, [public_provider(it) for it in load_providers()], headers=no_store)
                except Exception as e:
                    self._send(500, {"error": str(e)})
                return
            if p == "/api/ping":
                self._send(200, {"ok": True, "time": int(time.time() * 1000)}, headers=no_store); return
            if p.startswith("/images/"):
                self._serve_image(p); return
            if p.startswith("/thumbs/"):
                self._serve_thumb(p); return
            self._send(404, {"error": "not found"}); return

        if p.startswith("/static/"):
            if not self._require_login():
                return
            self._serve_static(p); return
        self._send(404, {"error": "not found"})

    # ---- NDJSON 流式响应辅助（手动实现 chunked 传输） ----

    def _emit(self, obj):
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(("%X\r\n" % len(line)).encode("ascii") + line + b"\r\n")
        self.wfile.flush()

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_DELETE(self):
        if not self._require_login(write=True):
            return
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
                    tpath = os.path.join(thumbs_dir(), thumb_name(it.get("filename", "")))
                    try:
                        if os.path.isfile(tpath): os.remove(tpath)
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
        if not self._require_login(write=True):
            return
        parsed = urllib.parse.urlparse(self.path)
        m = re.match(r"^/api/providers/([^/]+)$", parsed.path)
        if not m:
            self._send(404, {"error": "not found"}); return
        self._update_provider(m.group(1))

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        # 登录相关接口不要求会话
        if p == "/api/auth/setup":
            self._auth_setup(); return
        if p == "/api/auth/login":
            self._auth_login(); return
        if p == "/api/auth/logout":
            self._auth_logout(); return
        if not self._require_login(write=True):
            return
        if p == "/api/providers":
            self._create_provider(); return
        if p == "/api/fetch-models":
            self._fetch_models(); return
        if p == "/api/vertex-check":
            self._check_vertex(); return
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
        project_id = (prov.get("project_id") or "").strip()
        if api_type == "vertex":
            if not base:
                base = VERTEX_DEFAULT_LOCATION
        elif not base:
            self._send(400, {"error": "该供应商缺少 Base URL，请到供应商管理里补上。"}); return
        if not model:
            models = prov.get("models") or []
            model = models[0] if models else ""
        if not model:
            self._send(400, {"error": "该供应商还没有模型，请先添加。"}); return

        try:
            if api_type == "minimax" and not key:
                raise ValueError("MiniMax 渠道需要填写 API Key，请到渠道设置中补充")
            auth_token = key
            if api_type == "vertex":
                token, inferred_project = vertex_access_token(key)
                project_id = project_id or (inferred_project or "")
                if not project_id:
                    raise ValueError("Vertex 渠道需要填写 Project ID（或使用带 project_id 的服务账号）")
                auth_token = token
            endpoint, payload = build_image_request(
                api_type, base, model, mode, prompt, params, extra,
                data.get("image"), data.get("mask"), project_id=project_id,
            )
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        body = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if auth_token:
            hdrs["Authorization"] = "Bearer " + auth_token
        req = urllib.request.Request(endpoint, data=body, headers=hdrs, method="POST")

        # ---- 以下走 NDJSON 流式响应：阶段事件实时推送给前端 ----
        try:
            self._start_stream()
        except OSError:
            # 连响应头都写不出去，说明客户端早已断开：不要再去上游下单，省一次付费调用。
            self.close_connection = True
            return

        client_gone = False

        def emit(event):
            """推一条事件给前端。客户端断开后只停止写事件，绝不影响后续落盘。"""
            nonlocal client_gone
            if client_gone:
                return
            try:
                self._emit(event)
            except OSError:
                # 取消、关标签、切网都会走到这里（BrokenPipe / ConnectionReset 都是 OSError）。
                # 上游很可能已经出图并计费，所以只能停止写事件，图片必须照常保存——
                # 否则这一次调用就白花钱了。
                client_gone = True

        try:
            try:
                emit({"stage": "accepted", "time": int(time.time() * 1000)})
                emit({"stage": "upstream", "model": model, "endpoint": endpoint})
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        upstream = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as e:
                    try:
                        err = json.loads(e.read().decode("utf-8"))
                    except Exception:
                        err = None
                    emit({"stage": "error", "http": e.code,
                          "message": upstream_err_text(err, e.code)})
                    return
                except Exception as e:
                    emit({"stage": "error", "message": str(e)})
                    return

                upstream = normalize_image_response(upstream, api_type)
                items = upstream.get("data") or []
                emit({"stage": "saving", "count": len(items)})
                # MiniMax / Vertex 不接受 output_format；按返回的真实图片内容确定后缀。
                out_fmt = None if api_type in ("minimax", "vertex") else params.get("output_format")
                size = params.get("size", "")
                if api_type == "minimax":
                    size = payload.get("aspect_ratio") or (
                        "%sx%s" % (payload["width"], payload["height"]) if "width" in payload else "1:1"
                    )
                elif api_type == "vertex":
                    size = (
                        ((payload.get("generationConfig") or {}).get("imageConfig") or {}).get("aspectRatio")
                        or (payload.get("parameters") or {}).get("aspectRatio")
                        or "auto"
                    )
                saved = []
                # 取图与写盘都在锁外：按 URL 下载可能耗时数十秒，持锁会卡死整个应用。
                for it in items:
                    filename, local_url = save_image(it, out_fmt)
                    if not filename:
                        continue
                    gid = filename.rsplit(".", 1)[0]
                    saved.append({
                        "id": gid, "filename": filename, "url": local_url,
                        "thumb": thumb_url(filename),
                        "prompt": prompt, "model": model,
                        "size": size,
                        "output_format": (
                            filename.rsplit(".", 1)[-1] if api_type in ("minimax", "vertex") else out_fmt or ""
                        ),
                        "api_type": api_type,
                        "mode": mode, "created": int(time.time() * 1000),
                    })
                if saved:
                    # 只有索引的 读 -> 改 -> 写 需要加锁，避免并发丢条目。
                    with _lock:
                        gallery = load_gallery()
                        gallery.extend(saved)
                        save_gallery(gallery)
                emit({"stage": "done", "created": upstream.get("created"),
                      "usage": upstream.get("usage"), "items": saved})
            except Exception as e:
                emit({"stage": "error", "message": str(e)})
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

    def _provider_fields(self, body, existing_key=None):
        name = (body.get("name") or "").strip()
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        if not key and existing_key:
            # 编辑时留空 = 保持已保存的密钥不变（浏览器从不拿到密钥，也就不会回传）
            key = existing_key.strip()
        project_id = (body.get("project_id") or "").strip()
        api_type = body.get("api_type") or "openai"
        if api_type not in API_TYPES:
            raise ValueError("接口类型仅支持 OpenAI 兼容、MiniMax 原生或 Vertex AI")
        if api_type == "minimax" and not base:
            base = MINIMAX_DEFAULT_BASE
        if api_type == "vertex" and not base:
            base = VERTEX_DEFAULT_LOCATION
        if api_type == "vertex":
            raw_location = (body.get("base") or "").strip().rstrip("/")
            if key.startswith("{"):
                try:
                    parsed_key = json.loads(key)
                except json.JSONDecodeError as exc:
                    raise ValueError("服务账号 JSON 无法解析") from exc
                info = _require_service_account(parsed_key)
                key = json.dumps(info, ensure_ascii=False)
                if not project_id:
                    project_id = (info.get("project_id") or "").strip()
                if not raw_location and info.get("location"):
                    base = str(info.get("location")).strip()
            else:
                path = os.path.expanduser(key)
                if not key or not os.path.isfile(path):
                    raise ValueError("Vertex 渠道需要上传或粘贴有效的服务账号 JSON（若填服务器文件路径，请确认文件已存在）")
                info = _vertex_service_account_info(key)
                if not project_id:
                    project_id = (info.get("project_id") or "").strip()
                if not raw_location and info.get("location"):
                    base = str(info.get("location")).strip()
        raw = body.get("models") or []
        if not isinstance(raw, list):
            raw = []
        seen, models = set(), []
        for m in raw:
            m = str(m).strip()
            if m and m not in seen:
                seen.add(m); models.append(m)
        return name, base, key, models, api_type, project_id

    def _create_provider(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        try:
            name, base, key, models, api_type, project_id = self._provider_fields(body)
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        if not name or not base:
            self._send(400, {"error": "名称和接口地址（Vertex 为 Location）必填"}); return
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
            if api_type == "vertex":
                prov["project_id"] = project_id
            items.append(prov)
            save_providers(items)
            self._send(200, public_provider(prov))

    def _update_provider(self, pid):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        # 在锁内完成 读 -> 改 -> 写，避免并发修正
        with _lock:
            try:
                items = load_providers()
            except Exception as e:
                self._send(500, {"error": f"供应商读取失败: {e}"}); return
            found = next((it for it in items if it.get("id") == pid), None)
            if found is None:
                self._send(404, {"error": "供应商不存在"}); return
            try:
                name, base, key, models, api_type, project_id = self._provider_fields(
                    body, existing_key=found.get("key") or ""
                )
            except ValueError as exc:
                self._send(400, {"error": str(exc)}); return
            it = found
            if name: it["name"] = name
            if base: it["base"] = base
            it["key"] = key
            it["models"] = models
            if "api_type" in body:
                it["api_type"] = api_type
            if api_type == "vertex" or it.get("api_type") == "vertex":
                it["project_id"] = project_id
            elif "project_id" in it and api_type != "vertex":
                it.pop("project_id", None)
            save_providers(items)
            self._send(200, public_provider(found))

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
            save_providers(new_items)
            self._send(200, {"ok": True})

    def _check_vertex(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        key = (body.get("key") or "").strip()
        provider_id = (body.get("provider_id") or "").strip()
        if not key and provider_id:
            try:
                prov = next((it for it in load_providers() if it.get("id") == provider_id), None)
            except Exception as e:
                self._send(500, {"error": f"供应商读取失败: {e}"}); return
            if prov is None:
                self._send(404, {"error": "渠道不存在或已被删除"}); return
            key = (prov.get("key") or "").strip()
        try:
            result = check_vertex_account(
                key,
                body.get("project_id") or "",
                body.get("base") or body.get("location") or "",
                body.get("model") or "",
            )
        except ValueError as exc:
            self._send(400, {"error": str(exc)}); return
        self._send(200, result)

    def _fetch_models(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        api_type = body.get("api_type") or "openai"
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        provider_id = (body.get("provider_id") or "").strip()
        if provider_id:
            # 设置页里已保存的渠道：由服务端取 Base URL 和 Key，
            # 前端不必把密钥再传一遍。
            prov = next((it for it in load_providers() if it.get("id") == provider_id), None)
            if prov is None:
                self._send(404, {"error": "渠道不存在或已被删除"}); return
            api_type = prov.get("api_type") or "openai"
            base = (prov.get("base") or "").strip().rstrip("/")
            key = (prov.get("key") or "").strip()
        if api_type not in API_TYPES:
            self._send(400, {"error": "不支持的渠道接口类型"}); return
        if api_type == "minimax":
            # 官方图片接口未提供模型枚举端点，明确返回文档中的预设列表。
            self._send(200, {"models": list(MINIMAX_MODELS), "source": "preset"}); return
        if api_type == "vertex":
            self._send(200, {"models": list(VERTEX_MODELS), "source": "preset"}); return
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
        if prov.get("api_type") in ("minimax", "vertex"):
            self._send(400, {"error": "此渠道用于图片生成，请为灵感功能选择 OpenAI 兼容的文本模型渠道"}); return

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
    migrate_providers_encryption()
    with ThreadingHTTPServer((HOST, PORT), Handler) as httpd:
        shown = "127.0.0.1" if HOST in ("127.0.0.1", "localhost") else HOST
        print("Image Fantasy 已启动 ->  http://%s:%d" % (shown, PORT))
        print("图库目录: %s" % DATA_DIR)
        setup_token = ensure_setup_token()
        if setup_token:
            print("")
            print("首次使用：打开页面设置登录账号，需要填写下面的设置令牌（只在服务器上可见）：")
            print("    设置令牌: %s" % setup_token)
            print("    也保存在 %s，设置完成后自动删除。" % SETUP_TOKEN_FILE)
            print("")
        if HOST not in ("127.0.0.1", "localhost") and not COOKIE_SECURE:
            print("提示: 当前监听 %s。对公网开放时请放在 HTTPS 反向代理之后，" % HOST)
            print("      并设置 PUBLIC_ORIGIN=https://你的域名，否则密码与会话 Cookie 会以明文经过网络。")
        if TRUSTED_PROXIES:
            print("信任的反向代理: %s" % ", ".join(str(n) for n in TRUSTED_PROXIES))
        print("按 Ctrl+C 退出。")
        # Docker 下 stdout 不是终端，默认整块缓冲：立即刷出，设置令牌才能马上出现在 docker logs 里
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
