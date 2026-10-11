#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Fantasy —— 自托管图像生成 Web 应用（零依赖，仅用 Python 标准库）

特性：
  - 对接任意 OpenAI 兼容图像 API（gpt-image-1 及各类兼容渠道 / 代理）
  - 全参数暴露：prompt / size(512–4096, 32倍数, ≤3:1, 或 auto) / n / output_format /
    output_compression / response_format / watermark / prompt_extend(提示词润色) / 参考图编辑(edit)
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
  # 浏览器打开 http://127.0.0.1:8080
部署（Docker）：
  docker compose up -d
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

PORT = int(os.environ.get("PORT", "8080"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
GALLERY_FILE = os.path.join(DATA_DIR, "gallery.json")
PROVIDERS_FILE = os.path.join(DATA_DIR, "providers.json")
os.makedirs(IMAGES_DIR, exist_ok=True)

_lock = threading.Lock()


def load_gallery():
    try:
        with open(GALLERY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_gallery(items):
    with _lock:
        with open(GALLERY_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def load_providers():
    try:
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_providers(items):
    with _lock:
        with open(PROVIDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


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
        req = urllib.request.Request(url, headers={"User-Agent": "image-fantasy"})
        with urllib.request.urlopen(req, timeout=60) as resp:
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


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        elif p.startswith("/static/"):
            fpath = os.path.normpath(os.path.join(STATIC_DIR, p[len("/static/"):]))
            if not fpath.startswith(STATIC_DIR):
                self._send(403, {"error": "forbidden"}); return
            ctype = "text/html; charset=utf-8"
            if fpath.endswith(".js"): ctype = "text/javascript; charset=utf-8"
            elif fpath.endswith(".css"): ctype = "text/css; charset=utf-8"
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
            self._send(200, load_gallery())
        elif p == "/api/providers":
            self._send(200, load_providers())
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
        items = load_gallery()
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
        save_gallery(new_items)
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
        if p != "/api/generate":
            self._send(404, {"error": "not found"}); return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
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
        if not base:
            self._send(400, {"error": "该供应商缺少 Base URL，请到供应商管理里补上。"}); return
        if not model:
            models = prov.get("models") or []
            model = models[0] if models else ""
        if not model:
            self._send(400, {"error": "该供应商还没有模型，请先添加。"}); return

        # 构造上游请求体
        payload = {"model": model, "prompt": prompt}
        if params.get("size"): payload["size"] = params["size"]
        if params.get("output_format"): payload["output_format"] = params["output_format"]
        if params.get("response_format"): payload["response_format"] = params["response_format"]
        # watermark 显式传布尔：False = 去水印（是否生效取决于上游渠道）
        if params.get("watermark") is not None:
            payload["watermark"] = bool(params["watermark"])
        # prompt_extend：提示词润色开关（部分渠道支持，如参考文档默认 true）
        if params.get("prompt_extend") is not None:
            payload["prompt_extend"] = bool(params["prompt_extend"])
        # GPT-Image 兼容参数（其他模型可忽略）
        for k in ("quality", "style", "background"):
            if params.get(k): payload[k] = params[k]
        # 压缩率（仅 jpeg/webp 有效）
        if params.get("output_compression") not in (None, ""):
            try:
                payload["output_compression"] = max(0, min(100, int(params["output_compression"])))
            except Exception:
                pass
        # n：默认 1；部分渠道仅支持 1
        try:
            payload["n"] = max(1, int(params.get("n", 1) or 1))
        except Exception:
            payload["n"] = 1
        # 任意额外参数透传
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k and v not in (None, ""):
                    payload[k] = v
        if mode == "edit":
            if data.get("image"):
                payload["image"] = data["image"]
            if data.get("mask"):
                payload["mask"] = data["mask"]

        endpoint = base + ("/images/edits" if mode == "edit" else "/images/generations")
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
                        err = "HTTP " + str(e.code)
                    self._emit({"stage": "error", "http": e.code, "message": err})
                    return
                except Exception as e:
                    self._emit({"stage": "error", "message": str(e)})
                    return

                items = upstream.get("data") or []
                self._emit({"stage": "saving", "count": len(items)})
                out_fmt = params.get("output_format")
                saved = []
                gallery = load_gallery()
                for i, it in enumerate(items):
                    filename, local_url = save_image(it, i, out_fmt)
                    if not filename:
                        continue
                    gid = filename.rsplit(".", 1)[0]
                    entry = {
                        "id": gid, "filename": filename, "url": local_url,
                        "prompt": prompt, "model": model,
                        "size": params.get("size", ""), "output_format": out_fmt or "",
                        "mode": mode, "created": int(time.time() * 1000),
                    }
                    gallery.append(entry)
                    saved.append(entry)
                save_gallery(gallery)
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

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return None

    def _provider_fields(self, body):
        name = (body.get("name") or "").strip()
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        raw = body.get("models") or []
        if not isinstance(raw, list):
            raw = []
        seen, models = set(), []
        for m in raw:
            m = str(m).strip()
            if m and m not in seen:
                seen.add(m); models.append(m)
        return name, base, key, models

    def _create_provider(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        name, base, key, models = self._provider_fields(body)
        if not name or not base:
            self._send(400, {"error": "名称和 Base URL 必填"}); return
        items = load_providers()
        prov = {
            "id": "p_" + uuid.uuid4().hex[:8],
            "name": name, "base": base, "key": key,
            "models": models, "created": int(time.time() * 1000),
        }
        items.append(prov)
        save_providers(items)
        self._send(200, prov)

    def _update_provider(self, pid):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        name, base, key, models = self._provider_fields(body)
        items = load_providers()
        found = None
        for it in items:
            if it.get("id") == pid:
                if name: it["name"] = name
                if base: it["base"] = base
                it["key"] = key
                it["models"] = models
                found = it
                break
        if found is None:
            self._send(404, {"error": "供应商不存在"}); return
        save_providers(items)
        self._send(200, found)

    def _delete_provider(self, pid):
        items = load_providers()
        new_items = [it for it in items if it.get("id") != pid]
        if len(new_items) == len(items):
            self._send(404, {"error": "供应商不存在"}); return
        save_providers(new_items)
        self._send(200, {"ok": True})

    def _fetch_models(self):
        body = self._read_json_body()
        if body is None:
            self._send(400, {"error": "请求解析失败"}); return
        base = (body.get("base") or "").strip().rstrip("/")
        key = (body.get("key") or "").strip()
        if not base:
            self._send(400, {"error": "缺少 Base URL"}); return
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

    def log_message(self, fmt, *args):
        try:
            ua = (self.headers.get("User-Agent") or "")[:70]
            print("[req %s] %s  UA=%s" % (self.address_string(), fmt % args, ua), flush=True)
        except Exception:
            pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    with ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
        print("Image Fantasy 已启动 ->  http://0.0.0.0:%d" % PORT)
        print("图库目录: %s" % DATA_DIR)
        print("按 Ctrl+C 退出。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")
