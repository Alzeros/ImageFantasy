"""生成链路的健壮性回归：断连不丢图、图片强缓存、并发不撞名、锁不做 I/O。

对应四个曾经会真实造成损失的问题，全部用假上游，不访问外部接口。
"""
import base64
import http.client
import http.server
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import server
from test_server import PNG


class GenerationResilienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images = os.path.join(self.temp.name, "images")
        os.makedirs(self.images)
        self.paths = mock.patch.multiple(
            server,
            DATA_DIR=self.temp.name,
            IMAGES_DIR=self.images,
            GALLERY_FILE=os.path.join(self.temp.name, "gallery.json"),
            PROVIDERS_FILE=os.path.join(self.temp.name, "providers.json"),
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)

        self.upstream_response = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        fixture = self

        class FakeUpstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps(fixture.upstream_response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.upstream = self.start_server(FakeUpstream)
        self.app = self.start_server(server.Handler)
        self.base = "http://127.0.0.1:%s/v1" % self.upstream.server_port
        self.session = server.create_session()
        self.provider_id = self.create_provider()

    def start_server(self, handler):
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()

        def close():
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)
        self.addCleanup(close)
        return httpd

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        headers = {"Content-Type": "application/json", "Cookie": "if_session=" + self.session}
        conn.request(method, path, None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        content = response.read()
        status = response.status
        response_headers = dict(response.getheaders())
        conn.close()
        if path == "/api/generate" and status == 200:
            return status, [json.loads(line) for line in content.splitlines()], response_headers
        try:
            payload = json.loads(content)
        except ValueError:
            payload = content       # 图片与 HTML 不是 JSON，保留原始字节
        return status, payload, response_headers

    def create_provider(self):
        status, provider, _ = self.request("POST", "/api/providers", {
            "name": "fixture", "base": self.base, "key": "fake-test-key",
            "models": ["test-model"], "api_type": "openai",
        })
        self.assertEqual(status, 200)
        return provider["id"]

    def generate(self, params=None):
        return self.request("POST", "/api/generate", {
            "provider_id": self.provider_id, "model": "test-model",
            "prompt": "回归测试", "params": params or {},
        })

    def test_client_disconnect_after_upstream_still_saves_image(self):
        """客户端在 saving 阶段断开：上游已经计费，图片必须照样入库。"""
        real_emit = server.Handler._emit

        def broken_emit(handler, event):
            if isinstance(event, dict) and event.get("stage") == "saving":
                raise BrokenPipeError("client disconnected")
            return real_emit(handler, event)

        with mock.patch.object(server.Handler, "_emit", broken_emit):
            status, events, _ = self.generate()

        self.assertEqual(status, 200)
        self.assertNotIn("done", [e.get("stage") for e in events])
        gallery = server.load_gallery()
        self.assertEqual(len(gallery), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.images, gallery[0]["filename"])))

    def test_images_are_immutable_cached_but_pages_are_not(self):
        _, _, _ = self.generate()
        item = server.load_gallery()[0]
        status, _, headers = self.request("GET", item["url"])
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Cache-Control"), server.IMAGE_CACHE_CONTROL)
        self.assertIn("immutable", headers.get("Cache-Control", ""))

        status, _, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Cache-Control"), "no-cache")

    def test_same_millisecond_generation_does_not_overwrite_images(self):
        encoded = base64.b64encode(PNG).decode()
        with mock.patch.object(server.time, "time", return_value=1700000000.123):
            names = [server.save_image({"b64_json": encoded}, None)[0] for _ in range(5)]
        self.assertEqual(len(set(names)), 5)
        self.assertEqual(len(os.listdir(self.images)), 5)
        for name in names:
            self.assertRegex(name, r"^img_\d+_[0-9a-f]{8}\.png$")

    def test_gallery_stays_responsive_while_image_download_runs(self):
        """慢速图片下载不能卡住图库接口：锁内不允许有网络 I/O。"""
        self.upstream_response = {"data": [{"url": "https://cdn.example.com/a.png"}]}
        started = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        class BlockingOpener:
            def open(self, req, timeout=60):
                started.set()
                release.wait(10)
                response = io.BytesIO(PNG)
                response.headers = {"Content-Type": "image/png"}
                return response

        result = {}
        with mock.patch.object(server.urllib.request, "build_opener",
                               return_value=BlockingOpener()), \
                mock.patch.object(server, "_validate_url", return_value=None):
            worker = threading.Thread(target=lambda: result.update(
                data=self.generate()))
            worker.start()
            self.assertTrue(started.wait(5), "图片下载没有开始")
            began = time.time()
            status, _, _ = self.request("GET", "/api/gallery")
            elapsed = time.time() - began
            release.set()
            worker.join(10)

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 2, "图库接口被图片下载阻塞了")
        self.assertEqual(result["data"][0], 200)
        self.assertEqual(len(server.load_gallery()), 1)


if __name__ == "__main__":
    unittest.main()
