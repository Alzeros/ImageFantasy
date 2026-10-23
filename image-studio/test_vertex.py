"""Vertex AI Gemini 出图：请求构造、响应归一与 HTTP 生成流程测试。"""
import base64
import http.client
import http.server
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import server
from test_server import PNG


FAKE_SA = {
    "type": "service_account",
    "project_id": "demo-project",
    "private_key_id": "x",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----\n",
    "client_email": "demo@demo-project.iam.gserviceaccount.com",
    "token_uri": "https://oauth2.googleapis.com/token",
}


class VertexProtocolTests(unittest.TestCase):
    def build(self, params=None, extra=None, mode="generate", image=None, mask=None,
              model="gemini-2.5-flash-image", location="global", project_id="demo-project"):
        return server.build_image_request(
            "vertex", location, model, mode, "夏日乡野车站",
            params or {}, extra or {}, image, mask, project_id=project_id,
        )

    def test_global_and_regional_endpoints(self):
        endpoint, _ = self.build(location="global")
        self.assertEqual(
            endpoint,
            "https://aiplatform.googleapis.com/v1/projects/demo-project/locations/global/"
            "publishers/google/models/gemini-2.5-flash-image:generateContent",
        )
        endpoint, _ = self.build(location="us-central1")
        self.assertTrue(endpoint.startswith("https://us-central1-aiplatform.googleapis.com/"))
        self.assertIn("/locations/us-central1/", endpoint)

    def test_payload_uses_generate_content_and_aspect_ratio(self):
        _, payload = self.build(params={"aspect_ratio": "16:9", "n": 1})
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "夏日乡野车站")
        self.assertEqual(payload["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])
        self.assertEqual(payload["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")

    def test_auto_ratio_omits_image_config(self):
        _, payload = self.build(params={})
        self.assertNotIn("imageConfig", payload["generationConfig"])

    def test_rejects_edit_and_bad_ratio_and_multi(self):
        with self.assertRaisesRegex(ValueError, "文生图"):
            self.build(mode="edit", image="data:image/png;base64,AA")
        with self.assertRaisesRegex(ValueError, "画幅"):
            self.build(params={"aspect_ratio": "7:5"})
        with self.assertRaisesRegex(ValueError, "单张"):
            self.build(params={"n": 2})

    def test_normalizes_inline_data(self):
        encoded = base64.b64encode(PNG).decode()
        result = server.normalize_image_response({
            "candidates": [{
                "content": {"parts": [
                    {"text": "done"},
                    {"inlineData": {"mimeType": "image/png", "data": encoded}},
                ]}
            }]
        }, "vertex")
        self.assertEqual(result["data"], [{"b64_json": encoded}])

    def test_normalizes_snake_case_inline_data_and_data_url(self):
        encoded = base64.b64encode(PNG).decode()
        result = server.normalize_image_response({
            "candidates": [{
                "content": {"parts": [
                    {"inline_data": {"mime_type": "image/png", "data": "data:image/png;base64," + encoded}},
                ]}
            }]
        }, "vertex")
        self.assertEqual(result["data"], [{"b64_json": encoded}])

    def test_safety_and_missing_images_are_errors(self):
        with self.assertRaisesRegex(ValueError, "内容安全"):
            server.normalize_image_response({
                "candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "no"}]}}]
            }, "vertex")
        with self.assertRaisesRegex(ValueError, "拒绝"):
            server.normalize_image_response({"promptFeedback": {"blockReason": "OTHER"}}, "vertex")
        with self.assertRaises(ValueError):
            server.normalize_image_response({"candidates": []}, "vertex")

    def test_wrong_credential_file_is_explained_in_chinese(self):
        with self.assertRaisesRegex(ValueError, "gcloud 用户登录凭据"):
            server._require_service_account({
                "type": "authorized_user", "client_id": "x", "refresh_token": "y",
            })
        with self.assertRaisesRegex(ValueError, "client_email、token_uri"):
            server._require_service_account({"type": "service_account", "private_key": "k"})
        with self.assertRaisesRegex(ValueError, "CLIProxyAPI"):
            server._require_service_account({
                "type": "vertex",
                "provider": "vertex",
                "auth_index": "abc",
                "path": "/root/.cli-proxy-api/vertex-demo.json",
                "project_id": "demo-project",
                "email": "demo@demo-project.iam.gserviceaccount.com",
            })

    def test_cliproxyapi_wrapped_credential_is_unwrapped(self):
        wrapped = {
            "type": "vertex",
            "project_id": "wrapped-project",
            "location": "us-central1",
            "service_account": dict(FAKE_SA, project_id=""),
        }
        info = server._vertex_service_account_info(json.dumps(wrapped))
        self.assertEqual(info["type"], "service_account")
        self.assertEqual(info["project_id"], "wrapped-project")
        self.assertEqual(info["location"], "us-central1")
        self.assertEqual(info["client_email"], FAKE_SA["client_email"])

    def test_imagen_predict_request_and_response(self):
        endpoint, payload = self.build(
            model="imagen-4.0-generate-001",
            location="us-central1",
            params={"aspect_ratio": "16:9", "n": 1},
        )
        self.assertTrue(endpoint.endswith("/models/imagen-4.0-generate-001:predict"))
        self.assertEqual(payload["instances"], [{"prompt": "夏日乡野车站"}])
        self.assertEqual(payload["parameters"]["aspectRatio"], "16:9")
        encoded = base64.b64encode(PNG).decode()
        result = server.normalize_image_response({
            "predictions": [{"bytesBase64Encoded": encoded, "mimeType": "image/png"}]
        }, "vertex")
        self.assertEqual(result["data"], [{"b64_json": encoded}])

    def test_service_account_info_from_json_and_file(self):
        info = server._vertex_service_account_info(json.dumps(FAKE_SA))
        self.assertEqual(info["project_id"], "demo-project")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            json.dump(FAKE_SA, handle)
            path = handle.name
        self.addCleanup(lambda: os.unlink(path))
        self.assertEqual(server._vertex_service_account_info(path)["client_email"], FAKE_SA["client_email"])
        self.assertIsNone(server._vertex_service_account_info(""))
        with self.assertRaises(ValueError):
            server._vertex_service_account_info("{not-json")


class VertexHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        images = os.path.join(self.temp.name, "images")
        os.makedirs(images)
        self.paths = mock.patch.multiple(
            server,
            DATA_DIR=self.temp.name,
            IMAGES_DIR=images,
            GALLERY_FILE=os.path.join(self.temp.name, "gallery.json"),
            PROVIDERS_FILE=os.path.join(self.temp.name, "providers.json"),
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.requests = []
        self.upstream_status = 200
        self.upstream_response = {
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}
                ]}
            }]
        }
        fixture = self

        class FakeUpstream(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                fixture.requests.append((self.path, dict(self.headers), None))
                payload = json.dumps({
                    "name": "publishers/google/models/gemini-2.5-flash-image",
                    "displayName": "Gemini 2.5 Flash Image",
                }).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fixture.requests.append((self.path, dict(self.headers), json.loads(body)))
                payload = json.dumps(fixture.upstream_response).encode()
                self.send_response(fixture.upstream_status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.upstream = self.start_server(FakeUpstream)
        server.VERTEX_API_ROOT = "http://127.0.0.1:%s" % self.upstream.server_port
        self.addCleanup(lambda: setattr(server, "VERTEX_API_ROOT", None))
        self.token_patch = mock.patch.object(
            server, "vertex_access_token", return_value=("vertex-test-token", "demo-project")
        )
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)
        self.app = self.start_server(server.Handler)
        self.session = server.create_session()

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
        conn.close()
        if path == "/api/generate" and status == 200:
            return status, [json.loads(line) for line in content.splitlines()]
        return status, json.loads(content)

    def provider(self, **overrides):
        body = {
            "name": "vertex-fixture",
            "api_type": "vertex",
            "base": "global",
            "project_id": "demo-project",
            "key": json.dumps(FAKE_SA),
            "models": ["gemini-2.5-flash-image"],
        }
        body.update(overrides)
        status, provider = self.request("POST", "/api/providers", body)
        self.assertEqual(status, 200)
        return provider

    def test_rejects_key_that_is_not_service_account_json(self):
        status, error = self.request("POST", "/api/providers", {
            "name": "vertex", "api_type": "vertex", "base": "global", "key": "not-a-json-file",
        })
        self.assertEqual(status, 400)
        self.assertIn("JSON", error["error"])

    def test_create_provider_infers_project_from_json(self):
        status, provider = self.request("POST", "/api/providers", {
            "name": "vertex",
            "api_type": "vertex",
            "base": "",
            "key": json.dumps(FAKE_SA),
            "models": ["gemini-2.5-flash-image"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(provider["base"], "global")
        self.assertEqual(provider["project_id"], "demo-project")

    def test_fetch_models_returns_preset(self):
        status, data = self.request("POST", "/api/fetch-models", {"api_type": "vertex", "base": "global"})
        self.assertEqual(status, 200)
        self.assertEqual(data["source"], "preset")
        self.assertIn("gemini-2.5-flash-image", data["models"])

    def test_check_reads_model_without_generating(self):
        status, result = self.request("POST", "/api/vertex-check", {
            "key": json.dumps(FAKE_SA),
            "base": "global",
            "model": "gemini-2.5-flash-image",
        })
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(result["project_id"], "demo-project")
        self.assertEqual(result["location"], "global")
        path, headers, body = self.requests[0]
        self.assertIsNotNone(body)
        self.assertNotIn(":generateContent", path)
        self.assertTrue(path.endswith("/models/gemini-2.5-flash-image:countTokens"))
        self.assertEqual(headers["Authorization"], "Bearer vertex-test-token")

    def test_browser_never_receives_private_key(self):
        provider = self.provider()
        self.assertNotIn("key", provider)
        self.assertTrue(provider["has_key"])
        self.assertEqual(provider["key_email"], FAKE_SA["client_email"])
        status, items = self.request("GET", "/api/providers")
        self.assertEqual(status, 200)
        self.assertNotIn("PRIVATE KEY", json.dumps(items))
        self.assertNotIn("key", items[0])

    def test_update_without_key_keeps_stored_key(self):
        provider = self.provider()
        status, updated = self.request("PUT", "/api/providers/" + provider["id"], {
            "name": "renamed", "api_type": "vertex", "base": "global",
            "project_id": "demo-project", "key": "", "models": ["gemini-2.5-flash-image"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(updated["name"], "renamed")
        self.assertTrue(updated["has_key"])
        stored = next(it for it in server.load_providers() if it["id"] == provider["id"])
        self.assertEqual(json.loads(stored["key"])["client_email"], FAKE_SA["client_email"])

    def test_check_with_provider_id_uses_stored_key(self):
        provider = self.provider()
        status, result = self.request("POST", "/api/vertex-check", {
            "provider_id": provider["id"], "base": "global", "model": "gemini-2.5-flash-image",
        })
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])

    def test_generation_streams_and_saves(self):
        provider = self.provider()
        status, events = self.request("POST", "/api/generate", {
            "provider_id": provider["id"],
            "prompt": "夏日车站",
            "params": {"aspect_ratio": "16:9"},
        })
        self.assertEqual(status, 200)
        self.assertEqual([event["stage"] for event in events], ["accepted", "upstream", "saving", "done"])
        path, headers, payload = self.requests[0]
        self.assertIn(":generateContent", path)
        self.assertIn("/locations/global/", path)
        self.assertEqual(headers["Authorization"], "Bearer vertex-test-token")
        self.assertEqual(payload["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")
        item = events[-1]["items"][0]
        self.assertEqual(item["api_type"], "vertex")
        self.assertEqual(item["size"], "16:9")
        self.assertEqual(server.load_gallery(), [item])

    def test_edit_mode_rejected_before_upstream(self):
        provider = self.provider()
        status, error = self.request("POST", "/api/generate", {
            "provider_id": provider["id"],
            "prompt": "test",
            "mode": "edit",
            "image": "data:image/png;base64," + base64.b64encode(PNG).decode(),
        })
        self.assertEqual(status, 400)
        self.assertIn("文生图", error["error"])
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
