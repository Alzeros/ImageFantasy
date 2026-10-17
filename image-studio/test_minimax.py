"""MiniMax 原生协议、渠道兼容与 HTTP 生成流程测试，不访问外部 API。"""
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


class MiniMaxProtocolTests(unittest.TestCase):
    def build(self, params=None, extra=None, mode="generate", image=None, mask=None,
              model="image-01", base="https://api.minimax.cn/v1"):
        return server.build_image_request(
            "minimax", base, model, mode, "夏日乡野车站", params or {}, extra or {}, image, mask
        )

    def test_endpoint_accepts_host_v1_and_full_endpoint(self):
        for base in ("https://api.minimax.cn", "https://api.minimax.cn/v1/",
                     "https://api.minimax.cn/v1/image_generation"):
            with self.subTest(base=base):
                endpoint, _ = self.build(base=base)
                self.assertEqual(endpoint, "https://api.minimax.cn/v1/image_generation")
        endpoint, _ = self.build(base="https://proxy.example.com/minimax/v1")
        self.assertEqual(endpoint, "https://proxy.example.com/minimax/v1/image_generation")

    def test_maps_shared_controls_and_omits_openai_only_parameters(self):
        _, payload = self.build(params={
            "response_format": "b64_json", "aspect_ratio": "16:9", "size": "2720x1536",
            "watermark": False, "prompt_extend": True, "output_format": "webp",
            "quality": "high", "style": "vivid", "background": "transparent",
            "output_compression": 80, "n": 1,
        })
        self.assertEqual(payload, {
            "model": "image-01", "prompt": "夏日乡野车站", "response_format": "base64",
            "aspect_ratio": "16:9", "aigc_watermark": False, "prompt_optimizer": True, "n": 1,
        })

    def test_default_base64_and_explicit_url(self):
        self.assertEqual(self.build()[1]["response_format"], "base64")
        self.assertEqual(self.build(params={"response_format": "url"})[1]["response_format"], "url")

    def test_custom_dimensions_use_eight_pixel_steps(self):
        _, payload = self.build(params={"size": "520x1032"})
        self.assertEqual((payload["width"], payload["height"]), (520, 1032))
        self.assertNotIn("size", payload)

    def test_native_extras_preserve_seed_zero_and_false_values(self):
        _, payload = self.build(extra={"seed": 0, "prompt_optimizer": False, "n": 3})
        self.assertEqual(payload["seed"], 0)
        self.assertIs(payload["prompt_optimizer"], False)
        self.assertEqual(payload["n"], 3)

    def test_subject_reference_uses_data_url_and_same_endpoint(self):
        image = "data:image/png;base64," + base64.b64encode(PNG).decode()
        endpoint, payload = self.build(mode="edit", image=image)
        self.assertTrue(endpoint.endswith("/v1/image_generation"))
        self.assertEqual(payload["subject_reference"], [{"type": "character", "image_file": image}])
        self.assertNotIn("image", payload)

    def test_unsupported_mask_and_bad_references_are_rejected(self):
        for options in ({"mask": "mask"}, {"image": "data:image/webp;base64,AAAA"},
                        {"image": "data:image/png;base64,???"}, {}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(mode="edit", **options)

    def test_live_style(self):
        _, payload = self.build(model="image-01-live", params={
            "minimax_style": "水彩", "minimax_style_weight": 0.65
        })
        self.assertEqual(payload["style"], {"style_type": "水彩", "style_weight": 0.65})

    def test_invalid_native_parameters_fail_before_sending(self):
        cases = (
            {"params": {"size": "4096x4096"}}, {"params": {"size": "513x1024"}},
            {"params": {"size": "wrong"}}, {"extra": {"width": 1024}},
            {"extra": {"n": 10}}, {"extra": {"n": 1.5}}, {"extra": {"n": True}},
            {"extra": {"aspect_ratio": "5:4"}}, {"extra": {"prompt": "a" * 1501}},
            {"extra": {"prompt": " "}}, {"extra": {"response_format": "b64_json"}},
            {"model": "image-01-live", "params": {"size": "1024x1024"}},
            {"model": "image-01-live", "params": {"aspect_ratio": "21:9"}},
            {"params": {"minimax_style": "水彩"}},
            {"model": "image-01-live", "params": {"minimax_style": "水彩", "minimax_style_weight": 0}},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(**options)

    def test_normalizes_base64_url_and_data_url_responses(self):
        encoded = base64.b64encode(PNG).decode()
        for key, value, expected in (
            ("image_base64", encoded, {"b64_json": encoded}),
            ("image_base64", "data:image/png;base64," + encoded, {"b64_json": encoded}),
            ("image_urls", "https://cdn.example.com/image.png", {"url": "https://cdn.example.com/image.png"}),
        ):
            with self.subTest(key=key):
                result = server.normalize_image_response({
                    "data": {key: [value]}, "base_resp": {"status_code": 0}
                }, "minimax")
                self.assertEqual(result["data"], [expected])

    def test_does_not_duplicate_two_response_formats(self):
        response = {"data": {"image_base64": ["AAAA"], "image_urls": ["https://example.com/img"]}}
        self.assertEqual(len(server.normalize_image_response(response, "minimax")["data"]), 1)

    def test_business_errors_even_when_http_status_is_200(self):
        for code in (1002, 1004, 1008, 1026, 2013, 2049):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, str(code)):
                server.normalize_image_response({
                    "base_resp": {"status_code": code, "status_msg": "upstream error"}
                }, "minimax")

    def test_content_filter_and_missing_images_are_errors(self):
        with self.assertRaisesRegex(ValueError, "内容安全检查"):
            server.normalize_image_response({"data": {}, "metadata": {"failed_count": "1"}}, "minimax")
        for response in ({"data": {}}, {"data": {"image_base64": "wrong"}}, {"data": ["wrong"]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                server.normalize_image_response(response, "minimax")

    def test_existing_openai_payload_and_edits_remain_compatible(self):
        endpoint, payload = server.build_image_request("openai", "https://proxy.example.com/v1",
            "gpt-image-1", "edit", "test", {
                "size": "2048x2048", "response_format": "b64_json", "output_format": "webp",
                "watermark": False, "prompt_extend": False, "quality": "high", "style": "vivid",
                "background": "transparent", "output_compression": 70,
            }, {"seed": 0}, "reference", "mask")
        self.assertEqual(endpoint, "https://proxy.example.com/v1/images/edits")
        self.assertEqual(payload["image"], "reference")
        self.assertEqual(payload["mask"], "mask")
        self.assertEqual(payload["response_format"], "b64_json")
        self.assertEqual(payload["output_format"], "webp")
        self.assertEqual(payload["output_compression"], 70)
        self.assertIs(payload["prompt_extend"], False)
        self.assertEqual(payload["seed"], 0)
        response = {"data": [{"b64_json": "AAAA"}]}
        self.assertIs(server.normalize_image_response(response, "openai"), response)


class GenerationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        images = os.path.join(self.temp.name, "images")
        os.makedirs(images)
        self.paths = mock.patch.multiple(server, DATA_DIR=self.temp.name, IMAGES_DIR=images,
            GALLERY_FILE=os.path.join(self.temp.name, "gallery.json"),
            PROVIDERS_FILE=os.path.join(self.temp.name, "providers.json"))
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.requests = []
        self.upstream_status = 200
        self.upstream_response = {
            "id": "fake-task", "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_base64": [base64.b64encode(PNG).decode()]}
        }
        fixture = self

        class FakeUpstream(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                fixture.requests.append((self.path, dict(self.headers),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps(fixture.upstream_response).encode()
                self.send_response(fixture.upstream_status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.upstream = self.start_server(FakeUpstream)
        self.app = self.start_server(server.Handler)
        self.base = "http://127.0.0.1:%s/v1" % self.upstream.server_port

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
        conn.request(method, path, None if body is None else json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        content = response.read()
        status = response.status
        conn.close()
        if path == "/api/generate" and status == 200:
            return status, [json.loads(line) for line in content.splitlines()]
        return status, json.loads(content)

    def provider(self, api_type="minimax"):
        body = {"name": "fixture", "base": self.base, "key": "fake-test-key", "models": ["image-01"]}
        if api_type is not None:
            body["api_type"] = api_type
        status, provider = self.request("POST", "/api/providers", body)
        self.assertEqual(status, 200)
        return provider

    def test_native_generation_streams_saves_and_serves_image(self):
        provider = self.provider()
        status, events = self.request("POST", "/api/generate", {
            "provider_id": provider["id"], "prompt": "夏日车站",
            "params": {"aspect_ratio": "16:9", "response_format": "b64_json", "output_format": "webp"}
        })
        self.assertEqual(status, 200)
        self.assertEqual([event["stage"] for event in events], ["accepted", "upstream", "saving", "done"])
        path, headers, payload = self.requests[0]
        self.assertEqual(path, "/v1/image_generation")
        self.assertEqual(headers["Authorization"], "Bearer fake-test-key")
        self.assertEqual(payload["response_format"], "base64")
        item = events[-1]["items"][0]
        self.assertEqual(item["size"], "16:9")
        self.assertEqual(item["output_format"], "png")
        self.assertEqual(server.load_gallery(), [item])
        conn = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        conn.request("GET", item["url"])
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), PNG)
        conn.close()

    def test_native_business_error_is_not_saved(self):
        provider = self.provider()
        self.upstream_response = {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}}
        _, events = self.request("POST", "/api/generate", {"provider_id": provider["id"], "prompt": "test"})
        self.assertEqual(events[-1]["stage"], "error")
        self.assertIn("1008", events[-1]["message"])
        self.assertNotIn("saving", [event["stage"] for event in events])
        self.assertEqual(server.load_gallery(), [])

    def test_native_http_error_body_is_readable(self):
        provider = self.provider()
        self.upstream_status = 401
        self.upstream_response = {"base_resp": {"status_code": 2049, "status_msg": "invalid api key"}}
        _, events = self.request("POST", "/api/generate", {"provider_id": provider["id"], "prompt": "test"})
        self.assertEqual(events[-1]["http"], 401)
        self.assertIn("2049", events[-1]["message"])

    def test_invalid_dimensions_do_not_call_upstream(self):
        provider = self.provider()
        status, error = self.request("POST", "/api/generate", {
            "provider_id": provider["id"], "prompt": "test", "params": {"size": "4096x4096"}
        })
        self.assertEqual(status, 400)
        self.assertIn("2048", error["error"])
        self.assertEqual(self.requests, [])

    def test_provider_type_persists_and_old_update_keeps_it(self):
        provider = self.provider()
        body = {"name": "updated", "base": self.base, "key": "fake-test-key", "models": ["image-01-live"]}
        status, saved = self.request("PUT", "/api/providers/" + provider["id"], body)
        self.assertEqual(status, 200)
        self.assertEqual(saved["api_type"], "minimax")
        self.assertEqual(server.load_providers()[0]["api_type"], "minimax")

    def test_default_base_and_documented_preset_models(self):
        status, saved = self.request("POST", "/api/providers", {
            "name": "MiniMax", "api_type": "minimax", "key": "fake", "models": ["image-01"]
        })
        self.assertEqual(status, 200)
        self.assertEqual(saved["base"], "https://api.minimax.cn/v1")
        status, result = self.request("POST", "/api/fetch-models", {"api_type": "minimax"})
        self.assertEqual(status, 200)
        self.assertEqual(result["models"], ["image-01", "image-01-live"])
        self.assertEqual(result["source"], "preset")
        self.assertEqual(self.requests, [])

    def test_legacy_provider_without_type_still_uses_openai(self):
        provider = {"id": "legacy", "name": "legacy", "base": self.base,
                    "key": "fake", "models": ["gpt-image-1"]}
        server.save_providers([provider])
        self.upstream_response = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        _, events = self.request("POST", "/api/generate", {"provider_id": "legacy", "prompt": "test"})
        self.assertEqual(self.requests[0][0], "/v1/images/generations")
        self.assertEqual(events[-1]["stage"], "done")

    def test_unknown_protocol_is_rejected(self):
        status, _ = self.request("POST", "/api/providers", {
            "api_type": "unknown", "name": "bad", "base": self.base
        })
        self.assertEqual(status, 400)
        self.assertEqual(server.load_providers(), [])


if __name__ == "__main__":
    unittest.main()
