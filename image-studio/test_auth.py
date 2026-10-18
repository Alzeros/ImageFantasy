"""登录系统测试：首次设置、登录、锁定、会话、CSRF 与资源保护（仅标准库）。"""
import http.client
import ipaddress
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import server

PASSWORD = "s3cret-Pass!"


class AuthHTTPTests(unittest.TestCase):
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
            AUTH_FILE=os.path.join(self.temp.name, "auth.json"),
            SECURITY_LOG=os.path.join(self.temp.name, "security.log"),
            SETUP_TOKEN_FILE=os.path.join(self.temp.name, "setup-token"),
            # 不让运行环境里的变量影响测试
            TRUSTED_PROXIES=(),
            PUBLIC_ORIGIN="",
            COOKIE_SECURE=False,
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        # 会话与登录锁定是模块级状态，测试之间必须清空
        server._sessions.clear()
        server._login_fails.clear()
        self.addCleanup(server._sessions.clear)
        self.addCleanup(server._login_fails.clear)
        self.app = self.start_server(server.Handler)

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

    def request(self, method, path, body=None, cookie=None, headers=None, raw=None, conn=None):
        own = conn is None
        conn = conn or http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        hdrs = {"Content-Type": "application/json"}
        if cookie:
            hdrs["Cookie"] = "if_session=" + cookie
        hdrs.update(headers or {})
        data = raw if raw is not None else (None if body is None else json.dumps(body))
        conn.request(method, path, data, hdrs)
        response = conn.getresponse()
        content = response.read()
        set_cookie = response.getheader("Set-Cookie") or ""
        status = response.status
        response_headers = dict(response.getheaders())
        if own:
            conn.close()
        try:
            payload = json.loads(content)
        except ValueError:
            payload = content
        return status, payload, set_cookie, response_headers

    def token_from(self, set_cookie):
        self.assertTrue(set_cookie.startswith("if_session="), set_cookie)
        return set_cookie.split(";", 1)[0].split("=", 1)[1]

    def setup_body(self, username="admin", password=PASSWORD, **extra):
        body = {"username": username, "password": password, "setup_token": server.ensure_setup_token()}
        body.update(extra)
        return body

    def setup_account(self, username="admin", password=PASSWORD):
        status, body, cookie, _ = self.request("POST", "/api/auth/setup", self.setup_body(username, password))
        self.assertEqual(status, 200, body)
        return self.token_from(cookie)

    def login(self, username="admin", password=PASSWORD, headers=None):
        return self.request("POST", "/api/auth/login", {"username": username, "password": password},
                            headers=headers)

    # ---- 首次设置 ----

    def test_status_before_setup_creates_private_token_file(self):
        status, body, _, headers = self.request("GET", "/api/auth/status")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"configured": False, "logged_in": False})
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertTrue(os.path.isfile(server.SETUP_TOKEN_FILE))
        self.assertEqual(os.stat(server.SETUP_TOKEN_FILE).st_mode & 0o777, 0o600)
        # 令牌稳定：多次查询不会换掉已经打印出去的令牌
        first = server.ensure_setup_token()
        self.request("GET", "/api/auth/status")
        self.assertEqual(server.ensure_setup_token(), first)
        self.assertGreaterEqual(len(first), 16)

    def test_setup_requires_valid_token(self):
        for token in ("", "wrong-token-value-123"):
            with self.subTest(token=token):
                status, body, _, _ = self.request("POST", "/api/auth/setup",
                                                  {"username": "admin", "password": PASSWORD, "setup_token": token})
                self.assertEqual(status, 403)
                self.assertIn("设置令牌", body["error"])
                self.assertFalse(server.auth_configured())
        self.setup_account()
        self.assertTrue(server.auth_configured())
        self.assertFalse(os.path.exists(server.SETUP_TOKEN_FILE))

    def test_setup_creates_hashed_account_and_session(self):
        status, body, cookie, _ = self.request("POST", "/api/auth/setup", self.setup_body())
        self.assertEqual(status, 200)
        self.assertEqual(body["username"], "admin")
        token = self.token_from(cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertNotIn("Secure", cookie)
        with open(server.AUTH_FILE, encoding="utf-8") as f:
            record = json.load(f)
        self.assertEqual(record["username"], "admin")
        self.assertNotIn(PASSWORD, json.dumps(record))
        self.assertEqual(record["password"]["algo"], "pbkdf2_sha256")
        self.assertTrue(server.verify_password(PASSWORD, record["password"]))
        self.assertFalse(server.verify_password("wrong", record["password"]))
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 200)

    def test_auth_file_is_owner_only(self):
        self.setup_account()
        self.assertEqual(os.stat(server.AUTH_FILE).st_mode & 0o777, 0o600)

    def test_weak_passwords_rejected(self):
        for password, hint in (("short", "至少"), ("12345678", "纯数字"), ("abcdefgh", "纯字母"),
                               (" lead-space", "空白")):
            with self.subTest(password=password):
                status, body, _, _ = self.request("POST", "/api/auth/setup", self.setup_body(password=password))
                self.assertEqual(status, 400)
                self.assertIn(hint, body["error"])
        self.assertFalse(server.auth_configured())

    def test_invalid_usernames_rejected(self):
        for username, hint in (("  ", "账号"), ("a" * 65, "最多"), ("ad\nmin", "控制字符"), ("\ud800", "无效")):
            with self.subTest(username=username[:10]):
                status, body, _, _ = self.request("POST", "/api/auth/setup", self.setup_body(username=username))
                self.assertEqual(status, 400)
                self.assertIn(hint, body["error"])
        self.assertFalse(server.auth_configured())

    def test_concurrent_setup_only_one_wins(self):
        token = server.ensure_setup_token()
        barrier = threading.Barrier(2)
        results = {}

        def claim(name):
            barrier.wait()
            status, body, _, _ = self.request("POST", "/api/auth/setup",
                                              {"username": name, "password": PASSWORD, "setup_token": token})
            results[name] = status

        threads = [threading.Thread(target=claim, args=(n,)) for n in ("owner", "attacker")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        winners = [n for n, s in results.items() if s == 200]
        self.assertEqual(len(winners), 1, results)
        loser = ({"owner", "attacker"} - set(winners)).pop()
        self.assertIn(results[loser], (401, 403, 409))
        self.assertEqual(server._load_auth()["username"], winners[0])

    def test_second_setup_needs_login_and_current_password(self):
        self.setup_account(password="first-Pass1")
        status, body, _, _ = self.request("POST", "/api/auth/setup",
                                          {"username": "other", "password": "second-Pass2"})
        self.assertEqual(status, 401)
        self.assertIn("未登录", body["error"])
        self.assertEqual(server._load_auth()["username"], "admin")

        status, _, cookie, _ = self.login(password="first-Pass1")
        token = self.token_from(cookie)
        status, body, _, _ = self.request("POST", "/api/auth/setup",
                                          {"username": "admin", "password": "second-Pass2"}, cookie=token)
        self.assertEqual(status, 400)
        self.assertIn("当前密码", body["error"])
        status, body, cookie, _ = self.request("POST", "/api/auth/setup",
                                               {"username": "admin", "password": "second-Pass2",
                                                "current_password": "first-Pass1"}, cookie=token)
        self.assertEqual(status, 200, body)
        new_token = self.token_from(cookie)
        # 改密码后旧会话全部失效
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 401)
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=new_token)
        self.assertEqual(status, 200)

    def test_wrong_current_password_counts_towards_lockout(self):
        token = self.setup_account()
        for i in range(server.LOGIN_MAX_FAILS):
            status, _, _, _ = self.request("POST", "/api/auth/setup",
                                           {"username": "admin", "password": "new-Pass1",
                                            "current_password": "guess-%d" % i}, cookie=token)
            self.assertEqual(status, 400)
        status, _, _, _ = self.request("POST", "/api/auth/setup",
                                       {"username": "admin", "password": "new-Pass1",
                                        "current_password": PASSWORD}, cookie=token)
        self.assertEqual(status, 429)
        self.assertTrue(server.verify_password(PASSWORD, server._load_auth()["password"]))

    def test_corrupt_auth_file_never_allows_reset(self):
        with open(server.AUTH_FILE, "w", encoding="utf-8") as f:
            f.write("{not json")
        status, body, _, _ = self.request("GET", "/api/auth/status")
        self.assertTrue(body["configured"])
        status, body, _, _ = self.request("POST", "/api/auth/setup", self.setup_body())
        self.assertEqual(status, 500)
        self.assertIn("auth.json", body["error"])

    def test_deleting_auth_file_requires_new_token(self):
        old_token = server.ensure_setup_token()
        self.setup_account()
        os.remove(server.AUTH_FILE)
        status, body, _, _ = self.request("GET", "/api/auth/status")
        self.assertFalse(body["configured"])
        new_token = server.ensure_setup_token()
        self.assertNotEqual(new_token, old_token)
        status, _, _, _ = self.request("POST", "/api/auth/setup",
                                       {"username": "x", "password": PASSWORD, "setup_token": old_token})
        self.assertEqual(status, 403)
        self.setup_account()

    def test_unreadable_setup_token_gets_clean_error(self):
        with mock.patch.object(server, "ensure_setup_token", side_effect=PermissionError("/secret/dir")):
            status, body, _, _ = self.request("POST", "/api/auth/setup",
                                              {"username": "admin", "password": PASSWORD, "setup_token": "x" * 24})
        self.assertEqual(status, 500)
        self.assertIn("设置令牌", body["error"])
        self.assertNotIn("/secret", body["error"])

    def test_setup_save_failure_message_is_generic(self):
        with mock.patch.object(server, "_create_private_exclusive",
                               side_effect=PermissionError("[Errno 13] /secret/dir/.tmp_abc")):
            status, body, _, _ = self.request("POST", "/api/auth/setup", self.setup_body())
        self.assertEqual(status, 500)
        self.assertNotIn("/secret", body["error"])

    # ---- 登录 ----

    def test_login_success_and_failure_messages_are_identical(self):
        self.setup_account()
        status, body, _, _ = self.login(password="wrong-Pass1")
        self.assertEqual(status, 401)
        wrong_password = body["error"]
        status, body, _, _ = self.login(username="nobody")
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], wrong_password)

        status, body, cookie, _ = self.login()
        self.assertEqual(status, 200)
        token = self.token_from(cookie)
        status, body, _, _ = self.request("GET", "/api/auth/status", cookie=token)
        self.assertEqual(body, {"configured": True, "logged_in": True})

    def test_login_before_setup(self):
        status, body, _, _ = self.login()
        self.assertEqual(status, 400)
        self.assertTrue(body.get("setup"))

    def test_failures_lock_per_source_and_account(self):
        self.setup_account()
        for i in range(server.LOGIN_MAX_FAILS):
            status, _, _, _ = self.login(password="bad-Pass%d" % i)
            self.assertEqual(status, 401)
        # 锁定后即使密码正确也拒绝，并给出重试时间
        status, body, _, headers = self.login()
        self.assertEqual(status, 429)
        self.assertGreater(body["retry_after"], 0)
        self.assertEqual(headers.get("Retry-After"), str(body["retry_after"]))
        # 同一来源换个账号也不能绕过
        status, _, _, _ = self.login(username="other")
        self.assertEqual(status, 429)

    def test_lock_expires(self):
        self.setup_account()
        for i in range(server.LOGIN_MAX_FAILS):
            self.login(password="bad-Pass%d" % i)
        future = time.time() + server.LOGIN_LOCK_MAX + 1
        with mock.patch.object(server.time, "time", side_effect=lambda: future):
            status, _, cookie, _ = self.login()
        self.assertEqual(status, 200)
        self.assertTrue(cookie)

    def test_successful_login_clears_failures(self):
        self.setup_account()
        for i in range(server.LOGIN_MAX_FAILS - 1):
            self.login(password="bad-Pass%d" % i)
        status, _, _, _ = self.login()
        self.assertEqual(status, 200)
        for i in range(server.LOGIN_MAX_FAILS - 1):
            status, _, _, _ = self.login(password="bad-Pass%d" % i)
            self.assertEqual(status, 401)

    def test_forwarded_for_is_ignored_without_trusted_proxy(self):
        self.setup_account()
        for i in range(server.LOGIN_MAX_FAILS):
            self.login(password="bad-Pass%d" % i, headers={"X-Forwarded-For": "203.0.113.%d" % i})
        # 伪造来源 IP 不能绕开锁定
        status, _, _, _ = self.login(headers={"X-Forwarded-For": "198.51.100.7"})
        self.assertEqual(status, 429)

    def test_forwarded_for_is_used_from_trusted_proxy(self):
        self.setup_account()
        with mock.patch.object(server, "TRUSTED_PROXIES", (ipaddress.ip_network("127.0.0.0/8"),)):
            for i in range(server.LOGIN_MAX_FAILS):
                self.login(password="bad-Pass%d" % i, headers={"X-Forwarded-For": "203.0.113.9"})
            status, _, _, _ = self.login(headers={"X-Forwarded-For": "203.0.113.9"})
            self.assertEqual(status, 429)
            # 代理后面的其他访客不受影响；客户端自己伪造的最左侧地址不被采信
            status, _, _, _ = self.login(headers={"X-Forwarded-For": "203.0.113.9, 198.51.100.7"})
            self.assertEqual(status, 200)
            status, _, cookie, _ = self.login(headers={"X-Forwarded-Proto": "https",
                                                       "X-Forwarded-For": "198.51.100.8"})
            self.assertIn("Secure", cookie)

    def test_forwarded_addresses_with_ports(self):
        self.setup_account()
        with mock.patch.object(server, "TRUSTED_PROXIES", (ipaddress.ip_network("127.0.0.0/8"),)):
            for i in range(server.LOGIN_MAX_FAILS):
                self.login(password="bad-Pass%d" % i, headers={"X-Forwarded-For": "203.0.113.9:%d" % (40000 + i)})
            status, _, _, _ = self.login(headers={"X-Forwarded-For": "203.0.113.9:443"})
            self.assertEqual(status, 429)
            # 其他访客不受影响（不会退回到代理自己的地址）
            status, _, _, _ = self.login(headers={"X-Forwarded-For": "[2001:db8::7]:443"})
            self.assertEqual(status, 200)

    # ---- 会话 ----

    def test_logout_invalidates_session(self):
        token = self.setup_account()
        status, _, cookie, headers = self.request("POST", "/api/auth/logout", cookie=token)
        self.assertEqual(status, 200)
        self.assertIn("Max-Age=0", cookie)
        # 图片是长期强缓存，登出必须让浏览器清掉，否则缓存里的作品仍可被打开
        self.assertEqual(headers.get("Clear-Site-Data"), '"cache"')
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 401)

    def test_idle_and_absolute_expiry(self):
        token = self.setup_account()
        with server._sessions_lock:
            server._sessions[token]["seen"] = time.time() - server.SESSION_IDLE - 1
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 401)

        status, _, cookie, _ = self.login()
        token = self.token_from(cookie)
        with server._sessions_lock:
            server._sessions[token]["created"] = time.time() - server.SESSION_MAX - 1
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 401)

    def test_activity_refreshes_session_without_sending_cookies(self):
        token = self.setup_account()
        with server._sessions_lock:
            server._sessions[token]["seen"] = time.time() - 3601
        status, _, cookie, _ = self.request("GET", "/api/ping", cookie=token)
        self.assertEqual(status, 200)
        self.assertEqual(cookie, "")
        self.assertLess(time.time() - server._sessions[token]["seen"], 5)

    def test_no_session_cookie_leaks_across_keepalive_requests(self):
        # 回归：续期 Cookie 曾挂在连接对象上，经反向代理复用连接时会发给下一个匿名请求
        token = self.setup_account()
        with server._sessions_lock:
            server._sessions[token]["seen"] = time.time() - 3601
        conn = http.client.HTTPConnection("127.0.0.1", self.app.server_port, timeout=5)
        self.addCleanup(conn.close)
        for path in ("/", "/static/favicon.svg", "/images/missing.png"):
            self.request("GET", path, cookie=token, conn=conn)
        status, body, cookie, _ = self.request("GET", "/api/auth/status", conn=conn)
        self.assertEqual(status, 200)
        self.assertEqual(cookie, "")
        self.assertFalse(body["logged_in"])

    def test_unknown_session_rejected(self):
        self.setup_account()
        status, body, _, _ = self.request("GET", "/api/gallery", cookie="not-a-real-session")
        self.assertEqual(status, 401)
        self.assertEqual(body["login"], "/login")

    def test_other_cookies_do_not_break_session(self):
        token = self.setup_account()
        status, _, _, _ = self.request("GET", "/api/gallery", headers={
            "Cookie": 'prefs={"theme":"dark"}; a=b c; if_session=%s; x=a\\b' % token})
        self.assertEqual(status, 200)

    # ---- 资源保护 ----

    def test_all_private_endpoints_require_login(self):
        self.setup_account()
        calls = [
            ("GET", "/api/gallery", None),
            ("GET", "/api/providers", None),
            ("GET", "/api/ping", None),
            ("GET", "/api/auth/me", None),
            ("GET", "/images/whatever.png", None),
            ("GET", "/static/index.html", None),
            ("POST", "/api/providers", {"name": "x", "base": "https://example.com/v1"}),
            ("POST", "/api/fetch-models", {"base": "https://example.com/v1"}),
            ("POST", "/api/inspire", {"model": "m"}),
            ("POST", "/api/generate", {"prompt": "x"}),
            ("PUT", "/api/providers/p_1", {"name": "x"}),
            ("DELETE", "/api/providers/p_1", None),
            ("DELETE", "/api/gallery/img_1", None),
        ]
        for method, path, body in calls:
            with self.subTest(path=path, method=method):
                status, payload, _, _ = self.request(method, path, body)
                self.assertEqual(status, 401, payload)

    def test_public_pages(self):
        self.setup_account()
        for path in ("/", "/index.html", "/login", "/favicon.ico", "/static/favicon.svg",
                     "/static/login.css", "/static/login.js", "/api/auth/status"):
            with self.subTest(path=path):
                status, _, _, _ = self.request("GET", path)
                self.assertEqual(status, 200)

    def test_root_serves_login_page_only_when_signed_out(self):
        token = self.setup_account()
        status, payload, _, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"login.js", payload)
        self.assertNotIn(b"modelPicker", payload)
        status, payload, _, _ = self.request("GET", "/", cookie=token)
        self.assertIn(b"modelPicker", payload)

    # ---- 跨站请求 ----

    def test_csrf_origin_mismatch_blocked_for_writes(self):
        token = self.setup_account()
        status, body, _, _ = self.request("POST", "/api/providers", {"name": "x", "base": "https://e.com/v1"},
                                          cookie=token, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertIn("跨站", body["error"])
        status, body, _, _ = self.request("POST", "/api/providers", {"name": "x", "base": "https://e.com/v1"},
                                          cookie=token,
                                          headers={"Origin": "http://127.0.0.1:%d" % self.app.server_port})
        self.assertEqual(status, 200, body)
        # 只读请求不受 Origin 限制
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token,
                                       headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 200)

    def test_fetch_metadata_and_public_origin(self):
        token = self.setup_account()
        body = {"name": "x", "base": "https://e.com/v1"}
        status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                       headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)
        # 反向代理改写了 Host：浏览器标明同源即放行
        status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                       headers={"Origin": "https://img.example.com", "Sec-Fetch-Site": "same-origin"})
        self.assertEqual(status, 200)
        # 没有 Fetch Metadata 的旧浏览器：按配置的对外地址比对
        status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                       headers={"Origin": "https://img.example.com"})
        self.assertEqual(status, 403)
        with mock.patch.object(server, "PUBLIC_ORIGIN", "https://img.example.com"):
            status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                           headers={"Origin": "https://img.example.com"})
        self.assertEqual(status, 200)
        # 浏览器标记为跨站时，即使 Origin 与本站同域名也拒绝
        same_host = "http://127.0.0.1:%d" % self.app.server_port
        status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                       headers={"Origin": same_host, "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)
        # 已知本站走 HTTPS 时，同域名的 http 页面不能代为提交
        with mock.patch.object(server, "COOKIE_SECURE", True):
            status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                           headers={"Origin": same_host})
            self.assertEqual(status, 403)
            status, _, _, _ = self.request("POST", "/api/providers", body, cookie=token,
                                           headers={"Origin": "https://127.0.0.1:%d" % self.app.server_port})
            self.assertEqual(status, 200)

    def test_auth_endpoints_reject_cross_site(self):
        token = self.setup_account()
        evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
        status, _, cookie, _ = self.request("POST", "/api/auth/logout", cookie=token, headers=evil)
        self.assertEqual(status, 403)
        self.assertEqual(cookie, "")
        status, _, _, _ = self.request("GET", "/api/gallery", cookie=token)
        self.assertEqual(status, 200)
        status, _, _, _ = self.login(headers=evil)
        self.assertEqual(status, 403)
        status, _, _, _ = self.request("POST", "/api/auth/setup",
                                       {"username": "admin", "password": "new-Pass1", "current_password": PASSWORD},
                                       cookie=token, headers=evil)
        self.assertEqual(status, 403)

    # ---- 响应头 ----

    def test_security_headers(self):
        token = self.setup_account()
        status, _, _, headers = self.request("GET", "/api/auth/status")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertIn("frame-ancestors 'none'", headers.get("Content-Security-Policy", ""))
        self.assertEqual(headers.get("Referrer-Policy"), "no-referrer")
        self.assertNotIn("Strict-Transport-Security", headers)
        status, _, _, headers = self.request("GET", "/api/providers", cookie=token)
        self.assertEqual(headers.get("Cache-Control"), "no-store")

    def test_secure_cookie_rules(self):
        # 不受信任的来源声称 HTTPS：不采信
        status, _, cookie, _ = self.request("POST", "/api/auth/setup", self.setup_body(),
                                            headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(status, 200)
        self.assertNotIn("Secure", cookie)
        with mock.patch.object(server, "COOKIE_SECURE", True):
            status, _, cookie, headers = self.login()
        self.assertIn("Secure", cookie)
        self.assertIn("Strict-Transport-Security", headers)

    # ---- 异常输入 ----

    def test_malformed_auth_bodies_get_clean_errors(self):
        self.setup_account()
        bodies = [b"[]", b'"x"', b"123", b"true", b"{not json",
                  b'{"username": 5, "password": "s3cret-Pass!"}',
                  b'{"username": "\\ud800", "password": "s3cret-Pass!"}',
                  b"x" * (server.AUTH_BODY_MAX + 1)]
        for raw in bodies:
            for path in ("/api/auth/login", "/api/auth/setup"):
                with self.subTest(path=path, body=raw[:30]):
                    status, _, _, _ = self.request("POST", path, raw=raw)
                    self.assertIn(status, (400, 401))
        # 服务仍然正常
        status, _, _, _ = self.login()
        self.assertIn(status, (200, 429))

    def test_security_log_is_sanitized_and_private(self):
        # 旧版本可能以 0644 创建过日志文件
        with open(server.SECURITY_LOG, "w", encoding="utf-8") as f:
            f.write("")
        os.chmod(server.SECURITY_LOG, 0o644)
        self.setup_account()
        forged = "x\n[2026-10-18 03:00:00] 登录成功 来源=10.0.0.1 账号=admin"
        self.login(username=forged, password="bad-Pass1")
        with open(server.SECURITY_LOG, encoding="utf-8") as f:
            log = f.read()
        self.assertIn("登录失败", log)
        self.assertNotIn("\n[2026-10-18 03:00:00]", log)
        # 账号字段整体加引号，伪造的内容不会被当成独立字段
        self.assertIn('账号="x?[2026-10-18 03:00:00] 登录成功', log)
        self.assertNotIn("bad-Pass1", log)
        self.assertNotIn(PASSWORD, log)
        self.assertEqual(os.stat(server.SECURITY_LOG).st_mode & 0o777, 0o600)


class LoginLimiterTests(unittest.TestCase):
    def setUp(self):
        server._login_fails.clear()
        self.addCleanup(server._login_fails.clear)

    def test_failure_table_is_bounded_and_keeps_locked_sources(self):
        locked = ["ip:203.0.113.1"]
        for _ in range(server.LOGIN_MAX_FAILS):
            server.record_login_failure(locked)
        self.assertGreater(server.login_locked_for(locked), 0)
        with mock.patch.object(server, "LOGIN_TRACK_MAX", 10):
            for i in range(200):
                server.record_login_failure(server.login_key("spray-%d" % i, "198.51.100.%d" % (i % 250)))
            self.assertLessEqual(len(server._login_fails), 10)
        self.assertGreater(server.login_locked_for(locked), 0)

    def test_ipv6_sources_are_grouped_by_64(self):
        self.assertEqual(server.login_key("a", "2001:db8::1")[1], server.login_key("a", "2001:db8::ffff:1")[1])
        self.assertNotEqual(server.login_key("a", "2001:db8::1")[1], server.login_key("a", "2001:db8:0:1::1")[1])
        self.assertEqual(server.login_key("a", "::ffff:203.0.113.1")[1], "ip:203.0.113.1")

    def test_mapped_proxy_addresses_match(self):
        networks = server._parse_networks("::ffff:127.0.0.1, 172.16.0.0/12, bogus")
        self.assertTrue(server._ip_in("127.0.0.1", networks))
        self.assertTrue(server._ip_in("::ffff:172.17.0.1", networks))
        self.assertFalse(server._ip_in("8.8.8.8", networks))

    def test_long_usernames_are_truncated_in_keys(self):
        key = server.login_key("a" * 100000, "203.0.113.1")[0]
        self.assertLess(len(key), server.USERNAME_MAX_LEN + 40)


class PasswordTests(unittest.TestCase):
    def test_hash_roundtrip_and_salt(self):
        a = server.hash_password(PASSWORD, iterations=10000)
        b = server.hash_password(PASSWORD, iterations=10000)
        self.assertNotEqual(a["salt"], b["salt"])
        self.assertNotEqual(a["hash"], b["hash"])
        self.assertTrue(server.verify_password(PASSWORD, a))
        self.assertFalse(server.verify_password("s3cret-Pass", a))

    def test_malformed_records_never_raise(self):
        for record in (None, {}, {"algo": "md5"}, {"algo": server.PBKDF2_ALGO, "iterations": "x"},
                       {"algo": server.PBKDF2_ALGO, "iterations": 600000, "salt": "", "hash": ""},
                       {"algo": server.PBKDF2_ALGO, "iterations": 10 ** 9, "salt": "AAAA", "hash": "AAAA"}):
            with self.subTest(record=record):
                self.assertFalse(server.verify_password("anything", record))
        self.assertFalse(server.verify_password("\ud800", server.hash_password(PASSWORD, iterations=10000)))

    def test_unicode_password(self):
        record = server.hash_password("密码测试-Åbc123", iterations=10000)
        self.assertTrue(server.verify_password("密码测试-Åbc123", record))


class ExclusiveCreateTests(unittest.TestCase):
    def test_second_create_fails_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "auth.json")
            server._create_private_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                server._create_private_exclusive(path, b"second")
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"first")
            self.assertEqual(os.listdir(d), ["auth.json"])
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_falls_back_when_hard_links_are_unsupported(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(server.os, "link", side_effect=PermissionError("not supported")):
            path = os.path.join(d, "auth.json")
            server._create_private_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                server._create_private_exclusive(path, b"second")
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"first")
            self.assertEqual(os.listdir(d), ["auth.json"])
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
