"""图片 URL 下载回归测试（仅用标准库，不调用付费生图接口）。"""
import base64
import io
import os
import socket
import tempfile
import unittest
import urllib.request
from unittest import mock

import server


IMAGE_HOST = "dashscope-a717.oss-accelerate.aliyuncs.com"
IMAGE_URL = "https://" + IMAGE_HOST + "/image.png?Signature=keep%2Bthis%3D"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aE1sAAAAASUVORK5CYII="
)


def dns_answers(*addresses):
    return [
        (socket.AF_INET6 if ":" in addr else socket.AF_INET,
         socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, 443))
        for addr in addresses
    ]


class ImageURLTests(unittest.TestCase):
    def test_aliyun_proxy_dns_ipv4_and_ipv6(self):
        for addresses in (("198.18.0.112",), ("2001:2::70",),
                          ("198.18.0.112", "2001:2::70")):
            with self.subTest(addresses=addresses), mock.patch.object(
                server.socket, "getaddrinfo", return_value=dns_answers(*addresses)
            ):
                server._validate_url(IMAGE_URL)

    def test_public_ipv4_and_ipv6_for_other_providers(self):
        for address in ("8.8.8.8", "2606:4700:4700::1111"):
            with self.subTest(address=address), mock.patch.object(
                server.socket, "getaddrinfo", return_value=dns_answers(address)
            ):
                server._validate_url("https://images.example.com/image.png")

    def test_private_and_special_literals_never_use_dns(self):
        hosts = ("127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
                 "169.254.169.254", "0.0.0.0", "198.18.0.112", "224.0.0.1",
                 "[::1]", "[::]", "[fc00::1]", "[fe80::1]", "[ff02::1]",
                 "[2001:2::70]", "[::ffff:127.0.0.1]")
        with mock.patch.object(server.socket, "getaddrinfo") as resolve:
            for host in hosts:
                with self.subTest(host=host), self.assertRaises(ValueError):
                    server._validate_url("https://" + host + "/image.png")
            resolve.assert_not_called()

    def test_trusted_domain_does_not_allow_real_private_dns(self):
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1",
                        "fc00::1", "::ffff:192.168.1.1"):
            for answers in ((address,), ("198.18.0.112", address)):
                with self.subTest(answers=answers), mock.patch.object(
                    server.socket, "getaddrinfo", return_value=dns_answers(*answers)
                ), self.assertRaises(ValueError):
                    server._validate_url(IMAGE_URL)

    def test_fake_ip_exception_requires_https_and_domain_boundary(self):
        urls = ("http://" + IMAGE_HOST + "/image.png",
                "https://images.example.com/image.png",
                "https://notaliyuncs.com/image.png",
                "https://aliyuncs.com.attacker.example/image.png",
                "https://aliyuncs.com@attacker.example/image.png")
        with mock.patch.object(server.socket, "getaddrinfo",
                               return_value=dns_answers("198.18.0.112")):
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    server._validate_url(url)
            server._validate_url("https://" + IMAGE_HOST.upper() + "./image.png")

    def test_dns_failure_is_distinct_and_hides_signed_query(self):
        with mock.patch.object(server.socket, "getaddrinfo",
                               side_effect=socket.gaierror("DNS unavailable")):
            with self.assertRaisesRegex(ValueError, "域名解析失败") as caught:
                server._validate_url(IMAGE_URL)
        self.assertNotIn("Signature", str(caught.exception))

    def test_empty_dns_answer_is_blocked(self):
        with mock.patch.object(server.socket, "getaddrinfo", return_value=[]):
            with self.assertRaises(ValueError):
                server._validate_url(IMAGE_URL)

    def test_malformed_urls_and_unsafe_schemes(self):
        for url in ("file:///etc/passwd", "ftp://example.com/image.png", "https:///a",
                    "https://[invalid]/a", "https://example.com:invalid/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                server._validate_url(url)

    def test_blocked_error_hides_signed_query(self):
        with mock.patch.object(server.socket, "getaddrinfo",
                               return_value=dns_answers("10.0.0.1")):
            with self.assertRaises(ValueError) as caught:
                server._validate_url(IMAGE_URL)
        self.assertNotIn("Signature", str(caught.exception))

    def test_redirect_to_internal_address_is_blocked(self):
        redirect = server._ImageRedirectHandler()
        request = urllib.request.Request(IMAGE_URL)
        for status in (301, 302, 303, 307, 308):
            for target in ("http://127.0.0.1/admin", "http://[::1]/admin"):
                with self.subTest(status=status, target=target), self.assertRaises(ValueError):
                    redirect.redirect_request(request, None, status, "redirect", {}, target)

    def test_redirect_to_trusted_storage_keeps_signature(self):
        with mock.patch.object(server.socket, "getaddrinfo",
                               return_value=dns_answers("198.18.0.112")):
            result = server._ImageRedirectHandler().redirect_request(
                urllib.request.Request("https://images.example.com/start"),
                None, 302, "redirect", {}, IMAGE_URL
            )
        self.assertEqual(result.full_url, IMAGE_URL)

    def test_proxy_image_is_saved_with_original_signature(self):
        response = io.BytesIO(PNG)
        response.headers = {"Content-Type": "image/png"}
        opener = mock.Mock()
        opener.open.return_value = response
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            server, "IMAGES_DIR", directory
        ), mock.patch.object(server.socket, "getaddrinfo",
                             return_value=dns_answers("198.18.0.112", "2001:2::70")), \
                mock.patch.object(server.urllib.request, "build_opener", return_value=opener) as build:
            filename, local_url = server.save_image({"url": IMAGE_URL}, None)
            with open(os.path.join(directory, filename), "rb") as saved:
                self.assertEqual(saved.read(), PNG)
            self.assertEqual(local_url, "/images/" + filename)
            request = opener.open.call_args.args[0]
            self.assertEqual(request.full_url, IMAGE_URL)
            self.assertIsInstance(build.call_args.args[0], server._ImageRedirectHandler)

    def test_base64_images_still_save_without_network(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            server, "IMAGES_DIR", directory
        ), mock.patch.object(server.urllib.request, "build_opener") as build:
            filename, _ = server.save_image({"b64_json": base64.b64encode(PNG).decode()}, None)
            with open(os.path.join(directory, filename), "rb") as saved:
                self.assertEqual(saved.read(), PNG)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
