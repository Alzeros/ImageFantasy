import json, os, tempfile, unittest
import server


class SecretEncryptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = (server.DATA_DIR, server.PROVIDERS_FILE, server.SECRET_FILE, server._fernet_cache[0])
        server.DATA_DIR = self.tmp
        server.PROVIDERS_FILE = os.path.join(self.tmp, "providers.json")
        server.SECRET_FILE = os.path.join(self.tmp, "secret.key")
        server._fernet_cache[0] = None

    def tearDown(self):
        server.DATA_DIR, server.PROVIDERS_FILE, server.SECRET_FILE, server._fernet_cache[0] = self._old

    def test_roundtrip_and_not_plaintext_on_disk(self):
        server.save_providers([{"id": "a", "key": "sk-secret-123"}])
        raw = open(server.PROVIDERS_FILE, encoding="utf-8").read()
        self.assertNotIn("sk-secret-123", raw)
        self.assertEqual(server.load_providers()[0]["key"], "sk-secret-123")

    def test_migrates_plaintext(self):
        with open(server.PROVIDERS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"id": "a", "key": "plain-key"}], f)
        server.migrate_providers_encryption()
        self.assertNotIn("plain-key", open(server.PROVIDERS_FILE, encoding="utf-8").read())
        self.assertEqual(server.load_providers()[0]["key"], "plain-key")

    def test_wrong_secret_fails_clearly(self):
        server.save_providers([{"id": "a", "key": "k"}])
        os.remove(server.SECRET_FILE)
        server._fernet_cache[0] = None
        with self.assertRaises(OSError):
            server.load_providers()


if __name__ == "__main__":
    unittest.main()
