"""随机灵感与扩写的接口行为，不调用外部文本模型。"""
import io
import json
import unittest
from unittest import mock

import server


class InspirationTests(unittest.TestCase):
    def setUp(self):
        self.handler = server.Handler.__new__(server.Handler)
        self.handler._send = mock.Mock()
        self.providers = mock.patch.object(server, "load_providers", return_value=[{
            "id": "text", "base": "https://example.com/v1", "key": "test-key"
        }])
        self.providers.start()
        self.addCleanup(self.providers.stop)

    def run_action(self, action, prompt, content="新的提示词", finish_reason="stop"):
        body = {"provider_id": "text", "model": "test-model", "prompt": prompt}
        if action is not None:
            body["action"] = action
        self.handler._read_json_body = mock.Mock(return_value=body)
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
        }).encode())
        with mock.patch.object(server.urllib.request, "urlopen", return_value=response) as upstream:
            self.handler._generate_inspiration()
        self.upstream_calls = upstream.call_count
        payload = json.loads(upstream.call_args.args[0].data) if upstream.called else None
        return self.handler._send.call_args.args, payload

    def test_random_ignores_existing_prompt(self):
        (status, result), payload = self.run_action("random", "用户独有的旧构思")
        self.assertEqual(status, 200)
        self.assertEqual(result["prompt"], "新的提示词")
        self.assertNotIn("用户独有的旧构思", json.dumps(payload, ensure_ascii=False))
        self.assertIn("随机", payload["messages"][-1]["content"])

    def test_expand_uses_current_prompt(self):
        (status, _), payload = self.run_action("expand", "雨夜书店与一只橘猫")
        self.assertEqual(status, 200)
        self.assertIn("雨夜书店与一只橘猫", payload["messages"][-1]["content"])
        self.assertIn("保留", payload["messages"][-1]["content"])

    def test_empty_expansion_never_calls_upstream(self):
        (status, result), payload = self.run_action("expand", "  ")
        self.assertEqual(status, 400)
        self.assertIn("先输入", result["error"])
        self.assertIsNone(payload)

    def test_unknown_action_never_calls_upstream(self):
        (status, _), payload = self.run_action("unknown", "构思")
        self.assertEqual(status, 400)
        self.assertIsNone(payload)

    def test_empty_model_output_is_an_error(self):
        (status, result), _ = self.run_action("expand", "构思", content="  ")
        self.assertEqual(status, 502)
        self.assertIn("未返回有效提示词", result["error"])

    def test_older_clients_can_still_expand(self):
        (status, _), payload = self.run_action(None, "旧客户端构思")
        self.assertEqual(status, 200)
        self.assertIn("旧客户端构思", payload["messages"][-1]["content"])

    def test_random_has_separate_instructions_and_smaller_output_budget(self):
        _, random_payload = self.run_action("random", "")
        _, expand_payload = self.run_action("expand", "雨夜书店与橘猫")
        self.assertLess(random_payload["max_tokens"], expand_payload["max_tokens"])
        self.assertLess(len(random_payload["messages"][0]["content"]),
                        len(expand_payload["messages"][0]["content"]))
        self.assertNotEqual(random_payload["messages"][0]["content"],
                            expand_payload["messages"][0]["content"])

    def test_random_keeps_one_complete_idea_without_following_details(self):
        idea = "云海上的小村庄，居民乘着巨大的风筝出门。"
        (status, result), _ = self.run_action("random", "", idea + "镜头缓慢靠近村庄。" * 20)
        self.assertEqual(status, 200)
        self.assertEqual(result["prompt"], idea)
        self.assertEqual(self.upstream_calls, 1)

    def test_long_random_sentence_ends_at_a_clause_boundary(self):
        idea = "月球上的小书店，一位宇航员正在给机器人讲故事"
        (status, result), _ = self.run_action("random", "", idea + "，" + "非常丰富的画面细节" * 30)
        self.assertEqual(status, 200)
        self.assertEqual(result["prompt"], idea + "。")
        self.assertLessEqual(len(result["prompt"]), server.RANDOM_IDEA_MAX_CHARS)
        self.assertEqual(self.upstream_calls, 1)

    def test_unbroken_overlong_output_is_not_returned_as_a_cutoff_idea(self):
        (status, result), _ = self.run_action("random", "", "过长的描述" * 30)
        self.assertEqual(status, 502)
        self.assertIn("灵感过长", result["error"])
        self.assertNotIn("prompt", result)
        self.assertEqual(self.upstream_calls, 1)

    def test_expansion_retains_full_details(self):
        expanded = "雨夜书店里，一只橘猫趴在窗边。" + "街灯映着雨滴，木桌上散落几本书。" * 10
        self.assertGreater(len(expanded), server.RANDOM_IDEA_MAX_CHARS)
        (status, result), _ = self.run_action("expand", "雨夜书店与橘猫", expanded)
        self.assertEqual(status, 200)
        self.assertEqual(result["prompt"], expanded)

    def test_token_limit_does_not_leave_an_incomplete_final_clause(self):
        (status, result), _ = self.run_action(
            "random", "", "机器人在月球上经营一家书店，窗外有一群正在", finish_reason="length"
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["prompt"], "机器人在月球上经营一家书店。")

    def test_incomplete_idea_without_a_boundary_is_rejected(self):
        (status, result), _ = self.run_action("random", "", "一只正要", finish_reason="length")
        self.assertEqual(status, 502)
        self.assertIn("未能提取完整构思", result["error"])


if __name__ == "__main__":
    unittest.main()
