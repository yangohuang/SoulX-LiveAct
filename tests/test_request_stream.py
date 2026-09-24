import io
import unittest

import torch

from request_stream import iter_requests, request_key, reset_kv_caches


class RequestStreamTests(unittest.TestCase):
    def test_warm_request_reuses_preencoded_prompt_with_new_audio(self):
        first = {"prompt": "speaking", "cond_image": "face.png", "cond_audio": "one.wav"}
        second = {"prompt": "speaking", "cond_image": "face.png", "cond_audio": "two.wav"}
        context = object()
        catalog = {request_key(first): context}
        stream = io.StringIO('{"prompt":"speaking","cond_image":"face.png","cond_audio":"two.wav"}\n')
        requests = list(iter_requests([first], catalog, serve_stdin=True, stream=stream))
        self.assertEqual(requests, [(second, context)])

    def test_unknown_prompt_is_rejected(self):
        first = {"prompt": "speaking", "cond_image": "face.png", "cond_audio": "one.wav"}
        stream = io.StringIO('{"prompt":"dancing","cond_image":"face.png","cond_audio":"two.wav"}\n')
        with self.assertRaisesRegex(ValueError, "not pre-encoded"):
            list(iter_requests([first], {request_key(first): object()}, serve_stdin=True, stream=stream))

    def test_bad_warm_request_reports_error_and_keeps_service_alive(self):
        first = {"prompt": "speaking", "cond_image": "face.png", "cond_audio": "one.wav"}
        stream = io.StringIO('not-json\n{"prompt":"unknown"}\n'
                             '{"prompt":"speaking","cond_image":"face.png","cond_audio":"one.wav"}\n')
        errors = []
        requests = list(iter_requests([first], {request_key(first): "cached"},
                                      serve_stdin=True, stream=stream, on_error=errors.append))
        self.assertEqual(requests, [(first, "cached")])
        self.assertEqual(len(errors), 2)

    def test_batch_mode_preserves_input_order(self):
        items = [{"prompt": "a"}, {"prompt": "b"}]
        ctx = {request_key(item): item["prompt"] for item in items}
        self.assertEqual(list(iter_requests(items, ctx, serve_stdin=False, stream=io.StringIO())),
                         [(items[0], "a"), (items[1], "b")])

    def test_reused_kv_cache_is_cleared_between_requests(self):
        cache = {0: {0: {"k": torch.ones(2), "v": torch.ones(2),
                         "k_scale": torch.full((2,), 2.0), "v_scale": torch.full((2,), 2.0)}}}
        reset_kv_caches(cache)
        self.assertEqual(cache[0][0]["k"].tolist(), [0.0, 0.0])
        self.assertEqual(cache[0][0]["v_scale"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
