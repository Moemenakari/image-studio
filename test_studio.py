"""Tests with a fake network session, so no internet or API key is needed.

Run with:  python -m unittest test_studio.py
"""

import io
import json
import tempfile
import unittest
from pathlib import Path

import requests
from PIL import Image

from studio.pipeline import (ContentBlockedError, CorruptImageError, PollinationsProvider,
                             StudioError, build_request, generate, verify_image)


def png_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 40)).save(buf, "PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status=200, body=b"", text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text
        self.headers = headers or {"content-type": "image/png"}

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeSession:
    """Returns the queued responses (or raises queued exceptions) in order."""

    def __init__(self, *queue):
        self.queue = list(queue)
        self.calls = []

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append({"url": url, "params": params, "timeout": timeout, "stream": stream})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestBuildRequest(unittest.TestCase):
    def test_ratio_maps_to_exact_pixels(self):
        req = build_request("a cat", "16:9")
        self.assertEqual((req.width, req.height), (1344, 768))

    def test_style_is_appended(self):
        req = build_request("a city", style="cyberpunk")
        self.assertTrue(req.full_prompt.startswith("a city, cyberpunk"))

    def test_invalid_inputs(self):
        for kwargs in ({"prompt": "  "}, {"prompt": "x", "aspect_ratio": "2:1"},
                       {"prompt": "x", "style": "unknown"}, {"prompt": "x", "count": 9},
                       {"prompt": "x" * 1001}):
            with self.assertRaises(StudioError):
                build_request(**kwargs)


class TestVerify(unittest.TestCase):
    def test_full_image_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.png"
            p.write_bytes(png_bytes())
            self.assertEqual(verify_image(p), ("PNG", (64, 64)))

    def test_truncated_image_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cut.png"
            data = png_bytes((256, 256))
            p.write_bytes(data[: len(data) // 2])  # simulate a dropped connection
            with self.assertRaises(CorruptImageError):
                verify_image(p)


class TestGenerate(unittest.TestCase):
    def run_generate(self, session, count=1):
        req = build_request("a lighthouse", "1:1", count=count, seed=10)
        provider = PollinationsProvider()
        provider.min_interval = 0  # skip the free-tier wait in tests
        with tempfile.TemporaryDirectory() as d:
            results = generate(req, provider, session, output_dir=d, retry_delay=0)
            files_exist = [Path(r["file"]).exists() for r in results]
            meta = json.loads((Path(results[0]["file"]).parent / "metadata.json").read_text())
        return results, files_exist, meta

    def test_generates_multiple_images_with_different_seeds(self):
        session = FakeSession(FakeResponse(body=png_bytes()), FakeResponse(body=png_bytes()))
        results, exists, meta = self.run_generate(session, count=2)
        self.assertEqual([r["seed"] for r in results], [10, 11])
        self.assertTrue(all(exists))
        self.assertEqual(meta["request"]["width"], 1024)
        self.assertEqual(session.calls[0]["timeout"], (3.05, 120))
        self.assertTrue(session.calls[0]["stream"])

    def test_retries_busy_server_then_succeeds(self):
        session = FakeSession(FakeResponse(503, text="busy"), FakeResponse(body=png_bytes()))
        results, _, _ = self.run_generate(session)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(session.calls), 2)

    def test_free_tier_full_402_is_retried(self):
        full = FakeResponse(402, text='{"error":"Payment required"}',
                            headers={"content-type": "application/json", "retry-after": "0"})
        session = FakeSession(full, FakeResponse(body=png_bytes()))
        results, _, _ = self.run_generate(session)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(results[0]["format"], "PNG")

    def test_non_image_response_is_rejected(self):
        page = FakeResponse(200, body=b"<html>error</html>", headers={"content-type": "text/html"})
        session = FakeSession(page, FakeResponse(body=png_bytes()))
        results, _, _ = self.run_generate(session)
        self.assertEqual(len(session.calls), 2)

    def test_retries_corrupt_download(self):
        bad = png_bytes((256, 256))[:500]
        session = FakeSession(FakeResponse(body=bad), FakeResponse(body=png_bytes()))
        results, _, _ = self.run_generate(session)
        self.assertEqual(results[0]["format"], "PNG")

    def test_safety_block_is_not_retried(self):
        session = FakeSession(FakeResponse(400, text="Blocked by content moderation policy"))
        with self.assertRaises(ContentBlockedError):
            self.run_generate(session)
        self.assertEqual(len(session.calls), 1)

    def test_connect_timeout_fails_fast(self):
        session = FakeSession(requests.exceptions.ConnectTimeout("no route"))
        with self.assertRaises(StudioError):
            self.run_generate(session)
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
