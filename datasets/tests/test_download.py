"""Integration checks for dataset downloads, resume and binary restoration."""

import hashlib
import importlib.util
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


spec = importlib.util.spec_from_file_location("dataset_download", Path(__file__).resolve().parents[1] / "download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DatasetDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payloads = {"/first": b"first-part\x00", "/second": b"second-part\xff"}
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.requests.append((self.path, self.headers.get("Range")))
                data = cls.payloads[self.path]
                start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

            def log_message(self, *_args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        type(self).requests.clear()

    def asset(self, route, name):
        data = self.payloads[route]
        return {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": "http://127.0.0.1:%d%s" % (self.server.server_port, route)}

    def archive(self):
        combined = self.payloads["/first"] + self.payloads["/second"]
        return {"name": "fixture.zip", "size_bytes": len(combined), "sha256": hashlib.sha256(combined).hexdigest(), "assets": [self.asset("/first", "fixture.zip.part001"), self.asset("/second", "fixture.zip.part002")]}

    def test_restores_original_bytes_and_removes_verified_parts(self):
        result = download.restore_archive(self.archive(), self.directory)
        self.assertEqual(result.read_bytes(), self.payloads["/first"] + self.payloads["/second"])
        self.assertEqual(list(self.directory.iterdir()), [result])
        self.assertEqual(len(self.requests), 2)

    def test_resumes_an_interrupted_part_with_a_range_request(self):
        (self.directory / "resumed.7z.partial").write_bytes(self.payloads["/first"][:4])
        result = download.fetch_asset(self.asset("/first", "resumed.7z"), self.directory)
        self.assertEqual(result.read_bytes(), self.payloads["/first"])
        self.assertEqual(self.requests, [("/first", "bytes=4-")])

    def test_rejects_corrupt_download_and_preserves_it_as_failed(self):
        asset = self.asset("/first", "bad.7z")
        asset["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            download.fetch_asset(asset, self.directory)
        self.assertFalse((self.directory / "bad.7z").exists())
        self.assertEqual((self.directory / "bad.7z.failed").read_bytes(), self.payloads["/first"])

    def test_existing_verified_archive_needs_no_network(self):
        data = self.payloads["/first"] + self.payloads["/second"]
        (self.directory / "fixture.zip").write_bytes(data)
        result = download.restore_archive(self.archive(), self.directory, verify_only=True)
        self.assertEqual(result.read_bytes(), data)
        self.assertEqual(self.requests, [])

    def test_rejects_paths_outside_the_download_directory(self):
        for name in ("../outside", "C:/outside", "nested/file", "nested\\file"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.safe_path(self.directory, name)


if __name__ == "__main__":
    unittest.main()
