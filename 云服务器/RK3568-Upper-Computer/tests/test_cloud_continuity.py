import unittest
from pathlib import Path
from unittest import mock

import config
from backend.cloud_media import CloudMediaService


ROOT = Path(__file__).resolve().parents[1]


class CloudContinuityTests(unittest.TestCase):
    def test_cloud_whep_does_not_wait_for_local_publisher(self):
        service = CloudMediaService()
        service.ensure_tunnel = mock.Mock()
        service._publisher = mock.Mock()
        service._whep_request = mock.Mock(
            return_value=(
                "v=0\r\nanswer",
                {"Location": "/plate_ai/whep/session", "ETag": "etag"},
            )
        )
        try:
            with (
                mock.patch.object(config, "DEPLOYMENT_ROLE", "cloud"),
                mock.patch.object(
                    config, "MEDIA_MTX_READ_USERNAME", "reader"
                ),
                mock.patch.object(
                    config, "MEDIA_MTX_READ_PASSWORD", "secret"
                ),
            ):
                answer = service.create_answer(
                    "plate_ai", "v=0\r\noffer", "offer"
                )
        finally:
            service.stop()

        self.assertEqual(answer["type"], "answer")
        service._publisher.assert_not_called()

    def test_mediamtx_template_keeps_all_paths_available(self):
        text = (
            ROOT / "deploy" / "cloud" / "mediamtx.yml.template"
        ).read_text(encoding="utf-8")
        self.assertIn("alwaysAvailable: true", text)
        self.assertIn("- codec: H264", text)
        self.assertIn("hlsVariant: lowLatency", text)
        self.assertIn("webrtcLocalTCPAddress: :49200", text)
        self.assertIn("srtAddress: :8890", text)

    def test_nginx_hls_uses_jwt_auth_request(self):
        text = (
            ROOT / "deploy" / "cloud" / "nginx.conf.template"
        ).read_text(encoding="utf-8")
        self.assertIn("auth_request /internal/media-auth", text)
        self.assertIn("proxy_buffering off", text)
        self.assertIn("proxy_pass http://127.0.0.1:8888/;", text)
        self.assertIn("proxy_redirect / /media/hls/;", text)
        self.assertNotIn("rewrite ^/media/hls/", text)


if __name__ == "__main__":
    unittest.main()
