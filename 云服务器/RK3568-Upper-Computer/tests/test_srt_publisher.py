import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import config
from backend.cloud_media import NvencRtspPublisher, NvencSrtPublisher


class SrtPublisherTests(unittest.TestCase):
    def test_ffmpeg_fallback_command_uses_nvenc_mpegts_and_srt(self):
        with (
            mock.patch.object(config, "FFMPEG_PATH", "ffmpeg-custom"),
            mock.patch.object(
                NvencSrtPublisher,
                "_resolve_ffmpeg",
                return_value="C:/tools/ffmpeg.exe",
            ),
            mock.patch.object(
                NvencSrtPublisher,
                "_srt_url",
                return_value="srt://media.test:8890?mode=caller",
            ),
        ):
            command = NvencSrtPublisher("plate_ai")._ffmpeg_command()

        self.assertEqual(command[0], "C:/tools/ffmpeg.exe")
        self.assertIn("pipe:0", command)
        self.assertIn(config.WEBRTC_VIDEO_ENCODER, command)
        # FFmpeg's input-side nobuffer flag drops every packet from a forced
        # MJPEG image pipe on the Windows build used by the Agent.  The
        # process keeps running and NVENC consumes CPU, but SRT receives zero
        # media bytes, so this regression is otherwise difficult to detect.
        self.assertNotIn("nobuffer", command)
        self.assertIn("low_delay", command)
        self.assertIn("32", command)
        self.assertEqual(command[-2:], [
            "mpegts",
            "srt://media.test:8890?mode=caller",
        ])

    def test_pyav_srt_capability_is_checked_before_publishing(self):
        with mock.patch("backend.cloud_media.av") as pyav:
            pyav.protocols_available = {"http", "rtsp"}
            self.assertFalse(NvencSrtPublisher._pyav_supports_srt())
            pyav.protocols_available.add("srt")
            self.assertTrue(NvencSrtPublisher._pyav_supports_srt())

    def test_ffmpeg_srt_url_removes_unsupported_rw_timeout(self):
        publisher = NvencSrtPublisher("plate_ai")
        with mock.patch.object(
            publisher,
            "_srt_url",
            return_value=(
                "srt://media.test:8890?mode=caller"
                "&rw_timeout=500000&latency=200000"
            ),
        ):
            url = publisher._ffmpeg_srt_url()
        query = parse_qs(urlsplit(url).query)
        self.assertNotIn("rw_timeout", query)
        self.assertEqual(query["mode"], ["caller"])
        self.assertEqual(query["latency"], ["200000"])

    def test_srt_url_uses_live_latency_encryption_and_auth(self):
        with (
            mock.patch.object(config, "MEDIA_MTX_SRT_HOST", "media.test"),
            mock.patch.object(config, "MEDIA_MTX_SRT_PORT", 8890),
            mock.patch.object(config, "MEDIA_MTX_SRT_LATENCY_MS", 200),
            mock.patch.object(config, "MEDIA_MTX_PUBLISH_USERNAME", "pub"),
            mock.patch.object(
                config, "MEDIA_MTX_PUBLISH_PASSWORD", "p@ss:word"
            ),
            mock.patch.object(
                config, "MEDIA_MTX_SRT_PASSPHRASE", "strong-passphrase"
            ),
        ):
            url = NvencSrtPublisher("plate_ai")._srt_url()

        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.scheme, "srt")
        self.assertEqual(parsed.hostname, "media.test")
        self.assertEqual(query["transtype"], ["live"])
        self.assertEqual(query["latency"], ["200000"])
        self.assertEqual(
            query["streamid"],
            ["publish:plate_ai:pub:p@ss:word"],
        )
        self.assertEqual(query["passphrase"], ["strong-passphrase"])

    def test_errors_redact_media_secrets(self):
        with (
            mock.patch.object(
                config, "MEDIA_MTX_PUBLISH_PASSWORD", "publish-secret"
            ),
            mock.patch.object(
                config, "MEDIA_MTX_SRT_PASSPHRASE", "srt-secret"
            ),
        ):
            message = NvencRtspPublisher._sanitize_error(
                "failed publish-secret and srt-secret"
            )
        self.assertNotIn("publish-secret", message)
        self.assertNotIn("srt-secret", message)
        self.assertEqual(message.count("********"), 2)


if __name__ == "__main__":
    unittest.main()
