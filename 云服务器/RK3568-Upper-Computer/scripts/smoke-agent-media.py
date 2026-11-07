"""Publish a generated JPEG through the production NVENC/SRT path."""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="runtime/rendered/agent/agent.env",
        help="Agent environment file",
    )
    parser.add_argument("--stream", default="plate_ai")
    parser.add_argument("--seconds", type=float, default=8.0)
    return parser.parse_args()


def _generate_jpeg(ffmpeg_path):
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x123047:s=640x360",
        "-frames:v",
        "1",
        "-c:v",
        "mjpeg",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        ),
    )
    if not completed.stdout.startswith(b"\xff\xd8"):
        raise RuntimeError("FFmpeg did not generate a JPEG frame")
    return completed.stdout


def main():
    args = _arguments()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = root / env_path
    if not load_dotenv(env_path, override=True):
        raise SystemExit(f"Agent environment file not found: {env_path}")

    import config
    from backend import ingest
    from backend.cloud_media import NvencSrtPublisher

    jpeg = _generate_jpeg(config.FFMPEG_PATH)
    stop_refresh = threading.Event()

    def refresh_source():
        frame_id = 0
        while not stop_refresh.wait(0.1):
            frame_id += 1
            with ingest._lock:
                ingest._latest[args.stream] = {
                    "meta": {
                        "version": 1,
                        "stream_id": args.stream,
                        "mode": args.stream,
                        "frame_id": frame_id,
                        "overlays": [],
                        "metrics": {},
                    },
                    "jpeg": jpeg,
                    "received_monotonic": time.monotonic(),
                }

    refresher = threading.Thread(target=refresh_source, daemon=True)
    publisher = NvencSrtPublisher(args.stream)
    refresher.start()
    publisher.start()
    try:
        if not publisher.wait_ready(15):
            raise RuntimeError(
                publisher.last_error or "SRT publisher did not become ready"
            )
        print(f"SRT publisher ready for {args.stream}")
        deadline = time.monotonic() + max(1.0, args.seconds)
        consecutive_ready_samples = 0
        last_error = ""
        while time.monotonic() < deadline:
            if publisher.ready:
                consecutive_ready_samples += 1
            else:
                consecutive_ready_samples = 0
            if publisher.last_error:
                last_error = publisher.last_error
            time.sleep(0.1)
        stats = publisher.stats
        if stats["encoded_frames"] < 10 or consecutive_ready_samples < 5:
            raise RuntimeError(
                last_error or "SRT publisher did not remain ready"
            )
        print(
            "Synthetic media smoke passed: "
            f"frames={stats['encoded_frames']} "
            f"upload={stats['upload_mbps']:.3f}Mbps"
        )
    finally:
        publisher.stop()
        stop_refresh.set()
        refresher.join(timeout=1)
        with ingest._lock:
            ingest._latest.pop(args.stream, None)


if __name__ == "__main__":
    main()
