#!/usr/bin/env python3
"""Download, verify and restore dataset archives from GitHub Releases."""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


MANIFEST = Path(__file__).with_name("manifest.json")
BUFFER_SIZE = 8 * 1024 * 1024


def safe_path(directory, name):
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
        raise ValueError("Unsafe dataset filename: " + name)
    directory = directory.resolve()
    result = directory / name
    if result.is_symlink():
        raise ValueError("Dataset file must not be a symlink: " + name)
    return result


def matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == digest


def fetch_asset(asset, directory):
    target = safe_path(directory, asset["name"])
    size, digest = asset["size_bytes"], asset["sha256"]
    if matches(target, size, digest):
        print("Verified existing:", target.name, flush=True)
        return target
    partial = safe_path(directory, target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        offset = 0
    if offset < size:
        headers = {"User-Agent": "rk3568-dataset-downloader"}
        if offset:
            headers["Range"] = "bytes=%d-" % offset
        request = urllib.request.Request(asset["url"], headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            raise RuntimeError("Download HTTP %d for %s. Check the release URL and GitHub account visibility." % (error.code, target.name)) from error
        with response:
            if offset and response.status != 206:
                offset = 0
            if response.status == 206:
                expected_range = "bytes %d-" % offset
                if not response.headers.get("Content-Range", "").startswith(expected_range):
                    raise RuntimeError("Unexpected download range for " + target.name)
            print("Downloading:", target.name, "from byte", offset, flush=True)
            with partial.open("ab" if offset else "wb") as output:
                received = offset
                while True:
                    chunk = response.read(BUFFER_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > size:
                        raise RuntimeError("Download exceeded expected size: " + target.name)
                    output.write(chunk)
    if not matches(partial, size, digest):
        # A failed digest cannot be repaired by appending; allow the next run to retry.
        failed = safe_path(directory, target.name + ".failed")
        partial.replace(failed)
        raise RuntimeError("Size/SHA-256 mismatch: " + target.name + "; saved as " + failed.name)
    partial.replace(target)
    print("Downloaded and verified:", target.name, flush=True)
    return target


def restore_archive(archive, directory, keep_parts=False, verify_only=False):
    destination = safe_path(directory, archive["name"])
    if matches(destination, archive["size_bytes"], archive["sha256"]):
        print("Archive verified:", destination.name, flush=True)
        return destination
    if verify_only:
        raise RuntimeError("Archive missing or invalid: " + destination.name)
    parts = [fetch_asset(asset, directory) for asset in archive["assets"]]
    if len(parts) == 1 and parts[0] == destination:
        if not matches(destination, archive["size_bytes"], archive["sha256"]):
            raise RuntimeError("Archive SHA-256 mismatch: " + destination.name)
        return destination
    temporary = safe_path(directory, archive["name"] + ".restoring")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, BUFFER_SIZE)
    if not matches(temporary, archive["size_bytes"], archive["sha256"]):
        raise RuntimeError("Restored archive SHA-256 mismatch: " + destination.name)
    temporary.replace(destination)
    if not keep_parts:
        for part in parts:
            part.unlink()
    print("Restored and verified:", destination.name, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="Archive name; repeat to select several. Default: all archives.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("downloads"))
    parser.add_argument("--keep-parts", action="store_true", help="Keep split parts after a successful restore.")
    parser.add_argument("--verify-only", action="store_true", help="Verify restored archives without downloading.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {item["name"] for item in manifest["archives"]}
    selected = set(args.dataset or known)
    if selected - known:
        parser.error("Unknown dataset: " + ", ".join(sorted(selected - known)))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for archive in manifest["archives"]:
            if archive["name"] in selected:
                restore_archive(archive, args.output, args.keep_parts, args.verify_only)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print("Error:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
