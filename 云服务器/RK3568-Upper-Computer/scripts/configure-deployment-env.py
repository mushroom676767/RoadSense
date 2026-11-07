"""Add missing cloud/Agent deployment values to the ignored local .env."""

import argparse
import os
import secrets
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse(lines):
    values = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acme-email", required=True)
    parser.add_argument("--public-ip", default="203.0.113.10")
    parser.add_argument("--private-ip", default="10.0.0.10")
    args = parser.parse_args()
    env_path = ROOT / ".env"
    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else []
    )
    existing = parse(lines)
    additions = {
        "PUBLIC_IP": args.public_ip,
        "PRIVATE_IP": args.private_ip,
        "ACME_EMAIL": args.acme_email,
        "AGENT_ID": existing.get("AGENT_ID") or "rk3568-primary",
        "AGENT_TOKEN": existing.get("AGENT_TOKEN")
        or secrets.token_urlsafe(48),
        "MEDIA_MTX_ADMIN_USERNAME": existing.get(
            "MEDIA_MTX_ADMIN_USERNAME"
        )
        or "local-admin",
        "MEDIA_MTX_ADMIN_PASSWORD": existing.get(
            "MEDIA_MTX_ADMIN_PASSWORD"
        )
        or secrets.token_urlsafe(32),
        "MEDIA_MTX_SRT_PASSPHRASE": existing.get(
            "MEDIA_MTX_SRT_PASSPHRASE"
        )
        or secrets.token_urlsafe(24),
        "AGENT_LISTEN_HOST": existing.get("AGENT_LISTEN_HOST")
        or "192.168.137.1",
        "AGENT_SERVER_HOST": existing.get("AGENT_SERVER_HOST")
        or "0.0.0.0",
        "AGENT_DATABASE_PATH": existing.get("AGENT_DATABASE_PATH")
        or str(ROOT / "db" / "agent.db"),
        "AGENT_EVENT_DIR": existing.get("AGENT_EVENT_DIR")
        or str(ROOT / "db" / "agent-events"),
    }
    output = list(lines)
    for key, value in additions.items():
        if existing.get(key):
            continue
        output.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".env.",
        dir=env_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(output).rstrip() + "\n")
        os.replace(temporary, env_path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
    print("deployment values are configured in the ignored local .env")


if __name__ == "__main__":
    main()
