"""Render cloud and Agent templates without writing secrets to Git."""

import argparse
import base64
import os
import re
import stat
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")
TEMPLATE_GROUPS = {
    "cloud": (
        ROOT / "deploy" / "cloud",
        (
            "mediamtx.yml.template",
            "nginx.conf.template",
            "turnserver.conf.template",
            "rk3568-cloud.service.template",
            "rk3568-ip-cert.service.template",
            "rk3568-ip-cert.timer",
            "rk3568-renew-ip-cert",
            "cloud.env.template",
            "cert.env.template",
        ),
    ),
    "agent": (
        ROOT / "deploy" / "agent",
        ("agent.env.template",),
    ),
}


def _env(name, default=None):
    value = os.environ.get(name, default)
    if value is None or not str(value):
        raise ValueError(f"required environment variable is missing: {name}")
    return str(value)


def render_values(role):
    values = {
        "__PUBLIC_IP__": _env("PUBLIC_IP", "203.0.113.10"),
        "__SRT_PASSPHRASE__": _env("MEDIA_MTX_SRT_PASSPHRASE"),
        "__AGENT_ID__": _env("AGENT_ID", "rk3568-primary"),
        "__AGENT_TOKEN__": _env("AGENT_TOKEN"),
        "__PUBLISH_USER__": _env("MEDIA_MTX_PUBLISH_USERNAME"),
        "__PUBLISH_PASSWORD__": _env("MEDIA_MTX_PUBLISH_PASSWORD"),
    }
    if role in {"cloud", "all"}:
        read_user = _env("MEDIA_MTX_READ_USERNAME")
        read_password = _env("MEDIA_MTX_READ_PASSWORD")
        values.update({
            "__PRIVATE_IP__": _env("PRIVATE_IP"),
            "__READ_USER__": read_user,
            "__READ_PASSWORD__": read_password,
            "__READ_BASIC_AUTH__": base64.b64encode(
                f"{read_user}:{read_password}".encode("utf-8")
            ).decode("ascii"),
            "__ADMIN_USER__": _env("MEDIA_MTX_ADMIN_USERNAME"),
            "__ADMIN_PASSWORD__": _env("MEDIA_MTX_ADMIN_PASSWORD"),
            "__TURN_USER__": _env("WEBRTC_TURN_USERNAME"),
            "__TURN_PASSWORD__": _env("WEBRTC_TURN_CREDENTIAL"),
            "__JWT_SECRET_KEY__": _env("JWT_SECRET_KEY"),
            "__ADMIN_USERNAME__": _env("ADMIN_USERNAME"),
            "__ADMIN_PASSWORD__": _env("ADMIN_PASSWORD"),
            "__VISITOR_USERNAME__": _env("VISITOR_USERNAME", "visitor"),
            "__VISITOR_PASSWORD__": _env("VISITOR_PASSWORD"),
            "__ACME_EMAIL__": _env("ACME_EMAIL"),
        })
    if role in {"agent", "all"}:
        values.update({
        "__AGENT_SERVER_HOST__": _env(
            "AGENT_SERVER_HOST",
            "0.0.0.0",
        ),
        "__AGENT_LISTEN_HOST__": _env(
            "AGENT_LISTEN_HOST",
            "192.168.137.1",
        ),
        "__AGENT_DATABASE_PATH__": _env(
            "AGENT_DATABASE_PATH",
            str(ROOT / "db" / "agent.db"),
        ),
        "__AGENT_EVENT_DIR__": _env(
            "AGENT_EVENT_DIR",
            str(ROOT / "db" / "agent-events"),
        ),
        "__SSH_HOST__": _env("SSH_HOST", "192.168.137.216"),
        "__SSH_PORT__": _env("SSH_PORT", "22"),
        "__SSH_USER__": _env("SSH_USER", "cat"),
        "__SSH_PASSWORD__": _env("SSH_PASSWORD"),
        "__RK_WORK_DIR__": _env(
            "RK_WORK_DIR",
            "/home/cat/RK3568/final",
        ),
        "__PUBLISH_TOKEN__": _env("PUBLISH_TOKEN"),
        })
    return values


def render_group(role, output_dir, values):
    source_dir, names = TEMPLATE_GROUPS[role]
    role_output = output_dir / role
    role_output.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        source = source_dir / name
        target = role_output / name.removesuffix(".template")
        text = source.read_text(encoding="utf-8")
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        unresolved = sorted(set(PLACEHOLDER.findall(text)))
        if unresolved:
            raise ValueError(
                f"{source.name} has unresolved placeholders: "
                + ", ".join(unresolved)
            )
        target.write_text(text, encoding="utf-8", newline="\n")
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)
        written.append(target)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("cloud", "agent", "all"),
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "rendered",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = (ROOT / "runtime").resolve()
    if runtime not in output.parents and output != runtime:
        raise SystemExit("rendered secrets must stay under runtime/")

    roles = TEMPLATE_GROUPS if args.role == "all" else (args.role,)
    count = 0
    for role in roles:
        values = render_values(role)
        count += len(render_group(role, output, values))
    print(f"rendered {count} protected files under {output}")


if __name__ == "__main__":
    main()
