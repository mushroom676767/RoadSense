import sqlite3
from functools import wraps

import bcrypt
from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
    set_access_cookies,
)

import config
from backend.database import get_db


auth_bp = Blueprint("auth", __name__)
ADMIN_ROLE = "admin"
VIEWER_ROLE = "viewer"


def _hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _password_matches(password, password_hash):
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("ascii"),
        )
    except (AttributeError, UnicodeEncodeError, ValueError):
        return False


def bootstrap_admin():
    """Create the configured administrator once without resetting its password."""
    if not config.ADMIN_USERNAME or not config.ADMIN_PASSWORD:
        return False
    password_hash = _hash_password(config.ADMIN_PASSWORD)
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (config.ADMIN_USERNAME, password_hash, ADMIN_ROLE),
        )
        conn.execute(
            "UPDATE users SET role = ? WHERE username = ?",
            (ADMIN_ROLE, config.ADMIN_USERNAME),
        )
        return cursor.rowcount == 1


def bootstrap_visitor():
    """Create or synchronize the configured read-only visitor account."""
    if not config.VISITOR_USERNAME or not config.VISITOR_PASSWORD:
        return False
    if config.VISITOR_USERNAME == config.ADMIN_USERNAME:
        raise RuntimeError("访客用户名不能与管理员用户名相同")

    with get_db() as conn:
        visitor = conn.execute(
            "SELECT password_hash, role FROM users WHERE username = ?",
            (config.VISITOR_USERNAME,),
        ).fetchone()
        if visitor is None:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
                """,
                (
                    config.VISITOR_USERNAME,
                    _hash_password(config.VISITOR_PASSWORD),
                    VIEWER_ROLE,
                ),
            )
            return True

        updates = []
        values = []
        if not _password_matches(
            config.VISITOR_PASSWORD,
            visitor["password_hash"],
        ):
            updates.append("password_hash = ?")
            values.append(_hash_password(config.VISITOR_PASSWORD))
        if visitor["role"] != VIEWER_ROLE:
            updates.append("role = ?")
            values.append(VIEWER_ROLE)
        if updates:
            values.append(config.VISITOR_USERNAME)
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE username = ?",
                values,
            )
        return bool(updates)


def _current_role():
    identity = get_jwt_identity()
    with get_db() as conn:
        user = conn.execute(
            "SELECT role FROM users WHERE username = ?",
            (identity,),
        ).fetchone()
    return user["role"] if user is not None else None


def control_required():
    """Require an authenticated administrator for device mutations."""
    def decorator(function):
        @wraps(function)
        @jwt_required()
        def wrapped(*args, **kwargs):
            if _current_role() != ADMIN_ROLE:
                return jsonify({"error": "访客账号不能进行设备控制"}), 403
            return function(*args, **kwargs)

        return wrapped

    return decorator


@auth_bp.post("/api/register")
def register():
    if not config.ALLOW_REGISTER:
        return jsonify({"error": "公开注册已关闭"}), 403

    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if len(username) < 3 or len(username) > 64:
        return jsonify({"error": "用户名长度必须为3至64个字符"}), 400
    if len(password) < 8 or len(password) > 128:
        return jsonify({"error": "密码长度必须为8至128个字符"}), 400

    password_hash = _hash_password(password)
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, VIEWER_ROLE),
            )
    except sqlite3.IntegrityError:
        return jsonify({"error": "用户名已存在"}), 409
    return jsonify({"message": "注册成功"}), 201


@auth_bp.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    with get_db() as conn:
        user = conn.execute(
            """
            SELECT username, password_hash, role
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    password_valid = user is not None and _password_matches(
        password,
        user["password_hash"],
    )
    if not password_valid:
        return jsonify({"error": "用户名或密码错误"}), 401

    token = create_access_token(
        identity=user["username"],
        additional_claims={"role": user["role"]},
    )
    response = jsonify(
        {
            "token": token,
            "username": user["username"],
            "role": user["role"],
            "can_control": user["role"] == ADMIN_ROLE,
        }
    )
    set_access_cookies(response, token)
    return response


@auth_bp.get("/api/user")
@jwt_required()
def get_user():
    role = _current_role()
    if role is None:
        return jsonify({"error": "用户不存在"}), 401
    return jsonify(
        {
            "username": get_jwt_identity(),
            "role": role,
            "can_control": role == ADMIN_ROLE,
        }
    )
