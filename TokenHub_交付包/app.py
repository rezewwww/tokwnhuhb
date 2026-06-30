"""
AI Model API Router + Token 售卖平台
- 统一入口 POST /v1/chat/completions (OpenAI 兼容格式)
- 用户注册/登录、API Key 管理、用量计费、充值管理
"""
import os
import json
import time
import uuid
import hashlib
import logging
import re
import secrets
import random
import string
import threading
import base64
from functools import wraps
from datetime import datetime, timedelta

import bcrypt
import jwt
import requests
from flask import Flask, request, Response, jsonify, stream_with_context, send_file, g
from dotenv import load_dotenv, set_key

from models import get_db, init_db, DB_TYPE

# ========== 初始化 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(STATIC_DIR, ".env")

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("SECRET_KEY"):
    set_key(ENV_FILE, "SECRET_KEY", app.config["SECRET_KEY"])
    logger.info("已自动生成 SECRET_KEY 并写入 .env")

# ========== CORS ==========
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOWED_ORIGINS = [o.strip() for o in CORS_ALLOWED_ORIGINS if o.strip()]
if not CORS_ALLOWED_ORIGINS:
    CORS_ALLOWED_ORIGINS = ["*"]

@app.before_request
def handle_options():
    origin = request.headers.get("Origin", "")
    allow_origin = "*"
    if "*" not in CORS_ALLOWED_ORIGINS and origin:
        allow_origin = origin if origin in CORS_ALLOWED_ORIGINS else CORS_ALLOWED_ORIGINS[0]

    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = allow_origin
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, x-api-key, anthropic-version"
        resp.headers["Access-Control-Max-Age"] = "3600"
        return resp

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    allow_origin = "*"
    if "*" not in CORS_ALLOWED_ORIGINS and origin:
        allow_origin = origin if origin in CORS_ALLOWED_ORIGINS else CORS_ALLOWED_ORIGINS[0]
    response.headers["Access-Control-Allow-Origin"] = allow_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, x-api-key, anthropic-version"
    response.headers["Access-Control-Max-Age"] = "3600"
    response.headers["Vary"] = "Origin"
    return response

# 首次启动时自动初始化数据库
init_db()

# ========== 定价配置（单位：分/1K tokens） ==========
PRICING = {
    # OpenAI
    "gpt-4o":               {"input": 15,   "output": 60},
    "gpt-4o-mini":          {"input": 1,    "output": 4},
    "gpt-4.1":              {"input": 2,    "output": 8},
    "gpt-4.1-mini":         {"input": 1,    "output": 4},
    "o4-mini":              {"input": 2,    "output": 8},
    "o3":                   {"input": 10,   "output": 40},
    "o3-mini":              {"input": 2,    "output": 8},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 20, "output": 100},
    "claude-3-opus-20240229":     {"input": 100, "output": 500},
    "claude-opus-4-20250514":     {"input": 15,  "output": 75},
    "claude-3-5-haiku-20241022":  {"input": 1,   "output": 5},
    # Google
    "gemini-2.0-flash":     {"input": 1,    "output": 3},
    "gemini-1.5-pro":       {"input": 2,    "output": 8},
    "gemini-2.5-pro":       {"input": 2,    "output": 10},
    "gemini-2.5-flash":     {"input": 1,    "output": 2},
    "gemini-2.0-flash-lite": {"input": 1,   "output": 1},
    # DeepSeek
    "deepseek-v3-0324":     {"input": 1,    "output": 2},
    "deepseek-r1":          {"input": 1,    "output": 2},
    # Meta
    "llama-4-maverick":     {"input": 2,    "output": 6},
    "llama-4-scout":        {"input": 1,    "output": 3},
    "llama-3.3-70b":        {"input": 1,    "output": 3},
    # Mistral
    "mistral-large":        {"input": 4,    "output": 12},
    "mistral-small":        {"input": 1,    "output": 3},
    "mixtral-8x22b":        {"input": 2,    "output": 6},
    "codestral":            {"input": 2,    "output": 6},
    # Alibaba
    "qwen-max":             {"input": 2,    "output": 6},
    "qwen-plus":            {"input": 1,    "output": 3},
    "qwen-turbo":           {"input": 1,    "output": 2},
    # Other
    "command-r-plus":       {"input": 3,    "output": 15},
    "jamba-1.5-large":      {"input": 2,    "output": 8},
    "dbrx-instruct":        {"input": 2,    "output": 8},
}
DEFAULT_PRICING = {"input": 1, "output": 2}  # 未知模型默认

MIN_BALANCE_THRESHOLD = 1  # 最低余额阈值（分）

# ========== 接口限流（令牌桶，SQLite/PostgreSQL 持久化） ==========
RATE_LIMIT_WINDOW = 60        # 窗口时间（秒）
RATE_LIMIT_MAX = 60           # 每窗口最大请求数

def _rate_limit_check(identifier):
    """令牌桶限流检查，基于 SQLite/PostgreSQL 持久化，返回 (allowed: bool, retry_after: int)。"""
    now = time.time()
    window_key = f"ratelimit:{identifier}"
    db = get_db()
    try:
        row = db.execute("SELECT window_start, count FROM rate_limit_state WHERE key = ?", (window_key,)).fetchone()
        if row is None or now - row["window_start"] >= RATE_LIMIT_WINDOW:
            # 新窗口
            db.execute(
                "INSERT INTO rate_limit_state (key, window_start, count) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET window_start = ?, count = ?",
                (window_key, now, 1, now, 1)
            ) if DB_TYPE == "postgresql" else db.execute(
                "INSERT OR REPLACE INTO rate_limit_state (key, window_start, count) VALUES (?, ?, ?)",
                (window_key, now, 1)
            )
            db.commit()
            return True, 0
        if row["count"] >= RATE_LIMIT_MAX:
            retry_after = int(RATE_LIMIT_WINDOW - (now - row["window_start"]))
            return False, max(1, retry_after)
        new_count = row["count"] + 1
        db.execute("UPDATE rate_limit_state SET count = ? WHERE key = ?", (new_count, window_key))
        db.commit()
        return True, 0
    finally:
        db.close()


# ========== 自动重试 + 熔断（SQLite/PostgreSQL 持久化） ==========
RETRY_MAX = 2                     # 最大重试次数
RETRY_DELAYS = [1, 2]             # 重试间隔（秒）
BREAKER_THRESHOLD = 3             # 5 分钟内连续失败 N 次触发熔断
BREAKER_TIMEOUT = 300             # 熔断持续时间（秒）

def _check_breaker(model_id):
    db = get_db()
    try:
        row = db.execute("SELECT failures, last_failure, is_open, opened_at FROM circuit_breaker_state WHERE model = ?",
                         (model_id,)).fetchone()
        if not row:
            return True, None
        now = time.time()
        if row["is_open"] and now - row["opened_at"] < BREAKER_TIMEOUT:
            remaining = int(BREAKER_TIMEOUT - (now - row["opened_at"]))
            return False, f"模型 '{model_id}' 暂时不可用（熔断中），请 {remaining} 秒后重试"
        if row["is_open"] and now - row["opened_at"] >= BREAKER_TIMEOUT:
            # 半开状态：允许尝试
            db.execute("UPDATE circuit_breaker_state SET is_open = 0, failures = 0 WHERE model = ?", (model_id,))
            db.commit()
            return True, None
        return True, None
    finally:
        db.close()

def _record_failure(model_id):
    now = time.time()
    db = get_db()
    try:
        row = db.execute("SELECT failures FROM circuit_breaker_state WHERE model = ?", (model_id,)).fetchone()
        if not row:
            db.execute("INSERT INTO circuit_breaker_state (model, failures, last_failure) VALUES (?, ?, ?)",
                       (model_id, 1, now))
        else:
            new_failures = row["failures"] + 1
            is_open = 1 if new_failures >= BREAKER_THRESHOLD else 0
            opened_at = now if is_open else 0
            db.execute(
                "UPDATE circuit_breaker_state SET failures = ?, last_failure = ?, is_open = ?, opened_at = ? WHERE model = ?",
                (new_failures, now, is_open, opened_at, model_id)
            )
        db.commit()
    finally:
        db.close()

def _record_success(model_id):
    db = get_db()
    try:
        db.execute("DELETE FROM circuit_breaker_state WHERE model = ?", (model_id,))
        db.commit()
    finally:
        db.close()

def _with_retry_breaker(model_id, forward_fn, *args, **kwargs):
    allowed, msg = _check_breaker(model_id)
    if not allowed:
        return jsonify({"error": msg}), 503
    last_error = None
    for attempt in range(RETRY_MAX + 1):
        try:
            result = forward_fn(*args, **kwargs)
            _record_success(model_id)
            return result
        except Exception as e:
            last_error = e
            _record_failure(model_id)
            if attempt < RETRY_MAX:
                logger.warning(f"[熔断] 模型={model_id} 第{attempt+1}次失败，{RETRY_DELAYS[attempt]}s后重试: {e}")
                time.sleep(RETRY_DELAYS[attempt])
    _log_error(f"[熔断] 模型={model_id} 连续失败，已熔断 {BREAKER_TIMEOUT}s")
    return jsonify({"error": f"模型 '{model_id}' 当前不可用，请稍后重试"}), 503

# ========== 错误日志（内存） ==========
_error_log = []
_error_log_lock = threading.Lock()
ERROR_LOG_MAX = 200

def _log_error(msg, level="ERROR"):
    with _error_log_lock:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _error_log.append((ts, level, msg))
        if len(_error_log) > ERROR_LOG_MAX:
            _error_log[:] = _error_log[-ERROR_LOG_MAX:]


class _ErrorLogHandler(logging.Handler):
    """将 ERROR/CRITICAL 级别日志自动写入 _error_log。"""
    def emit(self, record):
        try:
            msg = self.format(record)
            _log_error(msg, level=record.levelname)
        except Exception:
            pass


# 安装自定义 Handler 自动捕获日志
_err_handler = _ErrorLogHandler()
_err_handler.setLevel(logging.ERROR)
_err_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_err_handler)

JWT_SECRET = os.getenv("JWT_SECRET", app.config["SECRET_KEY"])
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))


def make_jwt(user_id: int, username: str, is_admin: bool = False) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "is_admin": is_admin,
        "exp": int(time.time()) + JWT_EXPIRE_HOURS * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ========== 后端配置 ==========
BACKENDS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "prefixes": ["gpt-", "o1-", "o3-", "o4-"],
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "prefixes": ["claude-"],
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        "prefixes": ["gemini-"],
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "prefixes": ["deepseek-"],
    },
}

DEFAULT_MODELS = [
    # OpenAI
    {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
    {"id": "gpt-4.1", "object": "model", "owned_by": "openai"},
    {"id": "gpt-4.1-mini", "object": "model", "owned_by": "openai"},
    {"id": "o4-mini", "object": "model", "owned_by": "openai"},
    {"id": "o3", "object": "model", "owned_by": "openai"},
    {"id": "o3-mini", "object": "model", "owned_by": "openai"},
    # Anthropic
    {"id": "claude-3-5-sonnet-20241022", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-3-opus-20240229", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-opus-4-20250514", "object": "model", "owned_by": "anthropic"},
    {"id": "claude-3-5-haiku-20241022", "object": "model", "owned_by": "anthropic"},
    # Google
    {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
    {"id": "gemini-1.5-pro", "object": "model", "owned_by": "google"},
    {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
    {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
    {"id": "gemini-2.0-flash-lite", "object": "model", "owned_by": "google"},
    # DeepSeek
    {"id": "deepseek-v3-0324", "object": "model", "owned_by": "deepseek"},
    {"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"},
    # Meta
    {"id": "llama-4-maverick", "object": "model", "owned_by": "meta"},
    {"id": "llama-4-scout", "object": "model", "owned_by": "meta"},
    {"id": "llama-3.3-70b", "object": "model", "owned_by": "meta"},
    # Mistral
    {"id": "mistral-large", "object": "model", "owned_by": "mistral"},
    {"id": "mistral-small", "object": "model", "owned_by": "mistral"},
    {"id": "mixtral-8x22b", "object": "model", "owned_by": "mistral"},
    {"id": "codestral", "object": "model", "owned_by": "mistral"},
    # Alibaba
    {"id": "qwen-max", "object": "model", "owned_by": "alibaba"},
    {"id": "qwen-plus", "object": "model", "owned_by": "alibaba"},
    {"id": "qwen-turbo", "object": "model", "owned_by": "alibaba"},
    # Other
    {"id": "command-r-plus", "object": "model", "owned_by": "cohere"},
    {"id": "jamba-1.5-large", "object": "model", "owned_by": "ai21"},
    {"id": "dbrx-instruct", "object": "model", "owned_by": "databricks"},
]

# ========== 鉴权中间件 ==========

def require_user(f):
    """管理后台 JWT 鉴权装饰器。"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
        if not token:
            return jsonify({"error": "缺少认证令牌"}), 401
        payload = verify_jwt(token)
        if not payload:
            return jsonify({"error": "令牌无效或已过期"}), 401
        g.current_user = payload
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    """管理员权限装饰器。"""
    @wraps(f)
    @require_user
    def decorated(*args, **kwargs):
        db = get_db()
        user = db.execute("SELECT is_admin FROM users WHERE id = ?", (g.current_user["user_id"],)).fetchone()
        db.close()
        if not user or not user["is_admin"]:
            return jsonify({"error": "需要管理员权限"}), 403
        return f(*args, **kwargs)
    return decorated


def resolve_backend(model: str):
    """根据 model 字段解析对应的后端配置。"""
    model_lower = model.lower()
    for name, cfg in BACKENDS.items():
        for prefix in cfg["prefixes"]:
            if model_lower.startswith(prefix):
                if not cfg["api_key"]:
                    raise ValueError(f"后端 '{name}' 的 API Key 未配置")
                return name, cfg
    raise ValueError(f"未找到模型 '{model}' 对应的后端。支持的前缀: " +
                     ", ".join(p for c in BACKENDS.values() for p in c["prefixes"]))


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> int:
    """计算费用（分），采用向上取整。"""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    cost = (prompt_tokens / 1000.0) * pricing["input"] + (completion_tokens / 1000.0) * pricing["output"]
    return max(0, int(cost + 0.99999))  # 向上取整


# ========== 转发函数（从 api_router.py 迁移） ==========

def forward_openai(body: dict, backend_cfg: dict, stream: bool):
    headers = {
        "Authorization": f"Bearer {backend_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{backend_cfg['base_url'].rstrip('/')}/chat/completions"
    resp = requests.post(url, headers=headers, json=body, stream=stream, timeout=300)
    if resp.status_code != 200:
        return jsonify(resp.json()), resp.status_code
    if not stream:
        return jsonify(resp.json())
    def generate():
        for line in resp.iter_lines():
            if line:
                yield line.decode("utf-8") + "\n"
    return Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


def forward_anthropic(body: dict, backend_cfg: dict, stream: bool):
    headers = {
        "x-api-key": backend_cfg["api_key"],
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    messages = body.get("messages", [])
    system_msg = None
    anthropic_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_msg = content if isinstance(content, str) else content
        else:
            anthropic_messages.append({"role": role, "content": content})

    anthropic_body = {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens", 4096),
        "messages": anthropic_messages,
        "stream": stream,
    }
    if system_msg:
        anthropic_body["system"] = system_msg
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]
    if "stop" in body:
        anthropic_body["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    url = f"{backend_cfg['base_url'].rstrip('/')}/messages"
    resp = requests.post(url, headers=headers, json=anthropic_body, stream=stream, timeout=300)
    if resp.status_code != 200:
        return jsonify(resp.json()), resp.status_code

    if not stream:
        data = resp.json()
        content_blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        return jsonify({
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": data.get("stop_reason", "stop")}],
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
            },
        })

    def generate():
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                event_data = text[6:]
                try:
                    event = json.loads(event_data)
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta_text = event.get("delta", {}).get("text", "")
                        sse = json.dumps({
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.get("model"),
                            "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                        })
                        yield f"data: {sse}\n\n"
                    elif event_type == "message_stop":
                        sse = json.dumps({
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.get("model"),
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        })
                        yield f"data: {sse}\n\n"
                        yield "data: [DONE]\n\n"
                except json.JSONDecodeError:
                    pass

    return Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


def forward_gemini(body: dict, backend_cfg: dict, stream: bool):
    api_key = backend_cfg["api_key"]
    model = body.get("model")
    messages = body.get("messages", [])
    system_instruction = None
    contents = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": item["image_url"]["url"]}})
        else:
            parts = [{"text": str(content)}]
        if role == "system":
            system_instruction = {"parts": [{"text": content}]} if isinstance(content, str) else {"parts": parts}
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": parts})

    gemini_body = {"contents": contents}
    if system_instruction:
        gemini_body["systemInstruction"] = system_instruction

    generation_config = {}
    if "temperature" in body:
        generation_config["temperature"] = body["temperature"]
    if "top_p" in body:
        generation_config["topP"] = body["top_p"]
    if "max_tokens" in body:
        generation_config["maxOutputTokens"] = body["max_tokens"]
    if "stop" in body:
        stops = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        generation_config["stopSequences"] = stops
    if generation_config:
        gemini_body["generationConfig"] = generation_config

    alt_param = "&alt=sse" if stream else ""
    url = f"{backend_cfg['base_url'].rstrip('/')}/models/{model}:{'streamGenerateContent' if stream else 'generateContent'}?key={api_key}{alt_param}"
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=gemini_body, stream=stream, timeout=300)
    if resp.status_code != 200:
        return jsonify(resp.json()), resp.status_code

    if not stream:
        data = resp.json()
        candidates = data.get("candidates", [])
        text = ""
        finish_reason = "stop"
        if candidates:
            c = candidates[0]
            for part in c.get("content", {}).get("parts", []):
                text += part.get("text", "")
            finish_reason = c.get("finishReason", "stop").lower()
        usage = data.get("usageMetadata", {})
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": usage.get("promptTokenCount", 0),
                       "completion_tokens": usage.get("candidatesTokenCount", 0),
                       "total_tokens": usage.get("totalTokenCount", 0)},
        })

    def generate():
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                event_data = text[6:]
                try:
                    event = json.loads(event_data)
                    candidates = event.get("candidates", [])
                    if candidates:
                        c = candidates[0]
                        content_parts = c.get("content", {}).get("parts", [])
                        delta_text = "".join(p.get("text", "") for p in content_parts)
                        finish_reason = c.get("finishReason")
                        sse = {
                            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta_text},
                                         "finish_reason": finish_reason.lower() if finish_reason else None}],
                        }
                        yield f"data: {json.dumps(sse)}\n\n"
                        if finish_reason:
                            yield "data: [DONE]\n\n"
                except json.JSONDecodeError:
                    pass

    return Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ========== 静态文件 ==========

@app.route("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path, mimetype="text/html")
    return jsonify({"service": "ai-model-router", "status": "ok"})


@app.route("/<path:filename>")
def serve_static(filename):
    import mimetypes
    file_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        return send_file(file_path, mimetype=mime_type or "application/octet-stream")
    return jsonify({"error": "Not found"}), 404


# ========== OpenAI 兼容入口（带用户鉴权+计费） ==========

@app.route("/v1/models", methods=["GET"])
def list_models():
    db = get_db()
    disabled = {r["model"] for r in db.execute("SELECT model FROM disabled_models").fetchall()}
    breakers = {}
    for r in db.execute("SELECT model, is_open, opened_at FROM circuit_breaker_state").fetchall():
        breakers[r["model"]] = r
    db.close()
    enabled = []
    for m in DEFAULT_MODELS:
        if m["id"] in disabled:
            continue
        entry = dict(m)
        br = breakers.get(m["id"])
        if br and br["is_open"] and time.time() - br["opened_at"] < BREAKER_TIMEOUT:
            entry["status"] = "degraded"
        enabled.append(entry)
    return jsonify({"object": "list", "data": enabled})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """核心聊天接口：API Key 鉴权 + 余额校验 + 转发 + 扣费。"""
    # 1. 验证 API Key
    auth = request.headers.get("Authorization", "")
    api_key = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
    if not api_key:
        return jsonify({"error": "缺少 API Key，请在 Authorization Header 中提供 Bearer <key>"}), 401

    db = get_db()
    # API Key 加密校验：用 SHA256 hash 做快速查找，兼容旧明文 key
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_row = db.execute(
        "SELECT k.id, k.user_id, k.is_active, k.allowed_models, u.balance, u.username "
        "FROM api_keys k JOIN users u ON k.user_id = u.id "
        "WHERE k.key_value = ?", (api_key_hash,)
    ).fetchone()

    # 兼容旧明文 Key（key_value 非 hash 格式，逐一解密匹配）
    if not key_row:
        all_keys = db.execute(
            "SELECT k.id, k.user_id, k.is_active, k.allowed_models, k.key_value, u.balance, u.username "
            "FROM api_keys k JOIN users u ON k.user_id = u.id WHERE k.is_active = 1"
        ).fetchall()
        for k in all_keys:
            if k["key_value"] == api_key:
                key_row = k
                break

    if not key_row:
        db.close()
        return jsonify({"error": "无效的 API Key"}), 401
    if not key_row["is_active"]:
        db.close()
        return jsonify({"error": "该 API Key 已被禁用"}), 403

    user_id = key_row["user_id"]
    api_key_id = key_row["id"]
    balance = key_row["balance"]

    # 2. 余额预检
    if balance < MIN_BALANCE_THRESHOLD:
        db.close()
        return jsonify({"error": f"余额不足，当前余额 {balance/100:.2f} 元，请充值"}), 402

    # 2.5. 接口限流（IP + API Key 组合）
    client_ip = request.remote_addr or "unknown"
    rate_id = f"user_{user_id}"
    allowed, retry = _rate_limit_check(rate_id)
    if not allowed:
        db.close()
        resp = jsonify({"error": "请求过于频繁，请稍后再试", "retry_after": retry})
        resp.status_code = 429
        resp.headers["Retry-After"] = str(retry)
        return resp

    # 3. 解析请求
    body = request.get_json(silent=True)
    if not body:
        db.close()
        return jsonify({"error": "Invalid JSON body"}), 400

    model = body.get("model", "")
    if not model:
        db.close()
        return jsonify({"error": "Missing 'model' field"}), 400
    disabled = db.execute("SELECT 1 FROM disabled_models WHERE model = ?", (model,)).fetchone()
    if disabled:
        db.close()
        return jsonify({"error": "模型已下架"}), 400

    # API Key 白名单检查
    allowed_models_str = (key_row["allowed_models"] or "").strip()
    if allowed_models_str:
        allowed_list = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
        if model not in allowed_list:
            db.close()
            return jsonify({"error": "此 API Key 无权访问该模型"}), 403

    # 套餐额度检查
    sub_row = db.execute(
        "SELECT id, plan, model_quota, calls_used, status, expires_at FROM subscriptions "
        "WHERE user_id = ? AND status = 'active' ORDER BY expires_at DESC LIMIT 1",
        (user_id,)
    ).fetchone()
    if sub_row:
        # 检查是否过期
        if sub_row["expires_at"]:
            try:
                expires_dt = datetime.strptime(sub_row["expires_at"], "%Y-%m-%d %H:%M:%S")
                if expires_dt < datetime.now():
                    db.execute("UPDATE subscriptions SET status = 'expired' WHERE id = ?", (sub_row["id"],))
                    db.commit()
                    db.close()
                    return jsonify({"error": "套餐已过期，请重新订阅"}), 402
            except (ValueError, TypeError):
                pass
        if sub_row["calls_used"] >= sub_row["model_quota"]:
            db.close()
            return jsonify({"error": "套餐额度已用完，请升级或等待下月重置"}), 402
    else:
        # 无套餐用户按余额扣费（已有逻辑继续）
        pass

    stream = body.get("stream", False)

    # 4. 路由转发
    try:
        backend_name, backend_cfg = resolve_backend(model)
        logger.info(f"[user={user_id}] model={model} -> backend={backend_name}, stream={stream}")
    except ValueError as e:
        db.close()
        return jsonify({"error": str(e)}), 400

    # 非流式先获取响应再扣费；流式只能先预估扣费（暂不处理流式扣费，走非流式计费逻辑）
    if stream:
        try:
            def _do_stream_forward():
                if backend_name == "anthropic":
                    return forward_anthropic(body, backend_cfg, stream)
                elif backend_name == "gemini":
                    return forward_gemini(body, backend_cfg, stream)
                else:
                    return forward_openai(body, backend_cfg, stream)
            result = _with_retry_breaker(model, _do_stream_forward)
            # 检查熔断/失败
            if isinstance(result, tuple) and len(result) == 2 and result[1] >= 400:
                db.close()
                return result
            # 冻结-实扣模式：按 max_tokens 冻结预估费用，标记为 pending
            prompt_est_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
            prompt_tokens_est = max(1, prompt_est_chars // 4)
            max_tokens = body.get("max_tokens", 1024)
            completion_tokens_est = max_tokens
            cost = calc_cost(model, prompt_tokens_est, completion_tokens_est)
            # 冻结金额
            db.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (cost, user_id))
            db.execute(
                "INSERT INTO usage_records (user_id, api_key_id, model, backend, prompt_tokens, completion_tokens, cost, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                (user_id, api_key_id, model, backend_name, prompt_tokens_est, completion_tokens_est, cost)
            )
            # 套餐调用次数 +1
            sub_row_s = db.execute(
                "SELECT id FROM subscriptions WHERE user_id = ? AND status = 'active' LIMIT 1",
                (user_id,)
            ).fetchone()
            if sub_row_s:
                db.execute("UPDATE subscriptions SET calls_used = calls_used + 1 WHERE id = ?", (sub_row_s["id"],))
            db.commit()
            db.close()
            return result
        except Exception as e:
            db.close()
            logger.error(f"转发错误: {e}")
            return jsonify({"error": str(e)}), 500

    # 非流式：获取完整响应后计费（带重试+熔断）
    def _do_forward():
        if backend_name == "anthropic":
            return forward_anthropic(body, backend_cfg, False)
        elif backend_name == "gemini":
            return forward_gemini(body, backend_cfg, False)
        else:
            return forward_openai(body, backend_cfg, False)
    result = _with_retry_breaker(model, _do_forward)

    # 5. 解析 usage 并扣费
    if isinstance(result, tuple) and len(result) == 2 and result[1] >= 400:
        db.close()
        return result
    resp_data = result.get_json() if hasattr(result, "get_json") else {}
    usage = resp_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = calc_cost(model, prompt_tokens, completion_tokens)

    try:
        db.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (cost, user_id))
        db.execute(
            "INSERT INTO usage_records (user_id, api_key_id, model, backend, prompt_tokens, completion_tokens, cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, api_key_id, model, backend_name, prompt_tokens, completion_tokens, cost)
        )
        # 套餐调用次数 +1
        sub_row2 = db.execute(
            "SELECT id FROM subscriptions WHERE user_id = ? AND status = 'active' LIMIT 1",
            (user_id,)
        ).fetchone()
        if sub_row2:
            db.execute("UPDATE subscriptions SET calls_used = calls_used + 1 WHERE id = ?", (sub_row2["id"],))
        db.commit()
        # Webhook 用量事件
        _emit_usage_event(user_id, model, prompt_tokens + completion_tokens, cost)
        logger.info(f"[user={user_id}] 扣费 {cost} 分, 模型={model}, tokens=({prompt_tokens},{completion_tokens})")
    except Exception as e:
        db.rollback()
        logger.error(f"扣费失败: {e}")
    finally:
        db.close()

    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ai-model-router"})


# ========== 用户 API ==========

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    email = (body.get("email") or "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 3:
        return jsonify({"error": "用户名至少 3 个字符"}), 400
    if len(password) < 8:
        return jsonify({"error": "密码至少 8 个字符"}), 400
    if not re.search(r"[A-Z]", password):
        return jsonify({"error": "密码必须包含至少一个大写字母"}), 400
    if not re.search(r"[a-z]", password):
        return jsonify({"error": "密码必须包含至少一个小写字母"}), 400
    if not re.search(r"\d", password):
        return jsonify({"error": "密码必须包含至少一个数字"}), 400
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        return jsonify({"error": "密码必须包含至少一个特殊字符"}), 400

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        db.close()
        return jsonify({"error": "用户名已存在"}), 409

    # 生成唯一邀请码（8 位随机字母数字）
    invite_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    while db.execute("SELECT id FROM users WHERE invite_code = ?", (invite_code,)).fetchone():
        invite_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    # 处理邀请返利
    invited_by_code = (body.get("invite_code") or "").strip()
    invited_by_user_id = None
    if invited_by_code:
        inviter = db.execute("SELECT id FROM users WHERE invite_code = ?", (invited_by_code,)).fetchone()
        if inviter:
            invited_by_user_id = inviter["id"]

    # 生成邮箱验证令牌
    email_verify_token = secrets.token_urlsafe(32) if email else ""

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    # 首个注册用户自动成为管理员
    is_first = db.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"] == 0
    cursor = db.execute(
        "INSERT INTO users (username, password_hash, email, email_verified, email_verify_token, balance, is_admin, invite_code, invited_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (username, password_hash, email, 0, email_verify_token, 0, 1 if is_first else 0, invite_code, invited_by_user_id)
    )
    user_id = cursor.lastrowid

    # 发放邀请返利
    reward_new = 100  # 新用户奖励（分）
    reward_inviter = 200  # 邀请人奖励（分）
    if invited_by_user_id:
        db.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (reward_new, user_id))
        db.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (reward_inviter, invited_by_user_id))
        logger.info(f"[邀请] 新用户={user_id} 获 {reward_new} 分，邀请人={invited_by_user_id} 获 {reward_inviter} 分")

    db.commit()
    new_balance = db.execute("SELECT balance FROM users WHERE id = ?", (user_id,)).fetchone()["balance"]
    db.close()

    # Webhook 注册事件
    _emit_register_event(user_id, username)

    token = make_jwt(user_id, username, is_admin=is_first)
    return jsonify({"status": "ok", "token": token, "user": {"id": user_id, "username": username, "balance": new_balance, "is_admin": is_first}})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "用户名或密码错误"}), 401

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        db.close()
        return jsonify({"error": "用户名或密码错误"}), 401

    db.close()
    token = make_jwt(user["id"], user["username"], is_admin=bool(user["is_admin"]))
    return jsonify({
        "status": "ok", "token": token,
        "user": {
            "id": user["id"], "username": user["username"], "email": user["email"],
            "balance": user["balance"], "is_admin": bool(user["is_admin"]),
        }
    })


@app.route("/api/user/profile", methods=["GET"])
@require_user
def user_profile():
    db = get_db()
    user = db.execute("SELECT id, username, email, balance, is_admin, created_at FROM users WHERE id = ?",
                      (g.current_user["user_id"],)).fetchone()
    db.close()
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify({
        "id": user["id"], "username": user["username"], "email": user["email"],
        "balance": user["balance"], "is_admin": bool(user["is_admin"]),
        "created_at": user["created_at"],
    })


# ========== 邮箱验证 ==========

EMAIL_ENABLED = bool(os.getenv("SMTP_HOST", "").strip())

def _send_verify_email(to_email: str, token: str, username: str):
    """发送邮箱验证邮件。需要 SMTP 配置在 .env 中。"""
    if not EMAIL_ENABLED:
        logger.warning("SMTP 未配置，跳过发送验证邮件")
        return
    import smtplib
    from email.mime.text import MIMEText
    verify_url = f"{os.getenv('SITE_URL', 'http://localhost:5000')}/api/auth/verify-email?token={token}"
    msg = MIMEText(
        f"Hi {username},\n\n请点击以下链接验证您的邮箱：\n{verify_url}\n\n此链接 24 小时内有效。\n\nTokenHub 团队",
        "plain", "utf-8"
    )
    msg["Subject"] = "TokenHub - 邮箱验证"
    msg["From"] = os.getenv("SMTP_FROM", "noreply@tokenhub.com")
    msg["To"] = to_email
    try:
        with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", 465)), timeout=10) as smtp:
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            smtp.send_message(msg)
        logger.info(f"验证邮件已发送至 {to_email}")
    except Exception as e:
        logger.error(f"发送验证邮件失败: {e}")

@app.route("/api/auth/verify-email", methods=["GET"])
def verify_email():
    """通过 URL token 验证邮箱。"""
    token = request.args.get("token", "")
    if not token:
        return jsonify({"error": "缺少验证令牌"}), 400
    db = get_db()
    user = db.execute("SELECT id, username, email_verify_token FROM users WHERE email_verify_token = ?", (token,)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "无效的验证令牌"}), 404
    db.execute("UPDATE users SET email_verified = 1, email_verify_token = '' WHERE id = ?", (user["id"],))
    db.commit()
    db.close()
    return jsonify({"status": "ok", "message": f"邮箱验证成功，{user['username']}！"})

@app.route("/api/auth/send-verify-email", methods=["POST"])
@require_user
def send_verify_email():
    """发送邮箱验证邮件。"""
    db = get_db()
    user = db.execute("SELECT username, email, email_verified, email_verify_token FROM users WHERE id = ?",
                      (g.current_user["user_id"],)).fetchone()
    if not user or not user["email"]:
        db.close()
        return jsonify({"error": "未设置邮箱"}), 400
    if user["email_verified"]:
        db.close()
        return jsonify({"status": "ok", "message": "邮箱已验证"})
    token = user["email_verify_token"] or secrets.token_urlsafe(32)
    if not user["email_verify_token"]:
        db.execute("UPDATE users SET email_verify_token = ? WHERE id = ?", (token, g.current_user["user_id"]))
        db.commit()
    db.close()
    _send_verify_email(user["email"], token, user["username"])
    return jsonify({"status": "ok", "message": "验证邮件已发送"})

# ========== 忘记密码 ==========

@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    """发送密码重置邮件。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "请输入邮箱地址"}), 400

    db = get_db()
    user = db.execute("SELECT id, username, email FROM users WHERE email = ?", (email,)).fetchone()
    db.close()
    if not user:
        return jsonify({"status": "ok", "message": "如果该邮箱已注册，重置邮件已发送"})

    reset_payload = {
        "user_id": user["id"],
        "username": user["username"],
        "is_admin": False,
        "exp": int(time.time()) + 3600,
    }
    reset_token = jwt.encode(reset_payload, JWT_SECRET, algorithm="HS256")
    if EMAIL_ENABLED:
        import smtplib
        from email.mime.text import MIMEText
        reset_url = f"{os.getenv('SITE_URL', 'http://localhost:5000')}/#/reset-password?token={reset_token}"
        msg = MIMEText(
            f"Hi {user['username']},\n\n请点击以下链接重置密码：\n{reset_url}\n\n此链接 1 小时内有效。\n\nTokenHub 团队",
            "plain", "utf-8"
        )
        msg["Subject"] = "TokenHub - 密码重置"
        msg["From"] = os.getenv("SMTP_FROM", "noreply@tokenhub.com")
        msg["To"] = email
        try:
            with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", 465)), timeout=10) as smtp:
                smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
                smtp.send_message(msg)
            logger.info(f"密码重置邮件已发送至 {email}")
        except Exception as e:
            logger.error(f"发送重置邮件失败: {e}")
            return jsonify({"error": "发送邮件失败，请稍后重试"}), 500
    return jsonify({"status": "ok", "message": "如果该邮箱已注册，重置邮件已发送"})

@app.route("/api/auth/reset-password", methods=["POST"])
def reset_password():
    """重置密码。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400
    token = (body.get("token") or "").strip()
    new_password = (body.get("new_password") or "").strip()
    if not token or not new_password:
        return jsonify({"error": "令牌和新密码不能为空"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "密码至少 8 个字符"}), 400
    if not re.search(r"[A-Z]", new_password):
        return jsonify({"error": "密码必须包含至少一个大写字母"}), 400
    if not re.search(r"[a-z]", new_password):
        return jsonify({"error": "密码必须包含至少一个小写字母"}), 400
    if not re.search(r"\d", new_password):
        return jsonify({"error": "密码必须包含至少一个数字"}), 400
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", new_password):
        return jsonify({"error": "密码必须包含至少一个特殊字符"}), 400

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "令牌已过期"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "无效的令牌"}), 401

    db = get_db()
    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, payload["user_id"]))
    db.commit()
    db.close()
    return jsonify({"status": "ok", "message": "密码重置成功"})

# ========== API Key 管理 ==========

@app.route("/api/user/keys", methods=["GET"])
@require_user
def user_keys_list():
    db = get_db()
    keys = db.execute(
        "SELECT id, name, is_active, allowed_models, key_prefix, created_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
        (g.current_user["user_id"],)
    ).fetchall()
    db.close()
    return jsonify({"keys": [{
        "id": k["id"], "name": k["name"], "is_active": bool(k["is_active"]),
        "allowed_models": (k["allowed_models"] or "").strip(),
        "created_at": k["created_at"],
        "key_preview": k["key_prefix"] or f"sk-...{k['id']:04x}"
    } for k in keys]})


@app.route("/api/user/keys", methods=["POST"])
@require_user
def user_keys_create():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "Default").strip()
    allowed_models = (body.get("allowed_models") or "").strip()

    # 白名单去重、去空
    if allowed_models:
        parts = [m.strip() for m in allowed_models.replace("，", ",").split(",") if m.strip()]
        allowed_models = ",".join(sorted(set(parts)))

    # 生成 API Key: sk- + 随机 48 字符
    key_value = "sk-" + secrets.token_hex(24)
    key_prefix = key_value[:12]  # 保存明文前缀供 UI 展示
    key_hash = hashlib.sha256(key_value.encode()).hexdigest()  # SHA256 hash 用于快速查找

    db = get_db()
    cursor = db.execute(
        "INSERT INTO api_keys (user_id, key_value, key_prefix, name, allowed_models) VALUES (?, ?, ?, ?, ?)",
        (g.current_user["user_id"], key_hash, key_prefix, name, allowed_models)
    )
    key_id = cursor.lastrowid
    db.commit()
    db.close()

    return jsonify({
        "status": "ok",
        "key": {
            "id": key_id, "name": name, "key_value": key_value,
            "allowed_models": allowed_models,
            "created_at": int(time.time()),
        },
        "warning": "请立即复制保存 API Key，此后将不再展示完整 Key 值。"
    })


@app.route("/api/user/keys/<int:key_id>", methods=["DELETE"])
@require_user
def user_keys_delete(key_id):
    db = get_db()
    key = db.execute(
        "SELECT id FROM api_keys WHERE id = ? AND user_id = ?",
        (key_id, g.current_user["user_id"])
    ).fetchone()
    if not key:
        db.close()
        return jsonify({"error": "Key 不存在"}), 404

    db.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
    db.commit()
    db.close()
    return jsonify({"status": "ok", "message": "API Key 已删除"})


# ========== 用量查询 ==========

@app.route("/api/user/usage", methods=["GET"])
@require_user
def user_usage():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    per_page = min(per_page, 100)
    offset = (page - 1) * per_page

    db = get_db()
    total = db.execute("SELECT COUNT(*) as cnt FROM usage_records WHERE user_id = ?",
                       (g.current_user["user_id"],)).fetchone()["cnt"]
    records = db.execute(
        "SELECT id, model, backend, prompt_tokens, completion_tokens, cost, created_at "
        "FROM usage_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (g.current_user["user_id"], per_page, offset)
    ).fetchall()
    db.close()

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "records": [{
            "id": r["id"], "model": r["model"], "backend": r["backend"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            "cost": r["cost"], "created_at": r["created_at"],
        } for r in records]
    })


# ========== 定价查询 ==========

@app.route("/api/pricing", methods=["GET"])
def get_pricing():
    return jsonify({"pricing": PRICING})


# ========== 管理 API（从 api_router.py 迁移） ==========

_KEY_MAP = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
}


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "(未设置)"
    return key[:6] + "..." + key[-4:]


def _mask_key_str(key_id: int) -> str:
    return f"sk-...{key_id:04x}"


@app.route("/admin/keys", methods=["GET"])
@require_admin
def admin_get_keys():
    keys = {}
    for name, (key_env, url_env) in _KEY_MAP.items():
        keys[name] = {
            "api_key": _mask_key(os.getenv(key_env, "")),
            "base_url": os.getenv(url_env, ""),
            "configured": bool(os.getenv(key_env, "")),
        }
    return jsonify({"keys": keys})


@app.route("/admin/keys/<name>", methods=["GET"])
@require_admin
def admin_get_key_status(name):
    if name not in _KEY_MAP:
        return jsonify({"error": f"未知后端: {name}"}), 404
    key_env, _ = _KEY_MAP[name]
    return jsonify({"name": name, "configured": bool(os.getenv(key_env, ""))})


@app.route("/admin/keys", methods=["POST"])
@require_admin
def admin_update_keys():
    body = request.get_json(silent=True)
    if not body or "keys" not in body:
        return jsonify({"error": "缺少 'keys' 字段"}), 400

    keys_data = body["keys"]
    updated = []
    for name, kv in keys_data.items():
        if name not in _KEY_MAP:
            continue
        key_env, url_env = _KEY_MAP[name]
        api_key = kv.get("api_key", "")
        base_url = kv.get("base_url", "")
        if api_key:
            set_key(ENV_FILE, key_env, api_key)
            os.environ[key_env] = api_key
            BACKENDS[name]["api_key"] = api_key
            updated.append(f"{name} API Key")
        if base_url:
            set_key(ENV_FILE, url_env, base_url)
            os.environ[url_env] = base_url
            BACKENDS[name]["base_url"] = base_url
            updated.append(f"{name} Base URL")

    load_dotenv(override=True)
    logger.info(f"管理: 已更新 {', '.join(updated) if updated else '无变更'}")
    return jsonify({"status": "ok", "updated": updated})


@app.route("/admin/routes", methods=["GET"])
@require_admin
def admin_get_routes():
    routes = []
    for name, cfg in BACKENDS.items():
        for prefix in cfg["prefixes"]:
            routes.append({"model": prefix, "prefix": prefix, "backend": name})
    return jsonify({"routes": routes})


@app.route("/admin/routes", methods=["POST"])
def admin_add_route():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON"}), 400
    model = body.get("model", "").strip()
    backend = body.get("backend", "").strip()
    if not model or not backend:
        return jsonify({"error": "缺少 'model' 或 'backend' 字段"}), 400
    if backend not in BACKENDS:
        return jsonify({"error": f"未知后端: {backend}"}), 400

    if model.endswith("-*") or model.endswith("*"):
        prefix = model.rstrip("*")
    elif "-" in model:
        prefix = model.rsplit("-", 1)[0] + "-"
    else:
        prefix = model + "-"

    if prefix not in BACKENDS[backend]["prefixes"]:
        BACKENDS[backend]["prefixes"].append(prefix)
        logger.info(f"管理: 为 {backend} 添加路由前缀 '{prefix}'")
    return jsonify({"status": "ok", "prefix": prefix, "backend": backend})


@app.route("/admin/routes", methods=["DELETE"])
@require_admin
def admin_delete_route():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON"}), 400
    model = body.get("model", "").strip()
    backend = body.get("backend", "").strip()
    if not model or not backend:
        return jsonify({"error": "缺少 'model' 或 'backend' 字段"}), 400
    if backend not in BACKENDS:
        return jsonify({"error": f"未知后端: {backend}"}), 400

    if model.endswith("-*") or model.endswith("*"):
        prefix = model.rstrip("*")
    elif "-" in model:
        prefix = model.rsplit("-", 1)[0] + "-"
    else:
        prefix = model + "-"

    if prefix in BACKENDS[backend]["prefixes"]:
        BACKENDS[backend]["prefixes"].remove(prefix)
        logger.info(f"管理: 从 {backend} 删除路由前缀 '{prefix}'")
    return jsonify({"status": "ok", "prefix": prefix, "backend": backend})


# ========== 支付模块 ==========

# 充值套餐
RECHARGE_PLANS = [
    {"amount": 1000,  "label": "10元",   "desc": "适合轻度体验"},
    {"amount": 5000,  "label": "50元",   "desc": "适合日常使用"},
    {"amount": 10000, "label": "100元",  "desc": "适合高频调用"},
    {"amount": 50000, "label": "500元",  "desc": "适合企业用户"},
]


def _generate_order_no():
    """生成唯一订单号：TH + 时间戳(秒) + 随机6位字母数字"""
    rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"TH{int(time.time())}{rand}"


def _get_payment_config():
    """从 .env 读取支付配置。"""
    return {
        "alipay": {
            "app_id": os.getenv("ALIPAY_APP_ID", ""),
            "private_key": os.getenv("ALIPAY_PRIVATE_KEY", ""),
            "public_key": os.getenv("ALIPAY_PUBLIC_KEY", ""),
            "configured": bool(os.getenv("ALIPAY_APP_ID", "")),
        },
        "wechat": {
            "mch_id": os.getenv("WECHAT_MCH_ID", ""),
            "api_key": os.getenv("WECHAT_API_KEY", ""),
            "app_id": os.getenv("WECHAT_APP_ID", ""),
            "configured": bool(os.getenv("WECHAT_MCH_ID", "")),
        },
    }


@app.route("/api/recharge/plans", methods=["GET"])
def recharge_plans():
    """返回预设充值套餐列表。"""
    return jsonify({"plans": RECHARGE_PLANS})


@app.route("/api/payment/create", methods=["POST"])
@require_user
def payment_create():
    """用户创建充值订单。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    amount = body.get("amount", 0)
    method = (body.get("method") or "alipay").lower()

    if amount <= 0:
        return jsonify({"error": "充值金额必须大于 0"}), 400
    if method not in ("alipay", "wechat"):
        return jsonify({"error": "支付方式仅支持 alipay 或 wechat"}), 400

    order_no = _generate_order_no()
    user_id = g.current_user["user_id"]

    # 生成模拟支付 URL（本地环境无法调用真实支付 API）
    # 真实环境中这里应调用支付宝/微信的统一下单接口，获取支付链接或二维码链接
    base_url = f"{request.scheme}://{request.host}"
    pay_url = f"{base_url}/api/payment/mock/pay?order_no={order_no}&amount={amount}&method={method}"

    db = get_db()
    cursor = db.execute(
        "INSERT INTO recharge_records (user_id, amount, method, status, order_no, pay_url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, amount, method, "pending", order_no, pay_url)
    )
    record_id = cursor.lastrowid
    db.commit()
    db.close()

    logger.info(f"[支付] 创建订单: user_id={user_id}, order_no={order_no}, amount={amount}, method={method}")
    return jsonify({
        "status": "ok",
        "order_no": order_no,
        "amount": amount,
        "method": method,
        "pay_url": pay_url,
        "record_id": record_id,
    })


@app.route("/api/payment/status/<order_no>", methods=["GET"])
@require_user
def payment_status(order_no):
    """查询订单支付状态。"""
    db = get_db()
    record = db.execute(
        "SELECT id, user_id, amount, method, status, order_no, created_at "
        "FROM recharge_records WHERE order_no = ?", (order_no,)
    ).fetchone()
    db.close()

    if not record:
        return jsonify({"error": "订单不存在"}), 404

    # 普通用户只能查自己的订单
    if record["user_id"] != g.current_user["user_id"] and not g.current_user.get("is_admin"):
        return jsonify({"error": "无权查看该订单"}), 403

    return jsonify({
        "order_no": record["order_no"],
        "amount": record["amount"],
        "method": record["method"],
        "status": record["status"],
        "created_at": record["created_at"],
    })


@app.route("/api/payment/notify/alipay", methods=["POST"])
def payment_notify_alipay():
    """支付宝异步通知回调（模拟）。
    真实环境中需验证支付宝签名，然后更新订单状态。
    """
    # 支付宝通知参数为 form-data
    data = request.form.to_dict() if request.form else request.get_json(silent=True) or {}
    order_no = data.get("out_trade_no", "")
    trade_status = data.get("trade_status", "")

    if not order_no:
        return "failure", 400

    db = get_db()
    record = db.execute(
        "SELECT id, user_id, amount, status FROM recharge_records WHERE order_no = ? AND method = 'alipay'",
        (order_no,)
    ).fetchone()

    if not record:
        db.close()
        return "failure"

    if record["status"] == "completed":
        db.close()
        return "success"

    # 真实环境需验证签名 + 检查 trade_status == "TRADE_SUCCESS"
    # 此处模拟：直接标记为已完成
    db.execute(
        "UPDATE recharge_records SET status = 'completed' WHERE order_no = ?",
        (order_no,)
    )
    db.execute(
        "UPDATE users SET balance = balance + ? WHERE id = ?",
        (record["amount"], record["user_id"])
    )
    db.commit()
    # Webhook 充值事件
    _emit_recharge_event(record["user_id"], record["amount"], "alipay")
    logger.info(f"[支付回调] 支付宝订单完成: order_no={order_no}, user_id={record['user_id']}, amount={record['amount']}")
    db.close()

    return "success"


@app.route("/api/payment/notify/wechat", methods=["POST"])
def payment_notify_wechat():
    """微信支付异步通知回调（模拟）。
    真实环境中需验证微信签名，解析 XML，然后更新订单状态。
    """
    # 微信通知为 XML 格式
    data = request.get_json(silent=True) or {}
    order_no = data.get("out_trade_no", "")
    result_code = data.get("result_code", "")

    if not order_no:
        return "<xml><return_code><![CDATA[FAIL]]></return_code></xml>", 400

    db = get_db()
    record = db.execute(
        "SELECT id, user_id, amount, status FROM recharge_records WHERE order_no = ? AND method = 'wechat'",
        (order_no,)
    ).fetchone()

    if not record:
        db.close()
        return "<xml><return_code><![CDATA[FAIL]]></return_code></xml>"

    if record["status"] == "completed":
        db.close()
        return "<xml><return_code><![CDATA[SUCCESS]]></return_code></xml>"

    # 真实环境需验证签名 + 检查 result_code == "SUCCESS"
    # 此处模拟：直接标记为已完成
    db.execute(
        "UPDATE recharge_records SET status = 'completed' WHERE order_no = ?",
        (order_no,)
    )
    db.execute(
        "UPDATE users SET balance = balance + ? WHERE id = ?",
        (record["amount"], record["user_id"])
    )
    db.commit()
    # Webhook 充值事件
    _emit_recharge_event(record["user_id"], record["amount"], "wechat")
    logger.info(f"[支付回调] 微信订单完成: order_no={order_no}, user_id={record['user_id']}, amount={record['amount']}")
    db.close()

    return "<xml><return_code><![CDATA[SUCCESS]]></return_code></xml>"


@app.route("/api/payment/mock/complete/<order_no>", methods=["POST"])
@require_admin
def payment_mock_complete(order_no):
    """管理员模拟完成支付（开发测试用）。"""
    db = get_db()
    record = db.execute(
        "SELECT id, user_id, amount, status FROM recharge_records WHERE order_no = ?",
        (order_no,)
    ).fetchone()

    if not record:
        db.close()
        return jsonify({"error": "订单不存在"}), 404

    if record["status"] == "completed":
        db.close()
        return jsonify({"error": "订单已完成，无需重复操作"}), 400

    db.execute(
        "UPDATE recharge_records SET status = 'completed' WHERE order_no = ?",
        (order_no,)
    )
    db.execute(
        "UPDATE users SET balance = balance + ? WHERE id = ?",
        (record["amount"], record["user_id"])
    )
    db.commit()
    db.close()

    # Webhook 充值事件（手动补单）
    _emit_recharge_event(record["user_id"], record["amount"], "manual")
    logger.info(f"[模拟支付] 订单完成: order_no={order_no}, admin={g.current_user['username']}")
    return jsonify({"status": "ok", "message": f"订单 {order_no} 已模拟支付完成，用户余额已增加 {record['amount']/100:.2f} 元"})


@app.route("/api/admin/payment_config", methods=["GET"])
@require_admin
def payment_config_get():
    """管理员查看支付配置（敏感信息脱敏）。"""
    cfg = _get_payment_config()
    # 脱敏处理
    if cfg["alipay"]["private_key"]:
        cfg["alipay"]["private_key"] = "******"
    if cfg["alipay"]["public_key"]:
        cfg["alipay"]["public_key"] = "******"
    if cfg["wechat"]["api_key"]:
        cfg["wechat"]["api_key"] = "******"
    return jsonify({"config": cfg})


@app.route("/api/admin/payment_config", methods=["POST"])
@require_admin
def payment_config_set():
    """管理员配置支付参数，写入 .env 文件。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    updated = []
    alipay = body.get("alipay", {})
    wechat = body.get("wechat", {})

    if "app_id" in alipay and alipay["app_id"]:
        set_key(ENV_FILE, "ALIPAY_APP_ID", alipay["app_id"])
        os.environ["ALIPAY_APP_ID"] = alipay["app_id"]
        updated.append("支付宝 APP ID")
    if "private_key" in alipay and alipay["private_key"]:
        set_key(ENV_FILE, "ALIPAY_PRIVATE_KEY", alipay["private_key"])
        os.environ["ALIPAY_PRIVATE_KEY"] = alipay["private_key"]
        updated.append("支付宝私钥")
    if "public_key" in alipay and alipay["public_key"]:
        set_key(ENV_FILE, "ALIPAY_PUBLIC_KEY", alipay["public_key"])
        os.environ["ALIPAY_PUBLIC_KEY"] = alipay["public_key"]
        updated.append("支付宝公钥")

    if "mch_id" in wechat and wechat["mch_id"]:
        set_key(ENV_FILE, "WECHAT_MCH_ID", wechat["mch_id"])
        os.environ["WECHAT_MCH_ID"] = wechat["mch_id"]
        updated.append("微信商户号")
    if "api_key" in wechat and wechat["api_key"]:
        set_key(ENV_FILE, "WECHAT_API_KEY", wechat["api_key"])
        os.environ["WECHAT_API_KEY"] = wechat["api_key"]
        updated.append("微信 API 密钥")
    if "app_id" in wechat and wechat["app_id"]:
        set_key(ENV_FILE, "WECHAT_APP_ID", wechat["app_id"])
        os.environ["WECHAT_APP_ID"] = wechat["app_id"]
        updated.append("微信 App ID")

    load_dotenv(override=True)
    logger.info(f"[支付配置] 管理员 {g.current_user['username']} 更新了: {', '.join(updated) if updated else '无变更'}")
    return jsonify({"status": "ok", "updated": updated})


@app.route("/api/payment/mock/pay", methods=["GET"])
def payment_mock_pay_page():
    """模拟支付确认页面（仅开发测试使用）。"""
    order_no = request.args.get("order_no", "")
    method = request.args.get("method", "alipay")
    amount = request.args.get("amount", "0")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>模拟支付 - TokenHub</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0a0f;color:#e4e4ed;display:flex;justify-content:center;align-items:center;min-height:100vh}}
.card{{background:#16161e;border:1px solid #252538;border-radius:12px;padding:40px;max-width:400px;width:90%;text-align:center}}
h2{{font-size:20px;margin-bottom:16px}}
.order{{font-size:13px;color:#606078;margin-bottom:8px;word-break:break-all}}
.amount{{font-size:36px;font-weight:800;margin:16px 0;color:#22c55e}}
.method{{display:inline-block;padding:4px 12px;border-radius:99px;font-size:12px;font-weight:600;background:rgba(99,102,241,0.12);color:#6366f1;margin-bottom:20px}}
.btn{{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:12px 28px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:all 0.2s;font-family:inherit}}
.btn-success{{background:#22c55e;color:#fff}}
.btn-success:hover{{background:#16a34a}}
.btn-secondary{{background:transparent;color:#9898b0;border:1px solid #252538;margin-left:8px}}
.btn-secondary:hover{{color:#e4e4ed}}
.note{{font-size:12px;color:#606078;margin-top:20px}}
</style>
</head>
<body>
<div class="card">
  <h2>模拟支付确认</h2>
  <div class="order">订单号: {order_no}</div>
  <div class="amount">¥{int(amount)/100:.2f}</div>
  <div class="method">{'支付宝' if method=='alipay' else '微信支付'}</div>
  <div>
    <button class="btn btn-success" onclick="confirmPay()">确认支付</button>
    <button class="btn btn-secondary" onclick="window.close()">取消</button>
  </div>
  <div class="note">这是模拟支付页面，仅用于开发测试。<br>真实环境将跳转到{ '支付宝' if method=='alipay' else '微信支付' }收银台。</div>
</div>
<script>
async function confirmPay() {{
  try {{
    const notifyUrl = '{"/api/payment/notify/alipay" if method=="alipay" else "/api/payment/notify/wechat"}';
    const body = new URLSearchParams();
    body.append('out_trade_no', '{order_no}');
    body.append('trade_status', 'TRADE_SUCCESS');
    body.append('result_code', 'SUCCESS');
    const res = await fetch(notifyUrl, {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
    if (res.ok) {{ alert('支付成功！'); window.close(); }}
    else {{ alert('支付失败'); }}
  }} catch(e) {{ alert('请求失败: ' + e.message); }}
}}
</script>
</body></html>"""


@app.route("/api/user/recharge_records", methods=["GET"])
@require_user
def user_recharge_records():
    """查看充值记录。普通用户看自己的，管理员看全部。"""
    db = get_db()

    if g.current_user.get("is_admin"):
        records = db.execute(
            "SELECT r.id, r.user_id, u.username, r.amount, r.method, r.status, r.order_no, r.created_at "
            "FROM recharge_records r LEFT JOIN users u ON r.user_id = u.id "
            "ORDER BY r.created_at DESC LIMIT 200"
        ).fetchall()
    else:
        records = db.execute(
            "SELECT id, user_id, amount, method, status, order_no, created_at "
            "FROM recharge_records WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
            (g.current_user["user_id"],)
        ).fetchall()

    db.close()
    return jsonify({
        "records": [dict(r) for r in records]
    })


# ========== 模型广场 ==========

MODEL_META = {
    # OpenAI
    "gpt-4o": {
        "provider": "OpenAI", "context_length": 128000,
        "modalities": ["text", "image"],
        "description": "GPT-4o 是 OpenAI 最先进的多模态模型，支持文本和图像输入，在推理、创意写作和代码生成方面表现出色。",
    },
    "gpt-4o-mini": {
        "provider": "OpenAI", "context_length": 128000,
        "modalities": ["text", "image"],
        "description": "GPT-4o-mini 是 OpenAI 最具性价比的小型模型，速度快、成本低，适合日常对话和简单任务。",
    },
    "gpt-4.1": {
        "provider": "OpenAI", "context_length": 1000000,
        "modalities": ["text", "image"],
        "description": "GPT-4.1 是 OpenAI 最新旗舰模型，支持百万级上下文窗口，在编程和指令遵循方面大幅提升。",
    },
    "gpt-4.1-mini": {
        "provider": "OpenAI", "context_length": 1000000,
        "modalities": ["text", "image"],
        "description": "GPT-4.1-mini 是 GPT-4.1 的高效精简版，拥有百万级上下文窗口，兼具速度与性价比。",
    },
    "o4-mini": {
        "provider": "OpenAI", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "o4-mini 是 OpenAI o4 系列的高效推理模型，擅长复杂数学、编程和逻辑推理任务。",
    },
    "o3": {
        "provider": "OpenAI", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "o3 是 OpenAI 最强大的推理模型，在数学、科学和编程领域达到前沿水平。",
    },
    "o3-mini": {
        "provider": "OpenAI", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "o3-mini 是 o3 的精简推理版本，在保持推理能力的同时大幅降低成本和延迟。",
    },
    # Anthropic
    "claude-3-5-sonnet-20241022": {
        "provider": "Anthropic", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "Claude 3.5 Sonnet 是 Anthropic 的旗舰模型，在编码、写作和分析方面表现卓越，兼具速度与智能。",
    },
    "claude-3-opus-20240229": {
        "provider": "Anthropic", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "Claude 3 Opus 是 Anthropic 最强大的模型，擅长处理复杂的分析、长文档和多步骤任务。",
    },
    "claude-opus-4-20250514": {
        "provider": "Anthropic", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "Claude Opus 4 是 Anthropic 最新最强的模型，在复杂推理、长文档分析和编程方面全面领先。",
    },
    "claude-3-5-haiku-20241022": {
        "provider": "Anthropic", "context_length": 200000,
        "modalities": ["text", "image"],
        "description": "Claude 3.5 Haiku 是 Anthropic 最新一代快速模型，在保持极低延迟的同时显著提升智能水平。",
    },
    # Google
    "gemini-2.0-flash": {
        "provider": "Google", "context_length": 1048576,
        "modalities": ["text", "image", "audio"],
        "description": "Gemini 2.0 Flash 是 Google 最新的高效模型，拥有超长上下文窗口，适合处理和综合分析大量数据。",
    },
    "gemini-1.5-pro": {
        "provider": "Google", "context_length": 2097152,
        "modalities": ["text", "image", "video", "audio"],
        "description": "Gemini 1.5 Pro 是 Google 的中端多模态模型，支持文本、图像、音频和视频输入，上下文窗口可达 200 万 tokens。",
    },
    "gemini-2.5-pro": {
        "provider": "Google", "context_length": 1048576,
        "modalities": ["text", "image", "video", "audio"],
        "description": "Gemini 2.5 Pro 是 Google 最新的旗舰推理模型，在复杂推理、编程和多模态理解方面全面升级。",
    },
    "gemini-2.5-flash": {
        "provider": "Google", "context_length": 1048576,
        "modalities": ["text", "image", "video", "audio"],
        "description": "Gemini 2.5 Flash 是 Google 最新一代高效模型，兼具极快速度和强大推理能力，支持百万级上下文。",
    },
    "gemini-2.0-flash-lite": {
        "provider": "Google", "context_length": 1048576,
        "modalities": ["text", "image", "audio"],
        "description": "Gemini 2.0 Flash-Lite 是 Google 的超低成本模型，适合大规模简单任务和批量处理。",
    },
    # DeepSeek
    "deepseek-v3-0324": {
        "provider": "DeepSeek", "context_length": 128000,
        "modalities": ["text"],
        "description": "DeepSeek V3-0324 是 DeepSeek 最新版本对话模型，在前代基础上进一步提升了推理和编码能力。",
    },
    "deepseek-r1": {
        "provider": "DeepSeek", "context_length": 128000,
        "modalities": ["text"],
        "description": "DeepSeek R1 是深度求索的独立推理模型，专注于复杂数学、科学和逻辑推理场景。",
    },
    # Meta
    "llama-4-maverick": {
        "provider": "Meta", "context_length": 1000000,
        "modalities": ["text", "image"],
        "description": "Llama 4 Maverick 是 Meta 最新的多模态旗舰开源模型，支持百万级上下文和图像理解。",
    },
    "llama-4-scout": {
        "provider": "Meta", "context_length": 10000000,
        "modalities": ["text", "image"],
        "description": "Llama 4 Scout 拥有惊人的 1000 万上下文窗口，适合大规模文档分析和知识检索任务。",
    },
    "llama-3.3-70b": {
        "provider": "Meta", "context_length": 128000,
        "modalities": ["text"],
        "description": "Llama 3.3 70B 是 Meta 成熟的开源大语言模型，在通用任务上性能出色、稳定可靠。",
    },
    # Mistral
    "mistral-large": {
        "provider": "Mistral", "context_length": 128000,
        "modalities": ["text"],
        "description": "Mistral Large 是 Mistral AI 的顶级模型，擅长复杂推理、多语言任务和代码生成。",
    },
    "mistral-small": {
        "provider": "Mistral", "context_length": 32000,
        "modalities": ["text"],
        "description": "Mistral Small 是 Mistral AI 的高效轻量模型，适合日常对话和快速响应场景。",
    },
    "mixtral-8x22b": {
        "provider": "Mistral", "context_length": 64000,
        "modalities": ["text"],
        "description": "Mixtral 8x22B 是 Mistral AI 的混合专家模型，以稀疏激活实现高性能推理。",
    },
    "codestral": {
        "provider": "Mistral", "context_length": 256000,
        "modalities": ["text"],
        "description": "Codestral 是 Mistral AI 的专用编程模型，在代码生成、补全和重构方面表现出色。",
    },
    # Alibaba
    "qwen-max": {
        "provider": "Alibaba", "context_length": 128000,
        "modalities": ["text"],
        "description": "Qwen-Max 是阿里通义千问的最强模型，在中文理解、推理和创意写作方面表现出色。",
    },
    "qwen-plus": {
        "provider": "Alibaba", "context_length": 128000,
        "modalities": ["text"],
        "description": "Qwen-Plus 是阿里通义千问的中高端模型，兼具高效性和较强的理解能力。",
    },
    "qwen-turbo": {
        "provider": "Alibaba", "context_length": 128000,
        "modalities": ["text"],
        "description": "Qwen-Turbo 是阿里通义千问的极速版模型，延迟最低，适合高并发简单任务。",
    },
    # Other
    "command-r-plus": {
        "provider": "Cohere", "context_length": 128000,
        "modalities": ["text"],
        "description": "Command R+ 是 Cohere 的旗舰模型，专注于 RAG 和企业级检索增强生成场景。",
    },
    "jamba-1.5-large": {
        "provider": "AI21", "context_length": 256000,
        "modalities": ["text"],
        "description": "Jamba 1.5 Large 是 AI21 Labs 基于 Mamba-Transformer 混合架构的高效模型，支持长上下文。",
    },
    "dbrx-instruct": {
        "provider": "Databricks", "context_length": 32000,
        "modalities": ["text"],
        "description": "DBRX Instruct 是 Databricks 的混合专家开源模型，在编程和推理任务上表现出色。",
    },
}

MODALITY_ICONS = {
    "text": "T",
    "image": "IMG",
    "video": "VID",
    "audio": "AUD",
}


@app.route("/api/models", methods=["GET"])
def api_models():
    """返回模型列表，从定价表和路由表动态生成。"""
    models = []
    for m in DEFAULT_MODELS:
        model_id = m["id"]
        pricing = PRICING.get(model_id, DEFAULT_PRICING)
        meta = MODEL_META.get(model_id, {
            "provider": m.get("owned_by", "Unknown").title(),
            "context_length": 0,
            "modalities": ["text"],
            "description": "",
        })
        models.append({
            "id": model_id,
            "name": model_id,
            "provider": meta["provider"],
            "context_length": meta["context_length"],
            "input_price": pricing["input"],
            "output_price": pricing["output"],
            "modalities": meta["modalities"],
            "description": meta.get("description", ""),
        })
    return jsonify({"models": models})


# ========== 活动日志 ==========

@app.route("/api/user/activity", methods=["GET"])
@require_user
def user_activity():
    """最近的 API 调用记录，分页。"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 100)
    offset_val = (page - 1) * per_page

    db = get_db()
    if g.current_user.get("is_admin"):
        total = db.execute("SELECT COUNT(*) as cnt FROM usage_records").fetchone()["cnt"]
        records = db.execute(
            "SELECT id, user_id, model, backend, prompt_tokens, completion_tokens, cost, created_at "
            "FROM usage_records ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (per_page, offset_val)
        ).fetchall()
    else:
        total = db.execute(
            "SELECT COUNT(*) as cnt FROM usage_records WHERE user_id = ?",
            (g.current_user["user_id"],)
        ).fetchone()["cnt"]
        records = db.execute(
            "SELECT id, model, backend, prompt_tokens, completion_tokens, cost, created_at "
            "FROM usage_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (g.current_user["user_id"], per_page, offset_val)
        ).fetchall()

    db.close()

    def _status(cost_val):
        return "success" if cost_val > 0 else "error"

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "records": [{
            "id": r["id"],
            "model": r["model"],
            "backend": r["backend"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            "cost": r["cost"],
            "status": _status(r["cost"]),
            "created_at": r["created_at"],
        } for r in records]
    })


# ========== 用量统计（7天） ==========

@app.route("/api/user/stats", methods=["GET"])
@require_user
def user_stats():
    """近 7 天的每日用量统计。"""
    db = get_db()
    uid = g.current_user["user_id"]

    import datetime as dt_mod
    today = dt_mod.date.today()
    results = []
    for i in range(6, -1, -1):
        day = today - dt_mod.timedelta(days=i)
        day_str = day.isoformat()
        start_ts = f"{day_str} 00:00:00"
        end_ts = f"{day_str} 23:59:59"

        row = db.execute(
            "SELECT COUNT(*) as requests, COALESCE(SUM(cost), 0) as cost "
            "FROM usage_records WHERE user_id = ? AND created_at BETWEEN ? AND ?",
            (uid, start_ts, end_ts)
        ).fetchone()

        results.append({
            "date": day_str,
            "requests": row["requests"] or 0,
            "cost": row["cost"] or 0,
        })

    db.close()
    return jsonify({"daily": results})


# ========== 修改密码 ==========

@app.route("/api/user/change_password", methods=["POST"])
@require_user
def change_password():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    old_password = (body.get("old_password") or "").strip()
    new_password = (body.get("new_password") or "").strip()

    if not old_password or not new_password:
        return jsonify({"error": "旧密码和新密码不能为空"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "新密码至少 8 个字符"}), 400
    if not re.search(r"[A-Z]", new_password):
        return jsonify({"error": "密码必须包含至少一个大写字母"}), 400
    if not re.search(r"[a-z]", new_password):
        return jsonify({"error": "密码必须包含至少一个小写字母"}), 400
    if not re.search(r"\d", new_password):
        return jsonify({"error": "密码必须包含至少一个数字"}), 400
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", new_password):
        return jsonify({"error": "密码必须包含至少一个特殊字符"}), 400

    db = get_db()
    user = db.execute("SELECT password_hash FROM users WHERE id = ?",
                      (g.current_user["user_id"],)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "用户不存在"}), 404

    if not bcrypt.checkpw(old_password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        db.close()
        return jsonify({"error": "旧密码错误"}), 401

    new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
               (new_hash, g.current_user["user_id"]))
    db.commit()
    db.close()
    return jsonify({"status": "ok", "message": "密码修改成功"})


# ========== 管理员用户管理 ==========

@app.route("/api/admin/users", methods=["GET"])
@require_admin
def admin_users():
    db = get_db()
    users = db.execute(
        "SELECT id, username, email, balance, is_admin, created_at FROM users ORDER BY id"
    ).fetchall()
    db.close()
    return jsonify({
        "users": [{
            "id": u["id"], "username": u["username"], "email": u["email"],
            "balance": u["balance"], "is_admin": bool(u["is_admin"]),
            "created_at": u["created_at"],
        } for u in users]
    })


@app.route("/api/admin/users/<int:user_id>/toggle_admin", methods=["POST"])
@require_admin
def toggle_admin(user_id):
    db = get_db()
    user = db.execute("SELECT id, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "用户不存在"}), 404
    new_val = 0 if user["is_admin"] else 1
    db.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_val, user_id))
    db.commit()
    db.close()
    return jsonify({"status": "ok", "is_admin": bool(new_val)})


@app.route("/api/admin/users/<int:user_id>/recharge", methods=["POST"])
@require_admin
def admin_recharge(user_id):
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400
    try:
        amount = int(body.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "金额必须为整数"}), 400
    if amount <= 0:
        return jsonify({"error": "金额必须大于 0"}), 400

    db = get_db()
    user = db.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "用户不存在"}), 404
    db.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (amount, user_id))
    db.commit()
    new_balance = db.execute("SELECT balance FROM users WHERE id = ?", (user_id,)).fetchone()["balance"]
    db.close()
    return jsonify({"status": "ok", "balance": new_balance, "message": f"已为 {user['username']} 充值 {amount/100:.2f} 元"})


# ========== 健康检查增强 ==========

@app.route("/api/health/detail", methods=["GET"])
def health_detail():
    """检测各后端 API 连通性。"""
    status = {}
    for name, cfg in BACKENDS.items():
        if not cfg["api_key"]:
            status[name] = {"online": False, "message": "未配置 API Key"}
            continue
        headers = {}
        if name == "openai":
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
            url = f"{cfg['base_url'].rstrip('/')}/models"
        elif name == "anthropic":
            headers["x-api-key"] = cfg["api_key"]
            headers["anthropic-version"] = "2023-06-01"
            url = f"{cfg['base_url'].rstrip('/')}/models"
        elif name == "gemini":
            url = f"{cfg['base_url'].rstrip('/')}/models?key={cfg['api_key']}"
        elif name == "deepseek":
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
            url = f"{cfg['base_url'].rstrip('/')}/models"
        else:
            status[name] = {"online": False, "message": "未知后端"}
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            status[name] = {"online": resp.status_code < 500, "code": resp.status_code}
        except requests.RequestException as e:
            status[name] = {"online": False, "message": str(e)[:100]}
    return jsonify({"service": "ai-model-router", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "backends": status})

# ========== 模型管理 ==========

@app.route("/api/admin/models", methods=["GET"])
@require_admin
def admin_models():
    db = get_db()
    disabled = {r["model"] for r in db.execute("SELECT model FROM disabled_models").fetchall()}
    breakers = {}
    for r in db.execute("SELECT model, is_open, opened_at FROM circuit_breaker_state").fetchall():
        breakers[r["model"]] = r
    db.close()
    models = []
    for m in DEFAULT_MODELS:
        entry = dict(m)
        br = breakers.get(m["id"])
        entry["enabled"] = m["id"] not in disabled
        entry["degraded"] = bool(br and br["is_open"] and time.time() - br["opened_at"] < BREAKER_TIMEOUT)
        models.append(entry)
    return jsonify({"models": models})

@app.route("/api/admin/models/<model_id>/toggle", methods=["POST"])
@require_admin
def admin_model_toggle(model_id):
    valid_ids = {m["id"] for m in DEFAULT_MODELS}
    if model_id not in valid_ids:
        return jsonify({"error": "模型不存在"}), 404
    db = get_db()
    exists = db.execute("SELECT 1 FROM disabled_models WHERE model = ?", (model_id,)).fetchone()
    if exists:
        db.execute("DELETE FROM disabled_models WHERE model = ?", (model_id,))
        db.commit()
        db.close()
        logger.info(f"管理: 启用模型 {model_id}")
        return jsonify({"status": "ok", "enabled": True})
    else:
        db.execute("INSERT OR IGNORE INTO disabled_models (model) VALUES (?)", (model_id,))
        db.commit()
        db.close()
        logger.info(f"管理: 禁用模型 {model_id}")
        return jsonify({"status": "ok", "enabled": False})

# ========== 管理员监控面板 ==========

@app.route("/api/admin/stats", methods=["GET"])
@require_admin
def admin_stats():
    db = get_db()
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    today = time.strftime("%Y-%m-%d", time.localtime())

    # 总用户数
    total_users = db.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]

    # 今日新增
    today_new = db.execute(
        "SELECT COUNT(*) as cnt FROM users WHERE date(created_at) = ?", (today,)
    ).fetchone()["cnt"]

    # 今日调用次数 & 今日收入
    today_usage = db.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(cost), 0) as income "
        "FROM usage_records WHERE date(created_at) = ?", (today,)
    ).fetchone()
    today_calls = today_usage["cnt"]
    today_income = today_usage["income"]

    # 总调用次数 & 总收入
    total_usage = db.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(cost), 0) as income FROM usage_records"
    ).fetchone()
    total_calls = total_usage["cnt"]
    total_income = total_usage["income"]

    # 24h 在线 API Key 数
    online_keys = db.execute(
        "SELECT COUNT(DISTINCT api_key_id) as cnt FROM usage_records "
        "WHERE created_at >= datetime('now', '-1 day')"
    ).fetchone()["cnt"]

    # 近 7 天调用量
    daily_stats = []
    for i in range(6, -1, -1):
        d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
        row = db.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(cost), 0) as income "
            "FROM usage_records WHERE date(created_at) = ?", (d,)
        ).fetchone()
        daily_stats.append({"date": d, "calls": row["cnt"], "income": row["income"]})

    db.close()
    return jsonify({
        "total_users": total_users,
        "today_new_users": today_new,
        "today_calls": today_calls,
        "today_income": today_income,
        "total_calls": total_calls,
        "total_income": total_income,
        "online_keys": online_keys,
        "daily": daily_stats,
    })


# ========== 用量记录导出 ==========

@app.route("/api/user/usage_records/export", methods=["GET"])
@require_user
def export_usage_csv():
    db = get_db()
    user_id = g.current_user["user_id"]
    username = g.current_user.get("username", "user")

    # 获取用户全量用量记录
    records = db.execute(
        "SELECT model, backend, prompt_tokens, completion_tokens, cost, created_at "
        "FROM usage_records WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    db.close()

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["时间", "模型", "后端", "输入 Token", "输出 Token", "费用（分）"])
    for r in records:
        writer.writerow([
            r["created_at"], r["model"], r["backend"],
            r["prompt_tokens"], r["completion_tokens"], r["cost"]
        ])

    csv_content = output.getvalue()
    output.close()

    date_str = time.strftime("%Y%m%d", time.localtime())
    filename = f"usage_{username}_{date_str}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )



# ========== 套餐订阅 ==========

SUBSCRIPTION_PLANS = {
    "monthly":   {"price": 2900,  "quota": 10000, "days": 30, "label": "月费套餐"},
    "quarterly": {"price": 7900,  "quota": 35000, "days": 90, "label": "季费套餐"},
    "yearly":    {"price": 29900, "quota": 150000, "days": 365, "label": "年费套餐"},
}


@app.route("/api/subscribe", methods=["POST"])
@require_user
def subscribe():
    """订阅套餐，扣除余额创建订阅记录。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400

    plan = (body.get("plan") or "").strip().lower()
    if plan not in SUBSCRIPTION_PLANS:
        return jsonify({"error": "无效套餐，可选 monthly / quarterly / yearly"}), 400

    plan_info = SUBSCRIPTION_PLANS[plan]
    price = plan_info["price"]
    quota = plan_info["quota"]
    days = plan_info["days"]
    user_id = g.current_user["user_id"]

    db = get_db()

    # 检查余额
    user = db.execute("SELECT balance FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        db.close()
        return jsonify({"error": "用户不存在"}), 404
    if user["balance"] < price:
        db.close()
        return jsonify({"error": f"余额不足，需要 {price/100:.2f} 元，当前余额 {user['balance']/100:.2f} 元"}), 402

    # 停用旧套餐
    db.execute("UPDATE subscriptions SET status = 'cancelled' WHERE user_id = ? AND status = 'active'", (user_id,))

    # 计算到期时间
    now_dt = datetime.now()
    expires_dt = now_dt + timedelta(days=days)
    expires_str = expires_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 创建订阅
    cursor = db.execute(
        "INSERT INTO subscriptions (user_id, plan, model_quota, calls_used, status, started_at, expires_at) "
        "VALUES (?, ?, ?, 0, 'active', ?, ?)",
        (user_id, plan, quota, now_dt.strftime("%Y-%m-%d %H:%M:%S"), expires_str)
    )
    sub_id = cursor.lastrowid

    # 扣余额
    db.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (price, user_id))

    db.commit()
    new_balance = db.execute("SELECT balance FROM users WHERE id = ?", (user_id,)).fetchone()["balance"]
    db.close()

    logger.info(f"[订阅] user_id={user_id} 订阅 {plan} 套餐, 扣费 {price} 分")
    return jsonify({
        "status": "ok",
        "subscription": {
            "id": sub_id, "plan": plan, "plan_label": plan_info["label"],
            "model_quota": quota, "calls_used": 0,
            "price": price, "started_at": now_dt.isoformat(),
            "expires_at": expires_str,
        },
        "balance": new_balance,
    })


@app.route("/api/user/subscription", methods=["GET"])
@require_user
def user_subscription():
    """获取当前用户的套餐信息。"""
    db = get_db()
    sub = db.execute(
        "SELECT id, plan, model_quota, calls_used, status, started_at, expires_at "
        "FROM subscriptions WHERE user_id = ? AND status = 'active' ORDER BY expires_at DESC LIMIT 1",
        (g.current_user["user_id"],)
    ).fetchone()
    db.close()

    if not sub:
        return jsonify({"subscription": None})

    remaining = max(0, sub["model_quota"] - sub["calls_used"])
    plan_label = SUBSCRIPTION_PLANS.get(sub["plan"], {}).get("label", sub["plan"])

    return jsonify({
        "subscription": {
            "id": sub["id"], "plan": sub["plan"], "plan_label": plan_label,
            "model_quota": sub["model_quota"], "calls_used": sub["calls_used"],
            "remaining": remaining, "status": sub["status"],
            "started_at": sub["started_at"], "expires_at": sub["expires_at"],
        }
    })


# ========== 邀请返利 ==========

@app.route("/api/user/invite_info", methods=["GET"])
@require_user
def user_invite_info():
    """获取当前用户的邀请信息。"""
    db = get_db()
    user = db.execute(
        "SELECT invite_code, invited_by FROM users WHERE id = ?",
        (g.current_user["user_id"],)
    ).fetchone()

    if not user:
        db.close()
        return jsonify({"error": "用户不存在"}), 404

    # 统计邀请人数和返利
    invited_count = db.execute(
        "SELECT COUNT(*) as cnt FROM users WHERE invited_by = ?",
        (g.current_user["user_id"],)
    ).fetchone()["cnt"]

    # 累计返利 = 邀请人数 * 200 分
    total_reward = invited_count * 200

    db.close()
    return jsonify({
        "invite_code": user["invite_code"] or "",
        "invited_count": invited_count,
        "total_reward": total_reward,
    })


# ========== 系统公告 ==========

@app.route("/api/announcements", methods=["GET"])
def get_announcements():
    """获取所有激活的公告（无需登录）。"""
    db = get_db()
    announcements = db.execute(
        "SELECT id, content, created_at FROM announcements WHERE active = 1 ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    db.close()
    return jsonify({
        "announcements": [{
            "id": a["id"], "content": a["content"], "created_at": a["created_at"],
        } for a in announcements]
    })


@app.route("/api/admin/announcements", methods=["POST"])
@require_admin
def create_announcement():
    """管理员发布公告。"""
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "无效的请求体"}), 400
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify({"error": "公告内容不能为空"}), 400
    if len(content) > 500:
        return jsonify({"error": "公告内容不能超过 500 字"}), 400

    db = get_db()
    cursor = db.execute("INSERT INTO announcements (content, active) VALUES (?, 1)", (content,))
    a_id = cursor.lastrowid
    db.commit()
    db.close()

    logger.info(f"[公告] 管理员 {g.current_user['username']} 发布公告 #{a_id}")
    return jsonify({
        "status": "ok",
        "announcement": {"id": a_id, "content": content},
    })


@app.route("/api/admin/announcements/<int:a_id>", methods=["DELETE"])
@require_admin
def delete_announcement(a_id):
    """管理员删除公告（软删除，设 active=0）。"""
    db = get_db()
    a_record = db.execute("SELECT id FROM announcements WHERE id = ?", (a_id,)).fetchone()
    if not a_record:
        db.close()
        return jsonify({"error": "公告不存在"}), 404

    db.execute("UPDATE announcements SET active = 0 WHERE id = ?", (a_id,))
    db.commit()
    db.close()

    logger.info(f"[公告] 管理员 {g.current_user['username']} 删除公告 #{a_id}")
    return jsonify({"status": "ok", "message": "公告已删除"})


# ========== 错误日志 ==========

@app.route("/api/admin/error_logs", methods=["GET"])
@require_admin
def get_error_logs():
    """管理员查看最近错误日志。"""
    limit = request.args.get("limit", 50, type=int)
    limit = min(limit, 200)
    with _error_log_lock:
        logs = _error_log[-limit:]
    return jsonify({
        "logs": [{"timestamp": ts, "level": lv, "message": msg} for ts, lv, msg in logs]
    })


# ========== 套餐套餐价格查询 ==========

@app.route("/api/admin/logs/clear", methods=["POST"])
@require_admin
def clear_error_logs():
    """管理员清除所有错误日志。"""
    with _error_log_lock:
        count = len(_error_log)
        _error_log.clear()
    logger.info(f"[管理] 管理员 {g.current_user['username']} 清除 {count} 条错误日志")
    return jsonify({"status": "ok", "message": f"已清除 {count} 条日志"})


@app.route("/api/subscribe/plans", methods=["GET"])
def subscribe_plans():
    """返回套餐列表（无需登录，供前端展示）。"""
    plans = []
    for key, info in SUBSCRIPTION_PLANS.items():
        plans.append({
            "id": key,
            "label": info["label"],
            "price": info["price"],
            "quota": info["quota"],
            "days": info["days"],
        })
    return jsonify({"plans": plans})

# ========== Webhook 回调 ==========

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()

def _post_webhook(event: str, payload: dict):
    """发送 Webhook 通知。"""
    if not WEBHOOK_URL:
        return
    try:
        data = {"event": event, "timestamp": int(time.time()), "data": payload}
        resp = requests.post(WEBHOOK_URL, json=data, timeout=5)
        if resp.status_code >= 400:
            logger.warning(f"[Webhook] {event} 回调失败: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"[Webhook] {event} 发送异常: {e}")

def _emit_usage_event(user_id: int, model: str, tokens: int, cost: int):
    """发送用量事件 Webhook。"""
    _post_webhook("usage.created", {
        "user_id": user_id,
        "model": model,
        "total_tokens": tokens,
        "cost_cents": cost,
    })

def _emit_recharge_event(user_id: int, amount: int, method: str):
    """发送充值事件 Webhook。"""
    _post_webhook("recharge.completed", {
        "user_id": user_id,
        "amount_cents": amount,
        "method": method,
    })

def _emit_register_event(user_id: int, username: str):
    """发送注册事件 Webhook。"""
    _post_webhook("user.registered", {
        "user_id": user_id,
        "username": username,
    })

# ========== Swagger / OpenAPI 文档 ==========

@app.route("/api/docs/openapi.json", methods=["GET"])
def openapi_spec():
    return jsonify({
        "openapi": "3.0.3",
        "info": {
            "title": "TokenHub API",
            "version": "1.0.0",
            "description": "AI Model API Router + Token 售卖平台",
        },
        "servers": [{"url": os.getenv("SITE_URL", "http://localhost:5000")}],
        "paths": {
            "/v1/chat/completions": {
                "post": {
                    "summary": "核心聊天接口（OpenAI 兼容）",
                    "security": [{"ApiKeyAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatRequest"}}},
                    },
                    "responses": {
                        "200": {"description": "流式或非流式响应"},
                        "401": {"description": "无效 API Key"},
                        "402": {"description": "余额不足"},
                        "429": {"description": "限流"},
                    }
                }
            },
            "/api/auth/register": {
                "post": {
                    "summary": "用户注册",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"type": "object", "properties": {
                            "username": {"type": "string"}, "password": {"type": "string"},
                            "email": {"type": "string"}, "invite_code": {"type": "string"}
                        }, "required": ["username", "password"]}}},
                    },
                    "responses": {"200": {"description": "注册成功"}}
                }
            },
            "/api/auth/login": {
                "post": {
                    "summary": "用户登录",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"type": "object", "properties": {
                            "username": {"type": "string"}, "password": {"type": "string"}
                        }, "required": ["username", "password"]}}},
                    },
                    "responses": {"200": {"description": "登录成功"}}
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "获取可用模型列表",
                    "responses": {"200": {"description": "模型列表"}}
                }
            },
            "/api/pricing": {
                "get": {
                    "summary": "获取定价信息",
                    "responses": {"200": {"description": "定价信息"}}
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "Authorization", "description": "Bearer <your-api-key>"}
            },
            "schemas": {
                "ChatRequest": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "example": "gpt-4o-mini"},
                        "messages": {"type": "array"},
                        "stream": {"type": "boolean"},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "integer"},
                    },
                    "required": ["model", "messages"]
                }
            }
        }
    })

@app.route("/docs")
def swagger_ui():
    """Swagger UI 页面。"""
    spec_url = f"{os.getenv('SITE_URL', '')}/api/docs/openapi.json"
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>TokenHub API Docs</title>
    <link rel="stylesheet" href="https://registry.npmmirror.com/swagger-ui-dist/5.11.0/swagger-ui.css">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://registry.npmmirror.com/swagger-ui-dist/5.11.0/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({{ url: "{spec_url}", dom_id: "#swagger-ui" }});
    </script>
</body>
</html>"""

# ========== 启动入口 ==========

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    logger.info(f"AI Model Router + Token Platform 启动于 {host}:{port}")
    app.run(host=host, port=port, debug=debug)
