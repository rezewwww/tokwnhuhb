"""
Token 售卖平台 - 数据库模型
支持 SQLite 和 PostgreSQL，通过 DB_TYPE 环境变量切换。
"""
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.db")
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()  # sqlite | postgresql


def get_db():
    """获取数据库连接，根据 DB_TYPE 返回 SQLite 或 PostgreSQL 连接。"""
    if DB_TYPE == "postgresql":
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME", "tokenhub"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        return conn
    # SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库，创建所有表。"""
    conn = get_db()
    cursor = conn.cursor()

    if DB_TYPE == "postgresql":
        _init_postgresql(cursor)
    else:
        _init_sqlite(cursor)
        # 初始化 schema_version
        try:
            cursor.execute("INSERT INTO schema_version (id, version) VALUES (1, 1)")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()
    logger.info(f"数据库初始化完成 (type={DB_TYPE})")


def _init_sqlite(cursor):
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT DEFAULT '',
            email_verified INTEGER NOT NULL DEFAULT 0,
            email_verify_token TEXT DEFAULT '',
            balance INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            invite_code TEXT UNIQUE DEFAULT '',
            invited_by INTEGER DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_value TEXT UNIQUE NOT NULL,
            key_prefix TEXT DEFAULT '',
            name TEXT DEFAULT 'Default',
            is_active INTEGER NOT NULL DEFAULT 1,
            allowed_models TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS usage_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            api_key_id INTEGER,
            model TEXT NOT NULL,
            backend TEXT DEFAULT '',
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost INTEGER NOT NULL DEFAULT 0,
            status TEXT DEFAULT 'settled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        );

        CREATE TABLE IF NOT EXISTS recharge_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            method TEXT DEFAULT 'admin',
            status TEXT DEFAULT 'completed',
            order_no TEXT UNIQUE,
            pay_url TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan TEXT NOT NULL DEFAULT 'monthly',
            model_quota INTEGER NOT NULL DEFAULT 10000,
            calls_used INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rate_limit_state (
            key TEXT PRIMARY KEY,
            window_start REAL NOT NULL,
            count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            model TEXT PRIMARY KEY,
            failures INTEGER NOT NULL DEFAULT 0,
            last_failure REAL NOT NULL DEFAULT 0,
            is_open INTEGER NOT NULL DEFAULT 0,
            opened_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS disabled_models (
            model TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
        CREATE INDEX IF NOT EXISTS idx_api_keys_value ON api_keys(key_value);
        CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_records(user_id);
        CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_records(created_at);
        CREATE INDEX IF NOT EXISTS idx_recharge_user ON recharge_records(user_id);
        CREATE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions(user_id);
    """)

    migrations = [
        "ALTER TABLE recharge_records ADD COLUMN order_no TEXT UNIQUE",
        "ALTER TABLE recharge_records ADD COLUMN pay_url TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN invite_code TEXT UNIQUE DEFAULT ''",
        "ALTER TABLE users ADD COLUMN invited_by INTEGER DEFAULT NULL",
        "ALTER TABLE api_keys ADD COLUMN allowed_models TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN email_verify_token TEXT DEFAULT ''",
        "ALTER TABLE api_keys ADD COLUMN key_prefix TEXT DEFAULT ''",
        "ALTER TABLE usage_records ADD COLUMN status TEXT DEFAULT 'settled'",
    ]
    for m in migrations:
        try:
            cursor.execute(m)
        except sqlite3.OperationalError:
            pass


def _init_postgresql(cursor):
    """PostgreSQL 建表语句（使用 SERIAL 和 TEXT 类型）。"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT DEFAULT '',
            email_verified INTEGER NOT NULL DEFAULT 0,
            email_verify_token TEXT DEFAULT '',
            balance INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            invite_code TEXT UNIQUE DEFAULT '',
            invited_by INTEGER DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            key_value TEXT UNIQUE NOT NULL,
            key_prefix TEXT DEFAULT '',
            name TEXT DEFAULT 'Default',
            is_active INTEGER NOT NULL DEFAULT 1,
            allowed_models TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS usage_records (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            api_key_id INTEGER REFERENCES api_keys(id),
            model TEXT NOT NULL,
            backend TEXT DEFAULT '',
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost INTEGER NOT NULL DEFAULT 0,
            status TEXT DEFAULT 'settled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS recharge_records (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            amount INTEGER NOT NULL,
            method TEXT DEFAULT 'admin',
            status TEXT DEFAULT 'completed',
            order_no TEXT UNIQUE,
            pay_url TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            plan TEXT NOT NULL DEFAULT 'monthly',
            model_quota INTEGER NOT NULL DEFAULT 10000,
            calls_used INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS announcements (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rate_limit_state (
            key TEXT PRIMARY KEY,
            window_start DOUBLE PRECISION NOT NULL,
            count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS circuit_breaker_state (
            model TEXT PRIMARY KEY,
            failures INTEGER NOT NULL DEFAULT 0,
            last_failure DOUBLE PRECISION NOT NULL DEFAULT 0,
            is_open INTEGER NOT NULL DEFAULT 0,
            opened_at DOUBLE PRECISION NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS disabled_models (
            model TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verify_token TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_prefix TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'settled'")
    except Exception:
        pass
    # 初始化 schema_version
    try:
        cursor.execute("INSERT INTO schema_version (id, version) VALUES (1, 1) ON CONFLICT(id) DO NOTHING")
    except Exception:
        pass


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
