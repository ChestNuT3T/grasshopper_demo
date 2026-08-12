"""
=============================================================================
        SQLite 会话历史存储模块
=============================================================================
基于 SQLite（WAL 模式）的会话历史存储，替代脆弱的 FileChatMessageHistory。

设计要点：
    1. 事务保证：add_message 走 INSERT 事务，不会出现写一半损坏
    2. WAL 模式：读写不互斥，多线程并发安全
    3. 单连接 + 锁：同一进程共享一个 connection，写操作加锁串行化
    4. 实现 BaseChatMessageHistory 接口：兼容 RunnableWithMessageHistory

表结构：
    sessions(session_id PK, name, created_at, updated_at)
    messages(id PK, session_id, role, content, additional_kwargs, created_at)
=============================================================================
"""

import json
import sqlite3
import threading
import time
from typing import List, Optional, Dict, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.chat_history import BaseChatMessageHistory

from src.core.logger import logger


# =============================================================================
# 全局连接与锁（单例，进程内共享）
# =============================================================================
_db_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None
_DB_PATH: Optional[str] = None


def _get_connection() -> sqlite3.Connection:
    """
    获取全局 SQLite 连接（懒加载，线程安全）。

    Returns:
        sqlite3.Connection 实例，开启 WAL 模式与外键约束
    """
    global _connection, _DB_PATH
    if _connection is not None:
        return _connection

    with _db_lock:
        if _connection is not None:
            return _connection
        from config.settings import CHAT_DB_PATH

        CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DB_PATH = str(CHAT_DB_PATH)
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        _connection = conn
        logger.info(f"SQLite chat history initialized at {_DB_PATH}")
        return _connection


def _init_schema(conn: sqlite3.Connection) -> None:
    """初始化表结构与索引（幂等）。"""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            name        TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id        TEXT NOT NULL,
            role              TEXT NOT NULL,
            content           TEXT NOT NULL,
            additional_kwargs TEXT,
            created_at        REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id);
        """
    )
    conn.commit()


# =============================================================================
# 消息序列化/反序列化
# =============================================================================
def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """
    将 BaseMessage 序列化为可存入 SQLite 的字典。

    Args:
        message: LangChain 消息对象

    Returns:
        包含 role / content / additional_kwargs 的字典
    """
    msg_type = getattr(message, "type", "")
    if isinstance(message, HumanMessage) or msg_type == "human":
        role = "human"
    elif isinstance(message, AIMessage) or msg_type == "ai":
        role = "ai"
    elif isinstance(message, SystemMessage) or msg_type == "system":
        role = "system"
    else:
        role = msg_type or "unknown"

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    return {
        "role": role,
        "content": getattr(message, "content", ""),
        "additional_kwargs": json.dumps(additional_kwargs, ensure_ascii=False),
    }


def _deserialize_message(row: sqlite3.Row) -> BaseMessage:
    """
    将数据库行还原为 BaseMessage。

    Args:
        row: messages 表的查询行

    Returns:
        HumanMessage / AIMessage / SystemMessage 实例
    """
    role = row["role"]
    content = row["content"]
    raw_kwargs = row["additional_kwargs"]
    additional_kwargs = json.loads(raw_kwargs) if raw_kwargs else {}

    if role == "human":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "ai":
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    else:
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)


# =============================================================================
# SQLiteChatMessageHistory：实现 BaseChatMessageHistory 接口
# =============================================================================
class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """
    基于 SQLite 的会话历史存储。

    实现 langchain BaseChatMessageHistory 接口，可直接用于
    RunnableWithMessageHistory。每条消息一行 INSERT，事务保证不损坏。

    Attributes:
        session_id: 会话唯一标识
    """

    def __init__(self, session_id: str, name: Optional[str] = None):
        """
        Args:
            session_id: 会话 ID
            name: 会话名称（可选，首次创建时写入）
        """
        self.session_id = session_id
        self._ensure_session_record(name)

    def _ensure_session_record(self, name: Optional[str]) -> None:
        """确保 sessions 表中存在该会话记录（幂等，不存在则插入）。"""
        conn = _get_connection()
        now = time.time()
        display_name = name or session_id_to_name(self.session_id)
        with _db_lock:
            conn.execute(
                """
                INSERT INTO sessions (session_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (self.session_id, display_name, now, now),
            )
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        """
        读取该会话的全部消息（按写入顺序）。

        Returns:
            BaseMessage 列表
        """
        conn = _get_connection()
        rows = conn.execute(
            "SELECT role, content, additional_kwargs FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()
        return [_deserialize_message(row) for row in rows]

    def add_message(self, message: BaseMessage) -> None:
        """
        追加一条消息（INSERT 事务，不会损坏历史数据）。

        Args:
            message: 待追加的 LangChain 消息对象
        """
        data = _serialize_message(message)
        conn = _get_connection()
        now = time.time()
        with _db_lock:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, additional_kwargs, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.session_id, data["role"], data["content"], data["additional_kwargs"], now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, self.session_id),
            )
            conn.commit()

    def clear(self) -> None:
        """清空该会话的所有消息（保留 session 记录本身）。"""
        conn = _get_connection()
        with _db_lock:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (time.time(), self.session_id),
            )
            conn.commit()


# =============================================================================
# 会话级管理函数（供 main.py 使用）
# =============================================================================
def session_id_to_name(session_id: str) -> str:
    """将 session_id 转换为展示名称。"""
    return session_id.replace("_", " ").title()


def list_sessions() -> List[Dict[str, Any]]:
    """
    列出所有会话（按最近更新时间降序）。

    Returns:
        会话字典列表，每项含 id / name / messageCount / lastModified
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT s.session_id   AS id,
               s.name         AS name,
               s.updated_at   AS lastModified,
               COALESCE(m.cnt, 0) AS messageCount
        FROM sessions s
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS cnt
            FROM messages
            GROUP BY session_id
        ) m ON m.session_id = s.session_id
        ORDER BY s.updated_at DESC
        """
    ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"] or session_id_to_name(row["id"]),
            "messageCount": row["messageCount"],
            "lastModified": row["lastModified"],
        }
        for row in rows
    ]


def delete_session_record(session_id: str) -> bool:
    """
    删除会话及其全部消息（级联删除）。

    Args:
        session_id: 会话 ID

    Returns:
        是否删除了记录（False 表示会话不存在）
    """
    conn = _get_connection()
    with _db_lock:
        cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0


def get_session_message_count(session_id: str) -> int:
    """获取指定会话的消息数量。"""
    conn = _get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row["cnt"] if row else 0
