# app/memory.py
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

__all__ = [
    "get_history",
    "list_sessions",
    "drop_session",
    "drop_all_sessions",
    "EPHEMERAL_SESSION_ID",
]

MAX_SESSIONS: int = int(os.getenv("CHAT_MEMORY_MAX_SESSIONS", "200"))
MAX_MESSAGES_PER_SESSION: int = int(os.getenv("CHAT_MEMORY_MAX_MESSAGES", "60"))
SESSION_TTL_SECONDS: int = int(os.getenv("CHAT_MEMORY_TTL_SECONDS", "0"))
EPHEMERAL_SESSION_ID: str = "__ephemeral__"


@dataclass
class _SessionItem:
    history: ChatMessageHistory
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def trim(self, max_messages: int) -> None:
        if max_messages <= 0:
            return
        msgs = self.history.messages
        if len(msgs) > max_messages:
            excess = len(msgs) - max_messages
            del msgs[0:excess]


_SESSIONS: Dict[str, _SessionItem] = {}
_LOCK = threading.RLock()


def _purge_expired_sessions() -> None:
    now = time.time()
    to_delete: List[str] = []

    with _LOCK:
        if SESSION_TTL_SECONDS > 0:
            for sid, item in _SESSIONS.items():
                if (now - item.updated_at) > SESSION_TTL_SECONDS:
                    to_delete.append(sid)

        for sid in to_delete:
            _SESSIONS.pop(sid, None)

        if len(_SESSIONS) > MAX_SESSIONS:
            ordered: List[Tuple[str, _SessionItem]] = sorted(
                _SESSIONS.items(), key=lambda kv: kv[1].updated_at
            )
            overflow = len(_SESSIONS) - MAX_SESSIONS
            for i in range(overflow):
                sid, _ = ordered[i]
                _SESSIONS.pop(sid, None)


def _get_or_create_session(session_id: str) -> ChatMessageHistory:
    with _LOCK:
        item = _SESSIONS.get(session_id)
        if item is None:
            item = _SessionItem(history=ChatMessageHistory())
            _SESSIONS[session_id] = item

        item.touch()
        item.trim(MAX_MESSAGES_PER_SESSION)

    _purge_expired_sessions()
    return item.history


def get_history(session_id: str) -> BaseChatMessageHistory:
    if not session_id or session_id == EPHEMERAL_SESSION_ID:
        return ChatMessageHistory()
    return _get_or_create_session(session_id)


def list_sessions() -> List[str]:
    with _LOCK:
        return list(_SESSIONS.keys())


def drop_session(session_id: str) -> bool:
    with _LOCK:
        existed = session_id in _SESSIONS
        _SESSIONS.pop(session_id, None)
    return existed


def drop_all_sessions() -> None:
    with _LOCK:
        _SESSIONS.clear()