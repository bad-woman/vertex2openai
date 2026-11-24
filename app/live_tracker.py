"""
进行中请求实况脉冲注册表（Live Stream X-Ray & 服务端一键中止）。

纯内存轻量状态，空闲时返回空列表（前端脉冲舱自动折叠不占位）；
有请求在途时实时展示：模型名、当前凭证、实时阶段（选钥建连 / 深度思考 / 正文吐字 / 重试退避）、
已吐字数、估算吐字速度、抗截断守护标记，并支持通过 req_id 从控制台立即中止上游请求。
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class _LiveRequest:
    __slots__ = (
        "req_id",
        "model",
        "channel",
        "cred_label",
        "project_id",
        "phase",
        "is_stream",
        "sandbox_active",
        "started_mono",
        "started_wall",
        "connect_done_mono",
        "think_start_mono",
        "body_start_mono",
        "chars",
        "thought_chars",
        "retries",
        "note",
        "aborted",
        "cancel_callbacks",
    )

    def __init__(
        self,
        req_id: str,
        model: str,
        channel: str,
        is_stream: bool,
        sandbox_active: bool,
    ):
        now_m = time.monotonic()
        self.req_id = req_id
        self.model = model
        self.channel = channel
        self.cred_label = "准备选钥"
        self.project_id = ""
        self.phase = "connecting"
        self.is_stream = is_stream
        self.sandbox_active = sandbox_active
        self.started_mono = now_m
        self.started_wall = time.time()
        self.connect_done_mono: Optional[float] = None
        self.think_start_mono: Optional[float] = None
        self.body_start_mono: Optional[float] = None
        self.chars = 0
        self.thought_chars = 0
        self.retries = 0
        self.note = ""
        self.aborted = threading.Event()
        self.cancel_callbacks: List[Tuple[Optional[asyncio.AbstractEventLoop], Callable[[], Any]]] = []

    def to_public_dict(self) -> Dict[str, Any]:
        now_m = time.monotonic()
        elapsed = round(max(0.0, now_m - self.started_mono), 1)

        conn_sec: Optional[float] = None
        if self.connect_done_mono is not None:
            conn_sec = round(max(0.0, self.connect_done_mono - self.started_mono), 2)

        think_sec: Optional[float] = None
        if self.think_start_mono is not None:
            end_th = self.body_start_mono if self.body_start_mono is not None else now_m
            think_sec = round(max(0.0, end_th - self.think_start_mono), 1)

        tps_est: Optional[float] = None
        if self.body_start_mono is not None and self.chars > 0:
            gen_s = max(0.25, now_m - self.body_start_mono)
            # 中文与英文混合文本约 1.65 字符/token，仅用于实况脉冲实时展示
            tps_est = round((self.chars / 1.65) / gen_s, 1)

        return {
            "req_id": self.req_id,
            "model": self.model,
            "channel": self.channel,
            "cred_label": self.cred_label,
            "project_id": self.project_id,
            "phase": self.phase,
            "is_stream": self.is_stream,
            "sandbox_active": self.sandbox_active,
            "elapsed_sec": elapsed,
            "conn_sec": conn_sec,
            "think_sec": think_sec,
            "chars": self.chars,
            "thought_chars": self.thought_chars,
            "tps_est": tps_est,
            "retries": self.retries,
            "note": self.note,
            "aborted": self.aborted.is_set(),
            "started_at": self.started_wall,
        }


class LiveRequestTracker:
    """线程安全的在途请求实况注册表。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: Dict[str, _LiveRequest] = {}

    def start(
        self,
        *,
        model: str,
        channel: str = "express",
        is_stream: bool = True,
        sandbox_active: bool = False,
    ) -> str:
        req_id = secrets.token_hex(4)
        item = _LiveRequest(
            req_id=req_id,
            model=model or "unknown",
            channel=channel or "express",
            is_stream=bool(is_stream),
            sandbox_active=bool(sandbox_active),
        )
        with self._lock:
            self._active[req_id] = item
        return req_id

    def update(
        self,
        req_id: str,
        *,
        model: Optional[str] = None,
        channel: Optional[str] = None,
        cred_label: Optional[str] = None,
        project_id: Optional[str] = None,
        phase: Optional[str] = None,
        sandbox_active: Optional[bool] = None,
        add_chars: int = 0,
        add_thought_chars: int = 0,
        retries: Optional[int] = None,
        note: Optional[str] = None,
    ) -> None:
        if not req_id:
            return
        now_m = time.monotonic()
        with self._lock:
            item = self._active.get(req_id)
            if item is None:
                return
            if model is not None:
                item.model = model
            if channel is not None:
                item.channel = channel
            if cred_label is not None:
                item.cred_label = cred_label
            if project_id is not None:
                item.project_id = project_id
            if sandbox_active is not None:
                item.sandbox_active = bool(sandbox_active)
            if retries is not None:
                item.retries = int(retries)
            if note is not None:
                item.note = note
            if add_thought_chars > 0:
                item.thought_chars += int(add_thought_chars)
                if item.connect_done_mono is None:
                    item.connect_done_mono = now_m
                if item.think_start_mono is None:
                    item.think_start_mono = now_m
                if item.phase in ("connecting", "retrying"):
                    item.phase = "thinking"
            if add_chars > 0:
                item.chars += int(add_chars)
                if item.connect_done_mono is None:
                    item.connect_done_mono = now_m
                if item.body_start_mono is None:
                    item.body_start_mono = now_m
                item.phase = "streaming"
            if phase is not None:
                item.phase = phase
                if phase == "thinking":
                    if item.connect_done_mono is None:
                        item.connect_done_mono = now_m
                    if item.think_start_mono is None:
                        item.think_start_mono = now_m
                elif phase == "streaming":
                    if item.connect_done_mono is None:
                        item.connect_done_mono = now_m
                    if item.body_start_mono is None:
                        item.body_start_mono = now_m

    def register_cancel_callback(
        self,
        req_id: str,
        callback: Callable[[], Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        if not req_id:
            return
        with self._lock:
            item = self._active.get(req_id)
            if item is None:
                return
            if item.aborted.is_set():
                # 已处于中止态，立即触发
                try:
                    if loop is not None and not loop.is_closed():
                        loop.call_soon_threadsafe(callback)
                    else:
                        callback()
                except Exception:
                    pass
                return
            item.cancel_callbacks.append((loop, callback))

    def is_aborted(self, req_id: str) -> bool:
        if not req_id:
            return False
        with self._lock:
            item = self._active.get(req_id)
            return bool(item and item.aborted.is_set())

    def abort(self, req_id: str) -> bool:
        with self._lock:
            item = self._active.get(req_id)
            if item is None:
                return False
            item.aborted.set()
            item.note = "已收到控制台手动中止信号"
            callbacks = list(item.cancel_callbacks)
            item.cancel_callbacks.clear()

        for loop, cb in callbacks:
            try:
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(cb)
                else:
                    cb()
            except Exception:
                pass
        print(f"🛑 [实况中止] 控制台已下发中止信号，正在取消在途请求 #{req_id}（模型={item.model}）。")
        return True

    def finish(self, req_id: str) -> None:
        if not req_id:
            return
        with self._lock:
            self._active.pop(req_id, None)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = sorted(self._active.values(), key=lambda x: x.started_wall, reverse=True)
            return [it.to_public_dict() for it in items]

    def count(self) -> int:
        with self._lock:
            return len(self._active)


# 单例导出
live_tracker = LiveRequestTracker()
