import builtins
import time
import asyncio
import re
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from metrics_store import metrics_store

original_print = builtins.print
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class ProxyStats:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.retry_counts = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.lock = threading.Lock()

    def increment_total(self):
        with self.lock:
            self.total_requests += 1

    def add_error(self):
        with self.lock:
            self.error_requests += 1

    def add_retry(self):
        with self.lock:
            self.retry_counts += 1

    def add_success(self):
        """直接计一次成功请求（Cookie 通道不产生 token 统计行，成功数单独计入）。"""
        with self.lock:
            self.success_requests += 1

    def add_tokens(self, p_tokens, c_tokens, cached_tokens: int = 0):
        with self.lock:
            self.prompt_tokens += p_tokens
            self.cached_tokens += max(0, int(cached_tokens or 0))
            self.completion_tokens += c_tokens
            self.success_requests += 1

    def get_json_stats(self, range_key: str = "all") -> Dict[str, Any]:
        uptime = round(time.time() - self.start_time, 2)
        db_stats = metrics_store.get_overview_stats(
            range_key=range_key,
            uptime=uptime,
            records_version=call_recorder.version,
            records_count=call_recorder.count(),
        )
        # 兼容仅直接调用内存 stats.add_tokens() 而不经过 CallTraceTracker 的轻量单元测试场景
        if range_key == "all" and db_stats["total"] == 0:
            with self.lock:
                if self.total_requests > 0 or self.success_requests > 0 or self.error_requests > 0:
                    db_stats["total"] = self.total_requests
                    db_stats["success"] = self.success_requests
                    db_stats["error"] = self.error_requests
                    db_stats["retries"] = self.retry_counts
                    db_stats["prompt_tokens"] = self.prompt_tokens
                    db_stats["cached_tokens"] = self.cached_tokens
                    db_stats["completion_tokens"] = self.completion_tokens
        return db_stats


class CallRecorder:
    """线程安全的调用记录缓存（与 SQLite metrics_store 双向同步，最多保留最新 500 条，重启不丢）。"""

    def __init__(self, maxlen: int = 500):
        self.maxlen = maxlen
        self._records: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._seq: int = 0
        self.version: int = 0
        self._lock = threading.Lock()
        self.reload_from_store()

    def reload_from_store(self) -> None:
        persisted = metrics_store.load_recent_records(self.maxlen)
        with self._lock:
            self._records.clear()
            max_id = 0
            for item in persisted:
                self._records.append(item)
                try:
                    max_id = max(max_id, int(item.get("id") or 0))
                except (TypeError, ValueError):
                    pass
            self._seq = max(self._seq, max_id)
            self.version += 1

    def record(
        self,
        item: Dict[str, Any],
        *,
        channel: Optional[str] = None,
        created_at: Optional[float] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        entry = dict(item)
        ch = channel or entry.get("channel") or ("cookie" if entry.get("is_cookie") else "express")
        row_id = None
        if persist:
            row_id = metrics_store.record_call(entry, channel=ch, created_at=created_at)
        with self._lock:
            if row_id is not None and row_id > 0:
                self._seq = max(self._seq, row_id)
                entry["id"] = row_id
            else:
                self._seq += 1
                entry["id"] = self._seq
            self.version += 1
            self._records.appendleft(entry)
            return entry

    def clear(self) -> None:
        metrics_store.clear_call_records()
        with self._lock:
            self._records.clear()
            self.version += 1

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def get_payload(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._records)
            ver = self.version

        ok_cnt = sum(1 for r in items if r.get("group") == "ok")
        err_cnt = sum(1 for r in items if r.get("group") == "err")
        ttft_vals = [
            float(r["ttft"])
            for r in items
            if r.get("group") == "ok" and isinstance(r.get("ttft"), (int, float)) and r["ttft"] > 0
        ]
        tps_vals = [
            float(r["tps"])
            for r in items
            if r.get("group") == "ok" and isinstance(r.get("tps"), (int, float)) and r["tps"] > 0
        ]
        p_sum = sum(int(r.get("p_tk") or 0) for r in items if r.get("group") == "ok" and r.get("has_usage"))
        c_sum = sum(int(r.get("cached_tk") or 0) for r in items if r.get("group") == "ok" and r.get("has_usage"))
        avg_ttft = round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else None
        avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None
        avg_cache_pct = round((c_sum / p_sum) * 100, 1) if p_sum > 0 else None

        return {
            "version": ver,
            "total": len(items),
            "ok_count": ok_cnt,
            "err_count": err_cnt,
            "avg_ttft": avg_ttft,
            "avg_tps": avg_tps,
            "avg_cache_pct": avg_cache_pct,
            "records": items,
        }


call_recorder = CallRecorder(maxlen=500)
stats = ProxyStats()


class SSELogger:
    """把运行日志推给控制台的 SSE 订阅者（线程安全 + 有界队列防内存膨胀）。"""

    def __init__(self, max_history: int = 120, queue_size: int = 500):
        self.max_history = max_history
        self.queue_size = queue_size
        self.history: List[str] = []
        self._subscribers: List[tuple] = []       # [(queue, loop)]
        self._lock = threading.Lock()

    def subscribe(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        loop = loop or asyncio.get_running_loop()
        with self._lock:
            self._subscribers.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(sq, sl) for (sq, sl) in self._subscribers if sq is not q]

    def snapshot_history(self) -> List[str]:
        with self._lock:
            return list(self.history)

    @staticmethod
    def _offer(q: asyncio.Queue, msg: str) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(msg)
            except Exception:
                pass

    def push(self, plain_text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {plain_text}"

        with self._lock:
            self.history.append(formatted_msg)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            subscribers = list(self._subscribers)

        for q, loop in subscribers:
            try:
                if loop.is_closed():
                    continue
                loop.call_soon_threadsafe(self._offer, q, formatted_msg)
            except RuntimeError:
                pass


rt_logger = SSELogger()

_in_hook = threading.local()


def custom_print(*args, **kwargs):
    """把业务 print 同时推到控制台日志流。"""
    if getattr(_in_hook, "active", False):
        original_print(*args, **kwargs)
        return

    _in_hook.active = True
    try:
        import io
        buf = io.StringIO()
        kwargs_for_buffer = dict(kwargs)
        kwargs_for_buffer["file"] = buf

        raw_msg = ""
        try:
            original_print(*args, **kwargs_for_buffer)
            raw_msg = buf.getvalue().strip()
        except Exception:
            pass

        try:
            original_print(*args, **kwargs)
        except Exception:
            pass

        if not raw_msg:
            return

        try:
            rt_logger.push(ANSI_ESCAPE.sub('', raw_msg))
        except Exception:
            pass
    finally:
        _in_hook.active = False


builtins.print = custom_print
