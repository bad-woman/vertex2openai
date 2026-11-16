import builtins
import time
import asyncio
import re
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

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

    def add_tokens(self, p_tokens, c_tokens):
        with self.lock:
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            self.success_requests += 1

    def get_json_stats(self):
        with self.lock:
            return {
                "uptime": round(time.time() - self.start_time, 2),
                "total": self.total_requests,
                "success": self.success_requests,
                "error": self.error_requests,
                "retries": self.retry_counts,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "records_version": call_recorder.version,
                "records_count": call_recorder.count(),
            }


class CallRecorder:
    """轻量线程安全的最近调用记录环形缓冲区（默认最多保留 200 条，自动淘汰最旧记录，零磁盘开销）。"""

    def __init__(self, maxlen: int = 200):
        self.maxlen = maxlen
        self._records: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._seq: int = 0
        self.version: int = 0
        self._lock = threading.Lock()

    def record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            self.version += 1
            entry = dict(item)
            entry["id"] = self._seq
            self._records.appendleft(entry)
            return entry

    def clear(self) -> None:
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
        avg_ttft = round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else None
        avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None

        return {
            "version": ver,
            "total": len(items),
            "ok_count": ok_cnt,
            "err_count": err_cnt,
            "avg_ttft": avg_ttft,
            "avg_tps": avg_tps,
            "records": items,
        }


call_recorder = CallRecorder(maxlen=200)
stats = ProxyStats()


class SSELogger:
    """把运行日志推给控制台的 SSE 订阅者。

    P1-5 修复的三个隐患：
      - `asyncio.Queue.put_nowait` 不是线程安全的，而 push() 可能从任意线程被调用
        （图片压缩线程、to_thread 里的消息转换等）。改为 loop.call_soon_threadsafe。
      - 订阅者列表会被事件循环并发增删，遍历时可能 RuntimeError。改为持锁取快照。
      - 队列原本无界，慢客户端会把内存撑爆。改为有界 + 满则丢最旧。
    """

    def __init__(self, max_history: int = 100, queue_size: int = 500):
        self.max_history = max_history
        self.queue_size = queue_size
        self.history: List[str] = []
        self._subscribers: List[tuple] = []       # [(queue, loop)]
        self._lock = threading.Lock()

    # ---- 订阅管理 ----

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

    # ---- 推送 ----

    @staticmethod
    def _offer(q: asyncio.Queue, msg: str) -> None:
        """在事件循环线程里执行：队列满时丢最旧的一条，保证新日志能进来。"""
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
                # 事件循环已停止，忽略即可
                pass


rt_logger = SSELogger()

# 防止 print 钩子内部再触发 print 导致递归
_in_hook = threading.local()


def custom_print(*args, **kwargs):
    """把业务 print 同时推到控制台日志流。

    说明（P1-5 第一步）：仍保留对 builtins.print 的接管以兼容现有 6000 行代码里的
    print 调用，但已经**移除了用正则从日志文本反解 token 数**的逻辑——
    那与中文文案强耦合，改一个字就静默失效。现在由 api_helpers._record_usage()
    直接调用 stats.add_tokens()。后续可分批把 print 迁移到 logging。
    """
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
