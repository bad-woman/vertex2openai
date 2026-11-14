"""弱网可靠传输：服务端响应缓冲 + 序列号事件 + 断线重放 / 分片轮询拉取。"""

import asyncio
import hashlib
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import config as app_config
from runtime_state import app_state

DONE_FRAME = "data: [DONE]\n\n"


def _setting(key: str) -> Any:
    try:
        return app_state.get_setting(key, app_config.DEFAULT_SETTINGS.get(key))
    except Exception:
        return app_config.DEFAULT_SETTINGS.get(key)


def is_enabled() -> bool:
    return bool(_setting("resumable_enabled"))


def ttl_seconds() -> float:
    try:
        return max(30.0, float(_setting("resumable_ttl_seconds")))
    except (TypeError, ValueError):
        return float(app_config.DEFAULT_SETTINGS["resumable_ttl_seconds"])


def max_streams() -> int:
    try:
        return max(1, int(_setting("resumable_max_streams")))
    except (TypeError, ValueError):
        return int(app_config.DEFAULT_SETTINGS["resumable_max_streams"])


def compute_stream_id(headers: Any, api_key: str, fingerprint: str) -> Tuple[str, str]:
    """决定这次请求的缓冲区 ID。返回 (id, 来源说明)。"""
    for h in ("x-resume-id", "x-request-id", "x-idempotency-key", "idempotency-key"):
        try:
            v = (headers.get(h) or "").strip()
        except Exception:
            v = ""
        if v:
            digest = hashlib.sha256(f"{api_key}\x00{v}\x00{fingerprint}".encode("utf-8")).hexdigest()[:32]
            return f"cid-{digest}", f"客户端请求头 {h}"
    digest = hashlib.sha256(f"{api_key}\x00{fingerprint}".encode("utf-8")).hexdigest()[:32]
    return f"req-{digest}", "请求体指纹"


def is_keepalive_frame(text: str) -> bool:
    """判断是否为保活心跳帧（SSE 注释帧或假流式 keepalive 空数据帧）。"""
    s = (text or "").lstrip()
    if not s:
        return True
    if s.startswith(":"):
        return True
    if '"chatcmpl-keepalive"' in s or "'chatcmpl-keepalive'" in s:
        return True
    return False


def is_error_frame(text: str) -> bool:
    """判断 SSE 帧是否为上游报错帧（用于避免把纯报错流当成成功缓存重放）。"""
    s = (text or "").strip()
    if not s.startswith("data:"):
        return False
    payload_str = s[len("data:"):].strip()
    if not payload_str or payload_str == "[DONE]":
        return False
    try:
        obj = json.loads(payload_str)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    if "error" in obj and not obj.get("choices"):
        return True
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        delta = (choices[0] or {}).get("delta") or {}
        content = delta.get("content") or ""
        if isinstance(content, str) and any(
            tag in content for tag in ("[Studio 错误]", "[Studio API 错误]", "[Studio 权限错误]", "[Studio 提示] 上游")
        ):
            return True
    return False


def is_real_output_frame(text: str) -> bool:
    """判断 SSE 帧是否包含真正的模型产出（非空 content / reasoning_content / tool_calls，且非报错帧）。"""
    s = (text or "").strip()
    if not s.startswith("data:") or is_error_frame(text):
        return False
    payload_str = s[len("data:"):].strip()
    if not payload_str or payload_str == "[DONE]":
        return False
    try:
        obj = json.loads(payload_str)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    for ch in (obj.get("choices") or []):
        delta = (ch or {}).get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False


def make_keepalive_sse(model: str) -> str:
    """生成不带序号的假流式保活帧（直接发给在线客户端，不写入 ResumableBuffer）。"""
    keep_alive_data = {
        "id": "chatcmpl-keepalive",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "unknown",
        "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}],
    }
    return f"data: {json.dumps(keep_alive_data, ensure_ascii=False)}\n\n"


class ResumableBuffer:
    """一次回复的服务端权威副本（有序、带序号、可多次读取）。"""

    def __init__(self, stream_id: str, model: str = "", kind: str = "stream"):
        self.id = stream_id
        self.model = model
        self.kind = kind
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.last_access = self.created_at
        self.status = "running"
        self.error: Optional[str] = None
        self.events: List[str] = []
        self.payload: Optional[dict] = None
        self.status_code: int = 200
        self.fully_delivered = False
        self.delivered_at: Optional[float] = None
        self.has_model_output = False
        self.saw_midstream_error = False
        self.byte_size: int = 0
        self.readers = 0
        self.last_reader_left_at: Optional[float] = None
        self._cond = asyncio.Condition()

    async def append(self, frame: str, is_model_output: bool = False) -> int:
        async with self._cond:
            if is_model_output:
                self.has_model_output = True
            self.events.append(frame)
            self.byte_size += len(frame.encode("utf-8", errors="ignore"))
            self._cond.notify_all()
            return len(self.events) - 1

    async def finish(self, status: str = "done", error: Optional[str] = None,
                     payload: Optional[dict] = None, status_code: Optional[int] = None) -> None:
        async with self._cond:
            self.status = status
            self.error = error
            if payload is not None:
                self.payload = payload
                try:
                    self.byte_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                except Exception:
                    pass
            if status_code is not None:
                self.status_code = status_code
            self.finished_at = time.time()
            self._cond.notify_all()

    @property
    def next_seq(self) -> int:
        return len(self.events)

    @property
    def done(self) -> bool:
        return self.status != "running"

    def snapshot(self, start: int, limit: int) -> List[str]:
        start = max(0, start)
        return self.events[start:start + max(1, limit)]

    async def wait_for(self, seq: int, timeout: float) -> None:
        if timeout <= 0:
            return
        deadline = time.monotonic() + timeout
        async with self._cond:
            while len(self.events) <= seq and self.status == "running":
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remain)
                except asyncio.TimeoutError:
                    return

    async def wait_next(self, seq: int, timeout: float) -> Optional[str]:
        deadline = time.monotonic() + max(0.05, timeout)
        async with self._cond:
            while len(self.events) <= seq and self.status == "running":
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remain)
                except asyncio.TimeoutError:
                    return None
            if seq < len(self.events):
                self.last_access = time.time()
                return self.events[seq]
            return None

    async def iter_from(self, seq: int = 0) -> AsyncIterator[Tuple[int, str]]:
        i = max(0, seq)
        self.readers += 1
        try:
            while True:
                async with self._cond:
                    while len(self.events) <= i and self.status == "running":
                        await self._cond.wait()
                    if len(self.events) <= i:
                        break
                    frame = self.events[i]
                self.last_access = time.time()
                yield i, frame
                i += 1
        finally:
            self.readers = max(0, self.readers - 1)
            if self.readers == 0:
                self.last_reader_left_at = time.time()

    def mark_delivered(self) -> None:
        self.fully_delivered = True
        self.delivered_at = time.time()
        self.last_access = self.delivered_at

    def meta(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "events": len(self.events),
            "bytes": self.byte_size,
            "next": len(self.events),
            "done": self.done,
            "fully_delivered": self.fully_delivered,
            "has_model_output": self.has_model_output,
            "saw_midstream_error": self.saw_midstream_error,
            "readers": self.readers,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "age_seconds": round(time.time() - self.created_at, 1),
        }


class ResumableStore:
    """缓冲区登记处：并发安全、带 TTL 与容量上限。"""

    def __init__(self):
        self._buffers: Dict[str, ResumableBuffer] = {}
        self._lock = asyncio.Lock()
        self._pumps: Dict[str, asyncio.Task] = {}

    async def get(self, stream_id: str) -> Optional[ResumableBuffer]:
        async with self._lock:
            buf = self._buffers.get(stream_id)
            if buf:
                buf.last_access = time.time()
            return buf

    async def create(self, stream_id: str, model: str = "", kind: str = "stream") -> ResumableBuffer:
        async with self._lock:
            self._cancel_pump_locked(stream_id)
            buf = ResumableBuffer(stream_id, model=model, kind=kind)
            self._buffers[stream_id] = buf
            await self._evict_locked()
            return buf

    async def create_if_absent(self, stream_id: str, model: str = "",
                               kind: str = "stream") -> Tuple[ResumableBuffer, bool]:
        async with self._lock:
            buf = self._buffers.get(stream_id)
            if buf is not None:
                buf.last_access = time.time()
                return buf, False
            buf = ResumableBuffer(stream_id, model=model, kind=kind)
            self._buffers[stream_id] = buf
            await self._evict_locked()
            return buf, True

    def _cancel_pump_locked(self, stream_id: str) -> None:
        task = self._pumps.pop(stream_id, None)
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def drop(self, stream_id: str) -> bool:
        async with self._lock:
            self._cancel_pump_locked(stream_id)
            return self._buffers.pop(stream_id, None) is not None

    async def list_meta(self) -> List[dict]:
        async with self._lock:
            return [b.meta() for b in sorted(self._buffers.values(),
                                             key=lambda x: x.created_at, reverse=True)]

    def register_pump(self, stream_id: str, task: asyncio.Task) -> None:
        self._pumps[stream_id] = task

        def _cleanup_done(done_task: asyncio.Task, sid: str = stream_id) -> None:
            # 仅当当前注册表中的任务仍是自己时才移除，防止新请求覆盖同 stream_id 后被旧任务的回调误删！
            if self._pumps.get(sid) is done_task:
                self._pumps.pop(sid, None)

        task.add_done_callback(_cleanup_done)

    def cancel_all_pumps(self) -> None:
        for task in list(self._pumps.values()):
            if not task.done():
                task.cancel()
        self._pumps.clear()

    async def _evict_locked(self) -> None:
        limit = max_streams()
        max_total_bytes = 64 * 1024 * 1024  # 64MB 全局内存软顶保护（防超大图生图 base64 驻留撑爆容器）
        total_bytes = sum(b.byte_size for b in self._buffers.values())
        if len(self._buffers) <= limit and total_bytes <= max_total_bytes:
            return
        # 优先淘汰「客户端已完整签收 (fully_delivered)」的已完成缓冲，其次再淘汰尚未签收的旧已完成缓冲
        finished = sorted(
            (b for b in self._buffers.values() if b.done and b.readers == 0),
            key=lambda x: (0 if x.fully_delivered else 1, x.finished_at or x.created_at),
        )
        for buf in finished:
            if len(self._buffers) <= limit and total_bytes <= max_total_bytes:
                return
            self._cancel_pump_locked(buf.id)
            removed_buf = self._buffers.pop(buf.id, None)
            if removed_buf is not None:
                total_bytes -= removed_buf.byte_size
        while len(self._buffers) > limit:
            oldest = min(self._buffers.values(), key=lambda x: x.created_at)
            self._cancel_pump_locked(oldest.id)
            self._buffers.pop(oldest.id, None)

    async def sweep(self) -> int:
        ttl = ttl_seconds()
        now = time.time()
        removed = 0
        async with self._lock:
            for sid, buf in list(self._buffers.items()):
                if buf.readers > 0:
                    continue
                ref = buf.finished_at or buf.last_access
                if buf.status == "error" and (now - ref) > 5.0:
                    self._cancel_pump_locked(sid)
                    self._buffers.pop(sid, None)
                    removed += 1
                elif buf.done and buf.fully_delivered and (now - (buf.delivered_at or ref)) > min(ttl, 60.0):
                    # 客户端已完整收完 [DONE] 的缓冲无需傻留 30 分钟，60 秒后自动释放内存
                    self._cancel_pump_locked(sid)
                    self._buffers.pop(sid, None)
                    removed += 1
                elif buf.done and (now - ref) > ttl:
                    self._cancel_pump_locked(sid)
                    self._buffers.pop(sid, None)
                    removed += 1
                elif not buf.done and (now - buf.created_at) > max(ttl * 4, 1800):
                    self._cancel_pump_locked(sid)
                    self._buffers.pop(sid, None)
                    removed += 1
            await self._evict_locked()
        if removed:
            print(f"🧹 [续传缓冲] 已清理 {removed} 个过期或已签收的响应缓冲。")
        return removed

    async def sweeper_loop(self, interval: float = 60.0) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"⚠️ [续传缓冲] 清理任务异常（已忽略）：{type(e).__name__} - {e}")


resumable_store = ResumableStore()


class DetachedRequest:
    """带智能断连感知的请求包装器。"""

    def __init__(self, request: Any, buf: Optional[ResumableBuffer] = None):
        self._request = request
        self._buf = buf

    async def is_disconnected(self) -> bool:
        if self._buf is not None:
            if self._buf.status != "running":
                return True
            if self._buf.has_model_output:
                return False
            if self._buf.readers > 0:
                return False
        try:
            return await self._request.is_disconnected()
        except Exception:
            return False

    def __getattr__(self, item):
        return getattr(self._request, item)


def _as_text(chunk: Any) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def frame_with_id(seq: int, frame: str) -> str:
    if frame.startswith("data:"):
        return f"id: {seq}\n{frame}"
    return frame


async def pump_stream(buf: ResumableBuffer, body_iterator: AsyncIterator[Any]) -> None:
    saw_error = False
    try:
        async for chunk in body_iterator:
            text = _as_text(chunk)
            if not text or is_keepalive_frame(text):
                continue
            err_frame = is_error_frame(text)
            if err_frame:
                saw_error = True
                buf.saw_midstream_error = True
            is_real_output = is_real_output_frame(text)
            await buf.append(frame_with_id(buf.next_seq, text), is_model_output=is_real_output)

        if not buf.events or not buf.events[-1].strip().endswith("[DONE]"):
            await buf.append(frame_with_id(buf.next_seq, DONE_FRAME), is_model_output=False)

        if saw_error and not buf.has_model_output:
            await buf.finish("error", error="上游返回错误响应，未产出有效内容。")
            await resumable_store.drop(buf.id)
            print(f"⚠️ [续传缓冲] {buf.id} 上游返回错误且无有效内容，已清除该缓冲以便下次重试重新请求。")
        elif saw_error:
            # 半截断流报错：保留缓冲供携带 Last-Event-ID 的断线重连拉完剩余帧，但标记 saw_midstream_error 以便酒馆全新重Roll时重新请求上游
            await buf.finish("done", error="流式传输中途发生中断或错误。")
            print(f"⚠️ [续传缓冲] {buf.id} 流中途发生错误（已保留已产出的 {len(buf.events)} 帧供续传，全新重试将自动重请求上游）。")
        else:
            await buf.finish("done")
            print(f"📦 [续传缓冲] {buf.id} 上游已读完，共 {len(buf.events)} 个事件（{buf.byte_size / 1024:.1f} KB）已落缓冲。")
    except asyncio.CancelledError:
        await buf.finish("error", error="上游读取任务已取消。")
        async with resumable_store._lock:
            if resumable_store._buffers.get(buf.id) is buf:
                resumable_store._buffers.pop(buf.id, None)
        raise
    except Exception as e:
        msg = f"{type(e).__name__} - {e}"
        buf.saw_midstream_error = True
        print(f"❌ [续传缓冲] {buf.id} 上游读取失败：{msg}")
        err = {"error": {"message": str(e)[:1024], "type": "upstream_error", "code": 500}}
        await buf.append(frame_with_id(buf.next_seq, f"data: {json.dumps(err, ensure_ascii=False)}\n\n"))
        await buf.append(frame_with_id(buf.next_seq, DONE_FRAME))
        await buf.finish("error", error=msg)
        if not buf.has_model_output:
            await resumable_store.drop(buf.id)
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if aclose:
            try:
                await aclose()
            except Exception:
                pass


async def _cancel_if_abandoned(buf: ResumableBuffer, grace_seconds: float = 1.5) -> None:
    try:
        await asyncio.sleep(grace_seconds)
        if buf.readers <= 0 and not buf.done and not buf.has_model_output:
            async with resumable_store._lock:
                if resumable_store._buffers.get(buf.id) is buf:
                    print(f"🛑 [客户端取消] {buf.id} 前端已断开且尚未产出内容，立即终止后台上游请求与重试。")
                    resumable_store._cancel_pump_locked(buf.id)
                    resumable_store._buffers.pop(buf.id, None)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def sse_from_buffer(
    buf: ResumableBuffer,
    start_seq: int = 0,
    fastapi_request: Optional[Any] = None,
) -> AsyncIterator[str]:
    if start_seq > 0:
        print(f"↩️ [续传] {buf.id} 从事件 #{start_seq} 续传（缓冲区已有 {buf.next_seq} 个事件）。")
    delivered_done = False
    i = max(0, start_seq)
    buf.readers += 1
    try:
        while True:
            fake_on = bool(_setting("fake_streaming"))
            try:
                raw_interval = float(_setting("fake_streaming_interval") or 1.0)
            except (TypeError, ValueError):
                raw_interval = 1.0
            hb_interval = max(0.5, raw_interval) if fake_on else 3.0

            frame = await buf.wait_next(i, timeout=hb_interval)
            if frame is not None:
                yield frame
                if frame.strip().endswith("[DONE]") and buf.status != "error":
                    delivered_done = True
                i += 1
                continue

            if buf.done and i >= buf.next_seq:
                break

            if fastapi_request is not None:
                try:
                    if await fastapi_request.is_disconnected():
                        break
                except Exception:
                    pass

            if fake_on:
                yield make_keepalive_sse(buf.model)
            else:
                yield ": keep-alive\n\n"
    finally:
        buf.readers = max(0, buf.readers - 1)
        if buf.readers == 0:
            buf.last_reader_left_at = time.time()
            if not buf.done and not buf.has_model_output:
                asyncio.create_task(_cancel_if_abandoned(buf, grace_seconds=1.5))

    if delivered_done:
        buf.mark_delivered()