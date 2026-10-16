"""弱网可靠传输：服务端响应缓冲 + 序列号事件 + 断线重放 / 分片轮询拉取。

问题背景
--------
反代 → Google 的链路是好的，Google 的回复能完整到达反代；坏的是**前端 → 反代**这一段
（延迟抖动、丢包、连接中断）。原实现把上游的读取与前端连接绑死：
  - 前端连接一断，`fastapi_request.is_disconnected()` 变真，上游调用/重试被主动放弃；
  - 已经产出的增量只存在于那条 TCP 连接里，断了就永远找不回来，前端只能整轮重发
    （重新烧一次上游配额，且长回复在弱网下几乎不可能一次跑完）。

方案（不依赖单条长连接、不用心跳保活）
--------------------------------------
1. **响应落缓冲**：每个请求在服务端建一个缓冲区，上游产出的每个 SSE 帧按到达顺序
   追加进去并分配**单调递增的序列号 seq**。上游的读取由一个**后台任务**驱动，
   与前端连接完全解耦——前端断开不会取消它，Google 的回复总能被完整收下来。
2. **稳定请求 ID**：客户端应提供 `X-Idempotency-Key` / `X-Request-Id`，或在断线后
   回传响应头的 `X-Resume-Id`。没有显式 ID 时服务端总是生成新的随机 ID，避免把两个
   恰好相同的独立请求误合并。
3. **断线续传**：
   - SSE 路径：每帧前带 `id: <seq>`，重连时用 `Last-Event-ID` 或 `?resume_from=` 续取；
   - 轮询路径：`GET /v1/resumable/{stream_id}?from=<seq>&limit=&wait=` 拿一段分片（短连接，天生抗抖动），
     响应里带 `next` 游标与 `done` 标志，客户端按 seq 去重、按 done 判完成。
4. **幂等重放**：客户端用同一个幂等键原样重发时，若缓冲区里的回复**还没被完整交付**，
   就直接从 seq=0 重放已缓冲内容并继续跟随上游，**不再打一次 Google**；已经完整交付的，
   才当成一次新生成。
5. **生命周期与容量**：缓冲区在完成后保留 `resumable_ttl_seconds`，总数超过
   `resumable_max_streams` 时淘汰最旧的已完成缓冲；所有状态变更在锁/条件变量下进行。

为什么这能为支持续传协议的客户端提供完整一致的回复
-----------------------------------------------------------------------
回复的权威副本在服务端缓冲区里，且是**带序号的有序事件流**；前端连接只是这份副本的
一个游标读者。抖动 → 读者慢一点，数据仍在缓冲里等它；断线 → 换一条连接从 `next`
继续读，语义上等价于文件断点续传；同幂等键重发 → 命中同一缓冲区，得到同一份内容而
不是新的一次生成。服务重启、缓冲超限或客户端不执行续传协议时不保证恢复。
"""

import asyncio
import hashlib
import json
import re
import secrets
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import config as app_config
from runtime_state import app_state

# 终止帧：与 OpenAI 流一致，客户端见到它即可判定「本轮完整结束」
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


def _megabytes_setting(key: str, fallback_mb: int) -> int:
    try:
        return max(1, int(float(_setting(key))) * 1024 * 1024)
    except (TypeError, ValueError):
        return fallback_mb * 1024 * 1024


def max_stream_bytes() -> int:
    return _megabytes_setting("resumable_max_stream_mb", 16)


def max_total_bytes() -> int:
    return _megabytes_setting("resumable_max_total_mb", 128)


class ResumableCapacityError(RuntimeError):
    """服务端为了保护内存而拒绝创建或继续写入缓冲。"""


class ResumeIdentityMismatch(RuntimeError):
    """客户端拿一个续传 ID 去请求另一份内容或另一把 API Key。"""


_RESUME_ID_RE = re.compile(r"^(?:rsm|cid)-[A-Za-z0-9_-]{16,128}$")


def compute_stream_id(headers: Any, api_key: str, fingerprint: str) -> Tuple[str, str]:
    """决定这次请求的缓冲区 ID。返回 (id, 来源说明)。

    优先用客户端显式给的幂等键——客户端能准确声明“哪两次请求算同一次”。
    没有幂等键时必须生成新 ID，不能靠请求体猜测：两次相同提示词可能是两次独立生成。
    """
    try:
        resume_id = (headers.get("x-resume-id") or "").strip()
    except Exception:
        resume_id = ""
    # X-Resume-Id 是服务端响应给客户端的 opaque ID。路由层会继续校验 API Key
    # 和请求指纹，避免把它用到另一份请求上。
    if _RESUME_ID_RE.fullmatch(resume_id):
        return resume_id, "客户端回传 X-Resume-Id"

    for h in ("x-request-id", "x-idempotency-key", "idempotency-key"):
        try:
            v = (headers.get(h) or "").strip()
        except Exception:
            v = ""
        if v:
            # 幂等键只做“同一用户 + 同一请求体”之内的显式身份：混入 API Key 与请求指纹，
            # 避免不同用户复用同一键值时串缓冲（A 的回复被 B 读到），也避免同一键配不同
            # 请求体时误重放。客户端原样重发（同键 + 同体）仍稳定命中同一缓冲。
            digest = hashlib.sha256(f"{api_key}\x00{v}\x00{fingerprint}".encode("utf-8")).hexdigest()[:32]
            return f"cid-{digest}", f"客户端请求头 {h}"
    return f"rsm-{secrets.token_urlsafe(24)}", "新请求（未提供幂等键）"


class ResumableBuffer:
    """一次回复的服务端权威副本（有序、带序号、可多次读取）。"""

    def __init__(self, stream_id: str, model: str = "", kind: str = "stream",
                 owner_hash: str = "", fingerprint_hash: str = ""):
        self.id = stream_id
        self.model = model
        self.kind = kind                      # stream | json
        self.owner_hash = owner_hash
        self.fingerprint_hash = fingerprint_hash
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.last_access = self.created_at
        self.status = "running"               # running | done | error
        self.error: Optional[str] = None
        self.events: List[str] = []           # SSE 帧，下标即 seq
        self.byte_size = 0                     # events 在内存中的 UTF-8 字节数
        self.payload: Optional[dict] = None   # 非流式：完整 JSON 响应体
        self.status_code: int = 200
        self.fully_delivered = False          # 是否已有读者完整读到终止帧
        self.readers = 0
        self._cond = asyncio.Condition()

    # ---------- 写入端（后台泵任务持有） ----------

    async def append(self, frame: str, frame_bytes: int) -> int:
        async with self._cond:
            self.events.append(frame)
            self.byte_size += frame_bytes
            self._cond.notify_all()
            return len(self.events) - 1

    async def finish(self, status: str = "done", error: Optional[str] = None,
                     payload: Optional[dict] = None, status_code: Optional[int] = None) -> None:
        async with self._cond:
            self.status = status
            self.error = error
            if payload is not None:
                self.payload = payload
            if status_code is not None:
                self.status_code = status_code
            self.finished_at = time.time()
            self._cond.notify_all()

    # ---------- 读取端（可以有多个、可以中途离开再回来） ----------

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
        """等到出现 seq 号事件或本轮结束（或超时）。"""
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

    async def iter_from(self, seq: int = 0) -> AsyncIterator[Tuple[int, str]]:
        """从 seq 开始跟随读取，直到本轮结束且读完所有事件。"""
        i = max(0, seq)
        self.readers += 1
        try:
            while True:
                async with self._cond:
                    while len(self.events) <= i and self.status == "running":
                        await self._cond.wait()
                    if len(self.events) <= i:
                        break                     # 已结束且无新事件
                    frame = self.events[i]
                self.last_access = time.time()
                yield i, frame
                i += 1
        finally:
            self.readers = max(0, self.readers - 1)

    def mark_delivered(self) -> None:
        """读者已把终止帧交付出去：下一次同指纹请求应视为新生成，而不是重放。"""
        self.fully_delivered = True
        self.last_access = time.time()

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
        self._total_bytes = 0

    async def get(self, stream_id: str) -> Optional[ResumableBuffer]:
        async with self._lock:
            buf = self._buffers.get(stream_id)
            if buf:
                buf.last_access = time.time()
            return buf

    async def create_if_absent(self, stream_id: str, model: str = "",
                               kind: str = "stream", owner_hash: str = "",
                               fingerprint_hash: str = "") -> Tuple[ResumableBuffer, bool]:
        """原子地「占位建缓冲」。返回 (缓冲区, 是否是本次新建的)。

        `get` → 调上游 → `create` 三步之间有竞态窗口：并发的同一请求会各打一次上游，
        后建的缓冲还会把先建的覆盖掉（孤儿泵任务停机时也取消不到）。这里把「查 + 建」
        放进同一把锁：只有拿到 is_new=True 的那个请求去调上游，其余请求直接跟读同一份缓冲。
        """
        async with self._lock:
            buf = self._buffers.get(stream_id)
            if buf is not None:
                if (buf.owner_hash != owner_hash or
                        buf.fingerprint_hash != fingerprint_hash):
                    raise ResumeIdentityMismatch("续传 ID 与当前 API Key 或请求内容不匹配。")
                buf.last_access = time.time()
                return buf, False
            self._evict_finished_locked()
            if len(self._buffers) >= max_streams():
                raise ResumableCapacityError("续传缓冲数量已达上限；请稍后重试或调高上限。")
            buf = ResumableBuffer(stream_id, model=model, kind=kind,
                                  owner_hash=owner_hash, fingerprint_hash=fingerprint_hash)
            self._buffers[stream_id] = buf
            return buf, True

    async def append(self, buf: ResumableBuffer, frame: str, *, force: bool = False) -> int:
        """向仍被登记的缓冲写入一帧，并以字节为单位执行内存保护。"""
        frame_bytes = len(frame.encode("utf-8"))
        async with self._lock:
            if self._buffers.get(buf.id) is not buf:
                raise ResumableCapacityError("续传缓冲已被淘汰，已停止读取上游响应。")
            if not force:
                if buf.byte_size + frame_bytes > max_stream_bytes():
                    raise ResumableCapacityError("单条续传响应超过内存上限，已停止缓冲。")
                self._evict_finished_locked(required_bytes=frame_bytes)
                if self._total_bytes + frame_bytes > max_total_bytes():
                    raise ResumableCapacityError("续传缓冲总内存已达上限，已停止缓冲。")
            seq = await buf.append(frame, frame_bytes)
            self._total_bytes += frame_bytes
            return seq

    async def drop(self, stream_id: str) -> bool:
        async with self._lock:
            return self._remove_locked(stream_id, cancel_pump=True) is not None

    async def list_meta(self) -> List[dict]:
        async with self._lock:
            return [b.meta() for b in sorted(self._buffers.values(),
                                             key=lambda x: x.created_at, reverse=True)]

    def register_pump(self, stream_id: str, task: asyncio.Task) -> None:
        self._pumps[stream_id] = task
        def _forget(done_task: asyncio.Task, sid: str = stream_id) -> None:
            # 旧泵被取消后，不能把同一 stream_id 的新泵从登记表里误删。
            if self._pumps.get(sid) is done_task:
                self._pumps.pop(sid, None)
        task.add_done_callback(_forget)

    def cancel_all_pumps(self) -> None:
        """取消所有后台泵任务（服务停机时调用）。"""
        for task in list(self._pumps.values()):
            if not task.done():
                task.cancel()
        self._pumps.clear()

    def _remove_locked(self, stream_id: str, *, cancel_pump: bool) -> Optional[ResumableBuffer]:
        buf = self._buffers.pop(stream_id, None)
        if buf is None:
            return None
        self._total_bytes = max(0, self._total_bytes - buf.byte_size)
        if cancel_pump:
            task = self._pumps.pop(stream_id, None)
            if task and not task.done():
                task.cancel()
        return buf

    def _evict_finished_locked(self, required_bytes: int = 0) -> None:
        """只淘汰无人读取的已完成缓冲，绝不静默丢掉仍在生成的回复。"""
        finished = sorted((b for b in self._buffers.values() if b.done and b.readers == 0),
                          key=lambda x: x.finished_at or x.created_at)
        for buf in finished:
            if (len(self._buffers) < max_streams() and
                    self._total_bytes + required_bytes <= max_total_bytes()):
                return
            self._remove_locked(buf.id, cancel_pump=False)

    async def sweep(self) -> int:
        """清掉超过 TTL 的已完成缓冲（有读者在读的不动）。"""
        ttl = ttl_seconds()
        now = time.time()
        removed = 0
        async with self._lock:
            for sid, buf in list(self._buffers.items()):
                if buf.readers > 0:
                    continue
                ref = buf.finished_at or buf.last_access
                if buf.done and (now - ref) > ttl:
                    self._remove_locked(sid, cancel_pump=True)
                    removed += 1
                elif not buf.done and (now - buf.created_at) > max(ttl * 4, 1800):
                    # 极端兜底：上游永不结束的僵尸缓冲，必须一并取消泵任务。
                    self._remove_locked(sid, cancel_pump=True)
                    removed += 1
            self._evict_finished_locked()
        if removed:
            print(f"🧹 [续传缓冲] 已清理 {removed} 个过期响应缓冲。")
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
    """把上游调用与前端连接解绑的请求壳子。

    上游代码用 `fastapi_request.is_disconnected()` 决定要不要放弃调用/重试。
    弱网下前端连接随时会断，但**回复必须继续收完**放进缓冲区，所以这里恒返回 False；
    其余属性（app.state、headers 等）原样透传给被包装的真实 Request。
    """

    def __init__(self, request: Any):
        self._request = request

    async def is_disconnected(self) -> bool:
        return False

    def __getattr__(self, item):
        return getattr(self._request, item)


def _as_text(chunk: Any) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def frame_with_id(seq: int, frame: str) -> str:
    """给 SSE 数据帧加上事件 ID，供 Last-Event-ID 续传。

    注释帧（以 `:` 开头）不带 id —— 它不是事件，带 id 会污染客户端游标。
    """
    if frame.startswith("data:"):
        return f"id: {seq}\n{frame}"
    return frame


async def pump_stream(buf: ResumableBuffer, body_iterator: AsyncIterator[Any],
                      store: Optional[ResumableStore] = None) -> None:
    """后台把上游 SSE 帧抽进缓冲区。**不依赖前端连接**。

    每个 data 帧按落缓冲的顺序分配单调序列号并写入 `id:` 行，
    前端凭 Last-Event-ID / resume_from 续传时与这里的序号对齐。
    """
    store = store or resumable_store
    try:
        async for chunk in body_iterator:
            text = _as_text(chunk)
            if not text:
                continue
            if text.lstrip().startswith(":"):
                # 注释帧（旧心跳等）不是事件：不进缓冲、不占序号，保持 id == 下标
                continue
            await store.append(buf, frame_with_id(buf.next_seq, text))
        if not buf.events or not buf.events[-1].strip().endswith("[DONE]"):
            await store.append(buf, frame_with_id(buf.next_seq, DONE_FRAME))
        await buf.finish("done")
        print(f"📦 [续传缓冲] {buf.id} 上游已读完，共 {len(buf.events)} 个事件已完整落缓冲。")
    except ResumableCapacityError as e:
        # 不能再安全地保存全部响应，就终止上游；继续读取只会烧配额且无法续传。
        msg = str(e)
        print(f"⚠️ [续传缓冲] {buf.id} {msg}")
        err = {"error": {"message": msg, "type": "resumable_capacity", "code": 503}}
        try:
            await store.append(buf, frame_with_id(buf.next_seq, f"data: {json.dumps(err, ensure_ascii=False)}\n\n"), force=True)
            await store.append(buf, frame_with_id(buf.next_seq, DONE_FRAME), force=True)
        except ResumableCapacityError:
            pass
        await buf.finish("error", error=msg)
    except asyncio.CancelledError:
        await buf.finish("error", error="上游读取任务被取消。")
        raise
    except Exception as e:
        msg = f"{type(e).__name__} - {e}"
        print(f"❌ [续传缓冲] {buf.id} 上游读取失败：{msg}")
        err = {"error": {"message": str(e)[:1024], "type": "upstream_error", "code": 500}}
        try:
            await store.append(buf, frame_with_id(buf.next_seq, f"data: {json.dumps(err, ensure_ascii=False)}\n\n"), force=True)
            await store.append(buf, frame_with_id(buf.next_seq, DONE_FRAME), force=True)
        except ResumableCapacityError:
            pass
        await buf.finish("error", error=msg)
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if aclose:
            try:
                await aclose()
            except Exception:
                pass


async def sse_from_buffer(buf: ResumableBuffer, start_seq: int = 0) -> AsyncIterator[str]:
    """把缓冲区内容按 SSE 吐给当前这条前端连接（可从任意 seq 续取）。

    事件落缓冲时已带 `id: <seq>`（见 pump_stream），这里原样吐出，不再二次加工。
    """
    if start_seq > 0:
        print(f"↩️ [续传] {buf.id} 从事件 #{start_seq} 续传（缓冲区已有 {buf.next_seq} 个事件）。")
    delivered_done = False
    async for _seq, frame in buf.iter_from(start_seq):
        yield frame
        if frame.strip().endswith("[DONE]"):
            delivered_done = True
    if delivered_done:
        buf.mark_delivered()
