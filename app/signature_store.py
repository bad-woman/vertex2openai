"""思考签名（thought signature）旁路缓存。

背景（官方文档，核对于 2026-07-26）：
- 签名可以出现在任意 content part 上（text / functionCall）。
- **函数调用是强校验**：Gemini 3 拿不到签名直接 400。
- 纯文本不强校验，但省略会降低推理质量。
- 实在拿不到签名时，可把 thought_signature 设为 `skip_thought_signature_validator`
  跳过校验，官方明确这是最后手段、会损害模型表现。

旧实现把 base64 签名直接拼进 OpenAI 的 `tool_call_id`
（`{id}__thought__{base64}`）。签名通常几百到上千字符，很多前端对 tool_call_id
有长度限制或会截断，一旦截断回传时就解不出签名 → 400，且没有任何降级路径。

现在的策略是三层：
  1. 出站时生成短 id（≤40 字符），签名存进这里的 TTL LRU；回传时按 id 取回。
  2. 缓存未命中时，仍尝试解析旧的 `__thought__` 内嵌格式（向后兼容历史会话）。
  3. 都拿不到时，用官方哨兵值跳过校验，保证请求不会整体失败。

局限：仅进程内有效。多 worker 或重启后落到第 2/3 层。
这是有意的取舍——把签名持久化到磁盘意味着把模型内部状态落盘，收益不抵复杂度。
"""

import threading
import time
from collections import OrderedDict
from typing import Optional

# 官方允许的跳过校验哨兵值（会降低模型表现，仅作最后手段）
SKIP_VALIDATOR_SENTINEL = b"skip_thought_signature_validator"

DEFAULT_TTL_SECONDS = 2 * 60 * 60   # 2 小时：远长于一轮工具往返，远短于无限增长
DEFAULT_MAX_ENTRIES = 5000


class SignatureStore:
    """tool_call_id -> thought_signature 的短期缓存（线程安全）。"""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()

    def put(self, call_id: str, signature: Optional[bytes]) -> None:
        if not call_id or not signature:
            return
        now = time.time()
        with self._lock:
            self._data[call_id] = (now, signature)
            self._data.move_to_end(call_id)
            self._evict(now)

    def get(self, call_id: str) -> Optional[bytes]:
        if not call_id:
            return None
        now = time.time()
        with self._lock:
            item = self._data.get(call_id)
            if item is None:
                return None
            ts, sig = item
            if now - ts > self._ttl:
                self._data.pop(call_id, None)
                return None
            self._data.move_to_end(call_id)
            return sig

    def _evict(self, now: float) -> None:
        """调用方须持锁。先按 TTL 清理，再按容量淘汰最久未用的。"""
        while self._data:
            oldest_key = next(iter(self._data))
            ts, _ = self._data[oldest_key]
            if now - ts > self._ttl:
                self._data.pop(oldest_key, None)
                continue
            break
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._data), "ttl_seconds": self._ttl, "max_entries": self._max}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# 单例
signature_store = SignatureStore()
