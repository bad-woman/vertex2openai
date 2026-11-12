"""Short-lived fallback storage for Gemini thought-signature topology.

OpenAI-compatible responses carry signatures in ``extra_content.google``. This
store is deliberately only a fallback for clients that strip that extension. A
record has three possible states because ``None`` alone is ambiguous:

* ``SIGNED``: this exact function-call part carried a signature.
* ``UNSIGNED_FOLLOWER``: this was a later call in a parallel batch and must stay
  unsigned when replayed.
* ``UNKNOWN``: signature topology was lost; validation may need the documented
  skip sentinel for the first call of a step.
"""

import base64
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

SKIP_VALIDATOR_SENTINEL = b"skip_thought_signature_validator"

DEFAULT_TTL_SECONDS = 72 * 60 * 60
DEFAULT_MAX_ENTRIES = 20000


class SignatureState(str, Enum):
    SIGNED = "signed"
    UNSIGNED_FOLLOWER = "unsigned_follower"
    NO_THOUGHT = "no_thought"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignatureRecord:
    state: SignatureState
    signature: Optional[bytes] = None

    def __post_init__(self) -> None:
        if self.state is SignatureState.SIGNED and not self.signature:
            raise ValueError("signed signature records require signature bytes")
        if self.state is not SignatureState.SIGNED and self.signature is not None:
            raise ValueError("unsigned/unknown signature records cannot carry bytes")


class SignatureStore:
    """Thread-safe tool_call_id -> :class:`SignatureRecord` TTL/LRU cache."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, SignatureRecord]]" = OrderedDict()

    def put_record(self, call_id: str, record: SignatureRecord) -> None:
        if not call_id:
            return
        now = time.time()
        with self._lock:
            self._data[call_id] = (now, record)
            self._data.move_to_end(call_id)
            self._evict(now)

    def put(self, call_id: str, signature: Optional[bytes]) -> None:
        """Backward-compatible signed-record writer."""
        if signature:
            self.put_record(call_id, SignatureRecord(SignatureState.SIGNED, signature))

    def put_unsigned_follower(self, call_id: str) -> None:
        self.put_record(call_id, SignatureRecord(SignatureState.UNSIGNED_FOLLOWER))

    def put_no_thought(self, call_id: str) -> None:
        self.put_record(call_id, SignatureRecord(SignatureState.NO_THOUGHT))

    def put_unknown(self, call_id: str) -> None:
        self.put_record(call_id, SignatureRecord(SignatureState.UNKNOWN))

    def get_record(self, call_id: str) -> Optional[SignatureRecord]:
        if not call_id:
            return None
        now = time.time()
        with self._lock:
            item = self._data.get(call_id)
            if item is None:
                return None
            ts, record = item
            if now - ts > self._ttl:
                self._data.pop(call_id, None)
                return None
            self._data.move_to_end(call_id)
            return record

    def get(self, call_id: str) -> Optional[bytes]:
        """Backward-compatible accessor returning bytes only for signed calls."""
        record = self.get_record(call_id)
        if record and record.state is SignatureState.SIGNED:
            return record.signature
        return None

    def _evict(self, now: float) -> None:
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
            states = {state.value: 0 for state in SignatureState}
            for _, record in self._data.values():
                states[record.state.value] += 1
            return {"entries": len(self._data), "ttl_seconds": self._ttl,
                    "max_entries": self._max, "states": states}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


signature_store = SignatureStore()


class ImageSignatureStore:
    """线程安全的图片哈希 (SHA-256) -> 思考签名 (thought_signature) TTL/LRU 缓存。
    
    用于在多轮生图/修图对话中，自动为客户端回传的历史图片恢复原生 thought_signature，
    解决 OpenAI 格式客户端剥离 Google 扩展字段后上游报错 400 (Image part is missing a thought_signature)。
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()

    @staticmethod
    def _hash(data: Union[bytes, str]) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def put(self, image_data: Union[bytes, str], signature: Optional[Union[bytes, str]]) -> None:
        if not image_data or not signature:
            return
        if isinstance(signature, str):
            try:
                sig_bytes = base64.b64decode(signature)
            except Exception:
                sig_bytes = signature.encode("utf-8")
        else:
            sig_bytes = signature

        h = self._hash(image_data)
        now = time.time()
        with self._lock:
            self._data[h] = (now, sig_bytes)
            self._data.move_to_end(h)
            self._evict(now)

    def get(self, image_data: Union[bytes, str]) -> Optional[bytes]:
        if not image_data:
            return None
        h = self._hash(image_data)
        now = time.time()
        with self._lock:
            item = self._data.get(h)
            if item is None:
                return None
            ts, sig_bytes = item
            if now - ts > self._ttl:
                self._data.pop(h, None)
                return None
            self._data.move_to_end(h)
            return sig_bytes

    def _evict(self, now: float) -> None:
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


image_signature_store = ImageSignatureStore()

