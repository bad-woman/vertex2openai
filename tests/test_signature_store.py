# -*- coding: utf-8 -*-
"""P0-4：思考签名的短 id 运输与三层降级。"""
import time

import pytest

import message_processing as mp
from signature_store import SKIP_VALIDATOR_SENTINEL, SignatureStore, signature_store


class FakeFC:
    def __init__(self, name="get_weather", fc_id=None):
        self.name = name
        self.id = fc_id
        self.args = {"city": "Tokyo"}


class FakePart:
    def __init__(self, sig=None):
        self.thought_signature = sig


LONG_SIG = b"x" * 900       # 真实签名量级：几百到上千字节


def test_tool_call_id_is_short():
    """核心回归：旧实现把 base64 签名拼进 id，动辄上千字符会被前端截断。"""
    call_id = mp.build_tool_call_id(FakeFC(), FakePart(LONG_SIG))
    assert len(call_id) <= 40, f"tool_call_id 过长：{len(call_id)}"


def test_signature_roundtrip_via_store():
    call_id = mp.build_tool_call_id(FakeFC(fc_id="fc_abc"), FakePart(LONG_SIG))
    real_id, sig = mp.resolve_tool_call_signature(call_id, require_signature=True)
    assert real_id == "fc_abc"
    assert sig == LONG_SIG


def test_upstream_id_is_preserved():
    call_id = mp.build_tool_call_id(FakeFC(fc_id="fc_from_google"), FakePart(LONG_SIG))
    assert call_id == "fc_from_google"


def test_generated_id_when_upstream_has_none():
    call_id = mp.build_tool_call_id(FakeFC(fc_id=None), FakePart(LONG_SIG))
    assert call_id.startswith("call_")
    assert mp.resolve_tool_call_signature(call_id)[1] == LONG_SIG


def test_fallback_to_sentinel_for_gemini3():
    """缓存丢失（重启/多进程）时必须降级到官方哨兵，而不是让请求 400。"""
    signature_store.clear()
    real_id, sig = mp.resolve_tool_call_signature("call_never_seen", require_signature=True)
    assert real_id == "call_never_seen"
    assert sig == SKIP_VALIDATOR_SENTINEL


def test_no_sentinel_when_not_required():
    signature_store.clear()
    assert mp.resolve_tool_call_signature("call_never_seen", require_signature=False)[1] is None


def test_legacy_embedded_format_still_parsed():
    """升级前发出的历史会话仍要能工作。"""
    import base64
    legacy = "fc_old" + mp.LEGACY_THOUGHT_SEP + base64.b64encode(LONG_SIG).decode()
    real_id, sig = mp.resolve_tool_call_signature(legacy)
    assert real_id == "fc_old"
    assert sig == LONG_SIG


def test_embed_mode_produces_legacy_format():
    call_id = mp.build_tool_call_id(FakeFC(fc_id="fc_x"), FakePart(LONG_SIG), True)
    assert mp.LEGACY_THOUGHT_SEP in call_id
    assert mp.resolve_tool_call_signature(call_id)[1] == LONG_SIG


def test_requires_signature_only_for_gemini3():
    assert mp._requires_signature("gemini-3.6-flash") is True
    assert mp._requires_signature("gemini-2.5-flash") is False
    assert mp._requires_signature("") is False          # 未知模型不注入哨兵


def test_store_ttl_expiry():
    store = SignatureStore(ttl_seconds=0, max_entries=10)
    store.put("a", b"sig")
    time.sleep(0.01)
    assert store.get("a") is None


def test_store_lru_eviction():
    store = SignatureStore(ttl_seconds=3600, max_entries=3)
    for i in range(5):
        store.put(f"k{i}", f"s{i}".encode())
    assert store.get("k0") is None      # 最早的被淘汰
    assert store.get("k4") == b"s4"


def test_store_ignores_empty():
    store = SignatureStore()
    store.put("", b"x")
    store.put("k", None)
    assert store.get("k") is None
