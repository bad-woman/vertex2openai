# -*- coding: utf-8 -*-
"""Cookie(batchGraphql) 通道解析基线回归。"""
import json
import asyncio

import pytest

from models import OpenAIMessage
import upstreams.cookie_proxy as cp


def _collect(obj):
    return list(cp._extract_from_results(obj))


def _payload(parts=None, finish=None, usage=None, errors=None):
    result = {}
    data = {}
    if parts is not None:
        cand = {"content": {"parts": parts}}
        if finish:
            cand["finishReason"] = finish
        data["candidates"] = [cand]
    if usage is not None:
        data["usageMetadata"] = usage
    if data:
        result["data"] = data
    if errors is not None:
        result["errors"] = errors
    return {"results": [result]}


def test_extract_text_and_thought():
    events = _collect(_payload(parts=[
        {"text": "思考中", "thought": True},
        {"text": "正文"},
    ]))
    assert ("thought", "思考中") in events
    assert ("text", "正文") in events


def test_extract_finish_and_usage():
    events = _collect(_payload(parts=[{"text": "x"}], finish="STOP",
                               usage={"promptTokenCount": 3, "candidatesTokenCount": 5}))
    kinds = [e[0] for e in events]
    assert "finish" in kinds and "usage" in kinds


def test_extract_inline_image():
    events = _collect(_payload(parts=[{"inlineData": {"mimeType": "image/png", "data": "QUJD"}}]))
    assert events and events[0][0] == "image" and "data:image/png;base64,QUJD" in events[0][1]


def test_extract_graphql_errors():
    events = _collect(_payload(errors=[{"message": "quota exceeded"}]))
    assert events[0][0] == "error"


def test_extract_top_level_error():
    events = _collect({"error": {"message": "bad"}})
    assert events[0][0] == "error"


def test_iter_json_objects_split_chunks():
    """跨 chunk 的对象拼接 + 一个 chunk 多个对象。"""
    obj1 = json.dumps({"a": 1, "s": "包含 } 的字符串"})
    obj2 = json.dumps({"b": [1, 2, {"c": "x\"y"}]})
    stream_text = f"[{obj1},\n{obj2}]"

    class _F:
        def __init__(self, chunks):
            self._chunks = chunks

        async def aiter_text(self):
            for c in self._chunks:
                yield c

    async def run(chunks):
        out = []
        async for o in cp._iter_json_objects(_F(chunks)):
            out.append(o)
        return out

    # 一次给全
    got = asyncio.run(run([stream_text]))
    assert got == [json.loads(obj1), json.loads(obj2)]
    # 按 3 字符切碎
    pieces = [stream_text[i:i + 3] for i in range(0, len(stream_text), 3)]
    got2 = asyncio.run(run(pieces))
    assert got2 == [json.loads(obj1), json.loads(obj2)]


def test_convert_messages_roles_and_system():
    msgs = [
        OpenAIMessage(role="system", content="你是猫"),
        OpenAIMessage(role="user", content="hi"),
        OpenAIMessage(role="assistant", content="喵"),
    ]
    contents, system_text = cp._convert_messages_to_contents(msgs)
    assert system_text == "你是猫"
    assert [c["role"] for c in contents] == ["user", "model"]
    assert contents[0]["parts"] == [{"text": "hi"}]


def test_convert_messages_data_url_image():
    msgs = [OpenAIMessage(role="user", content=[
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ])]
    contents, _ = cp._convert_messages_to_contents(msgs)
    parts = contents[0]["parts"]
    assert {"text": "看图"} in parts
    assert any("inlineData" in p and p["inlineData"]["data"] == "QUJD" for p in parts)


def test_retry_and_cookie_error_detection():
    assert cp._is_retryable_error("HTTP 429 Resource exhausted")
    assert cp._is_retryable_error("The model is overloaded")
    assert not cp._is_retryable_error("invalid argument")
    assert cp._is_cookie_expired_error("Permission denied on aiplatform.endpoints.predict")
    assert not cp._is_cookie_expired_error("resource exhausted")
