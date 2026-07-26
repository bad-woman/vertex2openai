# -*- coding: utf-8 -*-
"""P0-3：流式并行函数调用。

回归的 bug：旧实现遇到第一个 function_call 就 break，其余并行调用被丢弃；
且 delta.tool_calls[].index 恒为 0，OpenAI 客户端会把它们合并成一个。
官方要求并行调用必须完整按 FC1,FC2,FR1,FR2 回传，缺一个就 400。
"""
import json

import pytest

from api_helpers import ToolCallIndexer, convert_chunk_to_openai


class FakeFC:
    def __init__(self, name, args=None, fc_id=None):
        self.name = name
        self.args = args or {}
        self.id = fc_id


class FakePart:
    def __init__(self, fc=None, text=None, sig=None):
        self.function_call = fc
        self.text = text
        self.thought_signature = sig
        self.thought = False


class FakeContent:
    def __init__(self, parts):
        self.parts = parts


class FakeCandidate:
    def __init__(self, parts, finish_reason=None):
        self.content = FakeContent(parts)
        self.finish_reason = finish_reason
        self.safety_ratings = None


class FakeChunk:
    def __init__(self, parts, finish_reason=None):
        self.candidates = [FakeCandidate(parts, finish_reason)]


def parse(sse: str) -> dict:
    return json.loads(sse[len("data: "):].strip())


def test_parallel_calls_in_one_chunk_all_emitted():
    chunk = FakeChunk([
        FakePart(FakeFC("get_weather", {"city": "Tokyo"}, "fc1"), sig=b"s1"),
        FakePart(FakeFC("get_time", {"tz": "JST"}, "fc2")),
    ])
    payload = parse(convert_chunk_to_openai(chunk, "gemini-3.6-flash", "resp1", 0,
                                            indexer=ToolCallIndexer()))
    calls = payload["choices"][0]["delta"]["tool_calls"]
    assert len(calls) == 2, "并行函数调用被丢弃了"
    assert [c["index"] for c in calls] == [0, 1]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert calls[0]["id"] != calls[1]["id"]


def test_index_is_stable_across_chunks():
    """并行调用可能分布在不同 chunk 里，序号必须由调用方持有。"""
    indexer = ToolCallIndexer()
    first = parse(convert_chunk_to_openai(
        FakeChunk([FakePart(FakeFC("a", fc_id="fa"))]), "m", "r", 0, indexer=indexer))
    second = parse(convert_chunk_to_openai(
        FakeChunk([FakePart(FakeFC("b", fc_id="fb"))]), "m", "r", 0, indexer=indexer))
    assert first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert second["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_indexer_is_per_candidate():
    indexer = ToolCallIndexer()
    assert indexer.next_index(0) == 0
    assert indexer.next_index(1) == 0
    assert indexer.next_index(0) == 1


def test_arguments_serialized():
    payload = parse(convert_chunk_to_openai(
        FakeChunk([FakePart(FakeFC("f", {"k": "v"}, "fc"))]), "m", "r", 0,
        indexer=ToolCallIndexer()))
    call = payload["choices"][0]["delta"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"k": "v"}


def test_text_chunk_unaffected():
    payload = parse(convert_chunk_to_openai(
        FakeChunk([FakePart(text="hello")]), "m", "r", 0, indexer=ToolCallIndexer()))
    assert "tool_calls" not in payload["choices"][0]["delta"]
    assert payload["choices"][0]["delta"]["content"] == "hello"
