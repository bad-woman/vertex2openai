# -*- coding: utf-8 -*-
"""Cookie 通道加固：finishReason 映射 / thought 健壮判定 / 拦截透出 /
连续同角色合并 / 参数注入对齐 / 无正文诊断（流式与非流式端到端）。"""
import json
import asyncio

import pytest

from models import OpenAIRequest, OpenAIMessage
from runtime_state import app_state
import upstreams.cookie_proxy as cp


# ---------- 纯函数 ----------

def test_map_finish_reason():
    assert cp._map_finish_reason("STOP") == "stop"
    assert cp._map_finish_reason("MAX_TOKENS") == "length"
    assert cp._map_finish_reason("SAFETY") == "content_filter"
    assert cp._map_finish_reason("PROHIBITED_CONTENT") == "content_filter"
    assert cp._map_finish_reason("RECITATION") == "content_filter"
    assert cp._map_finish_reason(None) == "stop"
    assert cp._map_finish_reason("SOMETHING_NEW") == "stop"


def test_is_thought_part_robust():
    assert cp._is_thought_part({"thought": True})
    assert cp._is_thought_part({"thought": "true"})
    assert cp._is_thought_part({"thought": "True"})
    assert not cp._is_thought_part({"thought": False})
    assert not cp._is_thought_part({"thought": "false"})  # 字符串 "false" 不能误判为思考
    assert not cp._is_thought_part({})


def test_extract_yields_any_finish_reason():
    obj = {"results": [{"data": {"candidates": [
        {"content": {"parts": [{"text": "t", "thought": True}]}, "finishReason": "PROHIBITED_CONTENT"},
    ]}}]}
    events = list(cp._extract_from_results(obj))
    assert ("finish", "PROHIBITED_CONTENT") in events


def test_extract_yields_blocked():
    obj = {"results": [{"data": {"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "x"}}}]}
    events = list(cp._extract_from_results(obj))
    assert events and events[0][0] == "blocked" and "SAFETY" in events[0][1]


def test_extract_filters_unspecified_noise():
    """batchGraphql 每块都带 *_UNSPECIFIED 枚举默认值，必须过滤，不能误判为拦截/结束。"""
    obj = {"results": [{"data": {
        "promptFeedback": {"blockReason": "BLOCKED_REASON_UNSPECIFIED"},
        "candidates": [{"content": {"parts": [{"text": "正文"}]},
                        "finishReason": "FINISH_REASON_UNSPECIFIED"}],
    }}]}
    events = list(cp._extract_from_results(obj))
    kinds = [e[0] for e in events]
    assert "blocked" not in kinds and "finish" not in kinds
    assert ("text", "正文") in events


def test_convert_messages_merges_consecutive_roles():
    msgs = [
        OpenAIMessage(role="user", content="a"),
        OpenAIMessage(role="user", content="b"),
        OpenAIMessage(role="assistant", content="c"),
        # 低层函数仍把未知角色折成 model；含工具的请求已在 chat_completions 入口被
        # has_tool_traffic() 拒绝（见 test_cookie_tool_rejection.py），这里只测合并逻辑
        OpenAIMessage(role="tool", content="d"),
        OpenAIMessage(role="user", content="e"),
    ]
    contents, _ = cp._convert_messages_to_contents(msgs)
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    texts0 = [p.get("text") for p in contents[0]["parts"]]
    assert "a" in texts0 and "b" in texts0


def test_summ_obj_strips_base64():
    s = cp._summ_obj({"inlineData": {"data": "A" * 200}})
    assert "A" * 200 not in s and "省略" in s


def test_raw_sampler_head_tail():
    sm = cp._RawSampler(keep=2)
    for i in range(6):
        sm.add({"i": i})
    d = sm.dump()
    assert "共 6 个对象" in d and '"i": 0' in d and '"i": 5' in d


# ---------- 参数注入对齐 ----------

def _req(model="gemini-3.6-flash", **kw):
    return OpenAIRequest(model=model, messages=[OpenAIMessage(role="user", content="hi")], **kw)


def test_body_no_fabricated_defaults():
    body = cp._build_batch_graphql_body("proj", "gemini-3.6-flash", _req())
    gc = body["variables"]["generationConfig"]
    assert "temperature" not in gc and "topP" not in gc and "maxOutputTokens" not in gc
    assert gc["thinkingConfig"]["thinkingLevel"] == "MEDIUM"


def test_body_respects_console_defaults():
    app_state.update_settings({"default_temperature": 0.6, "default_top_p": 0.9, "default_max_tokens": 4096})
    body = cp._build_batch_graphql_body("proj", "gemini-2.5-flash", _req(model="gemini-2.5-flash"))
    gc = body["variables"]["generationConfig"]
    assert gc["temperature"] == 0.6 and gc["topP"] == 0.9 and gc["maxOutputTokens"] == 4096
    # 3.6-flash：temperature/topP 被能力矩阵剥离，max_output_tokens 保留
    body2 = cp._build_batch_graphql_body("proj", "gemini-3.6-flash", _req())
    gc2 = body2["variables"]["generationConfig"]
    assert "temperature" not in gc2 and "topP" not in gc2 and gc2["maxOutputTokens"] == 4096


def test_body_request_overrides_and_max_completion_tokens():
    body = cp._build_batch_graphql_body("proj", "gemini-2.5-flash",
                                        _req(model="gemini-2.5-flash", temperature=1.3, max_completion_tokens=321))
    gc = body["variables"]["generationConfig"]
    assert gc["temperature"] == 1.3 and gc["maxOutputTokens"] == 321


def test_body_search_suffix_tool():
    r = _req(model="gemini-2.5-flash-search")
    body = cp._build_batch_graphql_body("proj", "gemini-2.5-flash", r)
    assert body["variables"]["tools"] == [{"googleSearch": {}}]


# ---------- 流式端到端（monkeypatch 执行器） ----------

class _FakeFastapiRequest:
    async def is_disconnected(self):
        return False


def _setup_cookie_auth():
    app_state.set_google_cookie("SAPISID=abc123; SID=x")
    app_state.set_project_id("proj-1")


def _stream_req(**kw):
    return OpenAIRequest(model="gemini-3.6-flash", stream=True,
                         messages=[OpenAIMessage(role="user", content="hi")], **kw)


async def _run_stream(monkeypatch, events, request_obj):
    """monkeypatch 掉底层执行器，收集 chat_completions 的 SSE 输出。"""
    async def fake_exec(client, headers, body, sampler=None):
        for e in events:
            if sampler is not None:
                sampler.add({"fake": True})
            yield e

    monkeypatch.setattr(cp, "_execute_stream_request_generator", fake_exec)
    _setup_cookie_auth()
    upstream = cp.CookieProxyUpstream()
    resp = await upstream.chat_completions(request_obj, _FakeFastapiRequest())
    chunks = []
    async for c in resp.body_iterator:
        chunks.append(c)
    return chunks


def _parse_sse(chunks):
    out = []
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[len("data: "):]))
    return out


def test_stream_normal_text(monkeypatch):
    chunks = asyncio.run(_run_stream(monkeypatch, [("text", "你好"), ("finish", "STOP")], _stream_req()))
    objs = _parse_sse(chunks)
    contents = [o["choices"][0]["delta"].get("content") for o in objs if o["choices"]]
    assert "你好" in contents
    finishes = [o["choices"][0].get("finish_reason") for o in objs if o["choices"]]
    assert "stop" in finishes
    assert any("[DONE]" in c for c in chunks)


def test_stream_starts_with_heartbeat(monkeypatch):
    """issue4：流式一开始就吐 SSE 心跳，尽快建连、防前端超时。"""
    chunks = asyncio.run(_run_stream(monkeypatch, [("text", "hi"), ("finish", "STOP")], _stream_req()))
    assert chunks and chunks[0].startswith(": keep-alive")


def test_stream_heartbeat_during_retry(monkeypatch):
    """issue4：首次可重试错误后的退避等待期间持续吐心跳，然后重试成功。"""
    calls = {"n": 0}

    async def fake_exec(client, headers, body, sampler=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("retryable_error", "429 resource exhausted")
        else:
            yield ("text", "重试后的正文")
            yield ("finish", "STOP")

    monkeypatch.setattr(cp, "_execute_stream_request_generator", fake_exec)
    # 缩短退避，加速测试
    app_state.update_settings({"retry_backoff_seconds": 1, "retry_max": 3})
    _setup_cookie_auth()

    async def run():
        resp = await cp.CookieProxyUpstream().chat_completions(_stream_req(), _FakeFastapiRequest())
        return [c async for c in resp.body_iterator]

    chunks = asyncio.run(run())
    assert sum(1 for c in chunks if c.startswith(": keep-alive")) >= 2  # 起始 + 退避期间
    full = "".join(
        o["choices"][0]["delta"].get("content") or ""
        for o in _parse_sse(chunks) if o.get("choices")
    )
    assert "重试后的正文" in full
    assert calls["n"] == 2


def test_stream_strips_thoughts_when_mode_off(monkeypatch):
    """核心修复：batchGraphql 忽略 includeThoughts，故 native_thinking_mode=off 时
    本通道在响应侧剥离思考块（即使上游仍回传 thought），只留正文。"""
    app_state.update_settings({"native_thinking_mode": "off"})
    chunks = asyncio.run(_run_stream(
        monkeypatch,
        [("thought", "原生思考应被剥离"), ("text", "这是正文"), ("finish", "STOP")],
        _stream_req(),
    ))
    objs = _parse_sse(chunks)
    reasonings = "".join(o["choices"][0]["delta"].get("reasoning_content") or "" for o in objs if o["choices"])
    contents = "".join(o["choices"][0]["delta"].get("content") or "" for o in objs if o["choices"])
    assert reasonings == ""          # 思考被剥离
    assert "这是正文" in contents     # 正文保留
    assert any("[DONE]" in c for c in chunks)


def test_stream_keeps_thoughts_when_mode_request(monkeypatch):
    app_state.update_settings({"native_thinking_mode": "request"})
    chunks = asyncio.run(_run_stream(
        monkeypatch,
        [("thought", "保留的思考"), ("text", "正文"), ("finish", "STOP")],
        _stream_req(),
    ))
    objs = _parse_sse(chunks)
    reasonings = "".join(o["choices"][0]["delta"].get("reasoning_content") or "" for o in objs if o["choices"])
    assert "保留的思考" in reasonings


def test_stream_thought_only_visible_diagnostic(monkeypatch, capsys):
    """用户报告的场景：只有思考、没有正文 → 必须给出可见提示与诊断日志，而非静默。"""
    before = cp.stats.get_json_stats()
    chunks = asyncio.run(_run_stream(
        monkeypatch, [("thought", "思考…"), ("finish", "SAFETY")], _stream_req()))
    objs = _parse_sse(chunks)
    reasonings = [o["choices"][0]["delta"].get("reasoning_content") for o in objs if o["choices"]]
    assert "思考…" in reasonings
    contents = "".join(o["choices"][0]["delta"].get("content") or "" for o in objs if o["choices"])
    assert "只返回了思考" in contents and "SAFETY" in contents
    finishes = [o["choices"][0].get("finish_reason") for o in objs if o["choices"]]
    assert "content_filter" in finishes
    out = capsys.readouterr().out
    assert "🔎" in out  # 诊断样本已落日志
    after = cp.stats.get_json_stats()
    assert after["error"] - before["error"] == 1
    assert any("[DONE]" in c for c in chunks)


def test_stream_totally_empty_not_silent(monkeypatch):
    """旧版 bug：上游空响应时一个字节都不发就关流；现在必须有明确输出与 [DONE]。"""
    chunks = asyncio.run(_run_stream(monkeypatch, [], _stream_req()))
    objs = _parse_sse(chunks)
    assert objs, "空响应不能再静默关流"
    contents = "".join(o["choices"][0]["delta"].get("content") or "" for o in objs if o["choices"])
    assert "未返回任何内容" in contents
    assert any("[DONE]" in c for c in chunks)


def test_stream_blocked_maps_content_filter(monkeypatch):
    chunks = asyncio.run(_run_stream(monkeypatch, [("blocked", "SAFETY（触发拦截）")], _stream_req()))
    objs = _parse_sse(chunks)
    contents = "".join(o["choices"][0]["delta"].get("content") or "" for o in objs if o["choices"])
    assert "promptFeedback 拦截" in contents
    finishes = [o["choices"][0].get("finish_reason") for o in objs if o["choices"]]
    assert "content_filter" in finishes


def test_stream_prefill_spliced_and_deduped(monkeypatch):
    """预填充先行发出；模型复述预填充开头时自动去重。"""
    app_state.update_settings({"prefill_mode": "smart"})
    req = OpenAIRequest(model="gemini-3.6-flash", stream=True, messages=[
        OpenAIMessage(role="user", content="hi"),
        OpenAIMessage(role="assistant", content="从前有座山，"),
    ])
    chunks = asyncio.run(_run_stream(
        monkeypatch, [("text", "从前有座山，山里有座庙。"), ("finish", "STOP")], req))
    objs = _parse_sse(chunks)
    full = "".join(o["choices"][0]["delta"].get("content") or "" for o in objs if o["choices"])
    assert full == "从前有座山，山里有座庙。"  # 预填充只出现一次


# ---------- 非流式端到端（monkeypatch httpx.AsyncClient） ----------

class _FakePostResp:
    def __init__(self, payload_text, status_code=200):
        self.status_code = status_code
        self.text = payload_text


class _FakeAsyncClient:
    payload = ""

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        return _FakePostResp(_FakeAsyncClient.payload)


def _batch_payload(parts, finish="STOP"):
    return json.dumps({"results": [{"data": {"candidates": [
        {"content": {"parts": parts}, "finishReason": finish}
    ]}}]})


def test_nonstream_thought_only_diagnostic(monkeypatch):
    _setup_cookie_auth()
    _FakeAsyncClient.payload = _batch_payload([{"text": "思考", "thought": True}], finish="SAFETY")
    monkeypatch.setattr(cp.httpx, "AsyncClient", _FakeAsyncClient)
    req = OpenAIRequest(model="gemini-3.6-flash", stream=False,
                        messages=[OpenAIMessage(role="user", content="hi")])
    resp = asyncio.run(cp.CookieProxyUpstream().chat_completions(req, _FakeFastapiRequest()))
    data = json.loads(resp.body)
    msg = data["choices"][0]["message"]
    assert "只返回了思考" in msg["content"]
    assert msg["reasoning_content"] == "思考"
    assert data["choices"][0]["finish_reason"] == "content_filter"


def test_nonstream_normal_with_prefill_dedup(monkeypatch):
    _setup_cookie_auth()
    app_state.update_settings({"prefill_mode": "smart"})
    _FakeAsyncClient.payload = _batch_payload([{"text": "从前有座山，山里有座庙。"}])
    monkeypatch.setattr(cp.httpx, "AsyncClient", _FakeAsyncClient)
    req = OpenAIRequest(model="gemini-3.6-flash", stream=False, messages=[
        OpenAIMessage(role="user", content="hi"),
        OpenAIMessage(role="assistant", content="从前有座山，"),
    ])
    resp = asyncio.run(cp.CookieProxyUpstream().chat_completions(req, _FakeFastapiRequest()))
    data = json.loads(resp.body)
    assert data["choices"][0]["message"]["content"] == "从前有座山，山里有座庙。"
    assert data["choices"][0]["finish_reason"] == "stop"
