# -*- coding: utf-8 -*-
"""预填充增强：原生透传 / 指令模板 / 思考压制 / 标准通道去重接线。"""
import json

from models import OpenAIRequest, OpenAIMessage
from message_processing import apply_prefill_compat, DEFAULT_PREFILL_INSTRUCTION, PrefillDeduper
from api_helpers import _prepend_prefill, _dedup_sse_chunk_content
import model_capabilities as mc
import upstreams.cookie_proxy as cp
from upstreams.express_sdk import _build_thinking_config
from runtime_state import app_state


def _msgs(*pairs):
    return [OpenAIMessage(role=r, content=c) for r, c in pairs]


# ---------- apply_prefill_compat 增强 ----------

def test_native_passthrough_when_model_last_allowed():
    msgs = _msgs(("user", "hi"), ("assistant", "从前有座山，"))
    new_msgs, prefill, active = apply_prefill_compat(msgs, "smart", allow_model_last=True)
    assert new_msgs is msgs            # 消息原样透传
    assert prefill == "从前有座山，"
    assert active is True


def test_conversion_when_model_last_forbidden():
    msgs = _msgs(("user", "hi"), ("assistant", "pf"))
    new_msgs, prefill, active = apply_prefill_compat(msgs, "smart", allow_model_last=False)
    assert new_msgs is not msgs and new_msgs[-1].role == "user"
    assert prefill == "pf" and active is True
    assert DEFAULT_PREFILL_INSTRUCTION.split("\n")[0][:6] in new_msgs[-1].content


def test_custom_instruction_template():
    msgs = _msgs(("user", "hi"), ("assistant", "pf"))
    new_msgs, _, _ = apply_prefill_compat(msgs, "smart", instruction_template="【自定义模板】继续：")
    assert "【自定义模板】继续：" in new_msgs[-1].content
    assert "pf" in new_msgs[-1].content
    assert DEFAULT_PREFILL_INSTRUCTION not in new_msgs[-1].content


def test_active_flags():
    # 无预填充 → False
    assert apply_prefill_compat(_msgs(("user", "hi")), "smart")[2] is False
    # minimal 模式检测到预填充 → True（用于思考压制联动）
    assert apply_prefill_compat(_msgs(("user", "hi"), ("assistant", "pf")), "minimal")[2] is True
    # off → False
    assert apply_prefill_compat(_msgs(("user", "hi"), ("assistant", "pf")), "off")[2] is False


# ---------- 思考压制（resolve_thinking + prefill_active） ----------

def _req(**kw):
    return OpenAIRequest(model=kw.pop("model", "gemini-3.6-flash"),
                         messages=[OpenAIMessage(role="user", content="hi")], **kw)


def test_suppress_g3_minimal_and_hidden():
    t = mc.resolve_thinking("gemini-3.6-flash", _req(), {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t == {"mode": "level", "level": "minimal", "include_thoughts": False}
    # pro 无 minimal → low
    t2 = mc.resolve_thinking("gemini-3.1-pro-preview", _req(), {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t2["level"] == "low" and t2["include_thoughts"] is False


def test_suppress_g25_budget_zero_or_floor():
    t = mc.resolve_thinking("gemini-2.5-flash", _req(model="gemini-2.5-flash"),
                            {"prefill_suppress_thinking": True, "thinking_g25_budget": 8192}, prefill_active=True)
    assert t == {"mode": "budget", "budget": 0, "include_thoughts": False}  # flash 全关，覆盖控制台预算
    t2 = mc.resolve_thinking("gemini-2.5-pro", _req(model="gemini-2.5-pro"),
                             {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t2 == {"mode": "budget", "budget": 128, "include_thoughts": False}  # pro 降到最低


def test_suppress_disabled_by_console():
    t = mc.resolve_thinking("gemini-3.6-flash", _req(), {"prefill_suppress_thinking": False}, prefill_active=True)
    assert t["level"] == "medium" and t["include_thoughts"] is True  # 不压制 → 走默认


def test_suppress_now_overrides_client_effort():
    # 变更：预填充压制现在优先于前端 effort（酒馆预设恒发 effort 也能卡住思维链）
    r = _req(reasoning_effort="high")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t["level"] == "minimal" and t["include_thoughts"] is False
    r2 = OpenAIRequest(model="gemini-2.5-flash", thinking_budget=4096,
                       messages=[OpenAIMessage(role="user", content="hi")])
    t2 = mc.resolve_thinking("gemini-2.5-flash", r2, {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t2["budget"] == 0 and t2["include_thoughts"] is False


def test_no_prefill_no_suppress():
    t = mc.resolve_thinking("gemini-3.6-flash", _req(), {"prefill_suppress_thinking": True}, prefill_active=False)
    assert t["level"] == "medium" and t["include_thoughts"] is True


# ---------- native_thinking_mode（request / off / console） ----------

def test_mode_off_forces_minimal_and_hides():
    # SillyTavern 恒发 xhigh；mode=off → 忽略前端、压 minimal、不返回思考（issue1+5 修复）
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"native_thinking_mode": "off", "thinking_g3_level": "high"})
    assert t == {"mode": "level", "level": "minimal", "include_thoughts": False}


def test_mode_off_g25_budget_zero_and_hide():
    r = OpenAIRequest(model="gemini-2.5-flash", reasoning_effort="xhigh",
                      messages=[OpenAIMessage(role="user", content="hi")])
    t = mc.resolve_thinking("gemini-2.5-flash", r, {"native_thinking_mode": "off"})
    assert t == {"mode": "budget", "budget": 0, "include_thoughts": False}


def test_mode_console_ignores_client_uses_console_level():
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"native_thinking_mode": "console", "thinking_g3_level": "low"})
    assert t == {"mode": "level", "level": "low", "include_thoughts": True}


def test_mode_console_falls_to_model_default():
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"native_thinking_mode": "console"})
    assert t["level"] == "medium" and t["include_thoughts"] is True


def test_mode_request_default_client_wins():
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"native_thinking_mode": "request", "thinking_g3_level": "minimal"})
    assert t["level"] == "high" and t["include_thoughts"] is True  # 前端 xhigh→high 优先


def test_backcompat_hide_thoughts_maps_to_off():
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"hide_thoughts": True})
    assert t["level"] == "minimal" and t["include_thoughts"] is False


def test_backcompat_force_console_maps_to_console():
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"thinking_force_console": True, "thinking_g3_level": "low"})
    assert t["level"] == "low" and t["include_thoughts"] is True


def test_effort_aliases_normalized():
    for raw, exp in [("xhigh", "high"), ("max", "high"), ("min", "minimal"), ("MEDIUM", "medium")]:
        r = _req(reasoning_effort=raw)
        assert mc.resolve_thinking("gemini-3.6-flash", r, {})["level"] == exp
    # auto/空 → 用模型默认
    assert mc.resolve_thinking("gemini-3.6-flash", _req(reasoning_effort="auto"), {})["level"] == "medium"


def test_suppress_ignores_client_effort_now():
    # 预填充压制现在即使前端发 effort 也生效（此前被 effort 旁路）
    r = _req(reasoning_effort="xhigh")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"prefill_suppress_thinking": True}, prefill_active=True)
    assert t == {"mode": "level", "level": "minimal", "include_thoughts": False}


# ---------- 通道接线 ----------

def test_cookie_thinking_config_suppressed():
    app_state.update_settings({"prefill_suppress_thinking": True})
    cfg = cp._build_thinking_config("gemini-3.6-flash", _req(), prefill_active=True)
    assert cfg == {"thinkingLevel": "MINIMAL", "includeThoughts": False}
    cfg2 = cp._build_thinking_config("gemini-2.5-flash", _req(model="gemini-2.5-flash"), prefill_active=True)
    assert cfg2 == {"thinkingBudget": 0, "includeThoughts": False}


def test_cookie_thinking_config_mode_off():
    app_state.update_settings({"native_thinking_mode": "off"})
    cfg = cp._build_thinking_config("gemini-3.6-flash", _req(reasoning_effort="xhigh"))
    assert cfg == {"thinkingLevel": "MINIMAL", "includeThoughts": False}


def test_cookie_body_thinking_suppressed():
    app_state.update_settings({"prefill_suppress_thinking": True})
    body = cp._build_batch_graphql_body("proj", "gemini-3.6-flash", _req(), prefill_active=True)
    tc = body["variables"]["generationConfig"]["thinkingConfig"]
    assert tc == {"thinkingLevel": "MINIMAL", "includeThoughts": False}


def test_express_thinking_config_suppressed():
    app_state.update_settings({"prefill_suppress_thinking": True})
    cfg = _build_thinking_config("gemini-3.6-flash", _req(), False, prefill_active=True)
    assert cfg == {"include_thoughts": False, "thinking_level": "minimal"}


def test_express_thinking_config_mode_off():
    app_state.update_settings({"native_thinking_mode": "off"})
    cfg = _build_thinking_config("gemini-3.6-flash", _req(reasoning_effort="xhigh"), False)
    assert cfg == {"include_thoughts": False, "thinking_level": "minimal"}


def test_cookie_native_passthrough_g25(monkeypatch):
    """2.5 模型经 Cookie 通道：消息不转换，contents 以 model 结尾。"""
    captured = {}
    orig = cp._build_batch_graphql_body

    def spy(project_id, model_name, request, prefill_active=False):
        body = orig(project_id, model_name, request, prefill_active)
        captured["contents"] = body["variables"]["contents"]
        captured["prefill_active"] = prefill_active
        return body

    monkeypatch.setattr(cp, "_build_batch_graphql_body", spy)

    async def fake_exec(client, headers, body, sampler=None):
        yield ("text", "山里有座庙。")
        yield ("finish", "STOP")

    monkeypatch.setattr(cp, "_execute_stream_request_generator", fake_exec)
    app_state.set_google_cookie("SAPISID=abc; SID=x")
    app_state.set_project_id("p")
    app_state.update_settings({"prefill_mode": "smart"})

    import asyncio

    class _FR:
        async def is_disconnected(self):
            return False

    req = OpenAIRequest(model="gemini-2.5-flash", stream=True, messages=[
        OpenAIMessage(role="user", content="hi"),
        OpenAIMessage(role="assistant", content="从前有座山，"),
    ])

    async def run():
        resp = await cp.CookieProxyUpstream().chat_completions(req, _FR())
        return [c async for c in resp.body_iterator]

    chunks = asyncio.run(run())
    assert captured["contents"][-1]["role"] == "model"  # 原生透传保留末尾 model
    assert captured["prefill_active"] is True
    full = ""
    for c in chunks:
        for line in c.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                o = json.loads(line[6:])
                if o.get("choices"):
                    full += o["choices"][0]["delta"].get("content") or ""
    assert full == "从前有座山，山里有座庙。"


# ---------- 标准通道输出去重 ----------

def test_prepend_prefill_dedup():
    d = {"choices": [{"message": {"role": "assistant", "content": "从前有座山，山里有座庙。"}}]}
    _prepend_prefill(d, "从前有座山，")
    assert d["choices"][0]["message"]["content"] == "从前有座山，山里有座庙。"


def _sse(content=None, finish=None, **delta_extra):
    delta = dict(delta_extra)
    if content is not None:
        delta["content"] = content
    return "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 1,
                                  "model": "m", "choices": [{"index": 0, "delta": delta,
                                                             "finish_reason": finish}]}) + "\n\n"


def test_dedup_sse_stream_echo():
    pf = "从前有座山，这是一段足够长的预填充开头内容啊"
    cont = "然后故事这样继续发展。" * 10  # 足够超过去重窗口
    d = PrefillDeduper(pf)
    out1 = _dedup_sse_chunk_content(_sse(content=pf[:10]), d)
    assert out1 is None  # 攒住
    out2 = _dedup_sse_chunk_content(_sse(content=pf[10:] + cont), d)
    payload = json.loads(out2[6:])
    assert payload["choices"][0]["delta"]["content"] == cont  # 复述的预填充被裁掉
    # 判定完成后透传
    out3 = _dedup_sse_chunk_content(_sse(content="尾部"), d)
    assert json.loads(out3[6:])["choices"][0]["delta"]["content"] == "尾部"


def _collect_sse_content(chunks):
    """把若干 SSE chunk 里的 delta.content 拼起来（放行时机可能变，总量不能变）。"""
    text = ""
    for c in chunks:
        if c is None:
            continue
        text += json.loads(c[6:])["choices"][0]["delta"].get("content") or ""
    return text


def test_dedup_sse_finish_forces_flush():
    """finish chunk 必须把攒着的正文一并放出，且总正文量守恒。

    P2-4 之后，与预填充无重叠的正文会更早放行（out1 不再必然是 None），
    因此断言改为检查**聚合结果**而不是某一个 chunk 的内容。
    """
    d = PrefillDeduper("很长很长很长很长很长很长的预填充")
    out1 = _dedup_sse_chunk_content(_sse(content="短"), d)
    out2 = _dedup_sse_chunk_content(_sse(content="回复", finish="stop"), d)
    assert _collect_sse_content([out1, out2]) == "短回复"
    assert json.loads(out2[6:])["choices"][0]["finish_reason"] == "stop"


def test_dedup_sse_strips_repeated_prefill_across_chunks():
    """模型复述预填充时，跨 chunk 也要裁干净。"""
    d = PrefillDeduper("从前有座山")
    outs = [_dedup_sse_chunk_content(_sse(content=c), d) for c in "从前有座山，山里有座庙"]
    assert _collect_sse_content(outs) == "，山里有座庙"


def test_dedup_sse_keeps_reasoning_chunks():
    d = PrefillDeduper("预填充预填充预填充预填充")
    out = _dedup_sse_chunk_content(_sse(reasoning_content="思考"), d)
    payload = json.loads(out[6:])
    assert payload["choices"][0]["delta"]["reasoning_content"] == "思考"
