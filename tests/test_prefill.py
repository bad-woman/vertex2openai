# -*- coding: utf-8 -*-
"""预填充兼容基线回归（当前行为；增强后会同步扩展）。"""
from models import OpenAIMessage
from message_processing import apply_prefill_compat


def _msgs(*pairs):
    return [OpenAIMessage(role=r, content=c) for r, c in pairs]


def test_off_mode_untouched():
    msgs = _msgs(("user", "hi"), ("assistant", "prefill"))
    out = apply_prefill_compat(msgs, "off")
    assert out[0] is msgs and out[1] == ""


def test_trailing_user_noop():
    msgs = _msgs(("user", "hi"))
    out = apply_prefill_compat(msgs, "smart")
    assert out[0] is msgs and out[1] == ""


def test_smart_converts_trailing_assistant():
    msgs = _msgs(("user", "hi"), ("assistant", "从前有座山"))
    new_msgs, prefill = apply_prefill_compat(msgs, "smart")[:2]
    assert prefill == "从前有座山"
    assert new_msgs[-1].role == "user"
    assert "从前有座山" in new_msgs[-1].content
    assert "hi" in new_msgs[-1].content  # 合并进上一条 user


def test_smart_appends_user_when_prev_not_str():
    msgs = [
        OpenAIMessage(role="user", content=[{"type": "text", "text": "hi"}]),
        OpenAIMessage(role="assistant", content="pf"),
    ]
    new_msgs, prefill = apply_prefill_compat(msgs, "smart")[:2]
    assert prefill == "pf"
    assert new_msgs[-1].role == "user" and isinstance(new_msgs[-1].content, str)
    assert len(new_msgs) == 2


def test_trailing_toolcall_untouched():
    msgs = [
        OpenAIMessage(role="user", content="hi"),
        OpenAIMessage(role="assistant", content=None,
                      tool_calls=[{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]),
    ]
    out = apply_prefill_compat(msgs, "smart")
    assert out[0] is msgs and out[1] == ""


def test_skips_trailing_empty_messages():
    msgs = _msgs(("user", "hi"), ("assistant", "pf"), ("assistant", "  "))
    new_msgs, prefill = apply_prefill_compat(msgs, "smart")[:2]
    assert prefill == "pf"
    assert all(m.role == "user" for m in new_msgs)


def test_minimal_appends_placeholder():
    msgs = _msgs(("user", "hi"), ("assistant", "pf"))
    new_msgs, prefill = apply_prefill_compat(msgs, "minimal")[:2]
    assert prefill == ""
    assert new_msgs[-1].role == "user"
    assert len(new_msgs) == 3


def test_empty_messages():
    out = apply_prefill_compat([], "smart")
    assert out[0] == [] and out[1] == ""
