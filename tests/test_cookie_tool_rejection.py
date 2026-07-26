# -*- coding: utf-8 -*-
"""P1-1：Cookie 通道对工具流量应显式报错，而不是静默发出错乱历史。"""
import pytest

from models import OpenAIMessage
from upstreams.cookie_proxy import _convert_messages_to_contents, has_tool_traffic


def test_detects_tool_role():
    msgs = [OpenAIMessage(role="user", content="hi"),
            OpenAIMessage(role="tool", content="{}", name="f", tool_call_id="c1")]
    assert has_tool_traffic(msgs) is True


def test_detects_assistant_tool_calls():
    msgs = [OpenAIMessage(role="assistant", content=None,
                          tool_calls=[{"id": "c1", "type": "function",
                                       "function": {"name": "f", "arguments": "{}"}}])]
    assert has_tool_traffic(msgs) is True


def test_detects_tools_declaration():
    assert has_tool_traffic([], tools=[{"type": "function",
                                        "function": {"name": "f"}}]) is True


def test_plain_chat_is_not_tool_traffic():
    msgs = [OpenAIMessage(role="system", content="sys"),
            OpenAIMessage(role="user", content="hi"),
            OpenAIMessage(role="assistant", content="hello")]
    assert has_tool_traffic(msgs) is False


def test_multipart_system_is_not_dropped():
    """分段 system（酒馆预设常见）此前会被整段丢弃。"""
    msgs = [OpenAIMessage(role="system",
                          content=[{"type": "text", "text": "A"},
                                   {"type": "text", "text": "B"}]),
            OpenAIMessage(role="user", content="hi")]
    _, system_text = _convert_messages_to_contents(msgs)
    assert "A" in system_text and "B" in system_text


def test_same_role_turns_merged():
    msgs = [OpenAIMessage(role="user", content="a"),
            OpenAIMessage(role="user", content="b")]
    contents, _ = _convert_messages_to_contents(msgs)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
