# -*- coding: utf-8 -*-
"""P1-3：OpenAI 字段兼容性。"""
import pytest

from models import OpenAIMessage, OpenAIRequest


def _req(**kw):
    # 显式构造 OpenAIMessage，不依赖 pydantic 的嵌套模型强制转换
    base = {"model": "gemini-2.5-flash",
            "messages": [OpenAIMessage(role="user", content="hi")]}
    base.update(kw)
    return OpenAIRequest(**base)


def test_stop_accepts_string():
    """OpenAI 允许 stop 为字符串，旧定义只收数组会 422。"""
    assert _req(stop="\n\n").stop == "\n\n"


def test_stop_accepts_list():
    assert _req(stop=["a", "b"]).stop == ["a", "b"]


def test_stop_normalized_to_list_in_config():
    from api_helpers import create_generation_config
    cfg = create_generation_config(_req(stop="STOP!"))
    assert cfg["stop_sequences"] == ["STOP!"]
    cfg = create_generation_config(_req(stop=["a", "b"]))
    assert cfg["stop_sequences"] == ["a", "b"]


def test_logprobs_bool_is_accepted():
    """OpenAI 的 logprobs 是 bool，Gemini 的是 int，两边语义不同。"""
    assert _req(logprobs=True).logprobs is True


def test_logprobs_bool_maps_to_gemini_shape():
    from api_helpers import create_generation_config
    cfg = create_generation_config(_req(logprobs=True, top_logprobs=3))
    assert cfg["response_logprobs"] is True
    assert cfg["logprobs"] == 3


def test_logprobs_int_passthrough():
    from api_helpers import create_generation_config
    cfg = create_generation_config(_req(logprobs=5))
    assert cfg["logprobs"] == 5


def test_stream_options_include_usage():
    from api_helpers import wants_usage
    assert wants_usage(_req(stream_options={"include_usage": True})) is True
    assert wants_usage(_req()) is False
