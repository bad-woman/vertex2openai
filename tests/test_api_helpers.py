# -*- coding: utf-8 -*-
"""标准通道参数构建基线回归（create_generation_config）。"""
from models import OpenAIRequest, OpenAIMessage
from api_helpers import create_generation_config
from runtime_state import app_state


def _req(model="gemini-2.5-flash", **kw):
    return OpenAIRequest(model=model, messages=[OpenAIMessage(role="user", content="hi")], **kw)


def test_sampling_passthrough_g25():
    cfg = create_generation_config(_req(temperature=0.7, top_p=0.9, max_tokens=100))
    assert cfg["temperature"] == 0.7 and cfg["top_p"] == 0.9 and cfg["max_output_tokens"] == 100


def test_sampling_stripped_g36():
    cfg = create_generation_config(_req(model="gemini-3.6-flash", temperature=0.7, top_p=0.9, n=2, max_tokens=50))
    assert "temperature" not in cfg and "top_p" not in cfg and "candidate_count" not in cfg
    assert cfg["max_output_tokens"] == 50


def test_console_defaults_injected_when_missing():
    app_state.update_settings({"default_temperature": 0.5, "default_top_p": 0.8, "default_max_tokens": 2000})
    cfg = create_generation_config(_req())
    assert cfg["temperature"] == 0.5 and cfg["top_p"] == 0.8 and cfg["max_output_tokens"] == 2000
    # 请求显式值优先于控制台
    cfg2 = create_generation_config(_req(temperature=1.2))
    assert cfg2["temperature"] == 1.2


def test_system_instruction_assembled():
    req = OpenAIRequest(model="gemini-2.5-flash", messages=[
        OpenAIMessage(role="system", content="A"),
        OpenAIMessage(role="system", content="B"),
        OpenAIMessage(role="user", content="hi"),
    ])
    cfg = create_generation_config(req)
    assert cfg["system_instruction"] == "A\nB"


def test_json_schema_mapping():
    cfg = create_generation_config(_req(response_format={
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": {"$schema": "http://x", "type": "object"}},
    }))
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_schema"] == {"type": "object"}


def test_image_model_config():
    app_state.update_settings({"image_size": "2K", "image_aspect_ratio": "16:9"})
    cfg = create_generation_config(_req(model="gemini-3-pro-image"))
    assert cfg["response_modalities"] == ["TEXT", "IMAGE"]
    assert cfg["image_config"].image_size == "2K"
    assert cfg["image_config"].aspect_ratio == "16:9"
    # 生图仅保留 google_search 工具
    assert cfg["tools"] == [{"google_search": {}}]
    assert "temperature" not in cfg


def test_safety_settings_include_jailbreak():
    cfg = create_generation_config(_req())
    cats = [s.category for s in cfg["safety_settings"]]
    assert "HARM_CATEGORY_JAILBREAK" in cats
    assert len(cats) == 5
