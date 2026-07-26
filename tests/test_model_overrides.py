# -*- coding: utf-8 -*-
"""按模型参数覆盖：存储、优先级合并、通道生效、清除。"""
import json

from models import OpenAIRequest, OpenAIMessage
from runtime_state import app_state
from api_helpers import create_generation_config
import upstreams.cookie_proxy as cp
import config as app_config


def _req(model, **kw):
    return OpenAIRequest(model=model, messages=[OpenAIMessage(role="user", content="hi")], **kw)


# ---------- 存储层 ----------

def test_set_get_clear_override():
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.3, "default_max_tokens": 500})
    ov = app_state.get_model_overrides()
    assert ov["gemini-2.5-flash"] == {"default_temperature": 0.3, "default_max_tokens": 500}
    assert app_state.clear_model_override("gemini-2.5-flash") is True
    assert "gemini-2.5-flash" not in app_state.get_model_overrides()
    assert app_state.clear_model_override("gemini-2.5-flash") is False  # 再清无效


def test_set_override_filters_unknown_keys():
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.9, "retry_max": 99, "bogus": 1})
    ov = app_state.get_model_overrides()["gemini-2.5-flash"]
    assert ov == {"default_temperature": 0.9}  # retry_max/bogus 非 PER_MODEL_KEYS 被过滤


def test_effective_settings_merge():
    app_state.update_settings({"default_temperature": 0.1, "image_size": "1K"})
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.8})
    eff = app_state.get_effective_settings("gemini-2.5-flash")
    assert eff["default_temperature"] == 0.8   # 专属覆盖
    assert eff["image_size"] == "1K"           # 未覆盖 → 全局
    eff2 = app_state.get_effective_settings("gemini-2.5-pro")
    assert eff2["default_temperature"] == 0.1  # 无专属 → 全局


def test_effective_settings_persist_only_per_model_keys():
    # 覆盖不影响基础设施级键
    app_state.update_settings({"retry_max": 7})
    app_state.set_model_override("gemini-2.5-flash", {"default_top_p": 0.5})
    eff = app_state.get_effective_settings("gemini-2.5-flash")
    assert eff["retry_max"] == 7


# ---------- 通道生效 ----------

def test_override_applies_in_express_config():
    app_state.update_settings({"default_temperature": None, "default_max_tokens": None})
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.42, "default_max_tokens": 1234})
    cfg = create_generation_config(_req("gemini-2.5-flash"))
    assert cfg["temperature"] == 0.42 and cfg["max_output_tokens"] == 1234
    # 另一模型不受影响
    cfg2 = create_generation_config(_req("gemini-2.5-pro"))
    assert "temperature" not in cfg2 or cfg2.get("temperature") is None


def test_override_request_still_wins():
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.42})
    cfg = create_generation_config(_req("gemini-2.5-flash", temperature=1.5))
    assert cfg["temperature"] == 1.5  # 单次请求 > 模型专属


def test_override_applies_in_cookie_body():
    app_state.update_settings({"default_temperature": None})
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.7, "default_max_tokens": 888})
    body = cp._build_batch_graphql_body("proj", "gemini-2.5-flash", _req("gemini-2.5-flash"))
    gc = body["variables"]["generationConfig"]
    assert gc["temperature"] == 0.7 and gc["maxOutputTokens"] == 888


def test_override_image_size_per_model():
    app_state.update_settings({"image_size": "1K"})
    app_state.set_model_override("gemini-3.1-flash-image", {"image_size": "4K", "image_aspect_ratio": "16:9"})
    body = cp._build_batch_graphql_body("proj", "gemini-3.1-flash-image", _req("gemini-3.1-flash-image"))
    img = body["variables"]["generationConfig"]["imageConfig"]
    assert img["imageSize"] == "4K" and img["aspectRatio"] == "16:9"
    # pro-image 无专属 → 用全局 1K
    body2 = cp._build_batch_graphql_body("proj", "gemini-3-pro-image", _req("gemini-3-pro-image"))
    assert body2["variables"]["generationConfig"]["imageConfig"]["imageSize"] == "1K"


def test_override_thinking_level_per_model():
    app_state.set_model_override("gemini-3.6-flash", {"thinking_g3_level": "high"})
    cfg = cp._build_thinking_config("gemini-3.6-flash", _req("gemini-3.6-flash"))
    assert cfg["thinkingLevel"] == "HIGH"
    # 无专属的 3.x 用默认 medium
    cfg2 = cp._build_thinking_config("gemini-3.5-flash", _req("gemini-3.5-flash"))
    assert cfg2["thinkingLevel"] == "MEDIUM"


def test_update_settings_does_not_wipe_overrides():
    app_state.set_model_override("gemini-2.5-flash", {"default_temperature": 0.3})
    # 普通 update_settings 传入 model_overrides 键也不应覆盖
    app_state.update_settings({"model_overrides": {}, "retry_max": 5})
    assert "gemini-2.5-flash" in app_state.get_model_overrides()
    assert app_state.get_setting("retry_max") == 5
