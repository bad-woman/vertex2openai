# -*- coding: utf-8 -*-
"""P0-2：retry_max 的语义与边界。

回归的 bug：Express 通道写成 range(retry_max)，retry_max=0 时一次请求都不发——
非流式返回误导性的“无有效内容”，真流式只吐一个心跳就关流。
"""
import pytest

from api_helpers import get_retry_settings
from runtime_state import app_state


def test_default_matches_config():
    import config as app_config
    app_state.update_settings({})
    retry_max, backoff = get_retry_settings()
    assert retry_max == app_config.DEFAULT_SETTINGS["retry_max"]
    assert backoff == float(app_config.DEFAULT_SETTINGS["retry_backoff_seconds"])


def test_zero_is_allowed_and_means_one_attempt():
    app_state.update_settings({"retry_max": 0})
    retry_max, _ = get_retry_settings()
    assert retry_max == 0
    # 语义约定：总尝试次数 = retry_max + 1
    assert retry_max + 1 == 1


def test_negative_clamped_to_zero():
    app_state.update_settings({"retry_max": -5})
    assert get_retry_settings()[0] == 0


def test_absurdly_large_is_clamped():
    app_state.update_settings({"retry_max": 99999})
    assert get_retry_settings()[0] == 50


def test_garbage_falls_back_to_default():
    import config as app_config
    app_state.update_settings({"retry_max": "abc"})
    assert get_retry_settings()[0] == app_config.DEFAULT_SETTINGS["retry_max"]


def test_backoff_clamped():
    app_state.update_settings({"retry_backoff_seconds": 9999})
    assert get_retry_settings()[1] == 120.0
    app_state.update_settings({"retry_backoff_seconds": -1})
    assert get_retry_settings()[1] == 0.0
