# -*- coding: utf-8 -*-
"""P0-1：思考档位的就近向下夹取。

回归的 bug：Pro 模型没有 minimal 档，旧实现对非法档位一律兜底到 "high"，
于是"控制台选 minimal + 强制用上方档位"在 3.1-pro 上实际发出 HIGH——
与用户想减少思考的意图完全相反。
"""
import pytest

import model_capabilities as mc

PRO = "gemini-3.1-pro-preview"
FLASH = "gemini-3.6-flash"
LITE = "gemini-3.5-flash-lite"


class Req:
    """模拟 SillyTavern：每次请求都恒发 reasoning_effort=xhigh。"""
    def __init__(self, effort="xhigh"):
        self.reasoning_effort = effort
        self.model_extra = {"reasoning_effort": effort} if effort else {}


def lvl(model, mode, console_level=None, effort="xhigh"):
    settings = {"native_thinking_mode": mode}
    if console_level:
        settings["thinking_g3_level"] = console_level
    return mc.resolve_thinking(model, Req(effort), settings)["level"]


@pytest.mark.parametrize("want,expect", [
    ("minimal", "low"),      # ← 核心回归：Pro 无 minimal，必须向下取 low，不能变成 high
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
])
def test_pro_console_level_clamps_downward(want, expect):
    assert lvl(PRO, "console", want) == expect


@pytest.mark.parametrize("want", ["minimal", "low", "medium", "high"])
def test_flash_console_level_is_exact(want):
    assert lvl(FLASH, "console", want) == want


def test_off_mode_uses_lowest_available_level():
    assert lvl(FLASH, "off") == "minimal"
    assert lvl(PRO, "off") == "low"          # Pro 最低是 low
    assert lvl(LITE, "off") == "minimal"


def test_off_mode_hides_thoughts():
    assert mc.resolve_thinking(FLASH, Req(), {"native_thinking_mode": "off"})["include_thoughts"] is False


def test_request_mode_still_follows_frontend():
    assert lvl(FLASH, "request") == "high"   # xhigh → high
    assert lvl(PRO, "request") == "high"


def test_unknown_level_word_clamps_down_not_up():
    """未知档位词按最高处理再向下夹，不应把 Pro 抬到 high 之外的非法值。"""
    assert lvl(PRO, "console", "ludicrous") in {"low", "medium", "high"}


def test_clamp_helper_directly():
    assert mc._clamp_level("minimal", {"low", "medium", "high"}) == "low"
    assert mc._clamp_level("high", {"minimal", "low"}) == "low"
    assert mc._clamp_level("medium", {"minimal", "low", "medium", "high"}) == "medium"
    assert mc._clamp_level("anything", set()) == "low"


def test_levels_sorted_by_strength_not_alphabet():
    """控制台下拉必须是 minimal→high，字典序会排成 high, low, medium, minimal。"""
    assert mc.capabilities_summary(FLASH)["thinking"]["levels"] == \
        ["minimal", "low", "medium", "high"]
