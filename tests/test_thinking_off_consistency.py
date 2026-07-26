"""F-4：budget=0（关闭思考）时不得再要求 include_thoughts。

实机验证时上游明确拒绝：
  400 Thinking_config.include_thoughts is only enabled when thinking is enabled.
出站参数是 {'include_thoughts': True, 'thinking_budget': 0} ——
也就是"用户主动关掉思考"这条路径在 2.5 系列上必然报错。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import model_capabilities as mc  # noqa: E402
from models import OpenAIMessage, OpenAIRequest  # noqa: E402


def _req(model, **extra):
    return OpenAIRequest(model=model,
                         messages=[OpenAIMessage(role="user", content="hi")],
                         **extra)


def test_explicit_zero_budget_disables_include_thoughts():
    t = mc.resolve_thinking("gemini-2.5-flash", _req("gemini-2.5-flash", thinking_budget=0), {})
    assert t["budget"] == 0
    assert t["include_thoughts"] is False


def test_minimal_effort_on_zero_capable_model_is_consistent():
    t = mc.resolve_thinking("gemini-2.5-flash",
                            _req("gemini-2.5-flash", reasoning_effort="minimal"), {})
    assert t["budget"] == 0 and t["include_thoughts"] is False


def test_console_zero_budget_is_consistent():
    t = mc.resolve_thinking("gemini-2.5-flash", _req("gemini-2.5-flash"),
                            {"thinking_g25_budget": 0})
    assert t["budget"] == 0 and t["include_thoughts"] is False


def test_pro_cannot_zero_so_thoughts_stay_on():
    """2.5-pro 最低 128、无法关闭，此时 include_thoughts 不该被误关。"""
    t = mc.resolve_thinking("gemini-2.5-pro", _req("gemini-2.5-pro", thinking_budget=0), {})
    assert t["budget"] == 128
    assert t["include_thoughts"] is True


@pytest.mark.parametrize("budget", [-1, 512, 4096])
def test_nonzero_budget_keeps_include_thoughts(budget):
    t = mc.resolve_thinking("gemini-2.5-flash",
                            _req("gemini-2.5-flash", thinking_budget=budget), {})
    assert t["include_thoughts"] is True


def test_suppress_path_unchanged():
    """关闭原生思考（mode=off）本来就同时关 include_thoughts，不应被本次改动影响。"""
    t = mc.resolve_thinking("gemini-2.5-flash", _req("gemini-2.5-flash"),
                            {"native_thinking_mode": "off"})
    assert t["budget"] == 0 and t["include_thoughts"] is False


def test_invariant_zero_budget_never_ships_with_thoughts():
    """不变量：任何组合下 budget==0 都不得带 include_thoughts=True。"""
    combos = [
        ({}, {"thinking_budget": 0}),
        ({"thinking_g25_budget": 0}, {}),
        ({"native_thinking_mode": "off"}, {}),
        ({}, {"reasoning_effort": "minimal"}),
        ({"native_thinking_mode": "console", "thinking_g25_budget": 0}, {"reasoning_effort": "high"}),
    ]
    for settings, extra in combos:
        for model in ("gemini-2.5-flash", "gemini-2.5-pro"):
            t = mc.resolve_thinking(model, _req(model, **extra), settings)
            if t.get("budget") == 0:
                assert t["include_thoughts"] is False, (model, settings, extra)
