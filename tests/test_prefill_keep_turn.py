"""prefill_mode="keep_turn"：保留 assistant 预填充轮次，只补一句极短 user 推动语。

3.x 拒绝的是"以 model 轮次**结尾**"，并不禁止 model 轮次出现在中间。
smart 模式把预填充塞进 user 消息，模型会把自己写的话当成用户给的参考文本，
倾向另起一句；keep_turn 让预填充留在模型自己的声音里。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from message_processing import (  # noqa: E402
    DEFAULT_KEEP_TURN_NUDGE,
    apply_prefill_compat,
)
from models import OpenAIMessage  # noqa: E402


def _msgs():
    return [OpenAIMessage(role="user", content="写个故事"),
            OpenAIMessage(role="assistant", content="从前有座山，")]


def test_keep_turn_preserves_assistant_turn():
    out, prefill, detected = apply_prefill_compat(_msgs(), mode="keep_turn", allow_model_last=False)
    assert [m.role for m in out] == ["user", "assistant", "user"]
    assert out[1].content == "从前有座山，", "预填充必须原样留在 model 轮次里"
    assert out[-1].content == DEFAULT_KEEP_TURN_NUDGE
    assert prefill == "从前有座山，" and detected is True


def test_keep_turn_never_ends_on_assistant():
    """唯一的硬约束：3.x 会拒绝以 assistant 结尾的请求。"""
    for extra in ([], [OpenAIMessage(role="assistant", content="")],
                  [OpenAIMessage(role="assistant", content=None),
                   OpenAIMessage(role="assistant", content="   ")]):
        out, _, _ = apply_prefill_compat(_msgs() + extra, mode="keep_turn", allow_model_last=False)
        assert out[-1].role == "user", f"尾随空消息 {extra} 导致以 assistant 结尾"


def test_keep_turn_does_not_leak_prefill_into_user_message():
    """与 smart 的关键区别：预填充不得出现在任何 user 消息里。"""
    out, _, _ = apply_prefill_compat(_msgs(), mode="keep_turn", allow_model_last=False)
    for m in out:
        if m.role == "user":
            assert "从前有座山" not in (m.content or "")


def test_smart_still_merges_into_user_message():
    """smart 行为保持不变（未被本次改动影响）。"""
    out, _, _ = apply_prefill_compat(_msgs(), mode="smart", allow_model_last=False)
    assert [m.role for m in out] == ["user"]
    assert "从前有座山" in out[0].content


def test_keep_turn_custom_nudge_template():
    out, _, _ = apply_prefill_compat(_msgs(), mode="keep_turn", allow_model_last=False,
                                     instruction_template="go on")
    assert out[-1].content == "go on"


def test_keep_turn_on_25_still_native_passthrough():
    """2.5 允许 model 结尾，应原样透传、不加推动语。"""
    out, prefill, _ = apply_prefill_compat(_msgs(), mode="keep_turn", allow_model_last=True)
    assert [m.role for m in out] == ["user", "assistant"]
    assert prefill == "从前有座山，"


def test_keep_turn_ignores_non_prefill_requests():
    msgs = [OpenAIMessage(role="user", content="hi")]
    out, prefill, detected = apply_prefill_compat(msgs, mode="keep_turn", allow_model_last=False)
    assert out == msgs and prefill == "" and detected is False


def test_keep_turn_skips_tool_call_tail():
    msgs = [OpenAIMessage(role="user", content="hi"),
            OpenAIMessage(role="assistant", content=None,
                          tool_calls=[{"id": "x", "type": "function",
                                       "function": {"name": "f", "arguments": "{}"}}])]
    out, prefill, detected = apply_prefill_compat(msgs, mode="keep_turn", allow_model_last=False)
    assert out == msgs and detected is False


def test_keep_turn_is_the_default_mode():
    """默认模式改为 keep_turn。

    实机对照（预设思维链预填充 `<thinking>\\n1.`，gemini-3.6-flash 各 3 次）：
      smart     切题 0/3，且预设要求的 <thinking> 格式全部丢失
      keep_turn 切题 3/3，格式完整
    预填充在本项目的主要用途就是用预设思维链顶掉原生思维链，smart 在这条主路径上失效。
    """
    import config as app_config
    assert app_config.DEFAULT_SETTINGS["prefill_mode"] == "keep_turn"


def test_default_mode_keeps_prefill_in_model_voice():
    """走默认配置时，预填充不得落进 user 消息。"""
    import config as app_config
    out, _, _ = apply_prefill_compat(
        _msgs(), mode=app_config.DEFAULT_SETTINGS["prefill_mode"], allow_model_last=False)
    assert out[1].role == "assistant" and out[1].content == "从前有座山，"
