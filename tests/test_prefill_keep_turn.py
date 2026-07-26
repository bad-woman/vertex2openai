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


def test_smart_remains_the_default_mode():
    """默认保持 smart。

    用真实酒馆预设（Izumi，思维链标签 <konatan_planning~>）实测 gemini-3.6-flash × 3：
      smart      重复开标签 0/3，思考语言正确 3/3
      keep_turn  重复开标签 3/3，思考语言正确 2/3
    真实预设的预填充多以完整句子收尾，keep_turn 追加的 user 推动语会让模型当成
    新一轮、把开标签重写一遍，且该重复去重逻辑抓不到（见下一条用例）。
    keep_turn 只在预填充停在半截 token 时更优，因此保留为可选项而非默认。
    """
    import config as app_config
    assert app_config.DEFAULT_SETTINGS["prefill_mode"] == "smart"


def test_dedup_cannot_catch_repeated_opening_tag():
    """锁住选择 smart 作默认的关键依据：重复的开标签去重逻辑救不了。"""
    from message_processing import strip_prefill_overlap
    prefill = "小此准备好啦。\n<konatan_planning~>\n¡Allá voy!\n"
    output = "<konatan_planning~>\n- Repaso de la situación actual"
    assert strip_prefill_overlap(prefill, output) == output   # 无重叠，原样返回
    assert (prefill + output).count("<konatan_planning~>") == 2
