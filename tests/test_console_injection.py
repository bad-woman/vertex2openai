"""控制台注入：给 RikkaHub 这类轻量前端补上 system 指令与预填充。

这些前端没有酒馆的预设系统，尤其**从不发送 assistant 预填充**，
于是破限最强的杠杆用不上，"预填充时压制原生思考"也永远不会触发。

本文件重点锁住四条护栏——没有它们，注入会和既有功能打架。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config as app_config  # noqa: E402
from message_processing import apply_console_injection, apply_prefill_compat  # noqa: E402
from models import OpenAIMessage  # noqa: E402

SYS = "你是角色扮演助手。"
PRE = "好的，我开始写。\n<think>"


def _client_msgs():
    """轻量前端的典型请求：只有 system + user，末尾一定是 user。"""
    return [OpenAIMessage(role="system", content="（前端自带）"),
            OpenAIMessage(role="user", content="我推开门走了进去。")]


def test_defaults_are_empty_so_feature_is_off():
    assert app_config.DEFAULT_SETTINGS["inject_system_instruction"] == ""
    assert app_config.DEFAULT_SETTINGS["inject_prefill"] == ""


def test_both_empty_is_a_noop():
    msgs = _client_msgs()
    out, notes = apply_console_injection(msgs, "", "")
    assert out is msgs and notes == []


def test_system_injected_after_client_system():
    out, notes = apply_console_injection(_client_msgs(), system_text=SYS)
    assert [m.role for m in out] == ["system", "system", "user"]
    assert out[0].content == "（前端自带）"      # 客户端的仍在场
    assert out[1].content == SYS                # 注入的排在其后
    assert any("system" in n for n in notes)


def test_system_injected_when_client_sent_none():
    msgs = [OpenAIMessage(role="user", content="hi")]
    out, _ = apply_console_injection(msgs, system_text=SYS)
    assert [m.role for m in out] == ["system", "user"]


def test_prefill_injected_as_assistant_turn():
    out, notes = apply_console_injection(_client_msgs(), prefill_text=PRE)
    assert [m.role for m in out] == ["system", "user", "assistant"]
    assert out[-1].content == PRE
    assert any("注入预填充" in n for n in notes)


# ---- 四条护栏 ----

def test_guard_client_prefill_is_not_overwritten():
    """酒馆已经发了预填充 → 不能再叠一段。"""
    msgs = _client_msgs() + [OpenAIMessage(role="assistant", content="客户端自己的预填充")]
    out, notes = apply_console_injection(msgs, prefill_text=PRE)
    assert [m.role for m in out] == ["system", "user", "assistant"]
    assert out[-1].content == "客户端自己的预填充"
    assert any("已自带预填充" in n for n in notes)


def test_guard_tools_skip_prefill_injection():
    out, notes = apply_console_injection(_client_msgs(), prefill_text=PRE, has_tools=True)
    assert all(m.role != "assistant" for m in out)
    assert any("函数调用" in n for n in notes)


def test_guard_image_model_skips_prefill_by_default():
    out, notes = apply_console_injection(_client_msgs(), prefill_text=PRE, is_image_model=True)
    assert all(m.role != "assistant" for m in out)
    assert any("生图" in n for n in notes)


def test_image_prefill_can_be_allowed():
    """预填充对生图有实际引导力，实测：同一句“画一只猫”，
    预填充承诺“纯黑白钢笔线稿”→ 输出线稿；不加 → 彩色写实照片。
    所以这条护栏是开关，不是硬拦。"""
    out, _ = apply_console_injection(_client_msgs(), prefill_text=PRE,
                                     is_image_model=True, allow_image_prefill=True)
    assert out[-1].role == "assistant" and out[-1].content == PRE


def test_image_toggle_defaults_off():
    assert app_config.DEFAULT_SETTINGS["inject_prefill_for_image"] is False


def test_image_toggle_does_not_bypass_other_guards():
    """放行生图不代表放行工具流量。"""
    out, notes = apply_console_injection(_client_msgs(), prefill_text=PRE, is_image_model=True,
                                         allow_image_prefill=True, has_tools=True)
    assert all(m.role != "assistant" for m in out)
    assert any("函数调用" in n for n in notes)


def test_guard_tools_still_allows_system_injection():
    """工具只挡预填充，system 指令不受影响。"""
    out, _ = apply_console_injection(_client_msgs(), system_text=SYS,
                                     prefill_text=PRE, has_tools=True)
    assert any(m.role == "system" and m.content == SYS for m in out)
    assert all(m.role != "assistant" for m in out)


def test_trailing_empty_messages_do_not_defeat_the_client_prefill_guard():
    """末尾跟着空消息时，仍要认出客户端已有预填充。"""
    msgs = _client_msgs() + [OpenAIMessage(role="assistant", content="客户端预填充"),
                             OpenAIMessage(role="assistant", content="   ")]
    out, notes = apply_console_injection(msgs, prefill_text=PRE)
    assert any("已自带预填充" in n for n in notes)
    assert sum(1 for m in out if m.role == "assistant") == 2   # 没有新增


# ---- 与下游兼容层的衔接 ----

def test_injected_prefill_flows_into_prefill_compat():
    """注入结果必须和"前端自发预填充"同形，下游四种模式原样复用。"""
    injected, _ = apply_console_injection(_client_msgs(), prefill_text=PRE)
    for mode in ("smart", "keep_turn", "minimal"):
        out, prefill, detected = apply_prefill_compat(injected, mode=mode, allow_model_last=False)
        assert detected is True, mode          # 这一步决定思考压制会不会触发
        assert out[-1].role == "user", mode    # 3.x 硬性要求
    # 2.5 走原生透传
    out, prefill, detected = apply_prefill_compat(injected, mode="smart", allow_model_last=True)
    assert detected is True and prefill == PRE


def test_injection_keys_are_per_model_overridable():
    """按模型专属很关键：只给角色扮演模型开，问答模型保持干净。"""
    assert "inject_system_instruction" in app_config.PER_MODEL_KEYS
    assert "inject_prefill" in app_config.PER_MODEL_KEYS


def test_whitespace_only_values_count_as_empty():
    msgs = _client_msgs()
    out, notes = apply_console_injection(msgs, system_text="   ", prefill_text="\n\t")
    assert out is msgs and notes == []


def test_image_models_get_an_image_specific_nudge():
    """生图模型必须换一句要图片的续写指令。

    通用那句是"从断点处无缝往下写"，生图模型会照办——继续写**文本**，
    实测结果是吐出一段 ASCII 字符画而不是图片。换成明确要图片的措辞后，
    smart / keep_turn 两种模式都能正常返回图片。
    """
    from upstreams.express_sdk import _prefill_tpl
    from message_processing import DEFAULT_IMAGE_PREFILL_NUDGE

    assert _prefill_tpl("", is_image_model=True) == DEFAULT_IMAGE_PREFILL_NUDGE
    assert "图片" in DEFAULT_IMAGE_PREFILL_NUDGE
    # 文本模型保持原样（空 = 让 apply_prefill_compat 用它自己的内置默认）
    assert _prefill_tpl("", is_image_model=False) == ""
    # 用户自定义优先，两类模型都不覆盖
    assert _prefill_tpl("我的模板", is_image_model=True) == "我的模板"
    assert _prefill_tpl("  我的模板  ", is_image_model=False) == "我的模板"


def test_both_upstreams_share_the_same_template_rule():
    from upstreams.express_sdk import _prefill_tpl as a
    from upstreams.cookie_proxy import _prefill_tpl as b
    for tpl in ("", "  ", "自定义"):
        for img in (True, False):
            assert a(tpl, img) == b(tpl, img)


def test_console_can_actually_save_injection_per_model():
    """后端把注入键放进 PER_MODEL_KEYS，控制台就必须真的能存/取它们。

    此前只改了后端：`PER_MODEL_KEYS` 含注入键，但控制台的
    saveModelOverride 不提交、applyModelParamFields 也不回显，
    说明文字却让用户去点“保存为该模型专属”——承诺了做不到的事。
    """
    main_py = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text(encoding="utf-8")
    for key in ("inject_system_instruction", "inject_prefill"):
        assert key in app_config.PER_MODEL_KEYS, key
        # 前端的 PER_MODEL_KEYS 常量（决定切换模型时的回显）
        js_list = main_py.split("const PER_MODEL_KEYS = [", 1)[1].split("]", 1)[0]
        assert key in js_list, f"前端 PER_MODEL_KEYS 缺 {key}"
        # 专属保存的 patch
        patch = main_py.split("async function saveModelOverride()", 1)[1].split("};", 1)[0]
        assert key in patch, f"saveModelOverride 未提交 {key}"


def test_backend_and_frontend_per_model_keys_agree():
    main_py = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text(encoding="utf-8")
    js_list = main_py.split("const PER_MODEL_KEYS = [", 1)[1].split("]", 1)[0]
    js_keys = {k.strip().strip("'\"") for k in js_list.split(",") if k.strip()}
    assert js_keys == set(app_config.PER_MODEL_KEYS), (
        f"前后端 PER_MODEL_KEYS 不一致：仅前端 {js_keys - set(app_config.PER_MODEL_KEYS)}，"
        f"仅后端 {set(app_config.PER_MODEL_KEYS) - js_keys}")
