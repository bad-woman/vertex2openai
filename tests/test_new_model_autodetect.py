"""新增模型时的自动识别边界。

设计目标是"往 vertexModels.json 里加个 ID 就能用"。这里锁住哪些能自动判对、
哪些判不了——判不了的必须有手动出口，否则加模型就得改代码。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config as app_config  # noqa: E402
import model_capabilities as mc  # noqa: E402


# ---- 能自动判对的部分 ----

@pytest.mark.parametrize("name,levels,default", [
    ("gemini-3.5-pro", {"low", "medium", "high"}, "high"),          # pro：无 minimal
    ("gemini-3.7-pro-preview", {"low", "medium", "high"}, "high"),
    ("gemini-4.0-pro", {"low", "medium", "high"}, "high"),
    ("gemini-3.9-flash", {"minimal", "low", "medium", "high"}, "medium"),
    ("gemini-3.9-flash-lite", {"minimal", "low", "medium", "high"}, "minimal"),
])
def test_thinking_levels_follow_the_pro_flash_split(name, levels, default):
    """档位靠名字里的 pro / flash-lite 判定，与版本号无关，新型号自动生效。"""
    prof = mc.get_profile(name)
    assert prof["thinking_levels"] == levels
    assert prof["default_level"] == default
    assert prof["family"] == "g3" and prof["thinking_kind"] == "level"


def test_new_models_keep_the_gemini3_hard_limits():
    prof = mc.get_profile("gemini-3.5-pro")
    assert "candidate_count" not in prof["allowed_sampling"]   # 3.x 一律不支持
    assert prof["requires_user_last_turn"] is True             # 影响预填充兼容
    assert prof["is_image"] is False
    assert prof["supports_search"] is True                     # /v1/models 会生成 -search 别名


def test_search_suffix_does_not_change_the_profile():
    assert mc.get_profile("gemini-3.5-pro") == mc.get_profile("gemini-3.5-pro-search")


# ---- 判不了的部分：必须有手动出口 ----

def test_version_heuristic_cannot_see_release_order():
    """版本号表达不了"号更小但发布更晚"。

    自动判定按 3.6+ / 3.5-flash-lite / 4.x 划线，所以一个日后才出的
    gemini-3.5-pro 会被当成"旧的 3.x"而放行采样参数——这是启发式的固有盲区，
    不是 bug，但必须能手动纠正。
    """
    auto = mc.get_profile("gemini-3.5-pro")
    assert "temperature" in auto["allowed_sampling"]      # 自动判定：放行


def test_sampling_policy_override_closes_the_gap():
    prof = mc.get_profile("gemini-3.5-pro")
    forced = mc.apply_sampling_policy(prof, {"sampling_policy": "deprecated"})
    assert "temperature" not in forced["allowed_sampling"]
    assert "top_p" not in forced["allowed_sampling"] and "top_k" not in forced["allowed_sampling"]
    assert forced["sampling_advice"] == "deprecated"

    kept = mc.apply_sampling_policy(mc.get_profile("gemini-3.6-flash"), {"sampling_policy": "allowed"})
    assert "temperature" in kept["allowed_sampling"]


def test_policy_defaults_to_auto_and_is_per_model():
    assert app_config.DEFAULT_SETTINGS["sampling_policy"] == "auto"
    assert "sampling_policy" in app_config.PER_MODEL_KEYS
    prof = mc.get_profile("gemini-3.5-pro")
    for value in ("auto", "", None, "垃圾值"):
        assert mc.apply_sampling_policy(prof, {"sampling_policy": value}) == prof


def test_policy_never_touches_image_models_or_candidate_count():
    img = mc.get_profile("gemini-3.1-flash-image")
    assert mc.apply_sampling_policy(img, {"sampling_policy": "allowed"})["allowed_sampling"] == set()
    g3 = mc.apply_sampling_policy(mc.get_profile("gemini-3.5-pro"), {"sampling_policy": "allowed"})
    assert "candidate_count" not in g3["allowed_sampling"]


def test_console_summary_reflects_the_override():
    """控制台显示的能力必须和实际下发一致，否则提示会骗人。"""
    auto = mc.capabilities_summary("gemini-3.5-pro")
    forced = mc.capabilities_summary("gemini-3.5-pro", {"sampling_policy": "deprecated"})
    assert auto["sampling_advice"] != forced["sampling_advice"]
    assert forced["sampling_advice"] == "deprecated"
