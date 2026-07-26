# -*- coding: utf-8 -*-
"""能力矩阵基线回归：家族识别 / 采样裁剪 / 思考解析 / 生图白名单。"""
import model_capabilities as mc


# ---------- 家族识别 ----------

def test_g36_flash_profile():
    p = mc.get_profile("gemini-3.6-flash")
    assert p["family"] == "g3" and p["thinking_kind"] == "level"
    assert p["default_level"] == "medium"
    assert "minimal" in p["thinking_levels"]
    # 3.6 起采样弃用
    assert "temperature" not in p["allowed_sampling"]
    assert "top_p" not in p["allowed_sampling"]
    assert "candidate_count" not in p["allowed_sampling"]
    assert "max_output_tokens" in p["allowed_sampling"]
    assert p["sampling_advice"] == "deprecated"
    assert p["requires_user_last_turn"] is True


def test_g35_flash_still_accepts_sampling():
    p = mc.get_profile("gemini-3.5-flash")
    assert "temperature" in p["allowed_sampling"]
    assert p["sampling_advice"] == "recommend_default"
    assert "candidate_count" not in p["allowed_sampling"]


def test_g35_flash_lite_deprecated():
    p = mc.get_profile("gemini-3.5-flash-lite")
    assert "temperature" not in p["allowed_sampling"]
    assert p["default_level"] == "minimal"


def test_g31_pro_no_minimal():
    p = mc.get_profile("gemini-3.1-pro-preview")
    assert p["default_level"] == "high"
    assert "minimal" not in p["thinking_levels"]
    assert "low" in p["thinking_levels"]


def test_g25_budgets():
    pro = mc.get_profile("gemini-2.5-pro")
    fl = mc.get_profile("gemini-2.5-flash")
    assert pro["thinking_kind"] == "budget" and pro["budget_min"] == 128 and not pro["budget_can_zero"]
    assert fl["budget_min"] == 0 and fl["budget_can_zero"]
    assert "temperature" in fl["allowed_sampling"]
    assert pro["requires_user_last_turn"] is False


def test_unknown_future_model_forward_safe():
    p = mc.get_profile("gemini-4.2-ultra")
    assert p["family"] == "g3"
    assert "temperature" not in p["allowed_sampling"]
    p2 = mc.get_profile("totally-unknown-model")
    assert p2["family"] == "g3" and "temperature" not in p2["allowed_sampling"]


def test_suffix_stripping():
    p = mc.get_profile("gemini-2.5-flash-search")
    assert p["family"] == "g25"


# ---------- 生图 ----------

def test_image_profiles():
    pro = mc.get_profile("gemini-3-pro-image")
    fl = mc.get_profile("gemini-3.1-flash-image")
    assert pro["is_image"] and fl["is_image"]
    assert len(pro["image_aspect_ratios"]) == 10
    # 14 = pro 的 10 种 + flash 新增的 1:4 / 4:1 / 1:8 / 8:1
    # （曾经是 15，多出的 "9:21" 无官方出处，已移除，见 model_capabilities 注释）
    assert len(fl["image_aspect_ratios"]) == 14
    assert "9:21" not in fl["image_aspect_ratios"]
    assert "1:8" in fl["image_aspect_ratios"] and "1:8" not in pro["image_aspect_ratios"]
    assert pro["image_sizes"] == {"1K", "2K", "4K"}
    assert fl["image_sizes"] == {"512", "1K", "2K", "4K"}
    assert pro["allowed_sampling"] == set()


def test_validate_aspect_ratio_fallback():
    assert mc.validate_aspect_ratio("gemini-3-pro-image", "1:8") is None
    assert mc.validate_aspect_ratio("gemini-3.1-flash-image", "1:8") == "1:8"
    assert mc.validate_aspect_ratio("gemini-3-pro-image", "16：9") == "16:9"  # 全角冒号归一化
    assert mc.validate_aspect_ratio("gemini-3-pro-image", None) is None


def test_resolve_image_size(fake_req):
    r = fake_req(model="gemini-3-pro-image")
    assert mc.resolve_image_size("gemini-3-pro-image", r, {"image_size": "4K"}) == "4K"
    # flash-image 独有 512；pro 不支持 512 → 回退 1K
    assert mc.resolve_image_size("gemini-3.1-flash-image", r, {"image_size": "512"}) == "512"
    assert mc.resolve_image_size("gemini-3-pro-image", r, {"image_size": "512"}) == "1K"
    # 请求级覆盖控制台
    r2 = fake_req(model_extra={"image_size": "2k"})
    assert mc.resolve_image_size("gemini-3-pro-image", r2, {"image_size": "4K"}) == "2K"
    assert mc.resolve_image_size("gemini-2.5-flash", r, {}) is None


# ---------- 思考解析 ----------

def test_resolve_thinking_g3_priority(fake_req):
    # 请求级 reasoning_effort > 控制台 > 默认
    r = fake_req(reasoning_effort="low")
    t = mc.resolve_thinking("gemini-3.6-flash", r, {"thinking_g3_level": "high"})
    assert t == {"mode": "level", "level": "low", "include_thoughts": True}
    r2 = fake_req()
    t2 = mc.resolve_thinking("gemini-3.6-flash", r2, {"thinking_g3_level": "high"})
    assert t2["level"] == "high"
    t3 = mc.resolve_thinking("gemini-3.6-flash", r2, {})
    assert t3["level"] == "medium"  # 模型默认


def test_resolve_thinking_g3_clamp(fake_req):
    # off → minimal；pro 无 minimal → low
    r = fake_req(reasoning_effort="off")
    assert mc.resolve_thinking("gemini-3.6-flash", r, {})["level"] == "minimal"
    assert mc.resolve_thinking("gemini-3.1-pro-preview", r, {})["level"] == "low"
    # 非法档位 → 回退 high
    r2 = fake_req()
    assert mc.resolve_thinking("gemini-3.6-flash", r2, {"thinking_g3_level": "bogus"})["level"] == "high"


def test_resolve_thinking_g25_budget(fake_req):
    r = fake_req()
    # 控制台预算
    t = mc.resolve_thinking("gemini-2.5-flash", r, {"thinking_g25_budget": 0})
    # F-4：budget=0 即关闭思考，此时上游拒绝 include_thoughts=True，必须同为 False。
    # （旧断言写的是 True，实机验证时这条路径稳定返回 400。）
    assert t == {"mode": "budget", "budget": 0, "include_thoughts": False}
    # pro 不可 0 → 抬到 128
    t2 = mc.resolve_thinking("gemini-2.5-pro", r, {"thinking_g25_budget": 0})
    assert t2["budget"] == 128
    # 超上限截断
    t3 = mc.resolve_thinking("gemini-2.5-flash", r, {"thinking_g25_budget": 999999})
    assert t3["budget"] == 24576
    # -1 动态透传
    t4 = mc.resolve_thinking("gemini-2.5-pro", r, {"thinking_g25_budget": -1})
    assert t4["budget"] == -1
    # 请求级 thinking_budget 覆盖控制台
    r2 = fake_req(model_extra={"thinking_budget": 2048})
    t5 = mc.resolve_thinking("gemini-2.5-flash", r2, {"thinking_g25_budget": 0})
    assert t5["budget"] == 2048


def test_resolve_thinking_none_for_image(fake_req):
    assert mc.resolve_thinking("gemini-3-pro-image", fake_req(), {}) == {"mode": None}


# ---------- 采样裁剪 ----------

def test_sanitize_sampling():
    prof = mc.get_profile("gemini-3.6-flash")
    cfg = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "candidate_count": 2,
           "max_output_tokens": 100, "system_instruction": "x"}
    mc.sanitize_sampling(cfg, prof)
    assert "temperature" not in cfg and "top_p" not in cfg and "candidate_count" not in cfg
    assert cfg["max_output_tokens"] == 100 and cfg["system_instruction"] == "x"

    prof25 = mc.get_profile("gemini-2.5-flash")
    cfg2 = {"temperature": 0.7, "candidate_count": 2}
    mc.sanitize_sampling(cfg2, prof25)
    assert cfg2 == {"temperature": 0.7, "candidate_count": 2}


def test_capabilities_summary_shape():
    s = mc.capabilities_summary("gemini-3.6-flash")
    assert s["thinking"]["kind"] == "level" and s["thinking"]["can_off"] is False
    s2 = mc.capabilities_summary("gemini-2.5-flash")
    assert s2["thinking"]["kind"] == "budget" and s2["thinking"]["can_off"] is True
