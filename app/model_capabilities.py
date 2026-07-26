"""
模型能力矩阵（按家族识别）

用于决定：
- 每个模型支持的思考方式（thinking_level / thinking_budget / 无）与合法档位
- 可用的采样参数集合（避免把不支持的参数发给模型导致 400）
- 生图模型支持的宽高比与分辨率白名单
- 是否要求最后一条消息为 user（Gemini 3.x 强制，否则 400 → 影响预填充）

设计原则：**按家族模式识别，而非写死每个型号**，因此往 vertexModels.json 里新增模型 ID
基本即插即用；未知/未来型号保守回退到“最新代（gemini-3）”档案。

依据（2026-07 官方文档核实）：
- thinking: ai.google.dev/gemini-api/docs/thinking
- 3.6/3.5 开发指南（采样弃用 + 预填充 400 + 去 candidate_count）
- 生图: ai.google.dev/gemini-api/docs/image-generation ；Agent Platform 3-pro-image / 3-1-flash-image
"""

import re
from typing import Any, Dict, Optional

from models import normalize_content_part

# 所有“采样类”参数键（用于按模型剥离）
SAMPLING_KEYS = {
    "temperature", "top_p", "top_k",
    "presence_penalty", "frequency_penalty",
    "candidate_count", "seed", "stop_sequences", "max_output_tokens",
}

# ---- 生图分辨率与比例白名单 ----
# 出处：ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image
#      「1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9 aspect ratios」（核对于 2026-07-26）
_PRO_IMAGE_ARS = {"1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
# 出处：ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image
#      「New 1:4, 4:1, 1:8 and 8:1 aspect ratios」（核对于 2026-07-26）
# 注意：曾经这里还有 "9:21"，但官方文档从未提及该比例。发送不支持的比例会 400，
#      而不在白名单里只会回退成“由模型决定”（不报错），因此按保守策略移除。
#      若日后真机验证可用，加回来并在此注明验证日期。
_FLASH_IMAGE_ARS = _PRO_IMAGE_ARS | {"1:4", "4:1", "1:8", "8:1"}

# ---- 思考档位阶梯（由低到高）----
# 出处：Agent Platform → Get started with Gemini 3（核对于 2026-07-26）
#   MINIMAL 仅 Gemini 3 Flash / 3.1 Flash-Lite 有；
#   MEDIUM 覆盖 Gemini 3 Flash / 3.1 Pro / 3.1 Flash-Lite；
#   LOW / HIGH 所有 3.x 都有。
_LEVEL_ORDER = ["minimal", "low", "medium", "high"]


def _clamp_level(level: str, levels: set) -> str:
    """把任意档位夹到该模型的合法档位集合内。

    规则：**优先向下就近取**，向下没有再向上。
    这条方向性很关键——用户把档位调低是为了减少思考，
    旧实现对非法档位一律兜底成 "high"，导致 Pro 模型上选 minimal 反而拿到 high，
    与用户意图完全相反（P0-1）。
    """
    if not levels:
        return "low"
    if level in levels:
        return level
    try:
        idx = _LEVEL_ORDER.index(level)
    except ValueError:
        idx = _LEVEL_ORDER.index("high")   # 未知词按最高处理，再向下夹
    for i in range(idx, -1, -1):           # 先向下找最接近的合法档
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    for i in range(idx + 1, len(_LEVEL_ORDER)):   # 向下没有再向上
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    return sorted(levels)[0]


def sort_levels(levels) -> list:
    """按强度而非字典序排列档位，供控制台下拉框使用。"""
    known = [lv for lv in _LEVEL_ORDER if lv in levels]
    extra = sorted(lv for lv in levels if lv not in _LEVEL_ORDER)
    return known + extra


def _strip_known_suffixes(name: str) -> str:
    """去掉别名后缀（-search / 分辨率 / 思考档位后缀），得到判断家族用的基础名。"""
    n = name.lower()
    for suf in ("-search", "-1k", "-2k", "-4k", "-512"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"-think-(minimal|low|medium|high|off|none)$", "", n)
    return n


def _temp_deprecated(name: str) -> bool:
    """temperature/top_p/top_k 是否已废弃。

    据官方 latest-model 文档：自 Gemini 3.6 Flash 与 Gemini 3.5 Flash-Lite 起，
    以及**所有更新/未来的 Gemini 模型**，这三个采样参数被废弃（现忽略、未来 400），
    需从请求中移除。更早的 3.x（3.0–3.5 非 lite）仍接受（但建议保持默认）。
    """
    n = name.lower()
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
    if not m:
        return True  # 未知/未来型号 → 前向安全，按已废弃处理
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    if major >= 4:
        return True
    if major == 3 and minor >= 6:
        return True
    if major == 3 and minor == 5 and "flash-lite" in n:
        return True
    return False


def get_profile(model_name: str) -> Dict[str, Any]:
    """返回给定模型的能力档案。"""
    raw = (model_name or "").lower()
    name = _strip_known_suffixes(raw)

    # ---- 生图模型 ----
    if "image" in name:
        is_lite = "flash-lite" in name
        is_flash = "flash" in name
        if is_lite:
            ars, sizes = _PRO_IMAGE_ARS, {"1K"}
        elif is_flash:
            ars, sizes = _FLASH_IMAGE_ARS, {"512", "1K", "2K", "4K"}
        else:  # pro-image 及未知生图
            ars, sizes = _PRO_IMAGE_ARS, {"1K", "2K", "4K"}
        return {
            "family": "image",
            "is_image": True,
            "thinking_kind": None,
            "allowed_sampling": set(),          # 生图剥离所有采样参数
            "image_aspect_ratios": ars,
            "image_sizes": sizes,
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    # ---- 文本 / 多模态 ----
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 3          # 未知 → 视作最新代
    minor = int(m.group(2)) if (m and m.group(2)) else 0
    is_pro = "pro" in name

    if major >= 3:
        # Gemini 3.x：思考用 thinking_level（取代 thinking_budget），且不支持 candidate_count。
        # temperature/top_p/top_k：自 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）已废弃，
        # 官方要求从请求移除（现忽略、未来 400）；更早的 3.x 仍可用但建议保持默认。
        levels = {"low", "medium", "high"} if is_pro else {"minimal", "low", "medium", "high"}
        if "flash-lite" in name:
            default_level = "minimal"
        elif is_pro:
            default_level = "high"
        else:
            default_level = "medium"
        allowed = set(SAMPLING_KEYS)
        allowed.discard("candidate_count")  # 所有 Gemini 3.x 不支持 candidate_count
        temp_dep = _temp_deprecated(name)
        if temp_dep:
            allowed -= {"temperature", "top_p", "top_k"}
        return {
            "family": "g3",
            "is_image": False,
            "thinking_kind": "level",
            "thinking_levels": levels,
            "default_level": default_level,
            "allowed_sampling": allowed,
            "sampling_advice": "deprecated" if temp_dep else "recommend_default",
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    if major == 2 and minor >= 5:
        # Gemini 2.5：thinking_budget；保留全部采样参数
        # ⚠️ 停用预告：gemini-2.5-pro / 2.5-flash / 2.5-flash-lite 官方停用日期 2026-10-16。
        #    届时本分支将没有活模型，可整体删除（连同 DEFAULT_SETTINGS.thinking_g25_budget）。
        # 预算区间出处：Thinking 文档预算表（核对于 2026-07-26）
        #   2.5 Pro        128 – 32768，不可关闭
        #   2.5 Flash        0 – 24576，0 = 关闭
        #   2.5 Flash-Lite 512 – 24576，0 = 关闭（非零最低 512，不是 0）
        if is_pro:
            budget_min, budget_max, can_zero = 128, 32768, False
        elif "flash-lite" in name:
            budget_min, budget_max, can_zero = 512, 24576, True
        else:
            budget_min, budget_max, can_zero = 0, 24576, True
        return {
            "family": "g25",
            "is_image": False,
            "thinking_kind": "budget",
            "budget_min": budget_min,
            "budget_max": budget_max,
            "budget_can_zero": can_zero,
            "allowed_sampling": set(SAMPLING_KEYS),
            "supports_search": True,
            "requires_user_last_turn": False,
        }

    # ---- 更早（2.0 / 1.5 等）：无思考、全采样、宽松 ----
    return {
        "family": "legacy",
        "is_image": False,
        "thinking_kind": None,
        "allowed_sampling": set(SAMPLING_KEYS),
        "supports_search": True,
        "requires_user_last_turn": False,
    }


# 客户端 reasoning_effort 归一化（SillyTavern 等会发 min/minimal/max/xhigh/auto 等）
_EFFORT_ALIASES = {
    "min": "minimal", "minimal": "minimal",
    "low": "low", "medium": "medium", "med": "medium",
    "high": "high", "max": "high", "xhigh": "high", "x-high": "high", "very_high": "high",
    "none": "off", "off": "off",
}


def _norm_effort(e: Optional[str]) -> Optional[str]:
    if not isinstance(e, str):
        return None
    s = e.strip().lower()
    if not s or s == "auto":
        return None  # auto/空 → 交给控制台/模型默认
    return _EFFORT_ALIASES.get(s, s)


def _effort(request: Any) -> Optional[str]:
    e = getattr(request, "reasoning_effort", None)
    if not e and getattr(request, "model_extra", None):
        e = request.model_extra.get("reasoning_effort")
    return _norm_effort(e)


def _extra(request: Any, key: str) -> Any:
    v = getattr(request, key, None)
    if v is None and getattr(request, "model_extra", None):
        v = request.model_extra.get(key)
    return v


def resolve_thinking(model_name: str, request: Any, settings: Dict[str, Any],
                     prefill_active: bool = False) -> Dict[str, Any]:
    """
    计算思考配置（中立结构，各通道再转成自己的线格式）。
    返回 {"mode": None} 或 {"mode":"level","level":..} 或 {"mode":"budget","budget":..}

    优先级（默认）：单次请求 reasoning_effort/thinking_budget > 控制台/该模型专属 > 家族默认。

    控制台 native_thinking_mode（原生思考控制，向后兼容旧 hide_thoughts/thinking_force_console 布尔）：
    - "request"（默认）：跟随请求 effort（无则控制台/模型默认），返回思考。
    - "off"（关闭原生思考，酒馆预设推荐）：把思考压到该模型最低（3.x=minimal/low；2.5-flash=0、
      2.5-pro=128），忽略前端 effort，并 include_thoughts=False。**重要：batchGraphql(Studio) 会忽略
      includeThoughts，故 Cookie 通道还需在响应侧剥离思考块（见 chat_completions）；把档位压到 minimal
      才是 Studio 下真正减少原生思考、避免重预设思考阶段被截断的关键（已真机验证）。**
    - "console"（强制控制台档位）：忽略前端 effort，用控制台/该模型专属档位（未设则模型默认），返回思考。

    另外 prefill_suppress_thinking（默认开）：检测到预填充时等效走 "off" 压制路径。
    """
    prof = get_profile(model_name)
    if prof["is_image"] or prof["thinking_kind"] is None:
        return {"mode": None}

    settings = settings or {}

    # 统一的“原生思考控制”模式：request（跟随请求）| off（关闭原生思考）| console（强制控制台档位）
    # 向后兼容旧布尔开关（上一版的 hide_thoughts / thinking_force_console）。
    mode = settings.get("native_thinking_mode")
    if mode not in ("request", "off", "console"):
        if settings.get("hide_thoughts"):
            mode = "off"
        elif settings.get("thinking_force_console"):
            mode = "console"
        else:
            mode = "request"

    # suppress = 关闭原生思考（压到最低 + 隐藏 + 忽略前端）。预填充压制也走这条路。
    suppress = bool(mode == "off" or (prefill_active and settings.get("prefill_suppress_thinking", True)))
    ignore_client = bool(suppress or mode == "console")

    req_effort = None if ignore_client else _effort(request)
    req_budget = None if ignore_client else _extra(request, "thinking_budget")
    include_thoughts = not suppress

    if prof["thinking_kind"] == "level":
        levels = prof["thinking_levels"]
        if suppress:
            # 压制：3.x 无法完全关闭思考 → 压到该模型最低合法档并隐藏
            return {"mode": "level", "level": _clamp_level("minimal", levels),
                    "include_thoughts": False}
        level = req_effort or settings.get("thinking_g3_level") or prof.get("default_level", "high")
        level = str(level).lower()
        if level in ("off", "none"):
            level = "minimal"
        # 统一就近向下夹取：Pro 上选 minimal 得到 low，而不是被抬成 high
        level = _clamp_level(level, levels)
        return {"mode": "level", "level": level, "include_thoughts": include_thoughts}

    # budget（2.5）
    bmin, bmax, can_zero = prof["budget_min"], prof["budget_max"], prof["budget_can_zero"]
    if suppress:
        # 2.5-flash 可预算 0 完全关闭；2.5-pro 最低 128（无法全关）
        budget = 0 if can_zero else bmin
        return {"mode": "budget", "budget": budget, "include_thoughts": False}
    rb = req_budget
    eff = req_effort
    if rb is not None:
        try:
            budget = int(rb)
        except (TypeError, ValueError):
            budget = -1
    elif eff == "low":
        budget = max(bmin, 1024)
    elif eff in ("medium", "high"):
        budget = -1
    elif eff == "minimal":
        budget = 0 if can_zero else bmin
    else:
        try:
            budget = int(settings.get("thinking_g25_budget", -1))
        except (TypeError, ValueError):
            budget = -1
    if budget == 0 and not can_zero:
        budget = bmin
    # 0 = 关闭思考，是合法值，不能被 max(bmin, ...) 抬成 512/128；仅夹取非 0 非 -1 的值。
    if budget not in (-1, 0):
        budget = max(bmin, min(bmax, budget))
    # F-4：budget=0 表示关闭思考，此时上游拒绝 include_thoughts=True
    # （"Thinking_config.include_thoughts is only enabled when thinking is enabled"），
    # 即"用户主动关思考"这条路径必然 400。关思考就必须同时不要思考摘要。
    if budget == 0:
        include_thoughts = False
    return {"mode": "budget", "budget": budget, "include_thoughts": include_thoughts}


def apply_sampling_policy(profile: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """用控制台的「采样参数处理」覆盖自动判定，返回修正后的档案副本。

    自动判定按版本号（3.6+ / 3.5-flash-lite / 4.x 起废弃），但版本号表达不了
    "号更小却发布更晚"的情况——比如日后出一个 gemini-3.5-pro，官方按"更新模型"
    废弃采样，版本判据却会放行。有了这个开关，加模型就不必改代码。

      auto       —— 用内置版本判定（默认）
      deprecated —— 强制剥离 temperature / top_p / top_k
      allowed    —— 强制保留（官方澄清某模型仍可调时用）
    """
    policy = str(settings.get("sampling_policy", "auto") or "auto").lower()
    if policy not in ("deprecated", "allowed") or profile.get("is_image"):
        return profile          # 生图本来就剥离全部采样，不受该开关影响
    prof = dict(profile)
    allowed = set(prof.get("allowed_sampling", set()))
    if policy == "deprecated":
        allowed -= {"temperature", "top_p", "top_k"}
        prof["sampling_advice"] = "deprecated"
    else:
        allowed |= {"temperature", "top_p", "top_k"}
        # candidate_count 是 3.x 的硬限制，不归这个开关管
        if prof.get("family") == "g3":
            allowed.discard("candidate_count")
        prof["sampling_advice"] = "recommend_default"
    prof["allowed_sampling"] = allowed
    return prof


def sanitize_sampling(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """按档案剥离不支持的采样参数（防止未来 3.x 传弃用参数直接 400）。"""
    allowed = profile.get("allowed_sampling", set())
    for key in list(config.keys()):
        if key in SAMPLING_KEYS and key not in allowed:
            config.pop(key, None)
    return config


def resolve_image_size(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """确定生图分辨率（大写 1K/2K/4K/512），按模型白名单校验，缺省回退。"""
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    sizes = prof["image_sizes"]
    settings = settings or {}
    raw = _extra(request, "image_size") or settings.get("image_size") or "1K"
    size = str(raw).upper().replace("0.5K", "512").replace("512PX", "512")
    if size in sizes:
        return size
    # 回退：就近向下取档（4K→2K→1K→512），向下没有再向上。
    # 旧实现写成 `for cand in (...): if cand in sizes: return "1K" if "1K" in sizes else cand`，
    # 因为 1K 在所有白名单里，任何非法值都返回 1K，注释里的“优先给相近的高档”从未生效。
    ladder = ["512", "1K", "2K", "4K"]
    idx = ladder.index(size) if size in ladder else len(ladder) - 1
    for i in range(idx, -1, -1):
        if ladder[i] in sizes:
            return ladder[i]
    for i in range(idx + 1, len(ladder)):
        if ladder[i] in sizes:
            return ladder[i]
    return "1K"


def _prompt_aspect_ratio(request: Any) -> Optional[str]:
    """从最后一条 user 消息文本里解析宽高比（--ar 优先，其次独立比例）。"""
    try:
        for msg in reversed(request.messages):
            if getattr(msg, "role", None) != "user":
                continue
            c = msg.content
            content = ""
            if isinstance(c, str):
                content = c
            elif isinstance(c, list):
                # F-1：先归一，否则 pydantic 生成的 ContentPartText 会被 isinstance(p, dict) 漏掉，
                # 导致列表形式内容里的 `--ar 16:9` 永远检测不到。
                parts = [normalize_content_part(p) for p in c]
                content = " ".join(
                    p.get("text", "") for p in parts
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            m = re.search(r"(?i)--ar\s*(\d+[:：]\d+)", content) or re.search(r"\b(\d+[:：]\d+)\b", content)
            return m.group(1) if m else None
    except Exception:
        return None
    return None


def resolve_aspect_ratio(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """
    生图宽高比解析（两通道共用）。优先级：
    请求额外字段 aspect_ratio/ar > OpenAI size 映射 > 提示词解析 > 控制台默认，
    最后按该生图模型白名单校验；不合法则返回 None（交给模型自动决定）。
    """
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    settings = settings or {}
    raw = _extra(request, "aspect_ratio") or _extra(request, "ar")
    if not raw:
        size_param = _extra(request, "size")
        if isinstance(size_param, str):
            if size_param == "1024x1024":
                raw = "1:1"
            elif size_param == "1024x768":
                raw = "4:3"
            elif size_param == "768x1024":
                raw = "3:4"
            elif ":" in size_param:
                raw = size_param
    if not raw:
        raw = _prompt_aspect_ratio(request)
    if not raw:
        raw = settings.get("image_aspect_ratio") or None
    return validate_aspect_ratio(model_name, raw)


def validate_aspect_ratio(model_name: str, ar: Optional[str]) -> Optional[str]:
    """校验宽高比是否在该生图模型白名单内；不在则返回 None（交给模型自动决定）。"""
    if not ar:
        return None
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    norm = str(ar).replace("：", ":").strip()
    return norm if norm in prof["image_aspect_ratios"] else None


def capabilities_summary(model_name: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """给控制台前端用的精简能力描述（决定显示/禁用哪些控件）。

    传入该模型生效的 settings 时，会把「采样参数处理」开关算进去，
    否则控制台显示的能力会和实际下发的参数对不上。
    """
    prof = get_profile(model_name)
    if settings:
        prof = apply_sampling_policy(prof, settings)
    thinking: Dict[str, Any] = {"kind": prof["thinking_kind"]}
    if prof["thinking_kind"] == "level":
        # 按强度排序（minimal→high），不要用字典序（会排成 high, low, medium, minimal）
        thinking["levels"] = sort_levels(prof["thinking_levels"])
        thinking["can_off"] = False
    elif prof["thinking_kind"] == "budget":
        thinking["budget_min"] = prof["budget_min"]
        thinking["budget_max"] = prof["budget_max"]
        thinking["can_off"] = prof["budget_can_zero"]
    return {
        "family": prof["family"],
        "is_image": prof["is_image"],
        "thinking": thinking,
        "sampling": sorted(prof["allowed_sampling"]),
        "sampling_advice": prof.get("sampling_advice"),
        "image_aspect_ratios": sorted(prof.get("image_aspect_ratios", set())),
        "image_sizes": sorted(prof.get("image_sizes", set())),
        "supports_search": prof["supports_search"],
    }
