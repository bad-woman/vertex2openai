"""
模型能力矩阵（按家族识别 · Gemini 3.x / 4.x 及生图家族）

用于决定：
- 每个模型支持的思考档位（thinking_level）与合法档位白名单
- 可用的采样参数集合（避免把已废弃的参数发给模型导致 400）
- 生图模型支持的宽高比与分辨率白名单
- 是否要求最后一条消息为 user（Gemini 3.x / 4.x 强制，否则 400 → 触发预填充兼容）
"""

import re
from typing import Any, Dict, Optional

from models import normalize_content_part

SAMPLING_KEYS = {
    "temperature", "top_p", "top_k",
    "presence_penalty", "frequency_penalty",
    "candidate_count", "seed", "stop_sequences", "max_output_tokens",
}

PENALTY_KEYS = {
    "presence_penalty", "frequency_penalty",
}

# 生图模型支持的 15 种官方宽高比（gemini-3-pro-image / gemini-3.1-flash-image / gemini-nano-banana-2.1）
_IMAGE_ARS = {
    "1:1", "3:2", "2:3", "3:4", "1:4", "4:1", "4:3", "4:5", "5:4",
    "1:8", "8:1", "9:16", "16:9", "21:9", "9:21",
}

_LEVEL_ORDER = ["minimal", "low", "medium", "high"]

# 3.6 及更早的非 Pro 机型实测支持 MINIMAL；3.7 及更新机型最低为 LOW
_MINIMAL_MAX_VERSION = (3, 6)


def _supports_minimal(major: int, minor: int, is_pro: bool) -> bool:
    if is_pro:
        return False
    return (major, minor) <= _MINIMAL_MAX_VERSION


def _clamp_level(level: str, levels: set) -> str:
    """把任意档位夹到该模型的合法档位集合内（优先向下就近取，向下没有再向上）。"""
    if not levels:
        return "low"
    if level in levels:
        return level
    try:
        idx = _LEVEL_ORDER.index(level)
    except ValueError:
        idx = _LEVEL_ORDER.index("high")
    for i in range(idx, -1, -1):
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    for i in range(idx + 1, len(_LEVEL_ORDER)):
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    return sorted(levels)[0]


def sort_levels(levels) -> list:
    known = [lv for lv in _LEVEL_ORDER if lv in levels]
    extra = sorted(lv for lv in levels if lv not in _LEVEL_ORDER)
    return known + extra


def _strip_known_suffixes(name: str) -> str:
    n = name.lower()
    for suf in ("-search", "-1k", "-2k", "-4k", "-512"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"-think-(minimal|low|medium|high|off|none)$", "", n)
    return n


def _temp_deprecated(name: str) -> bool:
    """自 Gemini 3.6 Flash 与 3.5 Flash-Lite 起（含 4.x 及更新模型），temperature/top_p/top_k 已废弃。"""
    n = name.lower()
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
    if not m:
        return True
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
    if "image" in name or "nano-banana" in name:
        is_nano = "nano-banana" in name
        is_lite = "flash-lite" in name
        is_flash = "flash" in name
        if is_nano:
            sizes = {"1K", "2K", "4K"}
            thinking_levels = {"minimal", "medium", "high"}
        elif is_lite:
            sizes = {"1K"}
            thinking_levels = {"minimal", "high"}
        elif is_flash:
            sizes = {"512", "1K", "2K", "4K"}
            thinking_levels = {"minimal", "high"}
        else:
            sizes = {"1K", "2K", "4K"}
            thinking_levels = {"high"}
        return {
            "family": "image",
            "is_image": True,
            "thinking_kind": "level",
            "thinking_levels": thinking_levels,
            "default_level": "high",
            "allowed_sampling": set(),
            "image_aspect_ratios": _IMAGE_ARS,
            "image_sizes": sizes,
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    # ---- 文本 / 多模态（Gemini 3.x / 4.x）----
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 3
    minor = int(m.group(2)) if (m and m.group(2)) else 0
    is_pro = "pro" in name

    levels = {"low", "medium", "high"}
    if _supports_minimal(major, minor, is_pro):
        levels = levels | {"minimal"}
    if "flash-lite" in name and "minimal" in levels:
        default_level = "minimal"
    elif is_pro:
        default_level = "high"
    else:
        default_level = "medium"

    allowed = set(SAMPLING_KEYS)
    allowed.discard("candidate_count")
    allowed -= PENALTY_KEYS
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
        return None
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
    """计算思考配置：request（跟随请求）| off（关闭原生思考）| console（强制控制台档位）。"""
    prof = get_profile(model_name)
    if prof["thinking_kind"] is None:
        return {"mode": None}

    settings = settings or {}
    mode = settings.get("native_thinking_mode")
    if mode not in ("request", "off", "console"):
        mode = "request"

    suppress = bool(mode == "off" or (prefill_active and settings.get("prefill_suppress_thinking", True)))
    ignore_client = bool(suppress or mode == "console")

    req_effort = None if ignore_client else _effort(request)
    include_thoughts = not suppress

    levels = prof["thinking_levels"]
    if suppress:
        return {
            "mode": "level",
            "level": _clamp_level("minimal", levels),
            "include_thoughts": False,
        }
    level = req_effort or settings.get("thinking_g3_level") or prof.get("default_level", "high")
    level = str(level).lower()
    if level in ("off", "none"):
        level = "minimal"
    level = _clamp_level(level, levels)
    return {"mode": "level", "level": level, "include_thoughts": include_thoughts}


def apply_sampling_policy(profile: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(settings.get("sampling_policy", "auto") or "auto").lower()
    if policy not in ("deprecated", "allowed") or profile.get("is_image"):
        return profile
    prof = dict(profile)
    allowed = set(prof.get("allowed_sampling", set()))
    if policy == "deprecated":
        allowed -= {"temperature", "top_p", "top_k"}
        prof["sampling_advice"] = "deprecated"
    else:
        allowed |= {"temperature", "top_p", "top_k"}
        allowed.discard("candidate_count")
        prof["sampling_advice"] = "recommend_default"
    prof["allowed_sampling"] = allowed
    return prof


def sanitize_sampling(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    allowed = profile.get("allowed_sampling", set())
    for key in list(config.keys()):
        if key in SAMPLING_KEYS and key not in allowed:
            config.pop(key, None)
    return config


def resolve_image_size(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    sizes = prof["image_sizes"]
    settings = settings or {}
    raw = _extra(request, "image_size") or settings.get("image_size") or "1K"
    size = str(raw).upper().replace("0.5K", "512").replace("512PX", "512")
    if size in sizes:
        return size
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
    try:
        for msg in reversed(request.messages):
            if getattr(msg, "role", None) != "user":
                continue
            c = msg.content
            content = ""
            if isinstance(c, str):
                content = c
            elif isinstance(c, list):
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
    if not ar:
        return None
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    norm = str(ar).replace("：", ":").strip()
    return norm if norm in prof["image_aspect_ratios"] else None


def capabilities_summary(model_name: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prof = get_profile(model_name)
    if settings:
        prof = apply_sampling_policy(prof, settings)
    thinking: Dict[str, Any] = {"kind": prof["thinking_kind"]}
    if prof["thinking_kind"] == "level":
        thinking["levels"] = sort_levels(prof["thinking_levels"])
        thinking["can_off"] = False
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
