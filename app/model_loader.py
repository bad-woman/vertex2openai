import httpx
import asyncio
import json
from pathlib import Path
from typing import Any, List, Dict, Optional

import config as app_config

_model_cache: Optional[Dict[str, Any]] = None
_local_fallback_cache: Optional[Dict[str, Any]] = None
_cache_lock = asyncio.Lock()
DEFAULT_MODELS_CONFIG_URL = "https://raw.githubusercontent.com/bad-woman/vertex2openai/main/vertexModels.json"
FALLBACK_MODELS_CONFIG_URLS = [
    "https://cdn.jsdelivr.net/gh/bad-woman/vertex2openai@main/vertexModels.json",
]

_LOCAL_MODEL_FILE_CANDIDATES = [
    Path.cwd() / "vertexModels.json",
    Path(__file__).resolve().parent / "vertexModels.json",
    Path(__file__).resolve().parent.parent / "vertexModels.json",
]

_EMPTY_CONFIG: Dict[str, Any] = {
    "models": [],
    "categories": [],
    "capability_overrides": {},
}


def _clean_capabilities_dict(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    if isinstance(raw.get("thinking_levels"), list):
        levels = [str(lv).strip().lower() for lv in raw["thinking_levels"] if isinstance(lv, str) and lv.strip()]
        if levels:
            out["thinking_levels"] = levels
    if "max_nonstream_output_tokens" in raw:
        val = raw.get("max_nonstream_output_tokens")
        if val is None:
            out["max_nonstream_output_tokens"] = None
        else:
            try:
                ival = int(val)
                if ival > 0:
                    out["max_nonstream_output_tokens"] = ival
            except (TypeError, ValueError):
                pass
    for bool_key in (
        "supports_search",
        "supports_stream_fc_args",
        "requires_user_last_turn",
        "is_image",
        "sampling_deprecated",
    ):
        if isinstance(raw.get(bool_key), bool):
            out[bool_key] = raw[bool_key]
    if isinstance(raw.get("image_sizes"), list):
        sizes = [str(s).strip() for s in raw["image_sizes"] if isinstance(s, str) and s.strip()]
        if sizes:
            out["image_sizes"] = sizes
    if isinstance(raw.get("image_aspect_ratios"), list):
        ars = [str(a).strip() for a in raw["image_aspect_ratios"] if isinstance(a, str) and a.strip()]
        if ars:
            out["image_aspect_ratios"] = ars
    if isinstance(raw.get("family"), str) and raw["family"].strip():
        out["family"] = raw["family"].strip().lower()
    return out


def _normalize_models_config(data: object) -> Optional[Dict[str, Any]]:
    """规范化模型配置，向后兼容扁平 `models` 数组与新版 `categories` 分组能力定义。"""
    if not isinstance(data, dict):
        return None

    raw_categories = data.get("categories")
    raw_models = data.get("models")
    if not isinstance(raw_categories, list) and not isinstance(raw_models, list):
        return None

    categories: List[Dict[str, Any]] = []
    capability_overrides: Dict[str, Dict[str, Any]] = {}
    ordered_models: List[str] = []
    seen_models: set[str] = set()

    if isinstance(raw_categories, list):
        for idx, cat in enumerate(raw_categories):
            if not isinstance(cat, dict):
                continue
            cat_models_raw = cat.get("models")
            if not isinstance(cat_models_raw, list):
                continue
            cat_id = str(cat.get("id") or f"group-{idx + 1}").strip()
            cat_label = str(cat.get("label") or cat_id).strip()
            cat_caps = _clean_capabilities_dict(cat.get("capabilities"))
            cat_models: List[str] = []
            for m in cat_models_raw:
                if not isinstance(m, str) or not m.strip():
                    continue
                mid = m.strip()
                if mid not in cat_models:
                    cat_models.append(mid)
                if mid not in seen_models:
                    ordered_models.append(mid)
                    seen_models.add(mid)
                if cat_caps:
                    low = mid.lower()
                    merged_cap = dict(capability_overrides.get(low, {}))
                    merged_cap.update(cat_caps)
                    capability_overrides[low] = merged_cap
            if cat_models:
                entry: Dict[str, Any] = {
                    "id": cat_id,
                    "label": cat_label,
                    "models": cat_models,
                }
                if cat_caps:
                    entry["capabilities"] = cat_caps
                categories.append(entry)

    if isinstance(raw_models, list):
        for m in raw_models:
            if not isinstance(m, str) or not m.strip():
                continue
            mid = m.strip()
            if mid not in seen_models:
                ordered_models.append(mid)
                seen_models.add(mid)

    # 可选的单模型能力覆盖（优先级高于分类级 capabilities）
    raw_per_model = data.get("model_capabilities") or data.get("capability_overrides")
    if isinstance(raw_per_model, dict):
        for model_key, cap_raw in raw_per_model.items():
            if not isinstance(model_key, str) or not model_key.strip():
                continue
            cleaned = _clean_capabilities_dict(cap_raw)
            if cleaned:
                low = model_key.strip().lower()
                merged_cap = dict(capability_overrides.get(low, {}))
                merged_cap.update(cleaned)
                capability_overrides[low] = merged_cap

    return {
        "models": ordered_models,
        "categories": categories,
        "capability_overrides": capability_overrides,
    }


def _merge_configs(remote: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """合并远程与本地模型配置：
    - 保留全部模型（合并去重）；
    - 本地分组定义与能力覆盖优先于远程同名模型，便于用户仅改本地 vertexModels.json 即可新增/归类模型；
    - 远程有但本地未归类的模型仍会保留在 models 中，由前端放入「其他模型」组并用正则兜底能力。
    """
    local_cats = local.get("categories") or []
    remote_cats = remote.get("categories") or []

    # 按分组 ID 合并：以本地分组顺序与归属为先，再补入远程分组中本地未收录的模型
    local_assigned: set[str] = {
        m for cat in local_cats for m in (cat.get("models") or [])
    }

    cats_by_id: Dict[str, Dict[str, Any]] = {}
    cat_order: List[str] = []

    for cat in local_cats:
        cid = cat["id"]
        cats_by_id[cid] = {
            "id": cid,
            "label": cat["label"],
            "models": list(cat.get("models") or []),
            **({"capabilities": dict(cat["capabilities"])} if cat.get("capabilities") else {}),
        }
        cat_order.append(cid)

    for cat in remote_cats:
        cid = cat["id"]
        r_models = [m for m in (cat.get("models") or []) if m not in local_assigned]
        if cid in cats_by_id:
            existing = cats_by_id[cid]
            for m in r_models:
                if m not in existing["models"]:
                    existing["models"].append(m)
            # 若远程分组有能力定义、本地该分组未定义某字段，则作为基础补充（本地字段优先）
            if cat.get("capabilities"):
                merged_caps = dict(cat["capabilities"])
                merged_caps.update(existing.get("capabilities") or {})
                existing["capabilities"] = merged_caps
        elif r_models:
            cats_by_id[cid] = {
                "id": cid,
                "label": cat["label"],
                "models": r_models,
                **({"capabilities": dict(cat["capabilities"])} if cat.get("capabilities") else {}),
            }
            cat_order.append(cid)

    merged_categories = [cats_by_id[cid] for cid in cat_order if cats_by_id[cid]["models"]]

    # 合并扁平模型列表：优先保留已分组顺序，再补入远程/本地未分组模型
    models: List[str] = []
    seen: set[str] = set()
    for cat in merged_categories:
        for m in cat["models"]:
            if m not in seen:
                models.append(m)
                seen.add(m)
    for src in (remote, local):
        for m in (src.get("models") or []):
            if m not in seen:
                models.append(m)
                seen.add(m)

    # 合并能力覆盖：远程为底，本地覆盖优先
    capability_overrides: Dict[str, Dict[str, Any]] = {}
    for k, v in (remote.get("capability_overrides") or {}).items():
        if isinstance(v, dict):
            capability_overrides[k] = dict(v)
    for k, v in (local.get("capability_overrides") or {}).items():
        if isinstance(v, dict):
            merged_cap = dict(capability_overrides.get(k, {}))
            merged_cap.update(v)
            capability_overrides[k] = merged_cap

    return {
        "models": models,
        "categories": merged_categories,
        "capability_overrides": capability_overrides,
    }


def _load_local_models_config() -> Dict[str, Any]:
    global _local_fallback_cache
    try:
        local_file = next((path for path in _LOCAL_MODEL_FILE_CANDIDATES if path.exists()), None)
        if local_file is None:
            searched = ", ".join(str(path) for path in _LOCAL_MODEL_FILE_CANDIDATES)
            print(f"❌ [模型配置] 未找到本地 vertexModels.json，已检查：{searched}")
            _local_fallback_cache = dict(_EMPTY_CONFIG)
            return _local_fallback_cache

        data = json.loads(local_file.read_text(encoding="utf-8"))
        normalized = _normalize_models_config(data)
        if normalized is not None:
            print(f"📦 [模型配置] 已使用本地模型配置文件：{local_file}。")
            _local_fallback_cache = normalized
            return normalized
        print(f"❌ [模型配置] 本地模型配置结构无效：{local_file}。")
    except Exception as exc:
        print(f"❌ [模型配置] 读取本地模型配置失败：{exc}")
    _local_fallback_cache = dict(_EMPTY_CONFIG)
    return _local_fallback_cache


def get_cached_model_capability_override(model_name: str) -> Optional[Dict[str, Any]]:
    """同步读取当前缓存（或本地兜底）中的模型分类能力覆盖，供 model_capabilities.get_profile 调用。"""
    key = (model_name or "").strip().lower()
    if not key:
        return None
    cfg = _model_cache
    if cfg is None:
        cfg = _local_fallback_cache if _local_fallback_cache is not None else _load_local_models_config()
    overrides = cfg.get("capability_overrides") if isinstance(cfg, dict) else None
    if not isinstance(overrides, dict):
        return None
    hit = overrides.get(key)
    return dict(hit) if isinstance(hit, dict) else None


async def fetch_and_parse_models_config() -> Optional[Dict[str, Any]]:
    primary_url = app_config.MODELS_CONFIG_URL or DEFAULT_MODELS_CONFIG_URL
    urls_to_try = [primary_url]
    for fallback_url in FALLBACK_MODELS_CONFIG_URLS:
        if fallback_url not in urls_to_try:
            urls_to_try.append(fallback_url)

    client_args = {"timeout": 20.0}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE

    async with httpx.AsyncClient(**client_args) as client:
        for index, url in enumerate(urls_to_try, start=1):
            print(f"🌐 [模型配置] 正在获取远程模型配置（第 {index} 个地址）：{url}")
            try:
                response = await client.get(url)
                response.raise_for_status()
                normalized = _normalize_models_config(response.json())
                if normalized is not None:
                    merged = _merge_configs(normalized, _load_local_models_config())
                    print(
                        f"✅ [模型配置] 远程模型配置加载成功，与本地合并后共 {len(merged['models'])} 个模型"
                        + (f"（{len(merged['categories'])} 个分类）" if merged.get("categories") else "")
                        + "。"
                    )
                    return merged
                print(f"⚠️ [模型配置] 远程模型配置结构无效，准备尝试下一个地址：{url}")
            except httpx.HTTPStatusError as exc:
                print(f"⚠️ [模型配置] 远程模型配置地址返回 HTTP {exc.response.status_code}，准备尝试下一个地址：{url}")
            except httpx.RequestError as exc:
                print(f"⚠️ [模型配置] 获取远程模型配置失败，准备尝试下一个地址。网络错误：{exc}")
            except json.JSONDecodeError as exc:
                print(f"⚠️ [模型配置] 远程模型配置不是有效 JSON，准备尝试下一个地址。解析错误：{exc}")
            except Exception as exc:
                print(f"⚠️ [模型配置] 加载远程模型配置时出现异常，准备尝试下一个地址：{exc}")

    print("⚠️ [模型配置] 所有远程模型配置地址都不可用，将回退到本地 vertexModels.json。")
    return _load_local_models_config()


async def get_models_config() -> Dict[str, Any]:
    global _model_cache
    async with _cache_lock:
        if _model_cache is None:
            print("📦 [模型配置] 缓存为空，正在初始化模型列表。")
            _model_cache = await fetch_and_parse_models_config()
            if _model_cache is None:
                print("⚠️ [模型配置] 模型配置初始化失败，当前模型列表为空。")
                _model_cache = dict(_EMPTY_CONFIG)
    return _model_cache


async def get_express_models() -> List[str]:
    config = await get_models_config()
    return config.get("models", [])


async def get_model_categories() -> List[Dict[str, Any]]:
    config = await get_models_config()
    return config.get("categories", [])


async def refresh_models_config_cache() -> bool:
    global _model_cache
    print("🔄 [模型配置] 正在刷新模型配置缓存。")
    async with _cache_lock:
        new_config = await fetch_and_parse_models_config()
        if new_config is not None:
            _model_cache = new_config
            print("✅ [模型配置] 模型配置缓存刷新成功。")
            return True
        print("❌ [模型配置] 模型配置缓存刷新失败。")
        return False
