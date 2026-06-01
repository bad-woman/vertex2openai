import httpx
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional

import config as app_config

_model_cache: Optional[Dict[str, List[str]]] = None
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


def _normalize_models_config(data: object) -> Optional[Dict[str, List[str]]]:
    if not isinstance(data, dict):
        return None

    models = data.get("models")
    if isinstance(models, list):
        return {"models": [str(model) for model in models if isinstance(model, str) and model.strip()]}

    legacy_express_models = data.get("vertex_express_models")
    if isinstance(legacy_express_models, list):
        return {"models": [str(model) for model in legacy_express_models if isinstance(model, str) and model.strip()]}

    return None


def _load_local_models_config() -> Dict[str, List[str]]:
    try:
        local_file = next((path for path in _LOCAL_MODEL_FILE_CANDIDATES if path.exists()), None)
        if local_file is None:
            searched = ", ".join(str(path) for path in _LOCAL_MODEL_FILE_CANDIDATES)
            print(f"❌ [模型配置] 未找到本地 vertexModels.json，已检查：{searched}")
            return {"models": []}

        data = json.loads(local_file.read_text(encoding="utf-8"))
        normalized = _normalize_models_config(data)
        if normalized is not None:
            print(f"📦 [模型配置] 已使用本地模型配置文件：{local_file}。")
            return normalized
        print(f"❌ [模型配置] 本地模型配置结构无效：{local_file}。")
    except Exception as exc:
        print(f"❌ [模型配置] 读取本地模型配置失败：{exc}")
    return {"models": []}


async def fetch_and_parse_models_config() -> Optional[Dict[str, List[str]]]:
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
                    print(f"✅ [模型配置] 远程模型配置加载成功，共 {len(normalized['models'])} 个模型。")
                    return normalized
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


async def get_models_config() -> Dict[str, List[str]]:
    global _model_cache
    async with _cache_lock:
        if _model_cache is None:
            print("📦 [模型配置] 缓存为空，正在初始化模型列表。")
            _model_cache = await fetch_and_parse_models_config()
            if _model_cache is None:
                print("⚠️ [模型配置] 模型配置初始化失败，当前模型列表为空。")
                _model_cache = {"models": []}
    return _model_cache


async def get_express_models() -> List[str]:
    config = await get_models_config()
    return config.get("models", [])


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
