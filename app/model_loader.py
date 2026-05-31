import httpx
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional

import config as app_config

_model_cache: Optional[Dict[str, List[str]]] = None
_cache_lock = asyncio.Lock()
_LOCAL_MODELS_FILE = Path(__file__).resolve().parent.parent / "vertexModels.json"


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
        data = json.loads(_LOCAL_MODELS_FILE.read_text(encoding="utf-8"))
        normalized = _normalize_models_config(data)
        if normalized is not None:
            print(f"📦 [模型配置] 已使用本地模型配置文件：{_LOCAL_MODELS_FILE.name}。")
            return normalized
        print(f"❌ [模型配置] 本地模型配置结构无效：{_LOCAL_MODELS_FILE}。")
    except Exception as exc:
        print(f"❌ [模型配置] 读取本地模型配置失败：{exc}")
    return {"models": []}


async def fetch_and_parse_models_config() -> Optional[Dict[str, List[str]]]:
    if not app_config.MODELS_CONFIG_URL:
        print("📦 [模型配置] MODELS_CONFIG_URL 未设置，直接使用本地 vertexModels.json。")
        return _load_local_models_config()

    print(f"🌐 [模型配置] 正在获取远程模型配置：{app_config.MODELS_CONFIG_URL}")

    client_args = {"timeout": 20.0}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE

    try:
        async with httpx.AsyncClient(**client_args) as client:
            response = await client.get(app_config.MODELS_CONFIG_URL)
            response.raise_for_status()
            normalized = _normalize_models_config(response.json())
            if normalized is not None:
                print(f"✅ [模型配置] 远程模型配置加载成功，共 {len(normalized['models'])} 个模型。")
                return normalized
            print("❌ [模型配置] 远程模型配置结构无效，将回退到本地配置。")
            return _load_local_models_config()
    except httpx.RequestError as exc:
        print(f"⚠️ [模型配置] 获取远程模型配置失败，将回退到本地配置。网络错误：{exc}")
        return _load_local_models_config()
    except json.JSONDecodeError as exc:
        print(f"⚠️ [模型配置] 远程模型配置不是有效 JSON，将回退到本地配置。解析错误：{exc}")
        return _load_local_models_config()
    except Exception as exc:
        print(f"⚠️ [模型配置] 加载远程模型配置时出现异常，将回退到本地配置：{exc}")
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
