import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_express_models, refresh_models_config_cache

router = APIRouter()
_last_model_fetch_time = 0


@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    global _last_model_fetch_time

    current_time = time.time()
    if current_time - _last_model_fetch_time > 3600:
        await refresh_models_config_cache()
        _last_model_fetch_time = current_time

    express_key_manager_instance = fastapi_request.app.state.express_key_manager
    has_express_key = express_key_manager_instance.get_total_keys() > 0
    raw_models = await get_express_models() if has_express_key else []

    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()

    def add_model(base_id: str):
        suffixes = [""]
        if "gemini" in base_id.lower() and "image" not in base_id.lower():
            suffixes.append("-search")

        for suffix in suffixes:
            final_id = f"{base_id}{suffix}"
            if final_id in processed_ids:
                continue
            final_model_list.append({
                "id": final_id,
                "object": "model",
                "created": int(current_time),
                "owned_by": "google",
                "permission": [],
                "root": base_id,
                "parent": None,
            })
            processed_ids.add(final_id)

    for model_id in raw_models:
        add_model(model_id)

    return {"object": "list", "data": sorted(final_model_list, key=lambda item: item["id"])}
