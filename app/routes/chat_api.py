import re
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from google import genai

from models import OpenAIRequest
from auth import get_api_key
from message_processing import create_gemini_prompt
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call,
)
from http_options import get_http_options

router = APIRouter()

LEGACY_EXPRESS_PREFIX = "[EXPRESS] "
LEGACY_PAY_PREFIX = "[PAY]"
OPENAI_DIRECT_SUFFIX = "-openai"
OPENAI_SEARCH_SUFFIX = "-openaisearch"


def _normalize_model_name(model_name: str) -> tuple[str, bool, str | None]:
    base_model_name = model_name

    if base_model_name.startswith(LEGACY_EXPRESS_PREFIX):
        base_model_name = base_model_name[len(LEGACY_EXPRESS_PREFIX):]

    if base_model_name.startswith(LEGACY_PAY_PREFIX):
        return base_model_name, False, "当前版本已经移除 Pay/Service Account 模式，请改用 Express Mode 模型名称。"

    if base_model_name.endswith(OPENAI_SEARCH_SUFFIX) or base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
        return base_model_name, False, "当前版本已经移除 -openai/-openaisearch 直连上游路径，请直接使用普通模型名或 -search 模型名。"

    is_grounded_search = base_model_name.endswith("-search")
    if is_grounded_search:
        base_model_name = base_model_name[:-len("-search")]

    return base_model_name, is_grounded_search, None


def _build_thinking_config(base_model_name: str, request: OpenAIRequest, is_image_model: bool) -> dict | None:
    if is_image_model:
        return None

    is_thinking_capable = False
    is_gemini_2_5 = False
    is_gemini_3_or_above = False

    version_match = re.search(r"gemini-(\d+)\.(\d+)|gemini-(\d+)", base_model_name.lower())
    if version_match:
        groups = version_match.groups()
        major = 0
        minor_val = 0.0

        if groups[2]:
            major = int(groups[2])
        elif groups[0] and groups[1]:
            major = int(groups[0])
            try:
                minor_val = float(groups[1])
            except ValueError:
                pass

        if major > 2 or (major == 2 and minor_val >= 5.0):
            is_thinking_capable = True
        if major == 2 and minor_val == 5.0:
            is_gemini_2_5 = True
        elif major >= 3:
            is_gemini_3_or_above = True

    if not is_thinking_capable:
        return None

    reasoning_effort = getattr(request, "reasoning_effort", None)
    if not reasoning_effort and hasattr(request, "model_extra") and request.model_extra:
        reasoning_effort = request.model_extra.get("reasoning_effort")

    thinking_config = {"include_thoughts": True}

    if is_gemini_3_or_above:
        import google.genai
        genai_version_str = getattr(google.genai, "__version__", "1.0.0")
        try:
            parts = genai_version_str.split(".")
            sdk_supports_level = (int(parts[0]) >= 2) or (int(parts[0]) == 1 and int(parts[1]) >= 51)
        except Exception:
            sdk_supports_level = False

        if sdk_supports_level:
            if reasoning_effort == "low":
                thinking_config["thinking_level"] = "low"
            elif reasoning_effort == "medium":
                thinking_config["thinking_level"] = "medium"
            else:
                thinking_config["thinking_level"] = "high"
        else:
            print(f"⚠️ [推理配置] 当前 google-genai 版本 {genai_version_str} 不支持 thinking_level，已自动跳过该参数。")
    elif is_gemini_2_5:
        if reasoning_effort == "low":
            thinking_config["thinking_budget"] = 1024
        else:
            thinking_config["thinking_budget"] = -1

    return thinking_config


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        base_model_name, is_grounded_search, model_error = _normalize_model_name(request.model)
        if model_error:
            print(f"❌ [模型名称] {model_error} 收到的模型名：{request.model}")
            return JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, model_error, "invalid_request_error"),
            )

        if express_key_manager_instance.get_total_keys() == 0:
            error_msg = "未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。"
            print(f"❌ [密钥配置] {error_msg}")
            return JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"),
            )

        key_tuple = express_key_manager_instance.get_express_api_key()
        if not key_tuple:
            error_msg = "没有可用的 Express API Key。"
            print(f"❌ [密钥配置] {error_msg}")
            return JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"),
            )

        _, express_api_key = key_tuple
        client_to_use = genai.Client(
            vertexai=True,
            api_key=express_api_key,
            http_options=get_http_options(),
        )
        print(f"🌐 [上游端点] 使用官方 Gemini Express Mode SDK 调用模型 {base_model_name}，不拼接 project/location。")

        is_image_model = "image" in request.model.lower()
        gen_config_dict = create_generation_config(request)
        thinking_config = _build_thinking_config(base_model_name, request, is_image_model)
        if thinking_config:
            gen_config_dict["thinking_config"] = thinking_config

        if is_grounded_search and not is_image_model:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]
            print(f"🔎 [搜索增强] 已为模型 {base_model_name} 启用 Google Search 工具。")

        return await execute_gemini_call(client_to_use, base_model_name, create_gemini_prompt, gen_config_dict, request)

    except Exception as exc:
        error_msg = f"处理 /v1/chat/completions 请求时发生未预期异常：{exc}"
        print(f"❌ [服务异常] {error_msg}")
        return JSONResponse(
            status_code=500,
            content=create_openai_error_response(500, error_msg, "server_error"),
        )
