import re
import google.genai
from fastapi import Request
from fastapi.responses import JSONResponse
from google import genai

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from api_helpers import (
    create_generation_config,
    execute_gemini_call,
    create_openai_error_response,
    execute_interaction_call  # 导入 Interactions API 处理器
)
from message_processing import create_gemini_prompt
from http_options import get_http_options

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
    if is_image_model: return None
    is_thinking_capable = False
    is_gemini_2_5 = False
    is_gemini_3_or_above = False
    version_match = re.search(r"gemini-(\d+)\.(\d+)|gemini-(\d+)", base_model_name.lower())
    if version_match:
        groups = version_match.groups()
        major = int(groups[2]) if groups[2] else int(groups[0])
        minor_val = float(groups[1]) if groups[1] else 0.0
        if major > 2 or (major == 2 and minor_val >= 5.0): is_thinking_capable = True
        if major == 2 and minor_val == 5.0: is_gemini_2_5 = True
        elif major >= 3: is_gemini_3_or_above = True

    if not is_thinking_capable: return None
    reasoning_effort = getattr(request, "reasoning_effort", None) or (request.model_extra.get("reasoning_effort") if hasattr(request, "model_extra") and request.model_extra else None)
    thinking_config = {"include_thoughts": True}

    if is_gemini_3_or_above:
        thinking_config["thinking_level"] = "low" if reasoning_effort == "low" else ("medium" if reasoning_effort == "medium" else "high")
    elif is_gemini_2_5:
        thinking_config["thinking_budget"] = 1024 if reasoning_effort == "low" else -1
    return thinking_config


# ==========================================================
# 🌟 新增：专门为 Interactions API (Omni模型) 适配的 Prompt 构造器
# ==========================================================
def create_interaction_prompt(messages: list) -> list:
    """将 OpenAI 消息结构转换为 Interactions API 要求的 Step 列表结构"""
    # 1. 先复用原有的强大处理器，它会帮我们自动下载图片、压缩、解析格式
    contents = create_gemini_prompt(messages)
    
    steps = []
    # 2. 将 Content 结构转化为 Pydantic 能够识别的 UserInputStep / ModelOutputStep 字典
    for content in contents:
        # 映射 role 到 type
        step_type = "user_input" if content.role == "user" else "model_output"
        steps.append({
            "type": step_type,
            "content": content.parts  # 直接透传已经处理好的 parts 数组
        })
        
    return steps


class ExpressSDKUpstream(BaseUpstream):
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        base_model_name, is_grounded_search, model_error = _normalize_model_name(request_obj.model)
        if model_error:
            return JSONResponse(status_code=400, content=create_openai_error_response(400, model_error, "invalid_request_error"))

        if express_key_manager_instance.get_total_keys() == 0:
            return JSONResponse(status_code=401, content=create_openai_error_response(401, "未配置 VERTEX_EXPRESS_API_KEY", "auth_error"))

        key_tuple = express_key_manager_instance.get_express_api_key()
        if not key_tuple:
            return JSONResponse(status_code=401, content=create_openai_error_response(401, "无可用 API Key", "auth_error"))
        _, express_api_key = key_tuple

        # ==========================================
        # 🌟 针对 Omni 模型触发新版 Interactions API
        # ==========================================
        is_omni = "omni" in base_model_name.lower()
        custom_headers = {"Api-Revision": "2026-05-20"} if is_omni else None

        client_to_use = genai.Client(
            vertexai=True,
            api_key=express_api_key,
            http_options=get_http_options(headers=custom_headers),
        )

        if is_omni:
            print(f"🌐 [上游端点] 检测到 Omni 模型，已启用 Interactions API 专属视频通道。")
            # 👇 核心修复点：将 create_gemini_prompt 替换为全新的 create_interaction_prompt
            return await execute_interaction_call(client_to_use, base_model_name, create_interaction_prompt, request_obj)

        # ====== 旧版生成模型逻辑 (Gemini 3.5/2.5 等) ======
        print(f"🌐 [上游端点] 使用官方 Gemini SDK 调用模型 {base_model_name}。")
        is_image_model = "image" in request_obj.model.lower()
        gen_config_dict = create_generation_config(request_obj)
        thinking_config = _build_thinking_config(base_model_name, request_obj, is_image_model)
        if thinking_config:
            gen_config_dict["thinking_config"] = thinking_config

        if is_grounded_search and not is_image_model:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]
            
        return await execute_gemini_call(client_to_use, base_model_name, create_gemini_prompt, gen_config_dict, request_obj)