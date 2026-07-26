import re
from functools import partial

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
)
from message_processing import (create_gemini_prompt, apply_prefill_compat,
                                apply_console_injection)
from http_options import get_http_options
import model_capabilities as mc
from runtime_state import app_state
import config as app_config

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


def _build_thinking_config(base_model_name: str, request: OpenAIRequest, is_image_model: bool,
                           prefill_active: bool = False) -> dict | None:
    """按模型能力档案 + 控制台设置 + 单次请求构建思考配置（SDK 线格式）。"""
    if is_image_model:
        return None

    settings = app_state.get_effective_settings(base_model_name)
    t = mc.resolve_thinking(base_model_name, request, settings, prefill_active=prefill_active)
    if t.get("mode") is None:
        return None

    thinking_config = {"include_thoughts": t.get("include_thoughts", True)}

    if t["mode"] == "level":
        genai_version_str = getattr(google.genai, "__version__", "1.0.0")
        try:
            parts = genai_version_str.split(".")
            sdk_supports_level = (int(parts[0]) >= 2) or (int(parts[0]) == 1 and int(parts[1]) >= 51)
        except Exception:
            sdk_supports_level = False

        if sdk_supports_level:
            thinking_config["thinking_level"] = t["level"]
        else:
            print(f"⚠️ [推理配置] 当前 google-genai 版本 {genai_version_str} 不支持 thinking_level，已自动跳过该参数。")
    else:  # budget（Gemini 2.5）
        thinking_config["thinking_budget"] = t["budget"]

    if app_state.get_setting("debug_outbound", False):
        print(f"🔎 [出站调试] Express 通道 模型={base_model_name} thinkingConfig={thinking_config}")

    return thinking_config


def _prefill_log(mode: str, prefill_text: str) -> str:
    """按模式说明预填充被怎么处理了——三种模式的差别对使用者影响很大。"""
    n = len(prefill_text)
    if mode == "keep_turn":
        return (f"🩹 [预填充兼容] 预填充（{n} 字）保留为 model 轮次，其后补一句续写推动语；"
                "输出会把它拼回开头。")
    if mode == "minimal":
        return ("🩹 [预填充兼容] 仅补占位 user 保证不报错；预填充**不会**拼回输出开头"
                "（预设的思考开标签可能因此缺失，酒馆正则会抓不到）。")
    return (f"🩹 [预填充兼容] 预填充（{n} 字）已并入末尾 user 消息作为续写指令，并将拼回输出开头。"
            "若预填充停在半截词/半截标签且模型接不上，可试试「保留模型轮次」。")


class ExpressSDKUpstream(BaseUpstream):
    """
    官方 API Key Express Mode 渠道处理器
    封装了原有的多密钥切匙、代理挂载以及 SDK 运行时调用
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        base_model_name, is_grounded_search, model_error = _normalize_model_name(request_obj.model)
        if model_error:
            print(f"❌ [模型名称] {model_error} 收到的模型名：{request_obj.model}")
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
        print(f"🌐 [上游端点] 使用官方 Gemini Express Mode SDK 调用模型 {base_model_name}。")

        profile = mc.get_profile(base_model_name)
        is_image_model = profile["is_image"]

        # 预填充智能兼容：按控制台模式 + 模型能力处理末尾 assistant 预填充（新模型自动生效）
        # - 2.5 及更早（允许 model 结尾）：原生透传，模型直接续写；
        # - 3.x（拒绝 model 结尾）：转成续写指令；
        # - 两者都会把预填充文本拼回输出开头（带去重）。
        prefill_text = ""
        prefill_active = False
        # 控制台注入（轻量前端用；两个字段都留空时是空操作）。
        # 必须在 apply_prefill_compat 之前，注入后的消息与前端自发预填充同形。
        _inj_settings = app_state.get_effective_settings(base_model_name)
        _injected, _inj_notes = apply_console_injection(
            request_obj.messages,
            system_text=_inj_settings.get("inject_system_instruction", ""),
            prefill_text=_inj_settings.get("inject_prefill", ""),
            has_tools=bool(getattr(request_obj, "tools", None)),
            is_image_model=is_image_model,
        )
        for _n in _inj_notes:
            print(_n)
        if _injected is not request_obj.messages:
            request_obj = request_obj.model_copy(update={"messages": _injected})

        _prefill_mode = app_state.get_setting("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
        if _prefill_mode != "off":
            new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not profile["requires_user_last_turn"],
                instruction_template=app_state.get_setting("prefill_instruction", ""),
            )
            if new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": new_msgs})
                print(_prefill_log(_prefill_mode, prefill_text))
            elif prefill_text:
                print(f"🩹 [预填充兼容] 该模型支持 model 结尾，预填充原生透传（{len(prefill_text)} 字），模型将直接续写。")
            else:
                # 没检测到预填充也要说一声：很多人以为整个预设就是预填充，
                # 实际只有「最后一条 assistant 消息」才算。没有它，压制原生思考也不会触发。
                print("ℹ️ [预填充兼容] 未检测到预填充（请求最后一条不是 assistant 消息）。"
                      "预设里的思维链指令属于 system/user 条目，不是预填充；"
                      "若要用预设思维链顶掉原生思考，需在预设末尾放一条 assistant 条目（通常是思考块的开标签）。")
            if prefill_active and app_state.get_setting("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        gen_config_dict = create_generation_config(request_obj)
        thinking_config = _build_thinking_config(base_model_name, request_obj, is_image_model,
                                                 prefill_active=prefill_active)
        if thinking_config:
            gen_config_dict["thinking_config"] = thinking_config

        if is_grounded_search and not is_image_model:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]
            print(f"🔎 [搜索增强] 已为模型 {base_model_name} 启用 Google Search 工具。")

        # 传入真实模型名：create_gemini_prompt 需要它来判断是否对缺失的思考签名
        # 启用官方哨兵（仅 Gemini 3.x 强校验，见 message_processing.resolve_tool_call_signature）
        prompt_func = partial(create_gemini_prompt, model_name=base_model_name)

        if app_state.get_setting("debug_outbound", False):
            _dbg = {k: v for k, v in gen_config_dict.items()
                    if k in ("temperature", "top_p", "top_k", "candidate_count",
                             "max_output_tokens", "stop_sequences", "thinking_config",
                             "response_modalities", "image_config")}
            print(f"🔎 [出站调试] Express 通道 生成参数={_dbg}")

        return await execute_gemini_call(
            client_to_use, base_model_name, prompt_func, gen_config_dict, request_obj,
            fastapi_request=fastapi_request, prefill_text=prefill_text,
        )