import re
from functools import partial

from typing import Any

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
from message_processing import (
    create_gemini_prompt,
    apply_prefill_compat,
    apply_console_injection,
    apply_unicode_input,
    prepare_anti_truncation_request,
    DEFAULT_IMAGE_PREFILL_NUDGE,
)
from http_options import get_http_options
import model_capabilities as mc
from runtime_state import app_state
import config as app_config
from schema_validation import SchemaValidationError, validate_request_schemas

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


def _prefill_tpl(user_template: str, is_image_model: bool) -> str:
    tpl = (user_template or "").strip()
    if tpl:
        return tpl
    return DEFAULT_IMAGE_PREFILL_NUDGE if is_image_model else ""


def _build_thinking_config(base_model_name: str, request: OpenAIRequest, is_image_model: bool,
                           prefill_active: bool = False) -> dict | None:
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

    if app_state.get_setting("debug_outbound", False):
        print(f"🔎 [出站调试] Express 通道 模型={base_model_name} thinkingConfig={thinking_config}")

    return thinking_config


def _prefill_log(mode: str, prefill_text: str) -> str:
    n = len(prefill_text)
    if mode == "keep_turn":
        return (f"🩹 [预填充兼容] 预填充（{n} 字）保留为 model 轮次，其后补一句续写推动语；"
                "输出会把它拼回开头。")
    if mode == "minimal":
        return ("🩹 [预填充兼容] 仅补占位 user 保证不报错；预填充**不会**拼回输出开头"
                "（预设的思考开标签可能因此缺失，酒馆正则会抓不到）。")
    return (f"🩹 [预填充兼容] 预填充（{n} 字）已并入末尾 user 消息作为续写指令，并将拼回输出开头。"
            "若预填充停在半截词/半截标签且模型接不上，可试试「保留模型轮次」。")


_ENDPOINT_LOGGED = False


def _log_resolved_endpoint(client: Any) -> None:
    global _ENDPOINT_LOGGED
    if _ENDPOINT_LOGGED:
        return
    _ENDPOINT_LOGGED = True
    try:
        api = client._api_client
        base_url = getattr(api._http_options, "base_url", None)
        api_version = getattr(api._http_options, "api_version", None)
        scope = "全局端点（URL 无 location 段）" if base_url == "https://aiplatform.googleapis.com/" \
            else "自定义/区域端点"
        print(f"🌐 [上游端点] 标准模式解析结果：base_url={base_url} api_version={api_version} "
              f"project={api.project} location={api.location} → {scope}。"
              "Express Key 模式下具体区域由 Google 后端路由，无法在客户端指定"
              "（如需强制指向某端点，可设环境变量 VERTEX_BASE_URL）。")
    except Exception as e:
        print(f"⚠️ [上游端点] 读取端点解析结果失败（不影响调用）：{e}")


def resolve_express_model_path(base_model_name: str, settings: dict,
                               key_project_id: str = "") -> str:
    if base_model_name.startswith(("projects/", "publishers/", "models/")):
        return base_model_name

    location = str(settings.get("express_location", "") or "").strip()
    if not location:
        return base_model_name

    project = (str(key_project_id or "").strip()
               or app_config.GOOGLE_PROJECT_ID
               or app_state.get_project_id() or "").strip()
    if not project:
        print("⚠️ [上游端点] 已选择钉定 location，但该 Key 没有可用的 Project ID"
              "（请在控制台「通道与凭证」页触发探测或手动填写该 Key 的 Project ID，"
              "也可设环境变量 GOOGLE_PROJECT_ID 兜底），"
              "本次退回默认路由。")
        return base_model_name

    return f"projects/{project}/locations/{location}/publishers/google/models/{base_model_name}"


class ExpressSDKUpstream(BaseUpstream):
    """
    官方 API Key Express Mode 渠道处理器
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        try:
            validate_request_schemas(request_obj)
        except SchemaValidationError as exc:
            return JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, str(exc), "invalid_request_error"),
            )

        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        base_model_name, is_grounded_search, model_error = _normalize_model_name(request_obj.model)
        if model_error:
            print(f"❌ [模型名称] {model_error} 收到的模型名：{request_obj.model}")
            return JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, model_error, "invalid_request_error"),
            )

        if express_key_manager_instance.get_total_keys() == 0:
            error_msg = ("未配置任何 Express API Key，无法调用 Gemini Express Mode"
                         "（请在控制台「通道与凭证」页添加，或设环境变量 VERTEX_EXPRESS_API_KEY）。")
            print(f"❌ [密钥配置] {error_msg}")
            return JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"),
            )

        key_record = express_key_manager_instance.get_express_record()
        if not key_record:
            error_msg = "没有可用的 Express API Key。"
            print(f"❌ [密钥配置] {error_msg}")
            return JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"),
            )

        express_api_key = key_record["key"]
        _inj_settings = app_state.get_effective_settings(base_model_name)
        model_to_call = resolve_express_model_path(base_model_name, _inj_settings,
                                                   key_project_id=key_record.get("project_id", ""))
        client_to_use = genai.Client(
            vertexai=True,
            api_key=express_api_key,
            http_options=get_http_options(),
        )
        _log_resolved_endpoint(client_to_use)
        print(f"🌐 [上游端点] 使用官方 Gemini Express Mode SDK 调用模型 {base_model_name}。")
        if model_to_call != base_model_name:
            print(f"🌐 [上游端点] 已钉定 location：{model_to_call}")

        profile = mc.get_profile(base_model_name)
        is_image_model = profile["is_image"]

        if bool(_inj_settings.get("unicode_input", False)) and not is_image_model:
            _uc_msgs, _uc_cnt = apply_unicode_input(request_obj.messages, enabled=True)
            if _uc_cnt > 0:
                request_obj = request_obj.model_copy(update={"messages": _uc_msgs})
                print(f"🔤 [Unicode 转码] 已对最后一条 user 消息完成紧凑转码（共 {_uc_cnt} 个字符）。")

        prefill_text = ""
        prefill_active = False
        _injected, _inj_notes = apply_console_injection(
            request_obj.messages,
            system_text=_inj_settings.get("inject_system_instruction", ""),
            prefill_text=_inj_settings.get("inject_prefill", ""),
            has_tools=bool(getattr(request_obj, "tools", None)),
            is_image_model=is_image_model,
            allow_image_prefill=bool(_inj_settings.get("inject_prefill_for_image", False)),
        )
        for _n in _inj_notes:
            print(_n)
        if _injected is not request_obj.messages:
            request_obj = request_obj.model_copy(update={"messages": _injected})

        _prefill_mode = _inj_settings.get("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
        if _prefill_mode != "off":
            new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not profile["requires_user_last_turn"],
                instruction_template=_prefill_tpl(_inj_settings.get("prefill_instruction", ""), is_image_model),
                cot_guard=bool(_inj_settings.get("prefill_cot_guard", True)) and not is_image_model,
            )
            if new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": new_msgs})
                print(_prefill_log(_prefill_mode, prefill_text))
            elif prefill_text:
                print(f"🩹 [预填充兼容] 该模型支持 model 结尾，预填充原生透传（{len(prefill_text)} 字），模型将直接续写。")
            else:
                print("ℹ️ [预填充兼容] 未检测到预填充（请求最后一条不是 assistant 消息）。"
                      "预设里的思维链指令属于 system/user 条目，不是预填充；"
                      "若要用预设思维链顶掉原生思考，需在预设末尾放一条 assistant 条目（通常是思考块的开标签）。")
            if prefill_active and _inj_settings.get("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        # ===== 合成工具抗截断包装（在预填充处理之后执行） =====
        _anti_mode = str(_inj_settings.get("anti_truncation_mode", "off") or "off").lower()
        anti_tool_name = None
        anti_stream_args = False
        if _anti_mode in ("streaming", "buffered") and not is_image_model and not is_grounded_search:
            request_obj, anti_tool_name, anti_stream_args, anti_note = prepare_anti_truncation_request(
                request_obj,
                mode=_anti_mode,
            )
            if anti_note:
                print(anti_note)

        gen_config_dict = create_generation_config(
            request_obj,
            stream_function_call_arguments=anti_stream_args,
        )
        thinking_config = _build_thinking_config(base_model_name, request_obj, is_image_model,
                                                 prefill_active=prefill_active)
        if thinking_config:
            gen_config_dict["thinking_config"] = thinking_config

        _tools_disabled = isinstance(request_obj.tool_choice, str) and request_obj.tool_choice.lower() == "none"
        if is_grounded_search and not _tools_disabled:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                if search_tool not in gen_config_dict["tools"]:
                    gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]
            print(f"🔎 [搜索增强] 已为模型 {base_model_name} 启用 Google Search 工具。")

        prompt_func = partial(create_gemini_prompt, model_name=base_model_name)

        if app_state.get_setting("debug_outbound", False):
            _dbg = {k: v for k, v in gen_config_dict.items()
                    if k in ("temperature", "top_p", "top_k", "candidate_count",
                             "max_output_tokens", "stop_sequences", "thinking_config",
                             "response_modalities", "image_config", "tool_config")}
            print(f"🔎 [出站调试] Express 通道 生成参数={_dbg}")

        return await execute_gemini_call(
            client_to_use, model_to_call, prompt_func, gen_config_dict, request_obj,
            fastapi_request=fastapi_request, prefill_text=prefill_text,
            anti_trunc_tool_name=anti_tool_name,
            fallback_model=(base_model_name if model_to_call != base_model_name else None),
            fallback_client_factory=(
                lambda: genai.Client(
                    vertexai=True,
                    api_key=express_api_key,
                    http_options=get_http_options(),
                )
            ),
        )