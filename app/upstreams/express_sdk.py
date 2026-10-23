from functools import partial
from typing import Any, Optional, Tuple

import google.genai
from fastapi import Request
from fastapi.responses import JSONResponse
from google import genai

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_unified_completion,
)
from message_processing import (
    create_gemini_prompt,
    apply_prefill_compat,
    apply_console_injection,
    DEFAULT_IMAGE_PREFILL_NUDGE,
)
from http_options import get_http_options
import model_capabilities as mc
from runtime_state import app_state
import config as app_config
from schema_validation import SchemaValidationError, validate_request_schemas


def _normalize_model_name(model_name: str) -> tuple[str, bool]:
    base_model_name = (model_name or "").strip()
    is_grounded_search = base_model_name.endswith("-search")
    if is_grounded_search:
        base_model_name = base_model_name[:-len("-search")]
    return base_model_name, is_grounded_search


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
              f"project={api.project} location={api.location} → {scope}。")
    except Exception as e:
        print(f"⚠️ [上游端点] 读取端点解析结果失败（不影响调用）：{e}")


def resolve_express_model_path(base_model_name: str, settings: dict,
                               key_project_id: str = "") -> str:
    if base_model_name.startswith(("projects/", "publishers/", "models/")):
        return base_model_name

    location = str(settings.get("express_location", "") or "").strip()
    if not location:
        return base_model_name

    # 严格只使用当前 Key 自身绑定的 Project ID，不借用 Cookie 通道的全局 Project ID（防跨号 403）
    project = str(key_project_id or "").strip()
    if not project:
        return base_model_name

    return f"projects/{project}/locations/{location}/publishers/google/models/{base_model_name}"


class ExpressSDKUpstream(BaseUpstream):
    """官方 API Key Express Mode 渠道处理器（支持多 Key 轮询与跨通道 Cookie 互备）。"""

    def prepare_express_context(self, orig_request_obj: OpenAIRequest) -> Tuple[Optional[dict], Optional[JSONResponse]]:
        """在独立副本上预处理消息、预填充与生成参数（不污染原请求对象，供多 Key 复用）。"""
        try:
            validate_request_schemas(orig_request_obj)
        except SchemaValidationError as exc:
            return None, JSONResponse(
                status_code=400,
                content=create_openai_error_response(400, str(exc), "invalid_request_error"),
            )

        base_model_name, is_grounded_search = _normalize_model_name(orig_request_obj.model)
        request_obj = orig_request_obj.model_copy(deep=True)
        _inj_settings = app_state.get_effective_settings(base_model_name)
        profile = mc.get_profile(base_model_name)
        is_image_model = profile["is_image"]

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
            if prefill_active and _inj_settings.get("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        gen_config_dict = create_generation_config(request_obj)
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
                             "response_modalities", "image_config")}
            print(f"🔎 [出站调试] Express 通道 生成参数={_dbg}")

        return {
            "request_obj": request_obj,
            "base_model_name": base_model_name,
            "inj_settings": _inj_settings,
            "gen_config_dict": gen_config_dict,
            "prompt_func": prompt_func,
            "prefill_text": prefill_text,
            "is_image_model": is_image_model,
        }, None

    def build_key_call_spec(self, ctx: dict, key_record: dict) -> dict:
        """为当前选中的 Express Key 构建专属 Client 与绑定了其 Project ID 的模型路径。"""
        express_api_key = key_record["key"]
        base_model_name = ctx["base_model_name"]
        model_to_call = resolve_express_model_path(
            base_model_name, ctx["inj_settings"], key_project_id=key_record.get("project_id", "")
        )
        client_to_use = genai.Client(
            vertexai=True,
            api_key=express_api_key,
            http_options=get_http_options(),
        )
        _log_resolved_endpoint(client_to_use)
        if model_to_call != base_model_name:
            print(f"🌐 [上游端点] Key #{key_record.get('index', 0) + 1} 已钉定 location：{model_to_call}")

        return {
            "client": client_to_use,
            "model_to_call": model_to_call,
            "fallback_model": (base_model_name if model_to_call != base_model_name else None),
            "fallback_client_factory": (
                lambda: genai.Client(
                    vertexai=True,
                    api_key=express_api_key,
                    http_options=get_http_options(),
                )
            ),
        }

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        return await execute_unified_completion(
            request_obj=request_obj,
            fastapi_request=fastapi_request,
            primary_channel="express",
            express_upstream=self,
        )
