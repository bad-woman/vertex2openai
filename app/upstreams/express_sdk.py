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
from message_processing import (create_gemini_prompt, apply_prefill_compat,
                                apply_console_injection, DEFAULT_IMAGE_PREFILL_NUDGE)
from http_options import get_http_options, should_use_priority_paygo
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
    """选用续写指令：生图模型在用户未自定义时改用要图片的那句。

    通用模板说的是"从断点处无缝往下写"，生图模型会老老实实继续写**文本**，
    结果吐出一段字符画而不是图片（实测复现）。用户填了自定义模板则以用户的为准。
    """
    tpl = (user_template or "").strip()
    if tpl:
        return tpl
    return DEFAULT_IMAGE_PREFILL_NUDGE if is_image_model else ""


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


_ENDPOINT_LOGGED = False


def _log_resolved_endpoint(client: Any) -> None:
    """把 SDK 实际解析出的上游端点打进日志（每进程一次）。

    为什么需要它：Express（api_key）模式下**无法**指定 location——SDK 里
    project/location 与 api_key 互斥，硬传会 `ValueError: Project/location and
    API key are mutually exclusive`。SDK 解析出的 base_url 就是**全局端点**
    https://aiplatform.googleapis.com/ ，project/location 均为 None，请求 URL 里
    根本没有 location 段。因此报错信息里出现的 `locations/xxx` 区域是 **Google 后端
    为该 Key 自行路由**的结果，不是本代理选的，客户端侧无从指定。
    打印出来便于随时核实（尤其怀疑"被路由到冷门区域"时）。
    """
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
    """把模型名解析成实际下发给 SDK 的 model 值。

    留空 express_location → 返回裸模型名，走 express 端点格式：
        https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent
      此时 location 由 Google 后端自行路由，**可能落到该模型不提供服务的区域并 404**
      （实测：gemini-2.5-pro 被路由到 asia-southeast1 → 404 not found，
       同一 Key 换成下面的完整路径即 200）。

    填了 express_location → 返回完整资源路径，让区域由我们钉定：
        projects/{project}/locations/{location}/publishers/google/models/{model}
      google-genai 的 t_model() 对以 "projects/" 开头的 model 原样透传，
      因此不需要（也不能）给 Client 传 location —— api_key 与 project/location 互斥。

    项目 ID 的取值顺序（多账号支持）：
      ① 本次选中的那个 Express Key **自己的** Project ID（启动时自动探测或人工填写）；
      ② 环境变量 GOOGLE_PROJECT_ID / 「通道与凭证」里的全局 Project ID（旧行为兜底）；
      ③ 都没有 → 退回裸模型名（并提示），绝不拼出半截路径。
    ①在多账号轮换时是必须的：Key 与项目一一对应，用别人的项目会 403（需要计费/无权）。
    """
    if base_model_name.startswith(("projects/", "publishers/", "models/")):
        return base_model_name          # 客户端已自带完整路径，尊重它

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
    封装了原有的多密钥切匙、代理挂载以及 SDK 运行时调用
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
        # 多账号：location 钉定用**这个 Key 自己的**项目，而不是 Cookie 通道的全局项目
        model_to_call = resolve_express_model_path(base_model_name, _inj_settings,
                                                   key_project_id=key_record.get("project_id", ""))
        priority_paygo = should_use_priority_paygo(model_to_call)

        client_to_use = genai.Client(
            vertexai=True,
            api_key=express_api_key,
            http_options=get_http_options(priority_paygo=priority_paygo),
        )
        _log_resolved_endpoint(client_to_use)
        print(f"🌐 [上游端点] 使用官方 Gemini Express Mode SDK 调用模型 {base_model_name}。")
        if model_to_call != base_model_name:
            print(f"🌐 [上游端点] 已钉定 location：{model_to_call}")
        if priority_paygo:
            print("🚦 [流量等级] global 请求已附加 Priority PayGo 请求头；实际是否命中以上游 traffic_type 为准。")
        else:
            print("ℹ️ [流量等级] 当前未钉定 global，使用普通请求。")

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

        _prefill_mode = app_state.get_setting("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
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

        _tools_disabled = isinstance(request_obj.tool_choice, str) and request_obj.tool_choice.lower() == "none"
        if is_grounded_search and not _tools_disabled:
            search_tool = {"google_search": {}}
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                if search_tool not in gen_config_dict["tools"]:
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
            client_to_use, model_to_call, prompt_func, gen_config_dict, request_obj,
            fastapi_request=fastapi_request, prefill_text=prefill_text,
            # 钉定路径万一不对（Project ID 与 Key 不同项目、该区域没有此模型），
            # 自动退回裸模型名重试一次，并改用不带 Priority 请求头的普通客户端。
            fallback_model=(base_model_name if model_to_call != base_model_name else None),
            fallback_client_factory=(
                lambda: genai.Client(
                    vertexai=True,
                    api_key=express_api_key,
                    http_options=get_http_options(priority_paygo=False),
                )
            ) if priority_paygo else None,
        )
