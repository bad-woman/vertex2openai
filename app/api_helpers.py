import html
import json
import time
import math
import random
import uuid
import asyncio
import httpx
import re
from typing import List, Dict, Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types

from models import OpenAIRequest
from message_processing import (
    convert_to_openai_format,
    _create_safety_ratings_html,
    strip_prefill_overlap,
    PrefillDeduper,
)
import config as app_config
import model_capabilities as mc
from runtime_state import app_state
from schema_validation import SchemaValidationError, validate_request_schemas
from logger import stats, call_recorder
from synthetic_wrapper import (
    _BUILTIN_SEARCH_NAMES,
    extract_synthetic_content,
    is_builtin_search_tool,
    is_malformed_tool_finish,
    is_stream_fc_args_unsupported_error,
    make_bad_woman_response_id,
    strip_stream_fc_args_from_config,
)


def _strip_json_schema_meta(node: Any) -> Any:
    """递归移除 JSON Schema 中的 $schema 元字段（保留 properties 字典内用户定义的同名属性键）。"""
    if isinstance(node, list):
        return [_strip_json_schema_meta(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k == "$schema":
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _strip_json_schema_meta(pv) for pk, pv in v.items()}
        else:
            out[k] = _strip_json_schema_meta(v)
    return out


def _sdk_supports_params_json_schema() -> bool:
    try:
        fields = getattr(types.FunctionDeclaration, "model_fields", None) or {}
        if "parameters_json_schema" in fields:
            return True
        ann = getattr(types.FunctionDeclaration, "__annotations__", None) or {}
        return "parameters_json_schema" in ann
    except Exception:
        return False


_USE_PARAMS_JSON_SCHEMA = _sdk_supports_params_json_schema()


def _sdk_supports_response_json_schema() -> bool:
    try:
        fields = getattr(types.GenerateContentConfig, "model_fields", None) or {}
        if "response_json_schema" in fields:
            return True
        ann = getattr(types.GenerateContentConfig, "__annotations__", None) or {}
        return "response_json_schema" in ann
    except Exception:
        return False


_USE_RESPONSE_JSON_SCHEMA = _sdk_supports_response_json_schema()


def _safety_score_enabled() -> bool:
    try:
        return bool(app_state.get_setting("safety_score", app_config.SAFETY_SCORE))
    except Exception:
        return bool(app_config.SAFETY_SCORE)


def get_retry_settings() -> tuple[int, float]:
    """读取重试配置。总请求次数 = retry_max + 1。"""
    try:
        retry_max = int(app_state.get_setting("retry_max", app_config.DEFAULT_SETTINGS["retry_max"]))
    except (TypeError, ValueError):
        retry_max = app_config.DEFAULT_SETTINGS["retry_max"]
    try:
        backoff = float(app_state.get_setting(
            "retry_backoff_seconds", app_config.DEFAULT_SETTINGS["retry_backoff_seconds"]))
    except (TypeError, ValueError):
        backoff = app_config.DEFAULT_SETTINGS["retry_backoff_seconds"]
    return max(0, min(50, retry_max)), max(0.0, min(120.0, backoff))


def _mask_key_short(key: str) -> str:
    k = str(key or "").strip()
    if len(k) <= 10:
        return "****"
    return f"{k[:6]}…{k[-4:]}"


def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    safe_message = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1***HIDDEN_API_KEY***", message)
    return {
        "error": {
            "message": safe_message,
            "type": error_type,
            "code": status_code,
            "param": None
        }
    }


def _diagnose_express_error(code: int, msg: str, status_str: str = "") -> str:
    """为 Express SDK 通道的常见报错生成清晰的中文诊断说明。"""
    low = f"{status_str} {msg}".lower()
    if any(k in low for k in (
        "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
        "finish_reason=safety", "image_safety", "recitation", "blocklist",
    )):
        return "请求或回复被 Google 安全策略拦截（不可重试），请检查提示词内容"
    if any(k in low for k in (
        "只返回了思考过程", "返回了空内容", "返回了空对象", "未返回任何有效内容",
        "解码格式异常", "malformed_response", "malformed_function_call",
    )):
        return "模型未输出有效正文（思考耗尽额度、返回空候选或解码格式异常，不可重试）"
    if any(k in low for k in (
        "remoteprotocolerror", "connecterror", "readerror", "writeerror",
        "proxyerror", "pooltimeout", "server disconnected", "connection reset",
    )):
        return "本地到 Google 上游的网络或代理连接中断"
    if "requires billing" in low or "billing to be enabled" in low:
        return "该项目未开启计费（Billing），请确认此 Key 关联的 Project ID 已启用计费，或更换/删除该 Key"
    if "api_key_invalid" in low or "api key not valid" in low or code == 401:
        return "Express API Key 无效或已失效（401），请在「通道与凭证」页检查或更换 Key"
    if code == 403 or "permission_denied" in low or ("permission" in low and "denied" in low):
        return "权限被拒绝（403）：该 API Key 无权访问目标项目/模型，或该 Key 所属项目受限"
    if code == 404 or "not found" in low:
        if "publisher model" in low or "locations/" in low:
            return "目标模型在当前区域（location）或项目中不存在（404），建议将「标准模式 location」设为 global 或默认"
        return "请求的模型名称不存在或尚未在 Express 通道开放（404）"
    if code == 429 or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "触发上游配额或并发频率限制（429 Resource Exhausted）"
    if code == 400 or "invalid_argument" in low:
        if "model turn" in low:
            return "当前模型不支持以 assistant 预填充结尾（400），请在控制台开启「预填充兼容模式」"
        if "thinking" in low:
            return "当前模型不支持所请求的思考档位或思考参数冲突（400）"
        return "请求参数不被上游模型接受（400 Invalid Argument）"
    if code in (500, 502, 503, 504) or "unavailable" in low or "overloaded" in low or "internal" in low:
        return f"Google 上游服务内部错误或暂时繁忙（HTTP {code}）"
    return f"上游调用异常（HTTP {code}）"


def extract_upstream_error(e: Exception) -> tuple[int, str]:
    """从上游异常里精准提取 (HTTP 状态码, 含中文诊断与完整 details 的清晰错误说明)。"""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    status_str = str(getattr(e, "status", "") or "").strip()

    clean_msg = getattr(e, "message", None)
    if not isinstance(clean_msg, str) or not clean_msg.strip():
        raw = str(e)
        m = (
            re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
            or re.search(r"'message'\s*:\s*\"((?:[^\"\\]|\\.)*)\"", raw)
            or re.search(r"'message'\s*:\s*'((?:[^'\\]|\\.)*)'", raw)
        )
        clean_msg = m.group(1) if m else raw

    details_obj = getattr(e, "details", None)
    extra_detail_str = ""
    if details_obj:
        try:
            if isinstance(details_obj, dict):
                err_inner = details_obj.get("error") if isinstance(details_obj.get("error"), dict) else details_obj
                det_list = err_inner.get("details")
                if det_list:
                    extra_detail_str = f" | details={json.dumps(det_list, ensure_ascii=False)[:1200]}"
                else:
                    d_str = json.dumps(details_obj, ensure_ascii=False)
                    if d_str and str(clean_msg) not in d_str:
                        extra_detail_str = f" | 详情: {d_str[:1200]}"
            elif isinstance(details_obj, list):
                extra_detail_str = f" | details={json.dumps(details_obj, ensure_ascii=False)[:1200]}"
            else:
                d_str = str(details_obj).strip()
                if d_str and str(clean_msg) not in d_str:
                    extra_detail_str = f" | 详情: {d_str[:1200]}"
        except Exception:
            pass

    clean_msg = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1***HIDDEN_API_KEY***", str(clean_msg).strip() + extra_detail_str)

    if not isinstance(code, int) or not (400 <= code <= 599):
        low = f"{type(e).__name__} {status_str} {clean_msg}".lower()
        if any(k in low for k in (
            "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
            "finish_reason=safety", "image_safety", "recitation", "blocklist",
        )):
            code = 400
        elif any(k in low for k in (
            "只返回了思考过程", "返回了空内容", "返回了空对象", "未返回任何有效内容",
            "解码格式异常", "malformed_response", "malformed_function_call", "unexpected_tool_call",
        )):
            code = 502
        elif "not found" in low or "404" in low:
            code = 404
        elif "permission" in low or "403" in low or "denied" in low:
            code = 403
        elif "unauthenticated" in low or "401" in low or "api_key_invalid" in low:
            code = 401
        elif "429" in low or "resource_exhausted" in low or "quota" in low or "too many requests" in low:
            code = 429
        elif "invalid" in low or "400" in low:
            code = 400
        elif "503" in low or "unavailable" in low or "overloaded" in low:
            code = 503
        elif any(k in low for k in (
            "remoteprotocolerror", "connecterror", "readerror", "writeerror",
            "proxyerror", "pooltimeout", "server disconnected", "connection reset", "502", "504",
        )):
            code = 502
        else:
            code = 500

    diag = _diagnose_express_error(code, clean_msg, status_str)
    status_tag = f" [{status_str}]" if status_str and status_str not in clean_msg else ""
    formatted = f"【{diag}】{status_tag} {clean_msg}"
    return code, formatted


def extract_retry_after_seconds(err: Any, msg: str = "") -> Optional[float]:
    """从上游 429 / 503 异常响应头或 RetryInfo 中提取 Google 明确返回的短建议秒数（仅在短于控制台设定秒数时作参考，不强制长阻塞）。"""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else getattr(err, "headers", None)
    if headers and hasattr(headers, "get"):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                val = float(str(ra).strip())
                if 0 < val <= 60:
                    return val
            except (TypeError, ValueError):
                pass

    raw_pool = f"{msg} {getattr(err, 'details', '')} {err}"
    for pat in (
        r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"',
        r"'retryDelay'\s*:\s*'(\d+(?:\.\d+)?)s'",
        r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b",
    ):
        m = re.search(pat, raw_pool, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0 < val <= 60:
                    return val
            except (TypeError, ValueError):
                pass
    return None


def _describe_empty_gemini_response(response_obj: Any, model_to_call: str) -> str:
    """当 Gemini 返回无有效正文的响应时，生成清晰的中文诊断详情。"""
    if response_obj is None:
        return f"模型 {model_to_call} 返回了空对象（None）。"

    pf = getattr(response_obj, "prompt_feedback", None)
    if pf and getattr(pf, "block_reason", None):
        br = getattr(pf.block_reason, "name", None) or str(pf.block_reason)
        br_msg = getattr(pf, "block_reason_message", "") or ""
        return f"请求被安全策略在提示词阶段拦截（block_reason={br} {br_msg}）。"

    candidates = getattr(response_obj, "candidates", None) or []
    if not candidates:
        return f"模型 {model_to_call} 未返回任何候选结果（candidates 为空，可能触发了静默过滤）。"

    cand = candidates[0]
    fr = getattr(cand, "finish_reason", None)
    fr_str = getattr(fr, "name", None) or str(fr or "未知")
    finish_msg = getattr(cand, "finish_message", None) or ""

    if any(r in fr_str.upper() for r in ("MALFORMED_RESPONSE", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL")):
        return f"模型 {model_to_call} 上游解码格式异常（finish_reason={fr_str} {finish_msg}）。"

    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    thought_chars = 0
    for p in parts:
        if getattr(p, "thought", None) is True and getattr(p, "text", None):
            thought_chars += len(str(p.text))

    if thought_chars > 0:
        return (
            f"模型 {model_to_call} 只返回了思考过程（共 {thought_chars} 字）、未输出任何正文"
            f"（finish_reason={fr_str} {finish_msg}）。"
            "这通常是因为原生思考在 HIGH 档位耗尽了输出额度或被截断，"
            "建议在控制台将「原生思考控制」设为「关闭原生思考」或调低思考档位。"
        )

    return (
        f"模型 {model_to_call} 返回了空内容（候选数={len(candidates)}，Parts数={len(parts)}，"
        f"finish_reason={fr_str} {finish_msg}）。"
    )


def is_retryable_exception(e: Exception) -> bool:
    """判断异常是否应按用户设置的最大次数进行退避重试。
    第 ① 类（本地 ValueError：提示词拦截 / 安全过滤 / 空候选 / 只思考无正文）绝不重试；
    第 ②、③、④ 类（网络/代理闪断 + Google 500/502/503/504/429）全部允许重试。
    """
    if getattr(e, "is_retryable", None) is True:
        return True
    if getattr(e, "is_retryable", None) is False and getattr(e, "is_auth_or_billing", None) is not None:
        return False

    if isinstance(e, ValueError):
        return False

    error_str = f"{type(e).__name__} {getattr(e, 'status', '')} {e}".lower()
    if any(k in error_str for k in (
        "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
        "finish_reason=safety", "image_safety", "blocklist",
        "只返回了思考过程", "返回了空内容", "返回了空对象",
    )):
        return False

    if isinstance(e, (httpx.TransportError, httpx.TimeoutException, TimeoutError, ConnectionError)):
        return True

    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503, 504):
        return True
    err_code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if err_code in (429, 500, 502, 503, 504):
        return True

    retry_keywords = (
        "429", "500", "502", "503", "504",
        "too many requests", "quota", "resource exhausted", "resource_exhausted", "rate limit",
        "internal", "backend error", "unavailable", "overloaded", "temporarily unavailable",
        "deadline_exceeded", "remoteprotocolerror", "connecterror", "readerror", "writeerror",
        "proxyerror", "pooltimeout", "server disconnected", "connection reset", "timed out", "timeout",
        "incomplete chunked read", "peer closed connection",
    )
    return any(k in error_str for k in retry_keywords)


def is_express_auth_or_billing_error(e: Exception, err_code: int) -> bool:
    """判断是否为单个 Express Key 的鉴权失效或所属项目欠费/无权限错误（用于在多凭证池中隔离该坏 Key 并切换到下一个）。"""
    if getattr(e, "is_auth_or_billing", None) is True:
        return True
    if isinstance(e, ValueError):
        return False
    low = f"{getattr(e, 'status', '')} {e}".lower()
    if err_code in (401, 403):
        return True
    return any(k in low for k in (
        "api_key_invalid", "api key not valid", "unauthenticated",
        "permission_denied", "requires billing", "billing to be enabled",
        "account_suspended", "consumer_invalid",
    ))


def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    system_texts = []
    for msg in request.messages:
        if msg.role in ("system", "developer") and msg.content:
            if isinstance(msg.content, str):
                system_texts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_texts.append(part.get("text", ""))
                    elif hasattr(part, "text") and isinstance(part.text, str):
                        system_texts.append(part.text)

    if system_texts:
        _is_image_name = mc.get_profile(request.model)["is_image"]
        _allow_sys = (not _is_image_name) or bool(
            app_state.get_setting("image_system_instruction", False))
        if _allow_sys:
            config["system_instruction"] = "\n".join(system_texts)

    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    elif getattr(request, "max_completion_tokens", None) is not None:
        config["max_output_tokens"] = request.max_completion_tokens

    if request.top_p is not None:
        config["top_p"] = request.top_p
    if request.top_k is not None:
        config["top_k"] = request.top_k
    if request.stop is not None:
        config["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else list(request.stop)
    if request.seed is not None:
        config["seed"] = request.seed
    if request.n is not None:
        config["candidate_count"] = request.n

    for _pk in ("presence_penalty", "frequency_penalty"):
        _pv = getattr(request, _pk, None)
        if _pv:
            if app_state.get_setting("debug_outbound", False):
                print(f"🔎 [出站调试] 已剥离 {_pk}={_pv}：新版 Gemini 不接受非零惩罚参数，该值不会发给 Google。")

    _logprobs = getattr(request, "logprobs", None)
    _top_logprobs = getattr(request, "top_logprobs", None)
    if getattr(request, "response_logprobs", None) is not None:
        config["response_logprobs"] = request.response_logprobs
    if isinstance(_logprobs, bool):
        if _logprobs:
            config["response_logprobs"] = True
            config["logprobs"] = _top_logprobs if isinstance(_top_logprobs, int) else 1
    elif isinstance(_logprobs, int):
        config["logprobs"] = _logprobs
    elif isinstance(_top_logprobs, int):
        config["logprobs"] = _top_logprobs

    if getattr(request, "response_format", None) is not None:
        fmt = request.response_format
        fmt_type = fmt.get("type", "") if isinstance(fmt, dict) else getattr(fmt, "type", "")
        if fmt_type == "json_object":
            config["response_mime_type"] = "application/json"
        elif fmt_type == "json_schema":
            config["response_mime_type"] = "application/json"
            json_schema_obj = fmt.get("json_schema") if isinstance(fmt, dict) else getattr(fmt, "json_schema", None)
            schema = None
            if isinstance(json_schema_obj, dict):
                schema = json_schema_obj.get("schema")
            if isinstance(schema, dict):
                schema = _strip_json_schema_meta(schema)
                if _USE_RESPONSE_JSON_SCHEMA:
                    config["response_json_schema"] = schema
                else:
                    config["response_schema"] = schema

    safety_threshold = "BLOCK_NONE"

    config["safety_settings"] = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]

    settings = app_state.get_effective_settings(request.model)
    profile = mc.apply_sampling_policy(mc.get_profile(request.model), settings)
    is_image_model = profile["is_image"]

    function_declarations = []
    declared_search = False
    for tool in (request.tools or []):
        if not isinstance(tool, dict):
            continue
        t_type = str(tool.get("type") or "").strip().lower().replace("-", "_")
        func_data = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if func_data is not None:
            fn_name = str(func_data.get("name") or "").strip()
        elif "function" not in tool and tool.get("name"):
            func_data = tool
            fn_name = str(tool.get("name") or "").strip()
        else:
            fn_name = ""

        if is_builtin_search_tool(tool):
            declared_search = True
            continue
        if func_data is not None and fn_name and (t_type in ("", "function") or "function" in tool or "parameters" in tool):
            declaration = {
                "name": fn_name,
                "description": func_data.get("description"),
            }
            parameters = func_data.get("parameters")
            if isinstance(parameters, dict):
                parameters = _strip_json_schema_meta(parameters)
                if not parameters:
                    parameters = {"type": "object", "properties": {}}
            if parameters is not None:
                if _USE_PARAMS_JSON_SCHEMA:
                    declaration["parameters_json_schema"] = parameters
                else:
                    # 旧版 OpenAPI Schema 对无参 OBJECT 要求省略 parameters，避免空 properties 触发 400
                    if isinstance(parameters, dict) and not parameters.get("properties") and str(parameters.get("type", "")).lower() == "object":
                        pass
                    else:
                        declaration["parameters"] = parameters
            declaration = {k: v for k, v in declaration.items() if v is not None}
            function_declarations.append(declaration)

    choice = request.tool_choice
    _tools_disabled = isinstance(choice, str) and choice.lower() == "none"
    _choice_name = ""
    if isinstance(choice, dict):
        _choice_fn = choice.get("function") if isinstance(choice.get("function"), dict) else {}
        _choice_name = str(_choice_fn.get("name") or choice.get("name") or "").strip()
    _declared_custom_names = {d["name"] for d in function_declarations}
    _forced_search = bool(
        _choice_name
        and _choice_name not in _declared_custom_names
        and _choice_name.lower().replace("-", "_") in _BUILTIN_SEARCH_NAMES
    )

    tools_list = []
    active_function_names: list[str] = []
    if is_image_model:
        config["response_modalities"] = ["TEXT", "IMAGE"]

        if "nano-banana" in (request.model or "").lower():
            config["safety_settings"].extend([
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold="OFF"),
            ])
            config["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH

        target_ar = mc.resolve_aspect_ratio(request.model, request, settings)
        image_size = mc.resolve_image_size(request.model, request, settings)
        image_config_args = {"image_size": image_size}
        if target_ar:
            image_config_args["aspect_ratio"] = target_ar

        config["image_config"] = types.ImageConfig(**image_config_args)
        if (declared_search or _forced_search) and not _tools_disabled and profile.get("supports_search"):
            tools_list = [{"google_search": {}}]

        for key in ["response_mime_type", "response_schema", "response_json_schema", "response_logprobs", "logprobs"]:
            config.pop(key, None)
    else:
        if config.get("temperature") is None and settings.get("default_temperature") is not None:
            config["temperature"] = settings["default_temperature"]
        if config.get("top_p") is None and settings.get("default_top_p") is not None:
            config["top_p"] = settings["default_top_p"]
        if config.get("max_output_tokens") is None and settings.get("default_max_tokens") is not None:
            config["max_output_tokens"] = settings["default_max_tokens"]

        if not _tools_disabled and _forced_search and profile.get("supports_search"):
            tools_list.append({"google_search": {}})
        elif function_declarations:
            tools_list.append({"function_declarations": function_declarations})
            active_function_names = [d["name"] for d in function_declarations]
        elif not _tools_disabled and declared_search and profile.get("supports_search"):
            tools_list.append({"google_search": {}})

    mc.sanitize_sampling(config, profile)

    if config.get("max_output_tokens") is not None and not is_image_model:
        _fake_on = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))
        _is_real_stream = bool(getattr(request, "stream", False) and not _fake_on)
        _clamped_max = mc.clamp_max_output_tokens(request.model or "", config["max_output_tokens"], is_real_stream=_is_real_stream)
        if _clamped_max is not None:
            config["max_output_tokens"] = _clamped_max
        else:
            config.pop("max_output_tokens", None)

    if tools_list:
        config["tools"] = tools_list

    tool_config = None
    # 仅在实际下发了 function_declarations 时才允许附带 function_calling_config，避免空 tools / 纯内置搜索触发 400
    if not is_image_model and active_function_names:
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            c_low = choice.lower()
            if c_low == "none":
                mode = "NONE"
            elif c_low == "auto":
                mode = "AUTO"
            elif c_low == "required":
                mode = "ANY"
        elif isinstance(choice, dict):
            if _choice_name and _choice_name in active_function_names:
                mode = "ANY"
                allowed_functions = [_choice_name]
            elif _choice_name and not _forced_search:
                mode = "AUTO"
        if mode:
            config_dict = {"mode": mode}
            if allowed_functions:
                config_dict["allowed_function_names"] = allowed_functions
            tool_config = {"function_calling_config": config_dict}

    if tool_config:
        config["tool_config"] = tool_config

    return config


def _summ_obj(obj: Any, width: int = 2000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = re.sub(r'"(data|thoughtSignature|thought_signature)":\s*"[A-Za-z0-9+/=_-]{40,}"', r'"\1": "<长Base64已省略>"', s)
    return s[:width] + ("…" if len(s) > width else "")


class _RawSampler:
    def __init__(self, keep: int = 2):
        self.keep = keep
        self.head: list = []
        self.tail: list = []
        self.count = 0

    def add(self, obj: Any):
        if not obj:
            return
        self.count += 1
        if len(self.head) < self.keep:
            self.head.append(_summ_obj(obj))
        else:
            self.tail.append(_summ_obj(obj))
            if len(self.tail) > self.keep:
                self.tail.pop(0)

    def dump(self) -> str:
        if self.count == 0:
            return "（上游未返回任何可解析的 JSON/候选对象）"
        lines = [f"共 {self.count} 个对象；首尾样本："]
        lines += [f"  ▸ {s}" for s in self.head]
        if self.tail:
            lines.append("  …")
            lines += [f"  ▸ {s}" for s in self.tail]
        return "\n".join(lines)


def extract_gemini_chunk_debug_dict(chunk: Any) -> dict:
    """提取 Google GenAI SDK 返回对象（GenerateContentResponse/Chunk）的关键诊断字典。"""
    if chunk is None:
        return {}
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        try:
            return chunk.model_dump(exclude_none=True, mode="json")
        except Exception:
            pass
    if hasattr(chunk, "to_json_dict"):
        try:
            return chunk.to_json_dict()
        except Exception:
            pass

    out: Dict[str, Any] = {}
    candidates = getattr(chunk, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)):
        cand_list = []
        for cand in candidates:
            c_dict: Dict[str, Any] = {
                "index": getattr(cand, "index", 0),
                "finishReason": str(getattr(cand, "finish_reason", "") or ""),
            }
            content = getattr(cand, "content", None)
            if content:
                c_dict["role"] = getattr(content, "role", "model")
                parts = getattr(content, "parts", None) or []
                part_list = []
                for p in parts:
                    p_dict: Dict[str, Any] = {}
                    if getattr(p, "thought", None) is True:
                        p_dict["thought"] = True
                    text_v = getattr(p, "text", None)
                    if text_v is not None:
                        p_dict["text"] = str(text_v)
                    fc = getattr(p, "function_call", None)
                    if fc is not None:
                        fc_args = getattr(fc, "args", {}) or {}
                        if hasattr(fc_args, "model_dump"):
                            try:
                                fc_args = fc_args.model_dump()
                            except Exception:
                                pass
                        p_dict["functionCall"] = {
                            "name": getattr(fc, "name", ""),
                            "args": fc_args,
                        }
                    sig = getattr(p, "thought_signature", None)
                    if sig:
                        p_dict["thoughtSignature"] = "<长Base64已省略>" if len(str(sig)) > 30 else str(sig)
                    inline = getattr(p, "inline_data", None)
                    if inline:
                        p_dict["inlineData"] = {"mimeType": getattr(inline, "mime_type", "")}
                    part_list.append(p_dict)
                c_dict["parts"] = part_list
            sr = getattr(cand, "safety_ratings", None)
            if sr:
                c_dict["safetyRatings"] = [str(r) for r in sr]
            cand_list.append(c_dict)
        out["candidates"] = cand_list

    pf = getattr(chunk, "prompt_feedback", None)
    if pf:
        br = getattr(pf, "block_reason", None)
        if br:
            out["promptFeedback"] = {"blockReason": str(br)}

    um = getattr(chunk, "usage_metadata", None)
    if um:
        out["usageMetadata"] = {
            "promptTokenCount": getattr(um, "prompt_token_count", 0),
            "thoughtsTokenCount": getattr(um, "thoughts_token_count", 0),
            "candidatesTokenCount": getattr(um, "candidates_token_count", 0),
            "totalTokenCount": getattr(um, "total_token_count", 0),
        }
    return out


def _format_preview(text: str, max_chars: int = 150) -> str:
    s = (text or "").replace("\r\n", "\\n").replace("\n", "\\n").replace("\t", "\\t")
    if len(s) <= max_chars:
        return s
    half = max(10, (max_chars - 3) // 2)
    return s[:half] + "..." + s[-half:]


def _plain_text_preview(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(p.get("text", ""))
            elif hasattr(p, "text"):
                out.append(getattr(p, "text", "") or "")
        return "\n".join(out)
    return str(content)


def print_outbound_debug_report(
    channel: str,
    model_name: str,
    gen_config_dict: dict,
    messages: list,
    prefill_text: str = "",
) -> None:
    """输出详尽、结构化的出站参数与提示词断点诊断日志。"""
    if not app_state.get_setting("debug_outbound", False):
        return

    num_msgs = len(messages) if messages else 0
    lines = [f"🔎 [出站调试 - {channel}] 模型={model_name} | 上下文消息共 {num_msgs} 轮"]

    # 1. 采样与生成核心参数
    cfg_items = []
    for k in ("temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "seed"):
        v = gen_config_dict.get(k)
        if v is not None:
            cfg_items.append(f"{k}={v}")
    tk = gen_config_dict.get("thinking_config")
    if tk:
        if isinstance(tk, dict):
            cfg_items.append(f"thinking_config={json.dumps(tk, ensure_ascii=False)}")
        else:
            cfg_items.append(f"thinking_config={tk}")
    if cfg_items:
        lines.append(f"  ⚙️ 生成配置: {', '.join(cfg_items)}")

    # 2. 工具与受限解码
    tools = gen_config_dict.get("tools") or []
    tool_decls = []
    for tg in tools:
        if isinstance(tg, dict):
            f_decls = tg.get("function_declarations") or tg.get("functionDeclarations") or []
            tool_decls.extend(f_decls)
        elif hasattr(tg, "function_declarations"):
            tool_decls.extend(getattr(tg, "function_declarations") or [])

    tc = gen_config_dict.get("tool_config") or gen_config_dict.get("toolConfig")
    tc_desc = ""
    if tc:
        fc_cfg = tc.get("function_calling_config") or tc.get("functionCallingConfig") or {}
        mode = fc_cfg.get("mode")
        allowed = fc_cfg.get("allowed_function_names") or fc_cfg.get("allowedFunctionNames")
        tc_desc = f"mode={mode}" + (f", allowed={allowed}" if allowed else "")

    if tool_decls or tc_desc:
        lines.append(f"  🛠️ 工具状态: 注入 {len(tool_decls)} 个工具" + (f" | 🔒 受限解码: {tc_desc}" if tc_desc else ""))
        for d in tool_decls:
            name = d.get("name") if isinstance(d, dict) else getattr(d, "name", "")
            params = (
                (d.get("parameters") or d.get("parameters_json_schema") or {})
                if isinstance(d, dict)
                else (getattr(d, "parameters", None) or getattr(d, "parameters_json_schema", None) or {})
            )
            props = params.get("properties") if isinstance(params, dict) else {}
            if isinstance(props, list):
                prop_keys = [item.get("key") for item in props if isinstance(item, dict) and item.get("key")]
            elif isinstance(props, dict):
                prop_keys = list(props.keys())
            else:
                prop_keys = []

            p_summary = f"params=[{', '.join(prop_keys)}]"
            lines.append(f"    ▸ [函数声明] {name} | {p_summary}")

    # 3. 预填充与末尾断点分析
    if prefill_text:
        lines.append(f"  🩹 预填充 prefill: 共 {len(prefill_text)} 字 | 断点尾部=\"{_format_preview(prefill_text, 100)}\"")

    # 4. 消息末尾切片预览
    if messages:
        first = messages[0]
        f_role = getattr(first, "role", "") if hasattr(first, "role") else (first.get("role") if isinstance(first, dict) else "")
        if f_role in ("system", "developer"):
            f_content = _plain_text_preview(getattr(first, "content", "") if hasattr(first, "content") else first.get("content"))
            lines.append(f"  📜 [第 1 轮 {f_role}]: \"{_format_preview(f_content, 120)}\"")

        last = messages[-1]
        l_role = getattr(last, "role", "unknown") if hasattr(last, "role") else (last.get("role", "unknown") if isinstance(last, dict) else "unknown")
        l_content = _plain_text_preview(getattr(last, "content", "") if hasattr(last, "content") else last.get("content", ""))
        lines.append(f"  📨 [末尾出站轮次 #{num_msgs} {l_role} | {len(l_content)} 字]: \"{_format_preview(l_content, 160)}\"")

    print("\n".join(lines))


def _um_get_int(um: Any, *keys: str) -> Optional[int]:
    """跨 SDK 版本安全读取 usage_metadata 字段（同时兼容对象属性、dict 与 Pydantic model_extra 未知驼峰字段）。"""
    if um is None:
        return None
    if isinstance(um, dict):
        for k in keys:
            v = um.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    extra = getattr(um, "model_extra", None)
    for k in keys:
        v = getattr(um, k, None)
        if v is None and isinstance(extra, dict):
            v = extra.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def _extract_raw_usage_fields(resp: Any) -> tuple[int, int, int, int, int]:
    """提取未经守恒折算的原始 (prompt_tk, explicit_th_tk, explicit_out_tk, tool_prompt_tk, total_tk)。"""
    um = resp.get("usageMetadata") if isinstance(resp, dict) else getattr(resp, "usage_metadata", None)
    if not um:
        return 0, 0, 0, 0, 0
    p_tk = _um_get_int(um, "prompt_token_count", "promptTokenCount") or 0
    th_tk = _um_get_int(
        um, "thoughts_token_count", "thoughtsTokenCount", "thought_token_count", "thoughtTokenCount"
    ) or 0
    out_tk = _um_get_int(um, "candidates_token_count", "candidatesTokenCount") or 0
    tool_p_tk = _um_get_int(um, "tool_use_prompt_token_count", "toolUsePromptTokenCount") or 0
    tot_tk = _um_get_int(um, "total_token_count", "totalTokenCount") or 0
    return p_tk, th_tk, out_tk, tool_p_tk, tot_tk


def _reconcile_usage_counts(
    p_tk: int,
    explicit_th: int,
    explicit_out: int,
    tool_p_tk: int,
    raw_tot: int,
    visible_chars: int = 0,
) -> tuple[int, int, int, int]:
    """基于 total_token_count 总量守恒校核并补齐隐藏思考或流式分片漏计的正文 Token：
    1. 当压制思考 (include_thoughts=False) 时，部分 Gemini 4 模型将 thoughts_token_count 填为 0 或 8（标记壳），
       而将真实的数百/数千内部推理 Token 计入 total_token_count —— 自动通过差额还原真实 th_tk；
    2. 当 stream_function_call_arguments 分片传输时，若 candidates_token_count 仅统计了收尾壳（<=32）而实际正文很长，
       且 explicit_th 已记录思考量，则将剩余差额归还给 out_tk。
    """
    th_tk = max(0, int(explicit_th or 0))
    out_tk = max(0, int(explicit_out or 0))
    p_tk = max(0, int(p_tk or 0))
    tool_p_tk = max(0, int(tool_p_tk or 0))
    raw_tot = max(0, int(raw_tot or 0))

    if raw_tot > 0 and p_tk > 0:
        residual_gen = max(0, raw_tot - p_tk - tool_p_tk)
        if residual_gen > (th_tk + out_tk):
            if th_tk > 16 and out_tk <= 32 and visible_chars > 120:
                out_tk = max(out_tk, residual_gen - th_tk)
            else:
                th_tk = max(th_tk, residual_gen - out_tk)

    comp_tk = th_tk + out_tk
    t_tk = max(raw_tot, p_tk + tool_p_tk + comp_tk) if (p_tk or comp_tk or raw_tot) else 0
    return p_tk, th_tk, out_tk, t_tk


def _extract_usage_detail(resp: Any, visible_chars: int = 0) -> tuple[int, int, int, int]:
    """从 Gemini 响应对象提取经总量守恒校核后的精确 (输入Tokens, 原生思考Tokens, 正文输出Tokens, 总Tokens)。"""
    p_tk, raw_th, raw_out, tool_p_tk, raw_tot = _extract_raw_usage_fields(resp)
    return _reconcile_usage_counts(p_tk, raw_th, raw_out, tool_p_tk, raw_tot, visible_chars=visible_chars)


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    p_tk, th_tk, out_tk, t_tk = _extract_usage_detail(resp)
    return p_tk, (th_tk + out_tk), t_tk


def _record_usage(resp: Any, visible_chars: int = 0) -> dict:
    usage_metadata = getattr(resp, "usage_metadata", None)
    traffic_type = getattr(usage_metadata, "traffic_type", None) if usage_metadata else None
    if traffic_type:
        value = getattr(traffic_type, "value", None) or str(traffic_type)
        print(f"🚦 [流量等级] 上游实际 traffic_type={value}")

    p_tk, th_tk, out_tk, t_tk = _extract_usage_detail(resp, visible_chars=visible_chars)
    comp_tk = th_tk + out_tk
    if p_tk or comp_tk:
        stats.add_tokens(p_tk, comp_tk)
        print(f"💰 [算力消耗统计] 输入: {p_tk} | 思考: {th_tk} | 正文: {out_tk} | 总计: {t_tk} Tokens")
    else:
        stats.add_success()
    res: Dict[str, Any] = {
        "prompt_tokens": p_tk,
        "completion_tokens": comp_tk,
        "total_tokens": t_tk,
        "thoughts_tokens": th_tk,
        "output_tokens": out_tk,
    }
    return res


def wants_usage(request_obj: Any) -> bool:
    opts = getattr(request_obj, "stream_options", None)
    if opts is None and getattr(request_obj, "model_extra", None):
        opts = request_obj.model_extra.get("stream_options")
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


def make_usage_chunk(response_id: str, model: str, usage: dict, created: Optional[int] = None) -> str:
    chunk = {"id": response_id, "object": "chat.completion.chunk",
             "created": int(created if created is not None else time.time()),
             "model": model, "choices": [], "usage": usage}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def is_gemini_response_valid(
    response: Any,
    synthetic_tool_name: str = "",
    prefill_text: str = "",
) -> bool:
    if response is None:
        return False
    if hasattr(response, "candidates") and response.candidates:
        saw_synthetic_call = False
        has_valid_real_part = False
        for cand in response.candidates:
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        if synthetic_tool_name and getattr(fc, "name", "") == synthetic_tool_name:
                            saw_synthetic_call = True
                            extracted = extract_synthetic_content(
                                getattr(fc, "args", None),
                                emit_tool_name=synthetic_tool_name,
                                prefill_text=prefill_text,
                            )
                            if extracted.strip() or (prefill_text and extracted):
                                return True
                            continue
                        return True
                    if getattr(part, "inline_data", None) is not None or getattr(part, "file_data", None) is not None:
                        return True
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip():
                        if getattr(part, "thought", None) is True:
                            continue
                        has_valid_real_part = True
            elif not synthetic_tool_name and hasattr(cand, "text") and isinstance(cand.text, str) and cand.text.strip():
                has_valid_real_part = True
        if saw_synthetic_call:
            # 命中了合成工具但提取正文为空且无真实工具调用：判定为无效以触发自动重试，防止多嘴普通文本被覆盖为空白回复
            return False
        if has_valid_real_part:
            return True
        return False
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
        return True
    return False


def _format_assembled_tool_call(
    call_dict: Dict[str, Any],
    tc_index: int,
    indexer: Optional["ToolCallIndexer"] = None,
    candidate_index: int = 0,
) -> Dict[str, Any]:
    """将聚合后的 function_call 转为 OpenAI tool_calls 增量对象，并同步签名与部分顺序。"""
    from message_processing import build_tool_call_id, thought_signature_extra
    from signature_store import SignatureState

    fake_fc = types.FunctionCall(
        id=call_dict.get("id"),
        name=call_dict.get("name") or "unknown",
        args=call_dict.get("args") or {},
    )
    fake_part = types.Part(
        function_call=fake_fc,
        thought_signature=call_dict.get("thought_signature"),
    )
    tc: Dict[str, Any] = {
        "index": tc_index,
        "id": build_tool_call_id(
            fake_fc, fake_part,
            missing_state=(SignatureState.UNSIGNED_FOLLOWER
                           if tc_index > 0 else SignatureState.NO_THOUGHT)),
        "type": "function",
        "function": {
            "name": fake_fc.name,
            "arguments": json.dumps(fake_fc.args, ensure_ascii=False) if fake_fc.args is not None else "{}",
        },
    }
    if indexer and tc_index == 0:
        indexer._first_tool_ids[candidate_index] = tc["id"]
    extra = thought_signature_extra(call_dict.get("thought_signature"))
    if extra:
        tc["extra_content"] = extra
    if indexer:
        indexer.record_tool_call(candidate_index, tc_index)
    return tc


class ToolCallIndexer:
    """一次流式响应内，为 tool_calls 分配稳定递增的 index，聚合分片函数参数，并实时拆包合成传输工具。"""

    def __init__(self, synthetic_tool_name: str = "", prefill_text: str = ""):
        self.synthetic_tool_name: str = (synthetic_tool_name or "").strip()
        self.prefill_text: str = prefill_text or ""
        self._next: Dict[int, int] = {}
        self._seen_tool_calls: set[int] = set()
        self._ordinary_parts: Dict[int, list] = {}
        self._part_order: Dict[int, list] = {}
        self._assemblers: Dict[int, Any] = {}
        self._pending_images: Dict[int, list[bytes]] = {}
        self._first_tool_ids: Dict[int, str] = {}
        self._extra_emitted: set[int] = set()

    def record_image_bytes(self, candidate_index: int, img_bytes: bytes) -> None:
        if img_bytes:
            self._pending_images.setdefault(candidate_index, []).append(img_bytes)

    def note_candidate_signature(self, candidate_index: int, sig: Optional[bytes]) -> None:
        if not sig:
            return
        self._assembler(candidate_index).note_turn_signature(sig)
        # 若首个工具调用已提前发出但当时尚未拿到尾随签名，立即补绑到缓存中
        first_id = self._first_tool_ids.get(candidate_index)
        if first_id:
            from signature_store import signature_store, SignatureRecord, SignatureState
            rec = signature_store.get_record(first_id)
            if rec is None or rec.state is not SignatureState.SIGNED:
                signature_store.put_record(first_id, SignatureRecord(SignatureState.SIGNED, sig))

    def bind_trailing_image_signature(self, candidate_index: int, sig: Optional[bytes]) -> None:
        if not sig:
            return
        self.note_candidate_signature(candidate_index, sig)
        imgs = self._pending_images.get(candidate_index)
        if imgs:
            from signature_store import image_signature_store
            for ib in imgs:
                if not image_signature_store.get(ib):
                    image_signature_store.put(ib, sig)

    def _assembler(self, candidate_index: int = 0):
        from message_processing import FunctionCallStreamAssembler
        asm = self._assemblers.get(candidate_index)
        if asm is None:
            asm = FunctionCallStreamAssembler(
                synthetic_tool_name=self.synthetic_tool_name,
                prefill_text=self.prefill_text,
            )
            self._assemblers[candidate_index] = asm
        return asm

    def feed_function_call(self, candidate_index: int, fc: Any, sig: Optional[bytes]) -> List[Dict[str, Any]]:
        return self._assembler(candidate_index).feed(fc, thought_signature=sig)

    def feed_function_call_ex(
        self, candidate_index: int, fc: Any, sig: Optional[bytes]
    ) -> tuple[List[Dict[str, Any]], str, Optional[bytes]]:
        return self._assembler(candidate_index).feed_with_synthetic(fc, thought_signature=sig)

    def has_synthetic_hit(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.synthetic_hit)

    def has_synthetic_halted(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.synthetic_halted)

    def synthetic_halt_reason(self, candidate_index: int = 0) -> str:
        asm = self._assemblers.get(candidate_index)
        return (asm.synthetic_halt_reason if asm else "") or ""

    def has_unclosed_synthetic(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.has_unclosed_synthetic())

    def flush_synthetic_tail(self, candidate_index: int = 0) -> str:
        asm = self._assemblers.get(candidate_index)
        return asm.flush_synthetic_tail() if asm else ""

    def pop_synth_signature(self, candidate_index: int = 0) -> Optional[bytes]:
        asm = self._assemblers.get(candidate_index)
        return asm.pop_synth_signature() if asm else None

    def flush_function_calls(self, candidate_index: int = 0) -> List[Dict[str, Any]]:
        asm = self._assemblers.get(candidate_index)
        return asm.flush() if asm else []

    def has_pending_function_call(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.has_pending())

    def has_received_synthetic_text(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.has_received_synthetic_text())

    def next_index(self, candidate_index: int = 0) -> int:
        i = self._next.get(candidate_index, 0)
        self._next[candidate_index] = i + 1
        self._seen_tool_calls.add(candidate_index)
        return i

    def record_tool_call(self, candidate_index: int, tool_index: int) -> None:
        self._part_order.setdefault(candidate_index, []).append(
            {"type": "tool_call", "index": tool_index})

    def record_ordinary_part(self, candidate_index: int, metadata: dict) -> int:
        items = self._ordinary_parts.setdefault(candidate_index, [])
        orders = self._part_order.setdefault(candidate_index, [])
        kind = metadata.get("kind")
        # 合并相邻的同类型（text / thought）普通增量块，避免上千个碎字典撑大收尾拓扑与下一轮历史解析开销
        if (
            items
            and orders
            and orders[-1].get("type") == "ordinary"
            and orders[-1].get("index") == len(items) - 1
            and kind in ("text", "thought")
            and items[-1].get("kind") == kind
            and not items[-1].get("thought_signature")
        ):
            prev = items[-1]
            add_txt = metadata.get("text") or ""
            if add_txt:
                prev["text"] = (prev.get("text") or "") + add_txt
            if metadata.get("thought_signature") and not prev.get("thought_signature"):
                prev["thought_signature"] = metadata["thought_signature"]
            return len(items) - 1

        index = len(items)
        items.append(dict(metadata))
        orders.append({"type": "ordinary", "index": index})
        return index

    def build_google_extra(self, candidate_index: int = 0) -> Dict[str, Any]:
        """构建精简后的单次收尾/工具拓扑元数据（仅在收尾帧或工具调用帧发送唯一一次）。"""
        metadata_to_emit, order_to_emit = self.topology(candidate_index)
        google_extra: Dict[str, Any] = {}
        for item in reversed(metadata_to_emit):
            if item.get("thought_signature"):
                google_extra["thought_signature"] = item["thought_signature"]
                google_extra["thought_signature_part"] = item.get("kind")
                break
        if metadata_to_emit:
            google_extra["ordinary_parts"] = metadata_to_emit
        if self.has_tool_calls(candidate_index) and order_to_emit:
            google_extra["part_order"] = order_to_emit
        return google_extra

    def mark_extra_emitted(self, candidate_index: int = 0) -> None:
        self._extra_emitted.add(candidate_index)

    def has_emitted_extra(self, candidate_index: int = 0) -> bool:
        return candidate_index in self._extra_emitted

    def topology(self, candidate_index: int = 0) -> tuple[list, list]:
        return (
            list(self._ordinary_parts.get(candidate_index, [])),
            list(self._part_order.get(candidate_index, [])),
        )

    def has_tool_calls(self, candidate_index: int = 0) -> bool:
        return candidate_index in self._seen_tool_calls


def convert_chunk_to_openai(chunk: Any, model_name: str, response_id: str, candidate_index: int = 0,
                            indexer: Optional[ToolCallIndexer] = None) -> str:
    from message_processing import (
        parse_gemini_response_for_reasoning_and_content,
        build_tool_call_id,
        thought_signature_extra,
        ordinary_part_metadata,
        _convert_image_to_markdown,
        _signature_bytes,
    )
    from signature_store import SignatureState
    delta_payload = {}
    openai_finish_reason = None

    if hasattr(chunk, "candidates") and chunk.candidates and len(chunk.candidates) > candidate_index:
        candidate = chunk.candidates[candidate_index]
        raw_gemini_finish_reason = getattr(candidate, "finish_reason", None)
        if raw_gemini_finish_reason:
            if hasattr(raw_gemini_finish_reason, "name"):
                raw_gemini_finish_reason_str = raw_gemini_finish_reason.name.upper()
            else:
                raw_gemini_finish_reason_str = str(raw_gemini_finish_reason).upper()

            if raw_gemini_finish_reason_str == "STOP":
                openai_finish_reason = "stop"
            elif raw_gemini_finish_reason_str in (
                "MAX_TOKENS",
                "MALFORMED_FUNCTION_CALL",
                "MALFORMED_RESPONSE",
                "UNEXPECTED_TOOL_CALL",
            ):
                openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str in (
                "SAFETY",
                "PROHIBITED_CONTENT",
                "RECITATION",
                "BLOCKLIST",
                "SPII",
                "IMAGE_SAFETY",
            ):
                openai_finish_reason = "content_filter"
            elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]:
                openai_finish_reason = "tool_calls"

        tool_call_deltas = []
        ordinary_metadata = []
        part_order = []
        ordinary_signature = None
        ordinary_signature_kind = None
        synthetic_text_deltas: List[str] = []

        def _emit_assembled_call(call_dict: Dict[str, Any]) -> None:
            tc_index = indexer.next_index(candidate_index) if indexer else len(tool_call_deltas)
            tc = _format_assembled_tool_call(call_dict, tc_index, indexer=indexer, candidate_index=candidate_index)
            part_order.append({"type": "tool_call", "index": tc_index})
            tool_call_deltas.append(tc)

        if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                sig = _signature_bytes(part, fc)
                if fc is None:
                    is_thought = getattr(part, "thought", None) is True
                    if (indexer and indexer.has_synthetic_hit(candidate_index)
                            and not is_thought
                            and getattr(part, "inline_data", None) is None
                            and getattr(part, "file_data", None) is None):
                        if sig:
                            indexer.note_candidate_signature(candidate_index, sig)
                            ordinary_signature = sig
                            ordinary_signature_kind = "text"
                        continue
                    text_value = getattr(part, "text", None)
                    if text_value is None and getattr(part, "inline_data", None) is not None:
                        inline = part.inline_data
                        if indexer and getattr(inline, "data", None):
                            indexer.record_image_bytes(candidate_index, inline.data)
                        text_value = _convert_image_to_markdown(inline.data, inline.mime_type)
                    elif text_value is None and getattr(part, "file_data", None) is not None:
                        text_value = f"![Image]({part.file_data.file_uri})"
                    text_value = "" if text_value is None else str(text_value)
                    if is_thought:
                        kind = "thought"
                    elif text_value == "" and sig:
                        kind = "signature_only"
                    else:
                        kind = "text"
                    metadata = ordinary_part_metadata(kind, text_value, sig)
                    if indexer:
                        ordinary_index = indexer.record_ordinary_part(candidate_index, metadata)
                    else:
                        ordinary_index = len(ordinary_metadata)
                    part_order.append({"type": "ordinary", "index": ordinary_index})
                    ordinary_metadata.append(metadata)
                    if sig:
                        if indexer:
                            indexer.bind_trailing_image_signature(candidate_index, sig)
                        ordinary_signature = sig
                        ordinary_signature_kind = kind
                    continue

                if indexer is not None:
                    completed_calls, synth_delta, synth_sig = indexer.feed_function_call_ex(candidate_index, fc, sig)
                    for c_item in completed_calls:
                        _emit_assembled_call(c_item)
                    if synth_delta or synth_sig:
                        if synth_delta:
                            synthetic_text_deltas.append(synth_delta)
                        metadata = ordinary_part_metadata("text", synth_delta, synth_sig)
                        ordinary_index = indexer.record_ordinary_part(candidate_index, metadata)
                        part_order.append({"type": "ordinary", "index": ordinary_index})
                        ordinary_metadata.append(metadata)
                        if synth_sig:
                            ordinary_signature = synth_sig
                            ordinary_signature_kind = "text"
                else:
                    _emit_assembled_call({
                        "id": getattr(fc, "id", None),
                        "name": getattr(fc, "name", "unknown"),
                        "args": getattr(fc, "args", None) or {},
                        "thought_signature": sig,
                    })

        if openai_finish_reason and indexer is not None and indexer.has_pending_function_call(candidate_index):
            for c_item in indexer.flush_function_calls(candidate_index):
                _emit_assembled_call(c_item)

        if indexer is not None and (openai_finish_reason or indexer.has_synthetic_halted(candidate_index)):
            tail_synth = indexer.flush_synthetic_tail(candidate_index)
            tail_sig = indexer.pop_synth_signature(candidate_index)
            if tail_synth or tail_sig:
                if tail_synth:
                    synthetic_text_deltas.append(tail_synth)
                metadata = ordinary_part_metadata("text", tail_synth, tail_sig)
                ordinary_index = indexer.record_ordinary_part(candidate_index, metadata)
                part_order.append({"type": "ordinary", "index": ordinary_index})
                ordinary_metadata.append(metadata)
                if tail_sig:
                    ordinary_signature = tail_sig
                    ordinary_signature_kind = "text"
            if indexer.has_synthetic_halted(candidate_index) and not indexer.has_tool_calls(candidate_index):
                openai_finish_reason = "stop"

        if indexer and indexer.has_synthetic_hit(candidate_index) and not indexer.has_tool_calls(candidate_index):
            if openai_finish_reason == "tool_calls":
                openai_finish_reason = "stop"
        elif (openai_finish_reason == "stop" and indexer
                and indexer.has_tool_calls(candidate_index)):
            openai_finish_reason = "tool_calls"

        if tool_call_deltas:
            delta_payload["tool_calls"] = tool_call_deltas

        reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)
        if indexer and indexer.has_synthetic_hit(candidate_index):
            normal_text = "".join(synthetic_text_deltas)

        if (openai_finish_reason and _safety_score_enabled()
                and hasattr(candidate, "safety_ratings") and candidate.safety_ratings):
            normal_text += _create_safety_ratings_html(candidate.safety_ratings)

        if reasoning_text:
            delta_payload["reasoning_content"] = reasoning_text
        if normal_text:
            delta_payload["content"] = normal_text
        elif tool_call_deltas:
            delta_payload["content"] = None
        elif not reasoning_text and openai_finish_reason is None:
            delta_payload["content"] = ""

        # 仅在收尾帧 (openai_finish_reason is not None) 或携带真实工具调用帧发送唯一一次拓扑扩展，
        # 彻底杜绝普通流式 token 帧每帧重复背负累积全文导致的 O(N^2) 响应体膨胀与酒馆越写越卡！
        should_emit_extra = bool(openai_finish_reason is not None or tool_call_deltas)
        if should_emit_extra:
            if indexer is not None:
                google_extra = indexer.build_google_extra(candidate_index)
                if not google_extra.get("thought_signature") and ordinary_signature:
                    google_extra.update((thought_signature_extra(
                        ordinary_signature, ordinary_signature_kind) or {}).get("google", {}))
                if google_extra:
                    delta_payload["extra_content"] = {"google": google_extra}
                    if openai_finish_reason is not None:
                        indexer.mark_extra_emitted(candidate_index)
            else:
                google_extra = {}
                for item in reversed(ordinary_metadata):
                    if item.get("thought_signature"):
                        google_extra["thought_signature"] = item["thought_signature"]
                        google_extra["thought_signature_part"] = item.get("kind")
                        break
                if not google_extra.get("thought_signature") and ordinary_signature:
                    google_extra.update((thought_signature_extra(
                        ordinary_signature, ordinary_signature_kind) or {}).get("google", {}))
                if ordinary_metadata:
                    google_extra["ordinary_parts"] = ordinary_metadata
                if tool_call_deltas and part_order:
                    google_extra["part_order"] = part_order
                if google_extra:
                    delta_payload["extra_content"] = {"google": google_extra}

    if not delta_payload and openai_finish_reason is None:
        delta_payload["content"] = ""

    chunk_data = {
        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
        "choices": [{"index": candidate_index, "delta": delta_payload, "finish_reason": openai_finish_reason}]
    }
    return f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"


async def _chunk_openai_response_dict_for_sse(
    openai_response_dict: Dict[str, Any],
    response_id_override: Optional[str] = None,
    model_name_override: Optional[str] = None,
    include_usage: bool = False,
):
    resp_id = response_id_override or openai_response_dict.get("id", f"chatcmpl-fakestream-{int(time.time())}")
    model_name = model_name_override or openai_response_dict.get("model", "unknown")
    created_time = openai_response_dict.get("created", int(time.time()))

    choices = openai_response_dict.get("choices", [])
    if not choices:
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    for choice_idx, choice in enumerate(choices):
        message = choice.get("message", {})
        final_finish_reason = choice.get("finish_reason", "stop")

        def _sse(delta):
            payload = {"id": resp_id, "object": "chat.completion.chunk", "created": created_time,
                       "model": model_name, "choices": [{"index": choice_idx, "delta": delta,
                                                         "finish_reason": None}]}
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        reasoning_content = message.get("reasoning_content", "")
        actual_content = message.get("content")
        if reasoning_content:
            yield _sse({"reasoning_content": reasoning_content})
            await asyncio.sleep(0.01)

        if actual_content is not None:
            content_to_chunk = actual_content
            if "![Image](data:image/" in content_to_chunk or "![Generated Image](data:" in content_to_chunk:
                chunk_size = max(1, len(content_to_chunk))
            else:
                chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1
            if not content_to_chunk and not reasoning_content:
                yield _sse({"content": ""})
            else:
                for i in range(0, len(content_to_chunk), chunk_size):
                    yield _sse({"content": content_to_chunk[i:i + chunk_size]})
                    if len(content_to_chunk) > chunk_size:
                        await asyncio.sleep(0.01)

        for tc_item_idx, tool_call_item in enumerate(message.get("tool_calls") or []):
            tool_delta = {
                "index": tc_item_idx,
                "id": tool_call_item["id"],
                "type": "function",
                "function": {"name": tool_call_item["function"]["name"], "arguments": ""},
            }
            if tool_call_item.get("extra_content") is not None:
                tool_delta["extra_content"] = tool_call_item["extra_content"]
            yield _sse({"tool_calls": [tool_delta]})
            await asyncio.sleep(0.01)
            yield _sse({"tool_calls": [{
                "index": tc_item_idx,
                "id": tool_call_item["id"],
                "function": {"arguments": tool_call_item["function"]["arguments"]},
            }]})
            await asyncio.sleep(0.01)

        fin_delta: Dict[str, Any] = {}
        if message.get("extra_content") is not None:
            fin_delta["extra_content"] = message["extra_content"]
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': fin_delta, 'finish_reason': final_finish_reason}]}, ensure_ascii=False)}\n\n"

    if include_usage:
        usage = openai_response_dict.get("usage")
        if isinstance(usage, dict):
            yield make_usage_chunk(resp_id, model_name, usage, created=created_time)

    yield "data: [DONE]\n\n"


def _prepend_prefill(openai_dict: Dict[str, Any], prefill_text: str) -> Dict[str, Any]:
    if not prefill_text:
        return openai_dict
    try:
        for choice in (openai_dict.get("choices") or []):
            msg = choice.get("message")
            if not isinstance(msg, dict) or msg.get("tool_calls"):
                continue
            existing = msg.get("content") or ""
            msg["content"] = prefill_text + strip_prefill_overlap(prefill_text, existing)
    except Exception:
        pass
    return openai_dict


def _dedup_sse_chunk_content(sse_line: str, deduper: PrefillDeduper, force_flush: bool = False) -> Optional[str]:
    if deduper.done and not force_flush:
        return sse_line
    try:
        payload = json.loads(sse_line[len("data: "):].strip())
        choices = payload.get("choices") or []
        if not choices:
            return sse_line
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        has_finish = bool(choice.get("finish_reason"))

        out = deduper.feed(content) if content else ""
        if has_finish or force_flush:
            out += deduper.flush()

        if out:
            delta["content"] = out
            choice["delta"] = delta
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if content is not None:
            delta.pop("content", None)
        if delta or has_finish:
            choice["delta"] = delta
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return None
    except Exception:
        return sse_line


def _strip_sse_reasoning_chunk(sse_line: str) -> Optional[str]:
    """在流内无感剥壳回退重连时，过滤重连流中重复生成的 reasoning_content 块。"""
    if '"reasoning_content":' not in sse_line:
        return sse_line
    try:
        payload = json.loads(sse_line[len("data: "):].strip())
        choices = payload.get("choices") or []
        if not choices:
            return sse_line
        choice = choices[0]
        delta = choice.get("delta") or {}
        delta.pop("reasoning_content", None)
        has_finish = choice.get("finish_reason") is not None
        has_meaningful = bool(
            delta.get("content")
            or delta.get("tool_calls")
            or delta.get("extra_content")
            or delta.get("role")
        )
        if has_meaningful or has_finish:
            choice["delta"] = delta
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return None
    except Exception:
        return sse_line


def _sse_chunk_has_visible_body(sse_line: str) -> bool:
    """判断 SSE 帧是否包含真实的可见正文 (content) 或真实工具调用 (tool_calls)，严格排除纯思考块。"""
    return bool(
        '"tool_calls":' in sse_line
        or ('"content":' in sse_line and '"content": ""' not in sse_line and '"content": null' not in sse_line)
    )


class _MalformedSandboxFallback(Exception):
    """内部控制流异常：当抗截断沙箱在尚未产出任何可见正文前触发 MALFORMED_FUNCTION_CALL 时，0 等待就地切换为无沙箱原生模式重连。"""
    pass


def is_location_pin_failure(err: Any) -> bool:
    """错误是否像"钉定的 projects/locations 路径不对"（而非模型或网络本身的问题）。"""
    e = str(err).lower()
    if "requires billing" in e or "billing to be enabled" in e:
        return True
    if ("not found" in e or "404" in e) and ("publisher model" in e or "projects/" in e):
        return True
    if ("permission_denied" in e or ("permission" in e and "denied" in e) or "403" in e) and "projects/" in e:
        return True
    return False


# ============================================================
# 统一多 Key 轮询 + 跨通道（Express ↔ Cookie）主备重试调度引擎
# ============================================================

def _emit_credential_invalid_warning(target: dict, err_code: int, err_formatted: str) -> str:
    """当某个 Key 或 Cookie 发生 401/403/欠费时，打印醒目红色告警并返回附加给用户的更换提醒。"""
    if target.get("channel") == "express":
        rec = target.get("key_record") or {}
        idx_num = rec.get("index", 0) + 1
        masked = _mask_key_short(rec.get("key", ""))
        pid = rec.get("project_id") or "未绑定"
        warn_line = (
            f"🔴 [密钥失效提醒] 第 #{idx_num} 个 Express API Key（{masked} | 项目={pid}）"
            f"鉴权或计费失败（HTTP {err_code}），本轮请求已自动隔离并跳过该 Key！"
            f"👉 请及时前往控制台「通道与凭证」页检查、删除或更换该 Key！原因：{err_formatted[:400]}"
        )
        user_tip = f"\n\n👉 【凭证更换提醒】第 #{idx_num} 个 Express API Key（{masked}）已失效或所属项目未开计费，请及时到控制台「通道与凭证」页删除或更换！"
    else:
        warn_line = (
            f"🔴 [Cookie 失效提醒] {target.get('label', 'Studio Cookie 直连')} "
            f"鉴权或项目计费失败（HTTP {err_code}），本轮请求已自动隔离并跳过 Cookie 通道！"
            f"👉 请及时前往控制台「通道与凭证」页重新获取并更新 Google Cookie（或检查 Project ID）！原因：{err_formatted[:400]}"
        )
        user_tip = "\n\n👉 【凭证更换提醒】Studio Cookie 登录态已失效或项目无权限，请及时到控制台「通道与凭证」页更新 Cookie 或切换通道！"
    print(warn_line)
    return user_tip


def _short_cred_name(target: Optional[dict]) -> str:
    if not target:
        return "—"
    if target.get("channel") == "express":
        rec = target.get("key_record") or {}
        return f"Key #{rec.get('index', 0) + 1}"
    return "🍪 Cookie"


def _format_thinking_for_record(tk_dict: Optional[dict]) -> tuple[str, str, bool]:
    """返回 (展示文案, 徽章样式 muted|accent|apricot, 是否流式回传原生思考)。"""
    if not tk_dict or tk_dict.get("mode") is None:
        return "无", "muted", False
    inc = bool(tk_dict.get("include_thoughts", True))
    lv = str(tk_dict.get("level") or "").strip().upper()
    if not inc:
        return (f"压制 ({lv})" if lv else "压制"), "muted", False
    if lv == "XHIGH":
        return "🧠 XHIGH", "apricot", True
    if lv:
        return lv, "accent", True
    return "自动", "accent", True


def _chunk_has_model_token(chunk: Any, include_thoughts: bool = True) -> tuple[bool, bool]:
    """检测上游流式 chunk 是否包含实际生成的模型 token（精确测量真实首字响应 TTFT，不受 96 字滑窗影响）。"""
    cands = getattr(chunk, "candidates", None) or []
    has_tok = False
    has_thought = False
    for cand in cands:
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        for p in parts:
            is_th = getattr(p, "thought", None) is True
            txt = getattr(p, "text", None)
            if is_th:
                if txt:
                    has_thought = True
                    if include_thoughts:
                        has_tok = True
            elif txt:
                has_tok = True
            else:
                fc = getattr(p, "function_call", None)
                if fc is not None and (
                    getattr(fc, "name", None)
                    or getattr(fc, "args", None)
                    or getattr(fc, "partial_args", None)
                ):
                    has_tok = True
                elif getattr(p, "inline_data", None) is not None or getattr(p, "file_data", None) is not None:
                    has_tok = True
    return has_tok, has_thought


class CallTraceTracker:
    """单次聊天请求的生命周期指标追踪器（幂等写入 CallRecorder，零重复记账）。"""

    def __init__(self, request_obj: OpenAIRequest, primary_channel: str):
        self.created_wall = time.time()
        self.start_mono = time.monotonic()
        self.primary_channel = primary_channel
        self.finalized = False

        raw_model = str(getattr(request_obj, "model", "") or "unknown").strip()
        base_m = raw_model
        while True:
            low_m = base_m.lower()
            if low_m.endswith("-search"):
                base_m = base_m[: -len("-search")]
                continue
            m_th = re.search(r"-think-(?:minimal|low|medium|high|xhigh|off|none)$", low_m)
            if m_th:
                base_m = base_m[: m_th.start()]
                continue
            break
        self.model_display = base_m or raw_model
        prof = mc.get_profile(self.model_display)
        self.is_image = bool(prof.get("is_image", False))
        self.is_stream_req = bool(getattr(request_obj, "stream", False))
        fake_on = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))

        eff_settings = app_state.get_effective_settings(self.model_display)
        # 粗略预判末尾是否为 assistant 预填充（后续 Express ctx 或 Cookie telemetry 会用精确结果覆盖）
        msgs = getattr(request_obj, "messages", None) or []
        rough_prefill = bool(
            msgs
            and str(getattr(msgs[-1], "role", "") or "") == "assistant"
            and eff_settings.get("prefill_mode", "smart") != "off"
        )
        tk_init = mc.resolve_thinking(raw_model, request_obj, eff_settings, prefill_active=rough_prefill)
        self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(tk_init)

        self.sandbox = bool(eff_settings.get("synthetic_sandbox_enabled", False) and not self.is_image)
        self.prefill = rough_prefill
        self.search = bool(raw_model.lower().endswith("-search"))
        self.tool_calls = False

        if self.is_image:
            self.mode_label = "🖼️ 生图假流" if self.is_stream_req else "🖼️ 生图非流"
            self.mode_class = "pill-accent" if self.is_stream_req else "pill-sky"
            self.is_unary = True
        elif not self.is_stream_req:
            self.mode_label = "📦 非流式"
            self.mode_class = "pill-sky"
            self.is_unary = True
        elif fake_on:
            self.mode_label = "💓 假流式"
            self.mode_class = "pill-accent"
            self.is_unary = True
        else:
            self.mode_label = "⚡ 真流式"
            self.mode_class = "pill-mint"
            self.is_unary = False

        self.retries = 0
        self.retry_wait_total = 0.0
        self.unique_creds: List[str] = []
        self.failed_steps: List[str] = []
        self.last_cred_short: str = ""

    def sync_from_express_ctx(self, ctx: Optional[dict]) -> None:
        if not ctx:
            return
        self.model_display = ctx.get("base_model_name") or self.model_display
        self.sandbox = bool(ctx.get("synthetic_tool_name"))
        self.prefill = bool(ctx.get("prefill_active", False))
        self.search = bool(ctx.get("has_search", False))
        tk_res = ctx.get("thinking_resolved")
        if tk_res is not None:
            self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(tk_res)

    def sync_from_cookie_telemetry(self, tel: Optional[dict]) -> None:
        if not tel:
            return
        if tel.get("base_model"):
            self.model_display = tel["base_model"]
        if "sandbox" in tel:
            self.sandbox = bool(tel["sandbox"])
        if "prefill" in tel:
            self.prefill = bool(tel["prefill"])
        if "search" in tel:
            self.search = bool(tel["search"])
        if tel.get("thinking_resolved") is not None:
            self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(
                tel["thinking_resolved"]
            )
        if "is_unary" in tel:
            self.is_unary = bool(tel["is_unary"])
        if tel.get("mode_label"):
            self.mode_label = tel["mode_label"]
            if "真流式" in self.mode_label:
                self.mode_class = "pill-mint"
            elif "假流" in self.mode_label:
                self.mode_class = "pill-accent"
            else:
                self.mode_class = "pill-sky"

    def note_attempt_start(self, target: dict) -> None:
        short = _short_cred_name(target)
        self.last_cred_short = short
        if short not in self.unique_creds:
            self.unique_creds.append(short)

    def note_attempt_failure(self, target: dict, err_code: int, wait_sec: Optional[float] = None) -> None:
        short = _short_cred_name(target)
        if wait_sec is not None:
            self.retries += 1
            self.retry_wait_total += wait_sec
            self.failed_steps.append(f"{html.escape(short)} (HTTP {err_code}，等 {wait_sec:.2f}s)")
        else:
            self.failed_steps.append(f"{html.escape(short)} (HTTP {err_code})")

    def _build_time_strs(self) -> tuple[str, str]:
        lt = time.localtime(self.created_wall)
        return time.strftime("%H:%M:%S", lt), time.strftime("%Y-%m-%d %H:%M:%S", lt)

    def finalize_success(
        self,
        target: dict,
        *,
        is_cookie: bool = False,
        ttft: Optional[float] = None,
        gen_time: float = 0.0,
        saw_thought_stream: bool = False,
        finish_reason: str = "stop",
        has_tool_calls: bool = False,
        has_usage: bool = False,
        p_tk: int = 0,
        th_tk: int = 0,
        out_tk: int = 0,
        out_chars: int = 0,
    ) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        short_cred = _short_cred_name(target)
        if target.get("channel") != self.primary_channel:
            cred_note = "跨通道兜底"
        elif is_cookie:
            cred_note = "Cookie 模式"
        else:
            note = str((target.get("key_record") or {}).get("note") or "").strip()
            cred_note = note[:16] if note else "标准模式"

        is_trunc = (str(finish_reason or "").lower() == "length")
        if is_trunc:
            badge_type = "trunc"
        elif self.retries > 0:
            badge_type = "retry_ok"
        else:
            badge_type = "ok"

        clean_ttft = round(float(ttft), 2) if (not self.is_unary and ttft is not None and ttft >= 0) else None
        clean_gen = round(max(0.0, float(gen_time)), 2)

        tps: Optional[float] = None
        tps_approx = False
        if has_usage:
            if not self.is_unary:
                streamed_tk = (th_tk + out_tk) if (self.include_thoughts and saw_thought_stream) else out_tk
                if clean_gen >= 0.12 and streamed_tk > 0:
                    tps = round(streamed_tk / clean_gen, 1)
                    tps_approx = False
                elif (th_tk + out_tk) > 0 and ((clean_ttft or 0.0) + clean_gen) >= 0.1:
                    tps = round((th_tk + out_tk) / ((clean_ttft or 0.0) + clean_gen), 1)
                    tps_approx = True
            else:
                tot_gen_tk = th_tk + out_tk
                if clean_gen >= 0.1 and tot_gen_tk > 0:
                    tps = round(tot_gen_tk / clean_gen, 1)
                    tps_approx = True

        # 组装清晰安全的展开详情
        parts_det: List[str] = []
        if self.failed_steps:
            chain_str = " → ".join(self.failed_steps + [f"<b>{html.escape(short_cred)} 成功</b>"])
            parts_det.append(
                f"🔄 <b>重试轨迹（重试 {self.retries} 次，退避共耗 {self.retry_wait_total:.1f}s）</b>：{chain_str}。"
            )
        else:
            parts_det.append(f"✅ <b>一次直通成功</b>：使用凭证 <b>{html.escape(short_cred)}</b>（{html.escape(cred_note)}）。")

        if not self.is_unary and clean_ttft is not None:
            parts_det.append(f"⏱️ 首字响应 <b>{clean_ttft:.2f}s</b>，首字后纯流式输出耗时 <b>{clean_gen:.2f}s</b>（请求总历时 {total_wall:.2f}s）。")
        else:
            parts_det.append(f"⏱️ 上游原生整包生成耗时 <b>{clean_gen:.2f}s</b>（无独立首字时间，请求总历时 {total_wall:.2f}s）。")

        if has_usage:
            spd_txt = f"，速度 <b>{'≈' if tps_approx else ''}{tps:.1f} tok/s</b>" if tps else ""
            parts_det.append(
                f"💰 Token 明细：输入 <b>{p_tk:,}</b> · 思考 <b>{th_tk:,}</b> · 正文 <b>{out_tk:,}</b>（共 {p_tk + th_tk + out_tk:,} Tokens{spd_txt}）。"
            )
        elif is_cookie:
            ch_txt = f"，正文共 <b>{out_chars:,}</b> 字符" if out_chars > 0 else ""
            parts_det.append(f"🍪 Cookie 私有接口不回传 Token 统计{ch_txt}。")

        if is_trunc:
            parts_det.append("⚠️ <b>达上限截断提醒</b>：finish_reason=length（撞上 max_tokens 限制提前停止），若正文未写完建议调大客户端最大回复长度。")

        t_short, t_full = self._build_time_strs()
        call_recorder.record({
            "time": t_short,
            "full_time": t_full,
            "group": "ok",
            "badge_type": badge_type,
            "retries": self.retries,
            "code": 200,
            "finish_reason": str(finish_reason or "stop"),
            "model": self.model_display,
            "sandbox": self.sandbox,
            "prefill": self.prefill,
            "search": self.search,
            "tool_calls": bool(has_tool_calls or finish_reason == "tool_calls"),
            "thinking": self.thinking_label,
            "think_style": self.think_style,
            "mode": self.mode_label,
            "mode_class": self.mode_class,
            "cred": short_cred,
            "cred_note": cred_note,
            "is_unary": self.is_unary,
            "ttft": clean_ttft,
            "gen_time": clean_gen,
            "total_time": total_wall,
            "is_cookie": is_cookie,
            "has_usage": has_usage,
            "p_tk": p_tk,
            "th_tk": th_tk,
            "out_tk": out_tk,
            "out_chars": out_chars,
            "tps": tps,
            "tps_approx": tps_approx,
            "detail": " ".join(parts_det),
        })

    def finalize_error(self, err_code: int, err_msg: str, preflight: bool = False) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        if preflight or not self.unique_creds:
            cred = "未发往上游"
            cred_note = "前置校验拦截"
        elif len(self.unique_creds) == 1:
            cred = self.unique_creds[0]
            cred_note = f"单凭证重试×{self.retries}" if self.retries > 0 else "未触发重试"
        else:
            cred = f"试 {len(self.unique_creds)} 凭证均败"
            cred_note = f"止于 {self.last_cred_short}"

        parts_det: List[str] = []
        if self.failed_steps:
            parts_det.append(f"🔄 <b>尝试轨迹</b>：{' → '.join(self.failed_steps)}。")
        clean_reason = html.escape(str(err_msg or "未知错误").strip()[:600])
        parts_det.append(f"❌ <b>失败原因（HTTP {err_code} · 总耗时 {total_wall:.2f}s）</b>：{clean_reason}")

        t_short, t_full = self._build_time_strs()
        call_recorder.record({
            "time": t_short,
            "full_time": t_full,
            "group": "err",
            "badge_type": "err",
            "retries": self.retries,
            "code": int(err_code or 500),
            "finish_reason": "error",
            "model": self.model_display,
            "sandbox": self.sandbox,
            "prefill": self.prefill,
            "search": self.search,
            "tool_calls": False,
            "thinking": self.thinking_label,
            "think_style": self.think_style,
            "mode": self.mode_label,
            "mode_class": self.mode_class,
            "cred": cred,
            "cred_note": cred_note,
            "is_unary": self.is_unary,
            "ttft": None,
            "gen_time": total_wall,
            "total_time": total_wall,
            "is_cookie": (cred == "🍪 Cookie"),
            "has_usage": False,
            "p_tk": 0,
            "th_tk": 0,
            "out_tk": 0,
            "out_chars": 0,
            "tps": None,
            "tps_approx": False,
            "detail": " ".join(parts_det),
        })

    def finalize_cancelled(self, partial_output: bool = False) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        cred = self.last_cred_short or "—"
        cred_note = "输出中途断开" if partial_output else "前端主动停止"
        msg = (
            f"🛑 <b>客户端中途断开（HTTP 499 · 已耗时 {total_wall:.2f}s）</b>：流式输出期间连接关闭，后台已安全回收上游流。"
            if partial_output
            else f"🛑 <b>前端手动停止（HTTP 499 · 已耗时 {total_wall:.2f}s）</b>：首包产出前检测到客户端断开，后台已立即取消上游任务与重试。"
        )
        t_short, t_full = self._build_time_strs()
        call_recorder.record({
            "time": t_short,
            "full_time": t_full,
            "group": "cancel",
            "badge_type": "cancel",
            "retries": self.retries,
            "code": 499,
            "finish_reason": "cancelled",
            "model": self.model_display,
            "sandbox": self.sandbox,
            "prefill": self.prefill,
            "search": self.search,
            "tool_calls": False,
            "thinking": self.thinking_label,
            "think_style": self.think_style,
            "mode": self.mode_label,
            "mode_class": self.mode_class,
            "cred": cred,
            "cred_note": cred_note,
            "is_unary": self.is_unary,
            "ttft": None,
            "gen_time": total_wall,
            "total_time": total_wall,
            "is_cookie": (cred == "🍪 Cookie"),
            "has_usage": False,
            "p_tk": 0,
            "th_tk": 0,
            "out_tk": 0,
            "out_chars": 0,
            "tps": None,
            "tps_approx": False,
            "detail": msg,
        })


async def execute_unified_completion(
    request_obj: OpenAIRequest,
    fastapi_request: Any,
    primary_channel: str = "express",
    express_upstream: Any = None,
    cookie_upstream: Any = None,
):
    """统一调度器：
    - 方案 A 主备顺序：以 primary_channel 为主通道，当 cross_channel_fallback 开启时将另一通道纳入轮询周期尾部；
    - 多 Key 自动轮询：每次重试自动切换到下一个健康的 Express Key（动态绑定各自的 Project ID）或 Cookie；
    - 坏凭证自动隔离与红色告警：401/403/欠费凭证当次隔离跳过并提示删除/更换；
    - 严格遵守 retry_backoff_seconds 退避间隔。
    """
    tracker = CallTraceTracker(request_obj, primary_channel)
    try:
        validate_request_schemas(request_obj)
    except SchemaValidationError as exc:
        tracker.finalize_error(400, str(exc), preflight=True)
        return JSONResponse(
            status_code=400,
            content=create_openai_error_response(400, str(exc), "invalid_request_error"),
        )

    if express_upstream is None:
        from upstreams.express_sdk import ExpressSDKUpstream
        express_upstream = ExpressSDKUpstream()
    if cookie_upstream is None:
        from upstreams.cookie_proxy import CookieProxyUpstream
        cookie_upstream = CookieProxyUpstream()

    express_key_manager = getattr(fastapi_request.app.state, "express_key_manager", None)
    express_records = express_key_manager.get_ordered_express_records() if express_key_manager else []
    express_targets = [
        {
            "channel": "express",
            "id": f"express:{rec['key']}",
            "key_record": rec,
            "label": f"Express Key #{rec.get('index', 0) + 1} ({_mask_key_short(rec['key'])} | 项目={rec.get('project_id') or '默认'})",
        }
        for rec in express_records
    ]

    cookie_ready, _, cookie_pid, _, cookie_err_code, cookie_err_msg = cookie_upstream.check_cookie_ready()
    cookie_target = {
        "channel": "cookie",
        "id": "cookie",
        "label": f"Studio Cookie 直连 (项目={cookie_pid})",
    } if cookie_ready else None

    cross_fallback = bool(app_state.get_setting("cross_channel_fallback", True))

    base_targets: List[dict] = []
    if primary_channel == "express":
        base_targets.extend(express_targets)
        if cross_fallback and cookie_target is not None:
            base_targets.append(cookie_target)
        if not base_targets:
            error_msg = (
                "未配置任何可用的 Express API Key"
                + ("（且未配置有效 Cookie 兜底）" if cross_fallback else "")
                + "，请在控制台「通道与凭证」页添加 Express Key。"
            )
            print(f"❌ [密钥配置] {error_msg}")
            tracker.finalize_error(401, error_msg, preflight=True)
            return JSONResponse(
                status_code=401,
                content=create_openai_error_response(401, error_msg, "authentication_error"),
            )
        if not express_targets and cookie_target is not None:
            print(
                "🔴 [密钥缺失提醒] 当前处于 Express 模式但未配置任何启用的 Express API Key！"
                "已自动跨通道切换至 Studio Cookie 直连兜底。👉 请及时在「通道与凭证」页添加 Key 或切换通道。"
            )
    else:
        if cookie_target is not None:
            base_targets.append(cookie_target)
        if cross_fallback and express_targets:
            base_targets.extend(express_targets)
            if cookie_target is None:
                print(
                    f"🔴 [Cookie 失效提醒] 当前处于 Cookie 直连模式，但 Cookie 未配置或无效（{cookie_err_msg}）！"
                    f"已自动跨通道切换至 Express API Key 池兜底。👉 请及时在「通道与凭证」页更新 Cookie！"
                )
        if not base_targets:
            print(f"❌ [Studio 鉴权/配置失败] 模型={request_obj.model} | {cookie_err_msg}")
            tracker.finalize_error(cookie_err_code, cookie_err_msg, preflight=True)
            return JSONResponse(
                status_code=cookie_err_code,
                content=create_openai_error_response(cookie_err_code, cookie_err_msg, "auth_error"),
            )

    if len(base_targets) > 1:
        pool_desc = " → ".join(t["label"] for t in base_targets)
        print(f"🔑 [凭证轮询池] 本次请求可用凭证共 {len(base_targets)} 个：{pool_desc}")

    express_ctx: Optional[dict] = None
    express_prompt_cache: Optional[List[types.Content]] = None
    express_nosandbox_ctx: Optional[dict] = None
    express_nosandbox_prompt_cache: Optional[List[types.Content]] = None
    sandbox_downgraded: bool = False

    async def _get_express_ctx(no_sandbox: bool = False):
        nonlocal express_ctx, express_prompt_cache, express_nosandbox_ctx, express_nosandbox_prompt_cache
        if no_sandbox:
            if express_nosandbox_ctx is None:
                ctx, err_resp = express_upstream.prepare_express_context(request_obj, disable_sandbox=True)
                if err_resp is not None:
                    return None, None, err_resp
                express_nosandbox_ctx = ctx
                tracker.sync_from_express_ctx(ctx)
                express_nosandbox_prompt_cache = await asyncio.to_thread(
                    ctx["prompt_func"], ctx["request_obj"].messages
                )
            return express_nosandbox_ctx, express_nosandbox_prompt_cache, None

        if express_ctx is None:
            ctx, err_resp = express_upstream.prepare_express_context(request_obj)
            if err_resp is not None:
                return None, None, err_resp
            express_ctx = ctx
            tracker.sync_from_express_ctx(ctx)
            express_prompt_cache = await asyncio.to_thread(
                ctx["prompt_func"], ctx["request_obj"].messages
            )
        return express_ctx, express_prompt_cache, None

    if any(t["channel"] == "express" for t in base_targets):
        _, _, init_err_resp = await _get_express_ctx()
        if init_err_resp is not None:
            tracker.finalize_error(getattr(init_err_resp, "status_code", 400), "请求参数校验失败", preflight=True)
            return init_err_resp

    dead_targets: set[str] = set()
    cursor = 0

    def _pick_next_target() -> Optional[dict]:
        nonlocal cursor
        n = len(base_targets)
        for offset in range(n):
            idx = (cursor + offset) % n
            cand = base_targets[idx]
            if cand["id"] not in dead_targets:
                cursor = (idx + 1) % n
                return cand
        return None

    def _peek_next_target() -> Optional[dict]:
        n = len(base_targets)
        for offset in range(n):
            idx = (cursor + offset) % n
            cand = base_targets[idx]
            if cand["id"] not in dead_targets:
                return cand
        return None

    async def _client_gone() -> bool:
        if fastapi_request is None:
            return False
        try:
            return await fastapi_request.is_disconnected()
        except Exception:
            return False

    max_retries, backoff_sec = get_retry_settings()
    is_image_request = mc.get_profile(request_obj.model)["is_image"]
    tried_target_ids_in_cycle: set[str] = set()

    def _compute_smart_retry_wait(
        base_sec: float,
        target_cur: dict,
        target_nxt: dict,
        upstream_retry_after: Optional[float] = None,
    ) -> float:
        """两档退避等待策略（适配快速模式公共池高频 429）：
        - 情况 A：切换到本轮周期内尚未试过的另一把可用凭证时，仅等 0.3~0.6s 立即换号敲门；
        - 情况 B：单凭证原地重试，或所有可用凭证已试满一整轮准备进入下一轮敲门时，
          清空本轮已试集合，并按控制台 retry_backoff_seconds（施加 0.8~1.2 随机抖动）退避等待
          （若上游明确给出了更短的 Retry-After 秒数则优先参考）。
        """
        nonlocal tried_target_ids_in_cycle
        if target_cur:
            tried_target_ids_in_cycle.add(target_cur["id"])

        switching_to_fresh_target = bool(
            target_nxt
            and target_cur
            and target_nxt["id"] != target_cur["id"]
            and target_nxt["id"] not in tried_target_ids_in_cycle
        )
        if switching_to_fresh_target:
            return round(random.uniform(0.30, 0.60), 2)

        # 所有存活凭证均已轮试完一圈（或仅配置了单个凭证）：重置本轮集合，执行完整周期退避
        tried_target_ids_in_cycle.clear()
        effective_base = float(base_sec)
        if upstream_retry_after is not None and 0.5 <= upstream_retry_after < effective_base:
            effective_base = upstream_retry_after
        if effective_base < 1.0:
            return round(effective_base, 2)
        return round(effective_base * random.uniform(0.80, 1.20), 2)

    # ============================================================
    # 分支一：流式请求（包含真流式与假流式，支持跨 Key / 跨通道无缝切换）
    # ============================================================
    if request_obj.stream:
        fake_stream_on = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))

        try:
            outer_keep_alive_interval = float(app_state.get_setting(
                "fake_streaming_interval", app_config.FAKE_STREAMING_INTERVAL_SECONDS) or 1.0)
        except (TypeError, ValueError):
            outer_keep_alive_interval = 1.0
        outer_keep_alive_interval = max(0.5, outer_keep_alive_interval)

        def _make_keepalive_chunk() -> str:
            if fake_stream_on or is_image_request:
                keep_alive_data = {
                    "id": "chatcmpl-keepalive",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_obj.model,
                    "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}],
                }
                return f"data: {json.dumps(keep_alive_data, ensure_ascii=False)}\n\n"
            return ": keep-alive\n\n"

        async def _unified_stream_generator():
            nonlocal sandbox_downgraded
            any_output_yielded = False
            emitted_reasoning_any = False
            try:
                if not (fake_stream_on or is_image_request):
                    yield ": keep-alive\n\n"

                for attempt in range(max_retries + 1):
                    if await _client_gone():
                        print(f"ℹ️ [客户端断开] 流式请求前检测到客户端已断开，停止调用模型 {request_obj.model}。")
                        tracker.finalize_cancelled(partial_output=any_output_yielded)
                        return

                    target = _pick_next_target()
                    if target is None:
                        break

                    tracker.note_attempt_start(target)
                    has_yielded_output = False
                    has_yielded_visible_body = False
                    attempt_failed = False
                    err_code = 500
                    err_formatted = ""
                    err_retry_after: Optional[float] = None
                    is_retryable = False
                    is_auth_or_billing = False

                    # ---------- 1A. 当前轮次使用 Express Key ----------
                    if target["channel"] == "express":
                        ctx, actual_prompt, _ = await _get_express_ctx(no_sandbox=sandbox_downgraded)
                        spec = express_upstream.build_key_call_spec(ctx, target["key_record"])
                        current_client = spec["client"]
                        model_to_call = spec["model_to_call"]
                        fallback_model = spec["fallback_model"]
                        fallback_client_factory = spec["fallback_client_factory"]
                        gen_config_dict = ctx["gen_config_dict"]
                        prefill_text = ctx["prefill_text"]
                        exp_req_obj = ctx["request_obj"]
                        synth_tool_name = ctx.get("synthetic_tool_name", "")

                        print(f"🚀 [上游请求] 正在使用 [{target['label']}] 调用 {model_to_call}（请求模型={request_obj.model}）。")

                        if fake_stream_on or is_image_request:
                            if is_image_request and attempt == 0:
                                print("🖼️ [生图保护] 图片模型请求已自动切换为假流式输出。")
                            api_call_task: Optional[asyncio.Task] = None
                            sampler = _RawSampler()

                            async def _cancel_task():
                                nonlocal api_call_task
                                if api_call_task is not None and not api_call_task.done():
                                    api_call_task.cancel()
                                    try:
                                        await api_call_task
                                    except (asyncio.CancelledError, Exception):
                                        pass

                            try:
                                raw_gemini_response = None
                                unary_cfg = strip_stream_fc_args_from_config(gen_config_dict)
                                fake_call_t0 = time.monotonic()
                                for _pin_try in range(2):
                                    fake_call_t0 = time.monotonic()
                                    api_call_task = asyncio.create_task(
                                        current_client.aio.models.generate_content(
                                            model=model_to_call,
                                            contents=actual_prompt,
                                            config=unary_cfg,
                                        )
                                    )
                                    elapsed_hb = 0.0
                                    while not api_call_task.done():
                                        await asyncio.sleep(0.5)
                                        elapsed_hb += 0.5
                                        if await _client_gone():
                                            print("ℹ️ [客户端断开] 假流式等待期间客户端已断开，正在取消上游任务。")
                                            await _cancel_task()
                                            tracker.finalize_cancelled(partial_output=False)
                                            return
                                        if not api_call_task.done() and elapsed_hb >= outer_keep_alive_interval:
                                            elapsed_hb = 0.0
                                            yield _make_keepalive_chunk()
                                    try:
                                        raw_gemini_response = await api_call_task
                                        sampler.add(extract_gemini_chunk_debug_dict(raw_gemini_response))
                                        break
                                    except Exception as e_pin:
                                        if (
                                            _pin_try == 0
                                            and fallback_model
                                            and model_to_call != fallback_model
                                            and is_location_pin_failure(e_pin)
                                        ):
                                            _, pin_msg = extract_upstream_error(e_pin)
                                            print(f"↩️ [上游端点回退] [{target['label']}] 钉定路径 {model_to_call} 失败（{pin_msg[:600]}），"
                                                  f"立即退回默认路由 {fallback_model} 重试一次。")
                                            model_to_call = fallback_model
                                            current_client = fallback_client_factory()
                                            continue
                                        raise
                                fake_unary_elapsed = max(0.0, time.monotonic() - fake_call_t0)

                                if hasattr(raw_gemini_response, "prompt_feedback") and \
                                   hasattr(raw_gemini_response.prompt_feedback, "block_reason") and \
                                   raw_gemini_response.prompt_feedback.block_reason:
                                    print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                                    raise ValueError(_describe_empty_gemini_response(raw_gemini_response, model_to_call))

                                if not is_gemini_response_valid(
                                    raw_gemini_response,
                                    synthetic_tool_name=synth_tool_name,
                                    prefill_text=prefill_text,
                                ):
                                    print(f"🔎 [Express 诊断] 模型={model_to_call} 空正文或无效响应。原始响应样本：\n{sampler.dump()}")
                                    err_v = ValueError(_describe_empty_gemini_response(raw_gemini_response, model_to_call))
                                    if synth_tool_name:
                                        err_v.is_retryable = True
                                        if not sandbox_downgraded and is_malformed_tool_finish(raw_gemini_response):
                                            sandbox_downgraded = True
                                            print(f"🩹 [抗截断自愈] 模型 {model_to_call} 假流式触发工具解码异常，后续重试已自动剥离 compose_reply 切换为原生模式。")
                                    raise err_v

                                openai_response_dict = convert_to_openai_format(
                                    raw_gemini_response,
                                    exp_req_obj.model,
                                    synthetic_tool_name=synth_tool_name,
                                    prefill_text=prefill_text,
                                )
                                _ch0 = (openai_response_dict.get("choices") or [{}])[0]
                                _msg0 = _ch0.get("message") or {}
                                _raw_gen_chars = len(_msg0.get("content") or "")
                                usage_info = _record_usage(raw_gemini_response, visible_chars=_raw_gen_chars)
                                if usage_info["prompt_tokens"] or usage_info["completion_tokens"]:
                                    _u_fake: Dict[str, Any] = {
                                        "prompt_tokens": usage_info["prompt_tokens"],
                                        "completion_tokens": usage_info["completion_tokens"],
                                        "total_tokens": usage_info["total_tokens"],
                                    }
                                    if usage_info["thoughts_tokens"] > 0:
                                        _u_fake["completion_tokens_details"] = {"reasoning_tokens": usage_info["thoughts_tokens"]}
                                    openai_response_dict["usage"] = _u_fake
                                _prepend_prefill(openai_response_dict, prefill_text)

                                _fin0 = _ch0.get("finish_reason") or "stop"
                                _has_tc0 = bool(_msg0.get("tool_calls"))
                                _out_c0 = len(_msg0.get("content") or "")

                                async for chunk_sse in _chunk_openai_response_dict_for_sse(
                                    openai_response_dict=openai_response_dict,
                                    include_usage=wants_usage(exp_req_obj),
                                ):
                                    has_yielded_output = True
                                    any_output_yielded = True
                                    yield chunk_sse
                                if app_state.get_setting("debug_outbound", False):
                                    print(f"🔎 [Express 响应原始结构] {sampler.dump()}")
                                tracker.finalize_success(
                                    target,
                                    is_cookie=False,
                                    ttft=None,
                                    gen_time=fake_unary_elapsed,
                                    finish_reason=_fin0,
                                    has_tool_calls=_has_tc0,
                                    has_usage=bool(usage_info["prompt_tokens"] or usage_info["completion_tokens"]),
                                    p_tk=usage_info["prompt_tokens"],
                                    th_tk=usage_info["thoughts_tokens"],
                                    out_tk=usage_info["output_tokens"],
                                    out_chars=_out_c0,
                                )
                                print(f"✅ [Express] {request_obj.model} | 假流式传输完成")
                                return

                            except asyncio.CancelledError:
                                await _cancel_task()
                                tracker.finalize_cancelled(partial_output=any_output_yielded)
                                raise
                            except Exception as e_fake:
                                attempt_failed = True
                                err_code, err_formatted = extract_upstream_error(e_fake)
                                err_retry_after = extract_retry_after_seconds(e_fake, err_formatted)
                                is_auth_or_billing = is_express_auth_or_billing_error(e_fake, err_code)
                                is_retryable = is_retryable_exception(e_fake)

                        else:
                            # Express 真流式
                            response_id_for_stream = make_bad_woman_response_id(
                                "chatcmpl-realstream", sandbox_enabled=bool(synth_tool_name)
                            )
                            stream_start_ts = time.time()
                            sampler = _RawSampler()
                            tool_indexer: Optional[ToolCallIndexer] = None
                            deduper: Optional[PrefillDeduper] = None
                            prefill_emitted: bool = not bool(prefill_text)
                            emitted_finish_reason: bool = False
                            try:
                                active_stream_cfg = gen_config_dict
                                for _pin_try in range(4):
                                    stream_gen_obj = None
                                    received_any_stream_chunk = False
                                    suppress_dup_reasoning = emitted_reasoning_any
                                    try:
                                        stream_attempt_t0 = time.monotonic()
                                        first_tok_mono: Optional[float] = None
                                        saw_thought_stream = False
                                        stream_gen_obj = await current_client.aio.models.generate_content_stream(
                                            model=model_to_call,
                                            contents=actual_prompt,
                                            config=active_stream_cfg,
                                        )
                                        tool_indexer = ToolCallIndexer(
                                            synthetic_tool_name=synth_tool_name,
                                            prefill_text=prefill_text,
                                        )
                                        deduper = PrefillDeduper(prefill_text) if (prefill_text and (exp_req_obj.n or 1) == 1) else None
                                        prefill_emitted = not bool(prefill_text)
                                        emitted_finish_reason = False
                                        saw_upstream_finish_reason = False
                                        silent_eof_truncated = False
                                        last_finish_reason_str = "stop"
                                        last_stream_chunk_obj = None
                                        raw_p_max, raw_th_max, raw_out_max, raw_tool_p_max, raw_tot_max = 0, 0, 0, 0, 0

                                        async for chunk_item_call in stream_gen_obj:
                                            received_any_stream_chunk = True
                                            last_stream_chunk_obj = chunk_item_call
                                            for _cand_fr_chk in (getattr(chunk_item_call, "candidates", None) or []):
                                                _fr_val = getattr(_cand_fr_chk, "finish_reason", None)
                                                if _fr_val is not None:
                                                    _fr_nm = str(getattr(_fr_val, "name", _fr_val) or "").strip().upper()
                                                    if _fr_nm and not _fr_nm.endswith("UNSPECIFIED"):
                                                        saw_upstream_finish_reason = True
                                                        break
                                            _has_m_tok, _has_m_th = _chunk_has_model_token(
                                                chunk_item_call, include_thoughts=tracker.include_thoughts
                                            )
                                            if _has_m_tok and first_tok_mono is None:
                                                first_tok_mono = time.monotonic()
                                            if _has_m_th:
                                                saw_thought_stream = True
                                            sampler.add(extract_gemini_chunk_debug_dict(chunk_item_call))
                                            if getattr(chunk_item_call, "usage_metadata", None):
                                                _up, _uth, _uout, _utp, _utot = _extract_raw_usage_fields(chunk_item_call)
                                                if _up > raw_p_max:
                                                    raw_p_max = _up
                                                if _uth > raw_th_max:
                                                    raw_th_max = _uth
                                                if _uout > raw_out_max:
                                                    raw_out_max = _uout
                                                if _utp > raw_tool_p_max:
                                                    raw_tool_p_max = _utp
                                                if _utot > raw_tot_max:
                                                    raw_tot_max = _utot

                                            pf = getattr(chunk_item_call, "prompt_feedback", None)
                                            if pf and getattr(pf, "block_reason", None) and not has_yielded_visible_body:
                                                print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                                                raise ValueError(_describe_empty_gemini_response(chunk_item_call, model_to_call))

                                            # 若尚未向客户端产出任何可见正文，但合成工具首帧/思考后立即撞上 MALFORMED_FUNCTION_CALL：
                                            # 拦截 premature finish 帧，0 等待就地剥离 compose_reply 无感回退原生流式！
                                            if (
                                                synth_tool_name
                                                and not sandbox_downgraded
                                                and not has_yielded_visible_body
                                                and not tool_indexer.has_received_synthetic_text(0)
                                                and is_malformed_tool_finish(chunk_item_call)
                                            ):
                                                _cand_chk = (getattr(chunk_item_call, "candidates", None) or [None])[0]
                                                _parts_chk = getattr(getattr(_cand_chk, "content", None), "parts", None) or []
                                                _has_inline_body = any(
                                                    (getattr(p, "function_call", None) is not None and (
                                                        getattr(p.function_call, "args", None) or getattr(p.function_call, "partial_args", None)
                                                    ))
                                                    or (not getattr(p, "thought", None) and getattr(p, "text", None))
                                                    for p in _parts_chk
                                                )
                                                if not _has_inline_body:
                                                    raise _MalformedSandboxFallback()

                                            num_candidates = len(chunk_item_call.candidates) if getattr(chunk_item_call, "candidates", None) else 1
                                            for ci in range(num_candidates):
                                                sse_chunk = convert_chunk_to_openai(
                                                    chunk_item_call, exp_req_obj.model, response_id_for_stream, ci,
                                                    indexer=tool_indexer,
                                                )
                                                if not has_yielded_visible_body and '"finish_reason": "content_filter"' in sse_chunk:
                                                    print(f"🔎 [Express 诊断] 模型={model_to_call} 触发安全过滤。原始响应样本：\n{sampler.dump()}")
                                                    raise ValueError(_describe_empty_gemini_response(chunk_item_call, model_to_call))

                                                if suppress_dup_reasoning:
                                                    sse_chunk = _strip_sse_reasoning_chunk(sse_chunk)
                                                    if sse_chunk is None:
                                                        continue

                                                if deduper is not None:
                                                    sse_chunk = _dedup_sse_chunk_content(sse_chunk, deduper)
                                                    if sse_chunk is None:
                                                        continue

                                                has_vis_in_chunk = _sse_chunk_has_visible_body(sse_chunk)
                                                has_th_in_chunk = ('"reasoning_content":' in sse_chunk)
                                                is_real_chunk = has_vis_in_chunk or has_th_in_chunk
                                                if is_real_chunk and not has_yielded_output:
                                                    has_yielded_output = True
                                                    any_output_yielded = True
                                                    print(f"⚡ [Express] {request_obj.model} | 连接建立，正在实时流式输出...")
                                                elif not has_yielded_output:
                                                    continue

                                                if has_th_in_chunk:
                                                    emitted_reasoning_any = True
                                                if has_vis_in_chunk:
                                                    has_yielded_visible_body = True

                                                if not prefill_emitted and '"content":' in sse_chunk and '"content": ""' not in sse_chunk and '"content": null' not in sse_chunk:
                                                    prefill_emitted = True
                                                    _pf = {
                                                        "id": response_id_for_stream,
                                                        "object": "chat.completion.chunk",
                                                        "created": int(time.time()),
                                                        "model": exp_req_obj.model,
                                                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}],
                                                    }
                                                    yield f"data: {json.dumps(_pf, ensure_ascii=False)}\n\n"
                                                if '"finish_reason": null' not in sse_chunk and '"finish_reason":' in sse_chunk:
                                                    emitted_finish_reason = True
                                                    if '"finish_reason": "length"' in sse_chunk:
                                                        last_finish_reason_str = "length"
                                                    elif '"finish_reason": "tool_calls"' in sse_chunk:
                                                        last_finish_reason_str = "tool_calls"
                                                yield sse_chunk
                                            if tool_indexer.has_synthetic_halted(0):
                                                _h_reason = tool_indexer.synthetic_halt_reason(0) or "闭合溢出"
                                                print(f"🛑 [抗截断保护] 检测到 {model_to_call} 触发熔断（{_h_reason}），已平滑切除并立即断开上游流。")
                                                last_finish_reason_str = "tool_calls" if tool_indexer.has_tool_calls(0) else "stop"
                                                break

                                        stream_eof_mono = time.monotonic()
                                        rem_calls = tool_indexer.flush_function_calls(0)
                                        rem_synth_parts = []
                                        _tail_synth = tool_indexer.flush_synthetic_tail(0)
                                        _tail_sig = tool_indexer.pop_synth_signature(0)
                                        if _tail_synth or _tail_sig:
                                            from message_processing import ordinary_part_metadata as _ord_meta_fn
                                            tool_indexer.record_ordinary_part(0, _ord_meta_fn("text", _tail_synth, _tail_sig))
                                        if _tail_synth:
                                            rem_synth_parts.append(_tail_synth)
                                        rem_tool_deltas = []
                                        for c_item in rem_calls:
                                            if synth_tool_name and c_item.get("name") == synth_tool_name:
                                                _extra_synth = extract_synthetic_content(
                                                    c_item.get("args"),
                                                    emit_tool_name=synth_tool_name,
                                                    prefill_text=prefill_text,
                                                )
                                                if _extra_synth:
                                                    from message_processing import ordinary_part_metadata as _ord_meta_fn
                                                    tool_indexer.record_ordinary_part(
                                                        0, _ord_meta_fn("text", _extra_synth, c_item.get("thought_signature"))
                                                    )
                                                    rem_synth_parts.append(_extra_synth)
                                            else:
                                                tc_idx = tool_indexer.next_index(0)
                                                rem_tool_deltas.append(
                                                    _format_assembled_tool_call(c_item, tc_idx, indexer=tool_indexer, candidate_index=0)
                                                )
                                        rem_synth = "".join(rem_synth_parts)
                                        if rem_synth or _tail_sig:
                                            if rem_synth and deduper is not None:
                                                rem_synth = deduper.feed(rem_synth)
                                            if rem_synth or _tail_sig:
                                                if rem_synth:
                                                    has_yielded_visible_body = True
                                                    if not has_yielded_output:
                                                        has_yielded_output = True
                                                        any_output_yielded = True
                                                        print(f"⚡ [Express] {request_obj.model} | 连接建立，正在实时流式输出...")
                                                if rem_synth and not prefill_emitted:
                                                    prefill_emitted = True
                                                    _pf = {
                                                        "id": response_id_for_stream,
                                                        "object": "chat.completion.chunk",
                                                        "created": int(time.time()),
                                                        "model": exp_req_obj.model,
                                                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}],
                                                    }
                                                    yield f"data: {json.dumps(_pf, ensure_ascii=False)}\n\n"
                                                _rem_delta: Dict[str, Any] = {}
                                                if rem_synth:
                                                    _rem_delta["content"] = rem_synth
                                                if _rem_delta and has_yielded_output:
                                                    _rem_chunk = {
                                                        "id": response_id_for_stream,
                                                        "object": "chat.completion.chunk",
                                                        "created": int(time.time()),
                                                        "model": exp_req_obj.model,
                                                        "choices": [{"index": 0, "delta": _rem_delta, "finish_reason": None}],
                                                    }
                                                    yield f"data: {json.dumps(_rem_chunk, ensure_ascii=False)}\n\n"

                                        if deduper is not None and not deduper.done:
                                            tail = deduper.flush()
                                            if tail:
                                                has_yielded_visible_body = True
                                                if not has_yielded_output:
                                                    has_yielded_output = True
                                                    any_output_yielded = True
                                                    print(f"⚡ [Express] {request_obj.model} | 连接建立，正在实时流式输出...")
                                                if not prefill_emitted:
                                                    prefill_emitted = True
                                                    _pf = {
                                                        "id": response_id_for_stream,
                                                        "object": "chat.completion.chunk",
                                                        "created": int(time.time()),
                                                        "model": exp_req_obj.model,
                                                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}],
                                                    }
                                                    yield f"data: {json.dumps(_pf, ensure_ascii=False)}\n\n"
                                                _tail = {
                                                    "id": response_id_for_stream,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": exp_req_obj.model,
                                                    "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                                                }
                                                yield f"data: {json.dumps(_tail, ensure_ascii=False)}\n\n"

                                        if rem_tool_deltas:
                                            has_yielded_visible_body = True
                                            first_flush_tc = not has_yielded_output
                                            if not has_yielded_output:
                                                has_yielded_output = True
                                                any_output_yielded = True
                                                print(f"⚡ [Express] {request_obj.model} | 连接建立，正在实时流式输出...")
                                            delta_tc: Dict[str, Any] = {"tool_calls": rem_tool_deltas, "content": None}
                                            if first_flush_tc:
                                                delta_tc["role"] = "assistant"
                                            _g_tc = tool_indexer.build_google_extra(0)
                                            if _g_tc:
                                                delta_tc["extra_content"] = {"google": _g_tc}
                                                tool_indexer.mark_extra_emitted(0)
                                            _eof_tc_chunk = {
                                                "id": response_id_for_stream,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": exp_req_obj.model,
                                                "choices": [{"index": 0, "delta": delta_tc, "finish_reason": None}],
                                            }
                                            yield f"data: {json.dumps(_eof_tc_chunk, ensure_ascii=False)}\n\n"

                                        # 真流式全量空正文拦截：严格检查 has_yielded_visible_body（即使前面已吐出过 reasoning_content 思考块，只要可见正文/工具为 0 字，绝不当成成功结束！）
                                        if not has_yielded_visible_body:
                                            if synth_tool_name and not sandbox_downgraded:
                                                raise _MalformedSandboxFallback()
                                            print(f"🔎 [Express 诊断] 模型={model_to_call} 真流式未产出任何可见正文。原始响应样本：\n{sampler.dump()}")
                                            desc_empty = _describe_empty_gemini_response(last_stream_chunk_obj, model_to_call)
                                            err_empty_stream = ValueError(desc_empty)
                                            if synth_tool_name:
                                                err_empty_stream.is_retryable = True
                                            raise err_empty_stream

                                        if has_yielded_output and not emitted_finish_reason:
                                            if tool_indexer.has_tool_calls(0):
                                                _fin_reason = "tool_calls"
                                            elif tool_indexer.has_synthetic_halted(0):
                                                _fin_reason = "stop"
                                            elif not saw_upstream_finish_reason:
                                                # 上游未下发任何 finish_reason 即静默结束流：按截断上报 finish_reason=length，避免客户端误以为完整收尾
                                                _fin_reason = "length"
                                                silent_eof_truncated = True
                                            else:
                                                _fin_reason = "stop"
                                            last_finish_reason_str = _fin_reason
                                            _fin_delta_eof: Dict[str, Any] = {}
                                            if not tool_indexer.has_emitted_extra(0):
                                                _g_eof = tool_indexer.build_google_extra(0)
                                                if _g_eof:
                                                    _fin_delta_eof["extra_content"] = {"google": _g_eof}
                                                    tool_indexer.mark_extra_emitted(0)
                                            _fin_chunk = {
                                                "id": response_id_for_stream,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": exp_req_obj.model,
                                                "choices": [{"index": 0, "delta": _fin_delta_eof, "finish_reason": _fin_reason}],
                                            }
                                            yield f"data: {json.dumps(_fin_chunk, ensure_ascii=False)}\n\n"
                                        elif has_yielded_output and not tool_indexer.has_emitted_extra(0):
                                            _g_late = tool_indexer.build_google_extra(0)
                                            if _g_late.get("thought_signature"):
                                                tool_indexer.mark_extra_emitted(0)
                                                _sig_chunk = {
                                                    "id": response_id_for_stream,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": exp_req_obj.model,
                                                    "choices": [{"index": 0, "delta": {"extra_content": {"google": _g_late}}, "finish_reason": None}],
                                                }
                                                yield f"data: {json.dumps(_sig_chunk, ensure_ascii=False)}\n\n"

                                        _ord_items, _ = tool_indexer.topology(0)
                                        _gen_only_chars = sum(
                                            len(it.get("text") or "") for it in _ord_items if it.get("kind") == "text"
                                        )
                                        _stream_chars = (len(prefill_text) if prefill_text else 0) + _gen_only_chars

                                        final_p_tk, final_th_tk, final_out_tk, final_t_tk = _reconcile_usage_counts(
                                            raw_p_max,
                                            raw_th_max,
                                            raw_out_max,
                                            raw_tool_p_max,
                                            raw_tot_max,
                                            visible_chars=_gen_only_chars,
                                        )
                                        final_comp_tk = final_th_tk + final_out_tk
                                        if final_p_tk > 0 or final_comp_tk > 0:
                                            stats.add_tokens(final_p_tk, final_comp_tk)
                                            print(f"💰 [算力消耗统计] 输入: {final_p_tk} | 思考: {final_th_tk} | 正文: {final_out_tk} | 总计: {final_t_tk} Tokens")
                                        else:
                                            stats.add_success()

                                        if wants_usage(exp_req_obj):
                                            _u_dict: Dict[str, Any] = {
                                                "prompt_tokens": final_p_tk,
                                                "completion_tokens": final_comp_tk,
                                                "total_tokens": final_t_tk,
                                            }
                                            if final_th_tk > 0:
                                                _u_dict["completion_tokens_details"] = {"reasoning_tokens": final_th_tk}
                                            yield make_usage_chunk(response_id_for_stream, exp_req_obj.model, _u_dict)

                                        yield "data: [DONE]\n\n"
                                        elapsed = time.time() - stream_start_ts
                                        if app_state.get_setting("debug_outbound", False):
                                            print(f"🔎 [Express 尾包原始结构] {sampler.dump()}")
                                        _s_ttft = max(0.0, first_tok_mono - stream_attempt_t0) if first_tok_mono is not None else None
                                        _s_gen = (
                                            max(0.0, stream_eof_mono - first_tok_mono)
                                            if first_tok_mono is not None
                                            else max(0.0, stream_eof_mono - stream_attempt_t0)
                                        )
                                        tracker.finalize_success(
                                            target,
                                            is_cookie=False,
                                            ttft=_s_ttft,
                                            gen_time=_s_gen,
                                            saw_thought_stream=saw_thought_stream,
                                            finish_reason=("tool_calls" if tool_indexer.has_tool_calls(0) else last_finish_reason_str),
                                            has_tool_calls=tool_indexer.has_tool_calls(0),
                                            has_usage=bool(final_p_tk > 0 or final_comp_tk > 0),
                                            p_tk=final_p_tk,
                                            th_tk=final_th_tk,
                                            out_tk=final_out_tk,
                                            out_chars=_stream_chars,
                                        )
                                        if tool_indexer.has_synthetic_halted(0):
                                            _h_fin = tool_indexer.synthetic_halt_reason(0) or "闭合溢出"
                                            print(f"⚠️ [Express] {request_obj.model} | 抗截断保护提前收口（{_h_fin}）| 耗时 {elapsed:.1f}s")
                                        elif silent_eof_truncated:
                                            print(f"⚠️ [Express] {request_obj.model} | 上游静默断流（未收到 finish_reason，正文可能不完整），已标 finish_reason=length | 耗时 {elapsed:.1f}s")
                                        elif last_finish_reason_str == "length":
                                            print(f"⚠️ [Express] {request_obj.model} | 流式传输结束（finish_reason=length，已达上限或截断）| 耗时 {elapsed:.1f}s")
                                        else:
                                            print(f"✅ [Express] {request_obj.model} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                                        return
                                    except _MalformedSandboxFallback:
                                        sandbox_downgraded = True
                                        ctx_fb, prompt_fb, _ = await _get_express_ctx(no_sandbox=True)
                                        if ctx_fb is not None and prompt_fb is not None:
                                            actual_prompt = prompt_fb
                                            gen_config_dict = ctx_fb["gen_config_dict"]
                                            active_stream_cfg = gen_config_dict
                                            prefill_text = ctx_fb["prefill_text"]
                                            exp_req_obj = ctx_fb["request_obj"]
                                            synth_tool_name = ""
                                            print(
                                                f"🩹 [抗截断自愈] 模型 {model_to_call} 在未产出可见正文前触发工具解码异常，"
                                                "0 等待剥离 compose_reply 无感回退原生流式重连..."
                                            )
                                            continue
                                        raise ValueError(_describe_empty_gemini_response(last_stream_chunk_obj, model_to_call))
                                    except Exception as e_pin_stream:
                                        _stripped_s_cfg = strip_stream_fc_args_from_config(active_stream_cfg)
                                        if (
                                            not has_yielded_visible_body
                                            and not received_any_stream_chunk
                                            and _stripped_s_cfg is not active_stream_cfg
                                            and is_stream_fc_args_unsupported_error(e_pin_stream)
                                        ):
                                            print(
                                                f"⚠️ [抗截断兼容] 模型 {model_to_call} 不支持 stream_function_call_arguments，"
                                                "已自动剥离分片参数标志并保持 compose_reply 重试。"
                                            )
                                            active_stream_cfg = _stripped_s_cfg
                                            continue
                                        if (
                                            _pin_try == 0
                                            and not has_yielded_visible_body
                                            and fallback_model
                                            and model_to_call != fallback_model
                                            and is_location_pin_failure(e_pin_stream)
                                        ):
                                            _, pin_msg = extract_upstream_error(e_pin_stream)
                                            print(f"↩️ [上游端点回退] [{target['label']}] 钉定路径 {model_to_call} 失败（{pin_msg[:600]}），"
                                                  f"立即退回默认路由 {fallback_model} 重试一次。")
                                            model_to_call = fallback_model
                                            current_client = fallback_client_factory()
                                            continue
                                        raise
                                    finally:
                                        if stream_gen_obj is not None and hasattr(stream_gen_obj, "aclose"):
                                            try:
                                                await stream_gen_obj.aclose()
                                            except Exception:
                                                pass
                            except asyncio.CancelledError:
                                print(f"ℹ️ [客户端断开] 真流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                                tracker.finalize_cancelled(partial_output=any_output_yielded)
                                raise
                            except Exception as e_stream_call:
                                err_code, err_formatted = extract_upstream_error(e_stream_call)
                                err_retry_after = extract_retry_after_seconds(e_stream_call, err_formatted)
                                if has_yielded_output:
                                    # 流式尾跳或中途异常中断前，强制冲刷滑动尾窗（96~640字）中已接收的状态栏文本与闭合标签，防止被静默活埋！
                                    if tool_indexer is not None:
                                        _salvage_parts: List[str] = []
                                        _s_tail = tool_indexer.flush_synthetic_tail(0)
                                        if _s_tail:
                                            _salvage_parts.append(_s_tail)
                                        for _sc in tool_indexer.flush_function_calls(0):
                                            if synth_tool_name and _sc.get("name") == synth_tool_name:
                                                _ex_s = extract_synthetic_content(
                                                    _sc.get("args"), emit_tool_name=synth_tool_name, prefill_text=prefill_text
                                                )
                                                if _ex_s:
                                                    _salvage_parts.append(_ex_s)
                                        _salvage_txt = "".join(_salvage_parts)
                                        if deduper is not None:
                                            if _salvage_txt:
                                                _salvage_txt = deduper.feed(_salvage_txt)
                                            if not deduper.done:
                                                _salvage_txt += deduper.flush()
                                        if _salvage_txt:
                                            if not prefill_emitted and prefill_text:
                                                prefill_emitted = True
                                                yield f"data: {json.dumps({'id': response_id_for_stream, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': exp_req_obj.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': prefill_text}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                                            yield f"data: {json.dumps({'id': response_id_for_stream, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': exp_req_obj.model, 'choices': [{'index': 0, 'delta': {'content': _salvage_txt}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                                            print(f"🛟 [尾窗抢救] 流式异常中断前已成功冲刷挽回滑动尾窗内剩余的 {len(_salvage_txt)} 字符。")
                                    if not emitted_finish_reason:
                                        yield f"data: {json.dumps({'id': response_id_for_stream, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': exp_req_obj.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]}, ensure_ascii=False)}\n\n"
                                    stats.add_error()
                                    tracker.note_attempt_failure(target, err_code)
                                    tracker.finalize_error(err_code, f"流式传输中途中断：{err_formatted}")
                                    print(f"❌ [Express 流式中断报错] 凭证=[{target['label']}] | 模型={request_obj.model} | HTTP {err_code} | {err_formatted[:2000]}")
                                    yield "data: [DONE]\n\n"
                                    return
                                attempt_failed = True
                                is_auth_or_billing = is_express_auth_or_billing_error(e_stream_call, err_code)
                                is_retryable = is_retryable_exception(e_stream_call)

                    # ---------- 1B. 当前轮次使用 Studio Cookie 直连 ----------
                    else:
                        print(f"🚀 [上游请求] 正在使用 [{target['label']}] 调用模型 {request_obj.model}。")
                        cookie_iter = None
                        cookie_tel: Dict[str, Any] = {}
                        try:
                            cookie_resp = await cookie_upstream.execute_once(
                                request_obj, fastapi_request, telemetry=cookie_tel
                            )
                            tracker.sync_from_cookie_telemetry(cookie_tel)
                            cookie_iter = cookie_resp.body_iterator
                            async for chunk_item in cookie_iter:
                                chunk_str = chunk_item.decode("utf-8", errors="replace") if isinstance(chunk_item, bytes) else str(chunk_item)
                                if chunk_str.strip().startswith("data:") and '"chatcmpl-keepalive"' not in chunk_str:
                                    has_yielded_output = True
                                    any_output_yielded = True
                                yield chunk_str
                            tracker.sync_from_cookie_telemetry(cookie_tel)
                            if cookie_tel.get("cancelled"):
                                tracker.finalize_cancelled(partial_output=any_output_yielded)
                            elif cookie_tel.get("midstream_error"):
                                m_code = int(cookie_tel.get("error_code") or 502)
                                tracker.note_attempt_failure(target, m_code)
                                tracker.finalize_error(m_code, f"Studio 流式中途报错：{cookie_tel['midstream_error']}")
                            else:
                                tracker.finalize_success(
                                    target,
                                    is_cookie=True,
                                    ttft=cookie_tel.get("ttft"),
                                    gen_time=float(cookie_tel.get("gen_time") or 0.0),
                                    saw_thought_stream=bool(cookie_tel.get("saw_thought_stream")),
                                    finish_reason=str(cookie_tel.get("finish_reason") or "stop"),
                                    has_tool_calls=bool(cookie_tel.get("tool_calls")),
                                    has_usage=bool(cookie_tel.get("has_usage")),
                                    p_tk=int(cookie_tel.get("p_tk") or 0),
                                    th_tk=int(cookie_tel.get("th_tk") or 0),
                                    out_tk=int(cookie_tel.get("out_tk") or 0),
                                    out_chars=int(cookie_tel.get("out_chars") or 0),
                                )
                            return
                        except asyncio.CancelledError:
                            tracker.sync_from_cookie_telemetry(cookie_tel)
                            tracker.finalize_cancelled(partial_output=any_output_yielded)
                            raise
                        except Exception as e_cookie:
                            tracker.sync_from_cookie_telemetry(cookie_tel)
                            err_code = getattr(e_cookie, "status_code", None) or getattr(e_cookie, "code", None) or 500
                            err_formatted = getattr(e_cookie, "message", None) or str(e_cookie)
                            err_retry_after = extract_retry_after_seconds(e_cookie, err_formatted)
                            if has_yielded_output:
                                stats.add_error()
                                tracker.note_attempt_failure(target, err_code)
                                tracker.finalize_error(err_code, f"Studio 流式中途报错：{err_formatted}")
                                print(f"❌ [Studio 流式中断报错] 凭证=[{target['label']}] | 模型={request_obj.model} | {err_formatted[:2000]}")
                                yield "data: [DONE]\n\n"
                                return
                            attempt_failed = True
                            is_auth_or_billing = bool(getattr(e_cookie, "is_auth_or_billing", False))
                            is_retryable = bool(getattr(e_cookie, "is_retryable", False))
                        finally:
                            if cookie_iter is not None and hasattr(cookie_iter, "aclose"):
                                try:
                                    await cookie_iter.aclose()
                                except Exception:
                                    pass

                    # ---------- 1C. 统一处理流式首包前的失败与多凭证退避轮换 ----------
                    if attempt_failed:
                        if is_auth_or_billing:
                            dead_targets.add(target["id"])
                            tip = _emit_credential_invalid_warning(target, err_code, err_formatted)
                            if "【凭证更换提醒】" not in err_formatted:
                                err_formatted = err_formatted + tip

                        next_target = _peek_next_target()
                        can_retry = (
                            attempt < max_retries
                            and next_target is not None
                            and (is_retryable or is_auth_or_billing)
                        )
                        if can_retry:
                            if await _client_gone():
                                print(f"ℹ️ [客户端断开] 重试前检测到客户端已断开，停止调用模型 {request_obj.model}。")
                                tracker.finalize_cancelled(partial_output=False)
                                return
                            stats.add_retry()
                            wait_target = _compute_smart_retry_wait(
                                backoff_sec, target, next_target, err_retry_after
                            )
                            tracker.note_attempt_failure(target, err_code, wait_sec=wait_target)
                            print(
                                f"⚠️ [自动重试] 流式模型={request_obj.model} | 凭证 [{target['label']}] 失败 (HTTP {err_code}) | "
                                f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {wait_target}s 后将切换至 [{next_target['label']}] | "
                                f"原因: {err_formatted[:1500]}"
                            )
                            waited = 0.0
                            hb_acc = 0.0
                            hb_step_target = outer_keep_alive_interval if (fake_stream_on or is_image_request) else 3.0
                            while waited < wait_target:
                                step = min(0.5, wait_target - waited)
                                await asyncio.sleep(step)
                                waited += step
                                hb_acc += step
                                if await _client_gone():
                                    print(f"ℹ️ [客户端断开] 退避等待期间客户端已断开，停止重试模型 {request_obj.model}。")
                                    tracker.finalize_cancelled(partial_output=False)
                                    return
                                if hb_acc >= hb_step_target:
                                    hb_acc = 0.0
                                    yield _make_keepalive_chunk()
                            continue

                        tracker.note_attempt_failure(target, err_code)
                        tracker.finalize_error(err_code, err_formatted)
                        stats.add_error()
                        print(f"❌ [流式请求失败] 模型={request_obj.model} | 凭证=[{target['label']}] | HTTP {err_code} | {err_formatted[:2000]}")
                        err_resp = create_openai_error_response(err_code, err_formatted[:2000], "upstream_error")
                        yield f"data: {json.dumps(err_resp, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            finally:
                if not tracker.finalized:
                    tracker.finalize_cancelled(partial_output=any_output_yielded)

        return StreamingResponse(_unified_stream_generator(), media_type="text/event-stream")

    # ============================================================
    # 分支二：非流式请求（支持多 Key 轮询与跨通道 Cookie 互备）
    # ============================================================
    last_err_code = 502
    last_err_formatted = f"模型 {request_obj.model} 上游未返回任何响应（重试已耗尽）。"

    for attempt in range(max_retries + 1):
        if await _client_gone():
            print(f"ℹ️ [客户端断开] 非流式请求前检测到客户端已断开，停止调用模型 {request_obj.model}。")
            tracker.finalize_cancelled(partial_output=False)
            return JSONResponse(
                status_code=499,
                content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
            )

        target = _pick_next_target()
        if target is None:
            break

        tracker.note_attempt_start(target)
        err_code = 500
        err_formatted = ""
        err_retry_after: Optional[float] = None
        is_retryable = False
        is_auth_or_billing = False

        # ---------- 2A. 当前轮次使用 Express Key ----------
        if target["channel"] == "express":
            ctx, actual_prompt, _ = await _get_express_ctx(no_sandbox=sandbox_downgraded)
            spec = express_upstream.build_key_call_spec(ctx, target["key_record"])
            current_client = spec["client"]
            model_to_call = spec["model_to_call"]
            fallback_model = spec["fallback_model"]
            fallback_client_factory = spec["fallback_client_factory"]
            gen_config_dict = ctx["gen_config_dict"]
            prefill_text = ctx["prefill_text"]
            exp_req_obj = ctx["request_obj"]
            synth_tool_name = ctx.get("synthetic_tool_name", "")

            print(f"🚀 [上游请求] 正在使用 [{target['label']}] 调用 {model_to_call}（请求模型={request_obj.model}）。")

            try:
                response_obj_call = None
                sampler = _RawSampler()
                unary_cfg = strip_stream_fc_args_from_config(gen_config_dict)
                ns_call_t0 = time.monotonic()
                for _pin_try in range(2):
                    ns_call_t0 = time.monotonic()
                    call_task = asyncio.create_task(
                        current_client.aio.models.generate_content(
                            model=model_to_call,
                            contents=actual_prompt,
                            config=unary_cfg,
                        )
                    )
                    try:
                        while not call_task.done():
                            await asyncio.sleep(0.5)
                            if await _client_gone():
                                print(f"ℹ️ [客户端断开] 非流式等待期间客户端已断开，取消调用模型 {model_to_call}。")
                                call_task.cancel()
                                try:
                                    await call_task
                                except (asyncio.CancelledError, Exception):
                                    pass
                                tracker.finalize_cancelled(partial_output=False)
                                return JSONResponse(
                                    status_code=499,
                                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                                )
                        response_obj_call = await call_task
                        sampler.add(extract_gemini_chunk_debug_dict(response_obj_call))
                        break
                    except asyncio.CancelledError:
                        if not call_task.done():
                            call_task.cancel()
                        tracker.finalize_cancelled(partial_output=False)
                        raise
                    except Exception as e_pin_call:
                        if (
                            _pin_try == 0
                            and fallback_model
                            and model_to_call != fallback_model
                            and is_location_pin_failure(e_pin_call)
                        ):
                            _, pin_msg = extract_upstream_error(e_pin_call)
                            print(f"↩️ [上游端点回退] [{target['label']}] 钉定路径 {model_to_call} 失败（{pin_msg[:600]}），"
                                  f"立即退回默认路由 {fallback_model} 重试一次。")
                            model_to_call = fallback_model
                            current_client = fallback_client_factory()
                            continue
                        raise
                ns_unary_elapsed = max(0.0, time.monotonic() - ns_call_t0)

                if hasattr(response_obj_call, "prompt_feedback") and \
                   hasattr(response_obj_call.prompt_feedback, "block_reason") and \
                   response_obj_call.prompt_feedback.block_reason:
                    print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                    raise ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))

                if not is_gemini_response_valid(
                    response_obj_call,
                    synthetic_tool_name=synth_tool_name,
                    prefill_text=prefill_text,
                ):
                    print(f"🔎 [Express 诊断] 模型={model_to_call} 空正文或无效响应。原始响应样本：\n{sampler.dump()}")
                    err_v = ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))
                    if synth_tool_name:
                        err_v.is_retryable = True
                        if not sandbox_downgraded and is_malformed_tool_finish(response_obj_call):
                            sandbox_downgraded = True
                            print(f"🩹 [抗截断自愈] 模型 {model_to_call} 非流式触发工具解码异常，后续重试已自动剥离 compose_reply 切换为原生模式。")
                    raise err_v

                openai_response_content = convert_to_openai_format(
                    response_obj_call,
                    exp_req_obj.model,
                    synthetic_tool_name=synth_tool_name,
                    prefill_text=prefill_text,
                )
                _ns_ch0 = (openai_response_content.get("choices") or [{}])[0]
                _ns_msg0 = _ns_ch0.get("message") or {}
                _ns_raw_gen_chars = len(_ns_msg0.get("content") or "")
                usage_info_ns = _record_usage(response_obj_call, visible_chars=_ns_raw_gen_chars)
                if "usage" in openai_response_content:
                    openai_response_content["usage"]["prompt_tokens"] = usage_info_ns["prompt_tokens"]
                    openai_response_content["usage"]["completion_tokens"] = usage_info_ns["completion_tokens"]
                    openai_response_content["usage"]["total_tokens"] = usage_info_ns["total_tokens"]
                    if usage_info_ns["thoughts_tokens"] > 0:
                        openai_response_content["usage"]["completion_tokens_details"] = {
                            "reasoning_tokens": usage_info_ns["thoughts_tokens"]
                        }
                _prepend_prefill(openai_response_content, prefill_text)
                if app_state.get_setting("debug_outbound", False):
                    print(f"🔎 [Express 响应原始结构] {sampler.dump()}")
                _ns_fin0 = _ns_ch0.get("finish_reason") or "stop"
                tracker.finalize_success(
                    target,
                    is_cookie=False,
                    ttft=None,
                    gen_time=ns_unary_elapsed,
                    finish_reason=_ns_fin0,
                    has_tool_calls=bool(_ns_msg0.get("tool_calls")),
                    has_usage=bool(usage_info_ns["prompt_tokens"] or usage_info_ns["completion_tokens"]),
                    p_tk=usage_info_ns["prompt_tokens"],
                    th_tk=usage_info_ns["thoughts_tokens"],
                    out_tk=usage_info_ns["output_tokens"],
                    out_chars=len(_ns_msg0.get("content") or ""),
                )
                print(f"✅ [Express] {request_obj.model} | 非流式响应完成")
                return JSONResponse(content=openai_response_content)

            except asyncio.CancelledError:
                tracker.finalize_cancelled(partial_output=False)
                raise
            except Exception as e_call:
                err_code, err_formatted = extract_upstream_error(e_call)
                err_retry_after = extract_retry_after_seconds(e_call, err_formatted)
                is_auth_or_billing = is_express_auth_or_billing_error(e_call, err_code)
                is_retryable = is_retryable_exception(e_call)

        # ---------- 2B. 当前轮次使用 Studio Cookie 直连 ----------
        else:
            print(f"🚀 [上游请求] 正在使用 [{target['label']}] 调用模型 {request_obj.model}。")
            cookie_ns_tel: Dict[str, Any] = {}
            try:
                cookie_ns_resp = await cookie_upstream.execute_once(
                    request_obj, fastapi_request, telemetry=cookie_ns_tel
                )
                tracker.sync_from_cookie_telemetry(cookie_ns_tel)
                if getattr(cookie_ns_resp, "status_code", 200) == 499 or cookie_ns_tel.get("cancelled"):
                    tracker.finalize_cancelled(partial_output=False)
                else:
                    tracker.finalize_success(
                        target,
                        is_cookie=True,
                        ttft=None,
                        gen_time=float(cookie_ns_tel.get("gen_time") or 0.0),
                        finish_reason=str(cookie_ns_tel.get("finish_reason") or "stop"),
                        has_tool_calls=bool(cookie_ns_tel.get("tool_calls")),
                        has_usage=bool(cookie_ns_tel.get("has_usage")),
                        p_tk=int(cookie_ns_tel.get("p_tk") or 0),
                        th_tk=int(cookie_ns_tel.get("th_tk") or 0),
                        out_tk=int(cookie_ns_tel.get("out_tk") or 0),
                        out_chars=int(cookie_ns_tel.get("out_chars") or 0),
                    )
                return cookie_ns_resp
            except asyncio.CancelledError:
                tracker.sync_from_cookie_telemetry(cookie_ns_tel)
                tracker.finalize_cancelled(partial_output=False)
                raise
            except Exception as e_cookie:
                tracker.sync_from_cookie_telemetry(cookie_ns_tel)
                err_code = getattr(e_cookie, "status_code", None) or getattr(e_cookie, "code", None) or 500
                err_formatted = getattr(e_cookie, "message", None) or str(e_cookie)
                err_retry_after = extract_retry_after_seconds(e_cookie, err_formatted)
                is_auth_or_billing = bool(getattr(e_cookie, "is_auth_or_billing", False))
                is_retryable = bool(getattr(e_cookie, "is_retryable", False))

        # ---------- 2C. 统一处理非流式失败与多凭证退避轮换 ----------
        if is_auth_or_billing:
            dead_targets.add(target["id"])
            tip = _emit_credential_invalid_warning(target, err_code, err_formatted)
            if "【凭证更换提醒】" not in err_formatted:
                err_formatted = err_formatted + tip

        last_err_code = err_code
        last_err_formatted = err_formatted

        next_target = _peek_next_target()
        can_retry = (
            attempt < max_retries
            and next_target is not None
            and (is_retryable or is_auth_or_billing)
        )
        if can_retry:
            if await _client_gone():
                tracker.finalize_cancelled(partial_output=False)
                return JSONResponse(
                    status_code=499,
                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                )
            stats.add_retry()
            wait_target = _compute_smart_retry_wait(
                backoff_sec, target, next_target, err_retry_after
            )
            tracker.note_attempt_failure(target, err_code, wait_sec=wait_target)
            print(
                f"⚠️ [自动重试] 非流式模型={request_obj.model} | 凭证 [{target['label']}] 失败 (HTTP {err_code}) | "
                f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {wait_target}s 后将切换至 [{next_target['label']}] | "
                f"原因: {err_formatted[:1500]}"
            )
            waited = 0.0
            while waited < wait_target:
                step = min(0.5, wait_target - waited)
                await asyncio.sleep(step)
                waited += step
                if await _client_gone():
                    print(f"ℹ️ [客户端断开] 非流式退避期间客户端已断开，停止重试模型 {request_obj.model}。")
                    tracker.finalize_cancelled(partial_output=False)
                    return JSONResponse(
                        status_code=499,
                        content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                    )
            continue

        tracker.note_attempt_failure(target, err_code)
        tracker.finalize_error(err_code, err_formatted)
        # 注意：非流式返回 status_code >= 400 时，main.py 的 stats_tracker_middleware 会统一计 1 次 error
        print(f"❌ [非流式请求失败] 模型={request_obj.model} | 凭证=[{target['label']}] | HTTP {err_code} | {err_formatted[:2000]}")
        return JSONResponse(
            status_code=err_code,
            content=create_openai_error_response(err_code, err_formatted[:2000], "upstream_error"),
        )

    tracker.finalize_error(last_err_code, last_err_formatted)
    return JSONResponse(
        status_code=last_err_code,
        content=create_openai_error_response(last_err_code, last_err_formatted[:2000], "upstream_error"),
    )