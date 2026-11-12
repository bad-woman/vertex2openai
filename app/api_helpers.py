import json
import time
import math
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
from logger import stats
from synthetic_wrapper import (
    extract_synthetic_content,
    make_bad_woman_response_id,
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
        elif any(k in low for k in ("只返回了思考过程", "返回了空内容", "返回了空对象", "未返回任何有效内容")):
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
        if msg.role == "system" and msg.content:
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
                schema = {k: v for k, v in schema.items() if k != "$schema"}
                config["response_schema"] = schema

    safety_threshold = "BLOCK_NONE"

    config["safety_settings"] = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]

    tools_list = []
    if request.tools:
        function_declarations = []
        for tool in request.tools:
            if tool.get("type") == "function":
                func_data = tool.get("function")
                if func_data:
                    declaration = {
                        "name": func_data.get("name"),
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
                            declaration["parameters"] = parameters
                    declaration = {k: v for k, v in declaration.items() if v is not None}
                    if declaration.get("name"):
                        function_declarations.append(declaration)
        if function_declarations:
            tools_list.append({"function_declarations": function_declarations})

    settings = app_state.get_effective_settings(request.model)
    profile = mc.apply_sampling_policy(mc.get_profile(request.model), settings)
    is_image_model = profile["is_image"]

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
        _search_names = {"google_search", "googlesearch", "web_search", "websearch", "search"}
        _declared_search = any(
            str(((tool.get("function") or {}).get("name") or "")).lower() in _search_names
            for tool in (request.tools or []) if isinstance(tool, dict)
        )
        _tools_disabled = isinstance(request.tool_choice, str) and request.tool_choice.lower() == "none"
        tools_list = ([{"google_search": {}}]
                      if _declared_search and not _tools_disabled and profile.get("supports_search")
                      else [])

        for key in ["response_mime_type", "response_schema", "response_logprobs", "logprobs"]:
            config.pop(key, None)
    else:
        if config.get("temperature") is None and settings.get("default_temperature") is not None:
            config["temperature"] = settings["default_temperature"]
        if config.get("top_p") is None and settings.get("default_top_p") is not None:
            config["top_p"] = settings["default_top_p"]
        if config.get("max_output_tokens") is None and settings.get("default_max_tokens") is not None:
            config["max_output_tokens"] = settings["default_max_tokens"]

    mc.sanitize_sampling(config, profile)

    if re.search(r"(?:^|/)gemini-4-", (request.model or "").lower()) and config.get("max_output_tokens"):
        # Gemini 4 官方支持流式与非流式单轮标准 64K (65536) 输出；Express SDK >= 2.30.0 超过 64K 时原生支持自动 Long Decode
        _fake_on = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))
        if _fake_on and not is_image_model:
            try:
                if int(config["max_output_tokens"]) > 65536:
                    config["max_output_tokens"] = 65536
            except (TypeError, ValueError):
                pass

    if tools_list:
        config["tools"] = tools_list

    tool_config = None
    if not is_image_model and (request.tool_choice or request.tools):
        choice = request.tool_choice
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
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"
                allowed_functions = [func_name]
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
        if f_role == "system":
            f_content = _plain_text_preview(getattr(first, "content", "") if hasattr(first, "content") else first.get("content"))
            lines.append(f"  📜 [第 1 轮 system]: \"{_format_preview(f_content, 120)}\"")

        last = messages[-1]
        l_role = getattr(last, "role", "unknown") if hasattr(last, "role") else (last.get("role", "unknown") if isinstance(last, dict) else "unknown")
        l_content = _plain_text_preview(getattr(last, "content", "") if hasattr(last, "content") else last.get("content", ""))
        lines.append(f"  📨 [末尾出站轮次 #{num_msgs} {l_role} | {len(l_content)} 字]: \"{_format_preview(l_content, 160)}\"")

    print("\n".join(lines))


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    um = getattr(resp, "usage_metadata", None)
    if not um:
        return 0, 0, 0
    p_tk = getattr(um, "prompt_token_count", 0) or 0
    c_tk = getattr(um, "candidates_token_count", 0) or 0
    t_tk = getattr(um, "total_token_count", None) or (p_tk + c_tk)
    return p_tk, c_tk, t_tk


def _record_usage(resp: Any) -> dict:
    usage_metadata = getattr(resp, "usage_metadata", None)
    traffic_type = getattr(usage_metadata, "traffic_type", None) if usage_metadata else None
    if traffic_type:
        value = getattr(traffic_type, "value", None) or str(traffic_type)
        print(f"🚦 [流量等级] 上游实际 traffic_type={value}")

    p_tk, c_tk, t_tk = _extract_usage(resp)
    if p_tk or c_tk:
        stats.add_tokens(p_tk, c_tk)
        print(f"💰 [算力消耗统计] 提示词: {p_tk} | 思考与生成: {c_tk} | 总计: {t_tk} Tokens")
    else:
        stats.add_success()
    return {"prompt_tokens": p_tk, "completion_tokens": c_tk, "total_tokens": t_tk}


def wants_usage(request_obj: Any) -> bool:
    opts = getattr(request_obj, "stream_options", None)
    if opts is None and getattr(request_obj, "model_extra", None):
        opts = request_obj.model_extra.get("stream_options")
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


def make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
    chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": model, "choices": [], "usage": usage}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def is_gemini_response_valid(response: Any, synthetic_tool_name: str = "") -> bool:
    if response is None:
        return False
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
        return True
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "text") and isinstance(cand.text, str) and cand.text.strip():
                return True
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        if synthetic_tool_name and getattr(fc, "name", "") == synthetic_tool_name:
                            if extract_synthetic_content(
                                getattr(fc, "args", None), emit_tool_name=synthetic_tool_name
                            ).strip():
                                return True
                            continue
                        return True
                    if getattr(part, "inline_data", None) is not None:
                        return True
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip():
                        if getattr(part, "thought", None) is True:
                            continue
                        return True
    return False


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

    def flush_synthetic_tail(self, candidate_index: int = 0) -> str:
        asm = self._assemblers.get(candidate_index)
        return asm.flush_synthetic_tail() if asm else ""

    def flush_function_calls(self, candidate_index: int = 0) -> List[Dict[str, Any]]:
        asm = self._assemblers.get(candidate_index)
        return asm.flush() if asm else []

    def has_pending_function_call(self, candidate_index: int = 0) -> bool:
        asm = self._assemblers.get(candidate_index)
        return bool(asm and asm.has_pending())

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
        index = len(items)
        items.append(metadata)
        self._part_order.setdefault(candidate_index, []).append(
            {"type": "ordinary", "index": index})
        return index

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
            elif raw_gemini_finish_reason_str == "MAX_TOKENS":
                openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str == "SAFETY":
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
            fake_fc = types.FunctionCall(
                id=call_dict.get("id"),
                name=call_dict.get("name") or "unknown",
                args=call_dict.get("args") or {},
            )
            fake_part = types.Part(
                function_call=fake_fc,
                thought_signature=call_dict.get("thought_signature"),
            )
            tc = {
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
            part_order.append({"type": "tool_call", "index": tc_index})
            if indexer:
                indexer.record_tool_call(candidate_index, tc_index)
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
            if tail_synth:
                synthetic_text_deltas.append(tail_synth)
                metadata = ordinary_part_metadata("text", tail_synth, None)
                ordinary_index = indexer.record_ordinary_part(candidate_index, metadata)
                part_order.append({"type": "ordinary", "index": ordinary_index})
                ordinary_metadata.append(metadata)
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

        google_extra = {}
        if indexer:
            metadata_to_emit, order_to_emit = indexer.topology(candidate_index)
        else:
            metadata_to_emit, order_to_emit = ordinary_metadata, part_order

        cumulative_signature = None
        cumulative_signature_kind = None
        for item in reversed(metadata_to_emit):
            if item.get("thought_signature"):
                cumulative_signature = item["thought_signature"]
                cumulative_signature_kind = item.get("kind")
                break
        if cumulative_signature:
            google_extra["thought_signature"] = cumulative_signature
            google_extra["thought_signature_part"] = cumulative_signature_kind
        elif ordinary_signature:
            google_extra.update((thought_signature_extra(
                ordinary_signature, ordinary_signature_kind) or {}).get("google", {}))
        if metadata_to_emit:
            google_extra["ordinary_parts"] = metadata_to_emit
        if order_to_emit:
            google_extra["part_order"] = order_to_emit
        if google_extra:
            delta_payload["extra_content"] = {"google": google_extra}

    if not delta_payload and openai_finish_reason is None:
        delta_payload["content"] = ""

    chunk_data = {
        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
        "choices": [{"index": candidate_index, "delta": delta_payload, "finish_reason": openai_finish_reason}]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"


async def _chunk_openai_response_dict_for_sse(
    openai_response_dict: Dict[str, Any],
    response_id_override: Optional[str] = None,
    model_name_override: Optional[str] = None
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
            return f"data: {json.dumps(payload)}\n\n"

        if message.get("extra_content") is not None:
            yield _sse({"extra_content": message["extra_content"]})

        reasoning_content = message.get("reasoning_content", "")
        actual_content = message.get("content")
        if reasoning_content:
            yield _sse({"reasoning_content": reasoning_content})
            await asyncio.sleep(0.01)

        if actual_content is not None:
            content_to_chunk = actual_content
            if "![Image](data:image/" in content_to_chunk:
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

        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {}, 'finish_reason': final_finish_reason}]})}\n\n"

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
    try:
        validate_request_schemas(request_obj)
    except SchemaValidationError as exc:
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
            return JSONResponse(
                status_code=cookie_err_code,
                content=create_openai_error_response(cookie_err_code, cookie_err_msg, "auth_error"),
            )

    if len(base_targets) > 1:
        pool_desc = " → ".join(t["label"] for t in base_targets)
        print(f"🔑 [凭证轮询池] 本次请求可用凭证共 {len(base_targets)} 个：{pool_desc}")

    express_ctx: Optional[dict] = None
    express_prompt_cache: Optional[List[types.Content]] = None

    async def _get_express_ctx():
        nonlocal express_ctx, express_prompt_cache
        if express_ctx is None:
            ctx, err_resp = express_upstream.prepare_express_context(request_obj)
            if err_resp is not None:
                return None, None, err_resp
            express_ctx = ctx
            express_prompt_cache = await asyncio.to_thread(
                ctx["prompt_func"], ctx["request_obj"].messages
            )
        return express_ctx, express_prompt_cache, None

    if any(t["channel"] == "express" for t in base_targets):
        _, _, init_err_resp = await _get_express_ctx()
        if init_err_resp is not None:
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

    # ============================================================
    # 分支一：流式请求（包含真流式与假流式，支持跨 Key / 跨通道无缝切换）
    # ============================================================
    if request_obj.stream:
        is_image_request = mc.get_profile(request_obj.model)["is_image"]
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
            if not (fake_stream_on or is_image_request):
                yield ": keep-alive\n\n"

            for attempt in range(max_retries + 1):
                if await _client_gone():
                    print(f"ℹ️ [客户端断开] 流式请求前检测到客户端已断开，停止调用模型 {request_obj.model}。")
                    return

                target = _pick_next_target()
                if target is None:
                    break

                has_yielded_output = False
                attempt_failed = False
                err_code = 500
                err_formatted = ""
                is_retryable = False
                is_auth_or_billing = False

                # ---------- 1A. 当前轮次使用 Express Key ----------
                if target["channel"] == "express":
                    ctx, actual_prompt, _ = await _get_express_ctx()
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
                            for _pin_try in range(2):
                                api_call_task = asyncio.create_task(
                                    current_client.aio.models.generate_content(
                                        model=model_to_call,
                                        contents=actual_prompt,
                                        config=gen_config_dict,
                                    )
                                )
                                elapsed_hb = 0.0
                                while not api_call_task.done():
                                    await asyncio.sleep(0.5)
                                    elapsed_hb += 0.5
                                    if await _client_gone():
                                        print("ℹ️ [客户端断开] 假流式等待期间客户端已断开，正在取消上游任务。")
                                        await _cancel_task()
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

                            if hasattr(raw_gemini_response, "prompt_feedback") and \
                               hasattr(raw_gemini_response.prompt_feedback, "block_reason") and \
                               raw_gemini_response.prompt_feedback.block_reason:
                                print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                                raise ValueError(_describe_empty_gemini_response(raw_gemini_response, model_to_call))

                            if not is_gemini_response_valid(raw_gemini_response, synthetic_tool_name=synth_tool_name):
                                print(f"🔎 [Express 诊断] 模型={model_to_call} 空正文或无效响应。原始响应样本：\n{sampler.dump()}")
                                err_v = ValueError(_describe_empty_gemini_response(raw_gemini_response, model_to_call))
                                if synth_tool_name:
                                    err_v.is_retryable = True
                                raise err_v

                            _record_usage(raw_gemini_response)
                            openai_response_dict = convert_to_openai_format(
                                raw_gemini_response,
                                exp_req_obj.model,
                                synthetic_tool_name=synth_tool_name,
                                prefill_text=prefill_text,
                            )
                            _prepend_prefill(openai_response_dict, prefill_text)

                            async for chunk_sse in _chunk_openai_response_dict_for_sse(openai_response_dict=openai_response_dict):
                                has_yielded_output = True
                                yield chunk_sse
                            if app_state.get_setting("debug_outbound", False):
                                print(f"🔎 [Express 响应原始结构] {sampler.dump()}")
                            print(f"✅ [Express] {request_obj.model} | 假流式传输完成")
                            return

                        except asyncio.CancelledError:
                            await _cancel_task()
                            raise
                        except Exception as e_fake:
                            attempt_failed = True
                            err_code, err_formatted = extract_upstream_error(e_fake)
                            is_auth_or_billing = is_express_auth_or_billing_error(e_fake, err_code)
                            is_retryable = is_retryable_exception(e_fake)

                    else:
                        # Express 真流式
                        response_id_for_stream = make_bad_woman_response_id(
                            "chatcmpl-realstream", sandbox_enabled=bool(synth_tool_name)
                        )
                        stream_start_ts = time.time()
                        sampler = _RawSampler()
                        try:
                            for _pin_try in range(2):
                                try:
                                    stream_gen_obj = await current_client.aio.models.generate_content_stream(
                                        model=model_to_call,
                                        contents=actual_prompt,
                                        config=gen_config_dict,
                                    )
                                    tool_indexer = ToolCallIndexer(
                                        synthetic_tool_name=synth_tool_name,
                                        prefill_text=prefill_text,
                                    )
                                    deduper = PrefillDeduper(prefill_text) if (prefill_text and (exp_req_obj.n or 1) == 1) else None
                                    prefill_emitted = not bool(prefill_text)
                                    emitted_finish_reason = False
                                    last_stream_chunk_obj = None
                                    final_p_tk, final_c_tk, final_t_tk = 0, 0, 0

                                    async for chunk_item_call in stream_gen_obj:
                                        last_stream_chunk_obj = chunk_item_call
                                        sampler.add(extract_gemini_chunk_debug_dict(chunk_item_call))
                                        if getattr(chunk_item_call, "usage_metadata", None):
                                            final_p_tk, final_c_tk, final_t_tk = _extract_usage(chunk_item_call)

                                        pf = getattr(chunk_item_call, "prompt_feedback", None)
                                        if pf and getattr(pf, "block_reason", None) and not has_yielded_output:
                                            print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                                            raise ValueError(_describe_empty_gemini_response(chunk_item_call, model_to_call))

                                        num_candidates = len(chunk_item_call.candidates) if getattr(chunk_item_call, "candidates", None) else 1
                                        for ci in range(num_candidates):
                                            sse_chunk = convert_chunk_to_openai(
                                                chunk_item_call, exp_req_obj.model, response_id_for_stream, ci,
                                                indexer=tool_indexer,
                                            )
                                            if not has_yielded_output and '"finish_reason": "content_filter"' in sse_chunk:
                                                print(f"🔎 [Express 诊断] 模型={model_to_call} 触发安全过滤。原始响应样本：\n{sampler.dump()}")
                                                raise ValueError(_describe_empty_gemini_response(chunk_item_call, model_to_call))

                                            if deduper is not None:
                                                sse_chunk = _dedup_sse_chunk_content(sse_chunk, deduper)
                                                if sse_chunk is None:
                                                    continue

                                            is_real_chunk = (
                                                '"reasoning_content":' in sse_chunk
                                                or '"tool_calls":' in sse_chunk
                                                or ('"content":' in sse_chunk and '"content": ""' not in sse_chunk and '"content": null' not in sse_chunk)
                                            )
                                            if is_real_chunk and not has_yielded_output:
                                                has_yielded_output = True
                                                print(f"⚡ [Express] {request_obj.model} | 连接建立，正在实时流式输出...")
                                            elif not has_yielded_output:
                                                continue

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
                                            yield sse_chunk
                                        if tool_indexer.has_synthetic_halted(0):
                                            print(f"🛑 [抗截断保护] 检测到 {model_to_call} 工具闭合溢出特征，已平滑切除并立即断开上游流。")
                                            break

                                    rem_synth = tool_indexer.flush_synthetic_tail(0)
                                    if rem_synth:
                                        if deduper is not None:
                                            rem_synth = deduper.feed(rem_synth)
                                        if rem_synth:
                                            if not has_yielded_output:
                                                has_yielded_output = True
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
                                            _rem_chunk = {
                                                "id": response_id_for_stream,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": exp_req_obj.model,
                                                "choices": [{"index": 0, "delta": {"content": rem_synth}, "finish_reason": None}],
                                            }
                                            yield f"data: {json.dumps(_rem_chunk, ensure_ascii=False)}\n\n"

                                    if deduper is not None and not deduper.done:
                                        tail = deduper.flush()
                                        if tail:
                                            if not has_yielded_output:
                                                has_yielded_output = True
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

                                    # 真流式全量空回拦截：无论是否开启沙箱，只要模型未产出任何有效正文/思考/工具调用，绝不把空壳当成功发出，立即报错（非沙箱模式不盲目重试）！
                                    if not has_yielded_output:
                                        print(f"🔎 [Express 诊断] 模型={model_to_call} 真流式未产出任何有效正文。原始响应样本：\n{sampler.dump()}")
                                        desc_empty = _describe_empty_gemini_response(last_stream_chunk_obj, model_to_call)
                                        err_empty_stream = ValueError(desc_empty)
                                        if synth_tool_name:
                                            err_empty_stream.is_retryable = True
                                        raise err_empty_stream

                                    if has_yielded_output and not emitted_finish_reason:
                                        _fin_reason = "tool_calls" if tool_indexer.has_tool_calls(0) else "stop"
                                        _fin_chunk = {
                                            "id": response_id_for_stream,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": exp_req_obj.model,
                                            "choices": [{"index": 0, "delta": {}, "finish_reason": _fin_reason}],
                                        }
                                        yield f"data: {json.dumps(_fin_chunk, ensure_ascii=False)}\n\n"

                                    if final_p_tk > 0 or final_c_tk > 0:
                                        stats.add_tokens(final_p_tk, final_c_tk)
                                        print(f"💰 [算力消耗统计] 提示词: {final_p_tk} | 思考与生成: {final_c_tk} | 总计: {final_t_tk} Tokens")
                                    else:
                                        stats.add_success()

                                    if wants_usage(exp_req_obj):
                                        yield make_usage_chunk(response_id_for_stream, exp_req_obj.model, {
                                            "prompt_tokens": final_p_tk,
                                            "completion_tokens": final_c_tk,
                                            "total_tokens": final_t_tk,
                                        })

                                    yield "data: [DONE]\n\n"
                                    elapsed = time.time() - stream_start_ts
                                    if app_state.get_setting("debug_outbound", False):
                                        print(f"🔎 [Express 尾包原始结构] {sampler.dump()}")
                                    print(f"✅ [Express] {request_obj.model} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                                    return
                                except Exception as e_pin_stream:
                                    if (
                                        _pin_try == 0
                                        and not has_yielded_output
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
                        except asyncio.CancelledError:
                            print(f"ℹ️ [客户端断开] 真流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                            raise
                        except Exception as e_stream_call:
                            err_code, err_formatted = extract_upstream_error(e_stream_call)
                            if has_yielded_output:
                                stats.add_error()
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
                    try:
                        cookie_resp = await cookie_upstream.execute_once(request_obj, fastapi_request)
                        cookie_iter = cookie_resp.body_iterator
                        async for chunk_item in cookie_iter:
                            chunk_str = chunk_item.decode("utf-8", errors="replace") if isinstance(chunk_item, bytes) else str(chunk_item)
                            if chunk_str.strip().startswith("data:") and '"chatcmpl-keepalive"' not in chunk_str:
                                has_yielded_output = True
                            yield chunk_str
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as e_cookie:
                        err_code = getattr(e_cookie, "status_code", None) or getattr(e_cookie, "code", None) or 500
                        err_formatted = getattr(e_cookie, "message", None) or str(e_cookie)
                        if has_yielded_output:
                            stats.add_error()
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
                            return
                        stats.add_retry()
                        print(
                            f"⚠️ [自动重试] 流式模型={request_obj.model} | 凭证 [{target['label']}] 失败 (HTTP {err_code}) | "
                            f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {backoff_sec}s 后将切换至 [{next_target['label']}] | "
                            f"原因: {err_formatted[:1500]}"
                        )
                        waited = 0.0
                        hb_acc = 0.0
                        hb_step_target = outer_keep_alive_interval if (fake_stream_on or is_image_request) else 3.0
                        while waited < backoff_sec:
                            step = min(0.5, backoff_sec - waited)
                            await asyncio.sleep(step)
                            waited += step
                            hb_acc += step
                            if await _client_gone():
                                print(f"ℹ️ [客户端断开] 退避等待期间客户端已断开，停止重试模型 {request_obj.model}。")
                                return
                            if hb_acc >= hb_step_target:
                                hb_acc = 0.0
                                yield _make_keepalive_chunk()
                        continue

                    stats.add_error()
                    print(f"❌ [流式请求失败] 模型={request_obj.model} | 凭证=[{target['label']}] | HTTP {err_code} | {err_formatted[:2000]}")
                    err_resp = create_openai_error_response(err_code, err_formatted[:2000], "upstream_error")
                    yield f"data: {json.dumps(err_resp, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(_unified_stream_generator(), media_type="text/event-stream")

    # ============================================================
    # 分支二：非流式请求（支持多 Key 轮询与跨通道 Cookie 互备）
    # ============================================================
    last_err_code = 502
    last_err_formatted = f"模型 {request_obj.model} 上游未返回任何响应（重试已耗尽）。"

    for attempt in range(max_retries + 1):
        if await _client_gone():
            print(f"ℹ️ [客户端断开] 非流式请求前检测到客户端已断开，停止调用模型 {request_obj.model}。")
            return JSONResponse(
                status_code=499,
                content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
            )

        target = _pick_next_target()
        if target is None:
            break

        err_code = 500
        err_formatted = ""
        is_retryable = False
        is_auth_or_billing = False

        # ---------- 2A. 当前轮次使用 Express Key ----------
        if target["channel"] == "express":
            ctx, actual_prompt, _ = await _get_express_ctx()
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
                for _pin_try in range(2):
                    call_task = asyncio.create_task(
                        current_client.aio.models.generate_content(
                            model=model_to_call,
                            contents=actual_prompt,
                            config=gen_config_dict,
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

                if hasattr(response_obj_call, "prompt_feedback") and \
                   hasattr(response_obj_call.prompt_feedback, "block_reason") and \
                   response_obj_call.prompt_feedback.block_reason:
                    print(f"🔎 [Express 诊断] 模型={model_to_call} 触发拦截。原始响应样本：\n{sampler.dump()}")
                    raise ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))

                if not is_gemini_response_valid(response_obj_call, synthetic_tool_name=synth_tool_name):
                    print(f"🔎 [Express 诊断] 模型={model_to_call} 空正文或无效响应。原始响应样本：\n{sampler.dump()}")
                    err_v = ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))
                    if synth_tool_name:
                        err_v.is_retryable = True
                    raise err_v

                _record_usage(response_obj_call)
                openai_response_content = convert_to_openai_format(
                    response_obj_call,
                    exp_req_obj.model,
                    synthetic_tool_name=synth_tool_name,
                    prefill_text=prefill_text,
                )
                _prepend_prefill(openai_response_content, prefill_text)
                if app_state.get_setting("debug_outbound", False):
                    print(f"🔎 [Express 响应原始结构] {sampler.dump()}")
                print(f"✅ [Express] {request_obj.model} | 非流式响应完成")
                return JSONResponse(content=openai_response_content)

            except asyncio.CancelledError:
                raise
            except Exception as e_call:
                err_code, err_formatted = extract_upstream_error(e_call)
                is_auth_or_billing = is_express_auth_or_billing_error(e_call, err_code)
                is_retryable = is_retryable_exception(e_call)

        # ---------- 2B. 当前轮次使用 Studio Cookie 直连 ----------
        else:
            print(f"🚀 [上游请求] 正在使用 [{target['label']}] 调用模型 {request_obj.model}。")
            try:
                return await cookie_upstream.execute_once(request_obj, fastapi_request)
            except asyncio.CancelledError:
                raise
            except Exception as e_cookie:
                err_code = getattr(e_cookie, "status_code", None) or getattr(e_cookie, "code", None) or 500
                err_formatted = getattr(e_cookie, "message", None) or str(e_cookie)
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
                return JSONResponse(
                    status_code=499,
                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                )
            stats.add_retry()
            print(
                f"⚠️ [自动重试] 非流式模型={request_obj.model} | 凭证 [{target['label']}] 失败 (HTTP {err_code}) | "
                f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {backoff_sec}s 后将切换至 [{next_target['label']}] | "
                f"原因: {err_formatted[:1500]}"
            )
            waited = 0.0
            while waited < backoff_sec:
                step = min(0.5, backoff_sec - waited)
                await asyncio.sleep(step)
                waited += step
                if await _client_gone():
                    print(f"ℹ️ [客户端断开] 非流式退避期间客户端已断开，停止重试模型 {request_obj.model}。")
                    return JSONResponse(
                        status_code=499,
                        content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                    )
            continue

        # 注意：非流式返回 status_code >= 400 时，main.py 的 stats_tracker_middleware 会统一计 1 次 error
        print(f"❌ [非流式请求失败] 模型={request_obj.model} | 凭证=[{target['label']}] | HTTP {err_code} | {err_formatted[:2000]}")
        return JSONResponse(
            status_code=err_code,
            content=create_openai_error_response(err_code, err_formatted[:2000], "upstream_error"),
        )

    return JSONResponse(
        status_code=last_err_code,
        content=create_openai_error_response(last_err_code, last_err_formatted[:2000], "upstream_error"),
    )