import json
import time
import math
import asyncio
import httpx
import re
from typing import List, Dict, Any, Callable, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    convert_to_openai_format,
    extract_reasoning_by_tags,
    _create_safety_ratings_html,
    strip_prefill_overlap,
    PrefillDeduper,
)
import config as app_config
from config import VERTEX_REASONING_TAG

import model_capabilities as mc
from runtime_state import app_state

from logger import stats


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
        backoff = float(app_config.DEFAULT_SETTINGS["retry_backoff_seconds"])
    return max(0, min(50, retry_max)), max(0.0, min(120.0, backoff))


class StreamingReasoningProcessor:
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self._reasoning_chunks = []
        self.partial_tag_buffer = ""

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content

        processed_content_chunks = []
        current_reasoning_chunks = []

        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content_chunks.append(self.tag_buffer[:-i])
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else:
                                self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content_chunks.append(self.tag_buffer)
                        self.tag_buffer = ""
                    break
                else:
                    processed_content_chunks.append(self.tag_buffer[:open_pos])
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else:
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self._reasoning_chunks.append(new_reasoning)
                                if new_reasoning:
                                    current_reasoning_chunks.append(new_reasoning)
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else:
                                self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self._reasoning_chunks.append(self.tag_buffer)
                            current_reasoning_chunks.append(self.tag_buffer)
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self._reasoning_chunks.append(final_reasoning_chunk)
                    if final_reasoning_chunk:
                        current_reasoning_chunks.append(final_reasoning_chunk)

                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False

        return "".join(processed_content_chunks), "".join(current_reasoning_chunks)

    def flush_remaining(self) -> tuple[str, str]:
        remaining_content_chunks = []
        if self.partial_tag_buffer:
            remaining_content_chunks.append(self.partial_tag_buffer)
            self.partial_tag_buffer = ""

        if not self.inside_tag:
            if self.tag_buffer:
                remaining_content_chunks.append(self.tag_buffer)
        else:
            if self.tag_buffer:
                self._reasoning_chunks.append(self.tag_buffer)
            self.inside_tag = False

        remaining_content = "".join(remaining_content_chunks)
        remaining_reasoning = "".join(self._reasoning_chunks)

        self.tag_buffer = ""
        self._reasoning_chunks.clear()

        return remaining_content, remaining_reasoning


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
    if "requires billing" in low or "billing to be enabled" in low:
        return "该项目未开启计费（Billing），请确认此 Key 关联的 Project ID 已启用计费，或将「标准模式 location」切回默认"
    if "api_key_invalid" in low or "api key not valid" in low or code == 401:
        return "Express API Key 无效或已失效（401），请在「通道与凭证」页检查或更换 Key"
    if code == 403 or "permission_denied" in low or ("permission" in low and "denied" in low):
        return "权限被拒绝（403）：该 API Key 无权访问目标项目/模型，或所填 Project ID 不属于该 Key"
    if code == 404 or "not found" in low:
        if "publisher model" in low or "locations/" in low:
            return "目标模型在当前区域（location）或项目中不存在（404），建议将「标准模式 location」设为 global 或默认"
        return "请求的模型名称不存在或尚未在 Express 通道开放（404）"
    if code == 429 or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return "触发上游配额或并发频率限制（429 Resource Exhausted），建议开启多 Key 轮询或降低并发"
    if code == 400 or "invalid_argument" in low:
        if "model turn" in low:
            return "当前模型不支持以 assistant 预填充结尾（400），请在控制台开启「预填充兼容模式」"
        if "thinking" in low:
            return "当前模型不支持所请求的思考档位或思考参数冲突（400）"
        return "请求参数不被上游模型接受（400 Invalid Argument）"
    if code in (500, 502, 503, 504) or "unavailable" in low or "overloaded" in low:
        return f"Google 上游服务繁忙或暂时不可用（HTTP {code}）"
    return f"上游调用异常（HTTP {code}）"


def extract_upstream_error(e: Exception) -> tuple[int, str]:
    """从上游异常里精准提取 (HTTP 状态码, 含中文诊断的清晰错误说明)。"""
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

    clean_msg = re.sub(r"([?&]key=)[^&\s'\"]+", r"\1***HIDDEN_API_KEY***", str(clean_msg).strip())

    if not isinstance(code, int) or not (400 <= code <= 599):
        low = f"{status_str} {clean_msg}".lower()
        if "not found" in low or "404" in low:
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


def is_retryable_exception(e):
    error_str = str(e).lower()
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [429, 503, 502]:
        return True
    if hasattr(e, "code") and e.code in [429, 503, 502]:
        return True
    if "429" in error_str or "503" in error_str or "too many requests" in error_str or "quota" in error_str:
        return True
    return False


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
                    if isinstance(parameters, dict) and "$schema" in parameters:
                        parameters = parameters.copy()
                        del parameters["$schema"]
                    if parameters is not None:
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


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    """从 SDK 响应里取 (prompt, completion, total) token 数。"""
    um = getattr(resp, "usage_metadata", None)
    if not um:
        return 0, 0, 0
    p_tk = getattr(um, "prompt_token_count", 0) or 0
    c_tk = getattr(um, "candidates_token_count", 0) or 0
    t_tk = getattr(um, "total_token_count", None) or (p_tk + c_tk)
    return p_tk, c_tk, t_tk


def _record_usage(resp: Any) -> dict:
    """记录 token 用量，并打印上游实际返回的流量等级。"""
    usage_metadata = getattr(resp, "usage_metadata", None)
    traffic_type = getattr(usage_metadata, "traffic_type", None) if usage_metadata else None
    if traffic_type:
        value = getattr(traffic_type, "value", None) or str(traffic_type)
        print(f"🚦 [流量等级] 上游实际 traffic_type={value}")

    p_tk, c_tk, t_tk = _extract_usage(resp)
    if p_tk or c_tk:
        stats.add_tokens(p_tk, c_tk)
        print(f"💰 [算力消耗统计] 提示词: {p_tk} | 思考与生成: {c_tk} | 总计: {t_tk} Tokens")
    return {"prompt_tokens": p_tk, "completion_tokens": c_tk, "total_tokens": t_tk}


def wants_usage(request_obj: Any) -> bool:
    """客户端是否通过 stream_options.include_usage 要求在流末尾附带用量。"""
    opts = getattr(request_obj, "stream_options", None)
    if opts is None and getattr(request_obj, "model_extra", None):
        opts = request_obj.model_extra.get("stream_options")
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


def make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
    """OpenAI 风格的用量尾块（choices 为空，仅携带 usage）。"""
    chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": model, "choices": [], "usage": usage}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def is_gemini_response_valid(response: Any) -> bool:
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
                    if getattr(part, "function_call", None) is not None:
                        return True
                    if getattr(part, "inline_data", None) is not None:
                        return True
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip():
                        return True
    return False


class ToolCallIndexer:
    """一次流式响应内，为 tool_calls 分配稳定递增的 index，并聚合分片函数参数（streamFunctionCallArguments）。"""

    def __init__(self):
        from message_processing import FunctionCallStreamAssembler
        self._next: Dict[int, int] = {}
        self._seen_tool_calls: set[int] = set()
        self._ordinary_parts: Dict[int, list] = {}
        self._part_order: Dict[int, list] = {}
        self._assemblers: Dict[int, FunctionCallStreamAssembler] = {}

    def _assembler(self, candidate_index: int = 0):
        from message_processing import FunctionCallStreamAssembler
        asm = self._assemblers.get(candidate_index)
        if asm is None:
            asm = FunctionCallStreamAssembler()
            self._assemblers[candidate_index] = asm
        return asm

    def feed_function_call(self, candidate_index: int, fc: Any, sig: Optional[bytes]) -> List[Dict[str, Any]]:
        return self._assembler(candidate_index).feed(fc, thought_signature=sig)

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
                                   if tc_index > 0 else SignatureState.UNKNOWN)),
                "type": "function",
                "function": {
                    "name": fake_fc.name,
                    "arguments": json.dumps(fake_fc.args, ensure_ascii=False) if fake_fc.args is not None else "{}",
                },
            }
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
                    text_value = getattr(part, "text", None)
                    if text_value is None and getattr(part, "inline_data", None) is not None:
                        inline = part.inline_data
                        text_value = _convert_image_to_markdown(inline.data, inline.mime_type)
                    elif text_value is None and getattr(part, "file_data", None) is not None:
                        text_value = f"![Image]({part.file_data.file_uri})"
                    text_value = "" if text_value is None else str(text_value)
                    if getattr(part, "thought", None) is True:
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
                        ordinary_signature = sig
                        ordinary_signature_kind = kind
                    continue

                if indexer is not None:
                    completed_calls = indexer.feed_function_call(candidate_index, fc, sig)
                    for c_item in completed_calls:
                        _emit_assembled_call(c_item)
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

        if (openai_finish_reason == "stop" and indexer
                and indexer.has_tool_calls(candidate_index)):
            openai_finish_reason = "tool_calls"

        if tool_call_deltas:
            delta_payload["tool_calls"] = tool_call_deltas

        reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)

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


def create_final_chunk(model: str, response_id: str, candidate_count: int = 1) -> str:
    choices = [{"index": i, "delta": {}, "finish_reason": "stop"} for i in range(candidate_count)]
    final_chunk_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": choices}
    return f"data: {json.dumps(final_chunk_data)}\n\n"


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
    """把预填充文本拼回到最终输出开头（预填充智能兼容用，带重叠去重）。"""
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
    """真流式预填充去重：改写单条 SSE chunk 的 delta.content。"""
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


async def gemini_fake_stream_generator(
    gemini_client_instance: Any,
    model_for_api_call: str,
    prompt_for_api_call: List[types.Content],
    gen_config_dict_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
    prefill_text: str = "",
    fastapi_request: Optional[Any] = None,
):
    print(f"🌊 [假流式] 已开始调用 Gemini 模型 {model_for_api_call}，客户端请求模型名为 {request_obj.model}。")

    max_retries, backoff_sec = get_retry_settings()

    async def _client_gone() -> bool:
        if fastapi_request is None:
            return False
        try:
            return await fastapi_request.is_disconnected()
        except Exception:
            return False

    try:
        outer_keep_alive_interval = float(app_state.get_setting(
            "fake_streaming_interval", app_config.FAKE_STREAMING_INTERVAL_SECONDS) or 1.0)
    except (TypeError, ValueError):
        outer_keep_alive_interval = 1.0
    outer_keep_alive_interval = max(0.5, outer_keep_alive_interval)

    api_call_task: Optional[asyncio.Task] = None
    raw_gemini_response = None
    last_error = None

    async def _cancel_api_task():
        nonlocal api_call_task
        if api_call_task is not None and not api_call_task.done():
            api_call_task.cancel()
            try:
                await api_call_task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        for attempt in range(max_retries + 1):
            if await _client_gone():
                print(f"ℹ️ [客户端断开] 假流式请求前检测到客户端已断开，停止调用模型 {model_for_api_call}。")
                return

            api_call_task = asyncio.create_task(
                gemini_client_instance.aio.models.generate_content(
                    model=model_for_api_call,
                    contents=prompt_for_api_call,
                    config=gen_config_dict_for_api_call,
                )
            )

            elapsed_since_hb = 0.0
            while not api_call_task.done():
                await asyncio.sleep(0.5)
                elapsed_since_hb += 0.5
                if await _client_gone():
                    print("ℹ️ [客户端断开] 假流式等待期间客户端已断开，正在取消上游任务。")
                    await _cancel_api_task()
                    return
                if not api_call_task.done() and elapsed_since_hb >= outer_keep_alive_interval:
                    elapsed_since_hb = 0.0
                    keep_alive_data = {
                        "id": "chatcmpl-keepalive",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request_obj.model,
                        "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(keep_alive_data)}\n\n"

            try:
                raw_gemini_response = await api_call_task
                break
            except asyncio.CancelledError:
                await _cancel_api_task()
                raise
            except Exception as e_call:
                last_error = e_call
                if is_retryable_exception(e_call) and attempt < max_retries:
                    if await _client_gone():
                        print("ℹ️ [客户端断开] 假流式异常后检测到客户端已断开，停止重试。")
                        return
                    stats.add_retry()
                    err_code, err_summary = extract_upstream_error(e_call)
                    print(f"⚠️ [自动重试] 假流式模型={request_obj.model} | HTTP {err_code} | "
                          f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {backoff_sec}s | {err_summary[:240]}")
                    waited = 0.0
                    elapsed_hb = 0.0
                    while waited < backoff_sec:
                        step = min(0.5, backoff_sec - waited)
                        await asyncio.sleep(step)
                        waited += step
                        elapsed_hb += step
                        if await _client_gone():
                            print("ℹ️ [客户端断开] 假流式退避期间客户端已断开，停止重试。")
                            return
                        if elapsed_hb >= outer_keep_alive_interval:
                            elapsed_hb = 0.0
                            keep_alive_data = {
                                "id": "chatcmpl-keepalive",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request_obj.model,
                                "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(keep_alive_data)}\n\n"
                    continue
                raise

        if raw_gemini_response is None:
            raise last_error or ValueError("上游未返回任何响应（重试已耗尽）。")

        if hasattr(raw_gemini_response, "prompt_feedback") and \
           hasattr(raw_gemini_response.prompt_feedback, "block_reason") and \
           raw_gemini_response.prompt_feedback.block_reason:
            raise ValueError(_describe_empty_gemini_response(raw_gemini_response, model_for_api_call))

        if not is_gemini_response_valid(raw_gemini_response):
            raise ValueError(_describe_empty_gemini_response(raw_gemini_response, model_for_api_call))

        _record_usage(raw_gemini_response)

        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        _prepend_prefill(openai_response_dict, prefill_text)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict
        ):
            yield chunk_sse

    except asyncio.CancelledError:
        print(f"ℹ️ [客户端断开] 假流式响应期间客户端已断开，正在清理模型 {request_obj.model} 的后台任务。")
        await _cancel_api_task()
        raise
    except Exception as e_outer_gemini:
        stats.add_error()
        err_code, err_formatted = extract_upstream_error(e_outer_gemini)
        print(f"❌ [Express 假流式报错] 模型={request_obj.model} (上游路径={model_for_api_call}) | "
              f"HTTP {err_code} | {err_formatted[:500]}")
        err_resp_sse = create_openai_error_response(err_code, err_formatted[:800], "upstream_error")
        json_payload_error = json.dumps(err_resp_sse, ensure_ascii=False)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt:
            raise


def is_location_pin_failure(err: Any) -> bool:
    """错误是否像"钉定的 projects/locations 路径不对"（而非模型或网络本身的问题）。"""
    e = str(err).lower()
    if "requires billing" in e or "billing to be enabled" in e:
        return True
    if ("not found" in e or "404" in e) and "publisher model" in e:
        return True
    if "permission_denied" in e and "projects/" in e:
        return True
    return False


async def execute_gemini_call(
    current_client: Any,
    model_to_call: str,
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]],
    gen_config_dict: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool = False,
    fastapi_request: Optional[Any] = None,
    prefill_text: str = "",
    fallback_model: Optional[str] = None,
    fallback_client_factory: Optional[Callable[[], Any]] = None,
):
    fallback_client = None

    def _get_fallback_client():
        nonlocal fallback_client
        if fallback_client_factory is None:
            return current_client
        if fallback_client is None:
            fallback_client = fallback_client_factory()
        return fallback_client

    actual_prompt_for_call = await asyncio.to_thread(prompt_func, request_obj.messages)
    print(f"🚀 [上游请求] 正在调用 Agent Platform Express Mode 模型 {model_to_call}，客户端请求模型名为 {request_obj.model}。")

    async def _client_gone() -> bool:
        """检测客户端是否已断开连接（用于在重试前及等待期间及时止损）。"""
        if fastapi_request is None:
            return False
        try:
            return await fastapi_request.is_disconnected()
        except Exception:
            return False

    if request_obj.stream:
        is_image_request = mc.get_profile(request_obj.model)["is_image"]

        if app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED) or is_image_request:
            if is_image_request:
                print("🖼️ [生图保护] 图片模型请求已自动切换为假流式输出，以避免上游流式限制。")
            return StreamingResponse(
                gemini_fake_stream_generator(
                    current_client, model_to_call, actual_prompt_for_call,
                    gen_config_dict, request_obj, is_auto_attempt, prefill_text=prefill_text,
                    fastapi_request=fastapi_request,
                ), media_type="text/event-stream"
            )
        else:  # True Streaming
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"

            async def _gemini_real_stream_generator_inner():
                nonlocal model_to_call, current_client
                max_retries, backoff_sec = get_retry_settings()
                has_yielded = False
                yield ": keep-alive\n\n"
                for attempt in range(max_retries + 1):
                    if await _client_gone():
                        print(f"ℹ️ [客户端断开] 真流式请求前检测到客户端已断开，停止调用模型 {model_to_call}。")
                        return
                    try:
                        stream_gen_obj = await current_client.aio.models.generate_content_stream(
                            model=model_to_call,
                            contents=actual_prompt_for_call,
                            config=gen_config_dict
                        )

                        tool_indexer = ToolCallIndexer()
                        deduper = PrefillDeduper(prefill_text) if (prefill_text and (request_obj.n or 1) == 1) else None
                        prefill_emitted = not bool(prefill_text)

                        final_p_tk, final_c_tk, final_t_tk = 0, 0, 0

                        async for chunk_item_call in stream_gen_obj:
                            if getattr(chunk_item_call, "usage_metadata", None):
                                final_p_tk, final_c_tk, final_t_tk = _extract_usage(chunk_item_call)

                            num_candidates = len(chunk_item_call.candidates) if getattr(chunk_item_call, "candidates", None) else 1
                            for ci in range(num_candidates):
                                has_yielded = True
                                sse_chunk = convert_chunk_to_openai(
                                    chunk_item_call, request_obj.model, response_id_for_stream, ci,
                                    indexer=tool_indexer)
                                if not prefill_emitted and '"content":' in sse_chunk and '"content": ""' not in sse_chunk and '"content": null' not in sse_chunk:
                                    prefill_emitted = True
                                    _pf = {
                                        "id": response_id_for_stream,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": request_obj.model,
                                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(_pf, ensure_ascii=False)}\n\n"
                                if deduper is not None:
                                    sse_chunk = _dedup_sse_chunk_content(sse_chunk, deduper)
                                    if sse_chunk is None:
                                        continue
                                yield sse_chunk

                        if not prefill_emitted:
                            prefill_emitted = True
                            has_yielded = True
                            _pf = {
                                "id": response_id_for_stream,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request_obj.model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(_pf, ensure_ascii=False)}\n\n"

                        if deduper is not None and not deduper.done:
                            tail = deduper.flush()
                            if tail:
                                _tail = {
                                    "id": response_id_for_stream,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": request_obj.model,
                                    "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(_tail, ensure_ascii=False)}\n\n"

                        if final_p_tk > 0 or final_c_tk > 0:
                            stats.add_tokens(final_p_tk, final_c_tk)
                            print(f"💰 [算力消耗统计] 提示词: {final_p_tk} | 思考与生成: {final_c_tk} | 总计: {final_t_tk} Tokens")

                        if wants_usage(request_obj):
                            yield make_usage_chunk(response_id_for_stream, request_obj.model, {
                                "prompt_tokens": final_p_tk,
                                "completion_tokens": final_c_tk,
                                "total_tokens": final_t_tk,
                            })

                        yield "data: [DONE]\n\n"
                        return

                    except asyncio.CancelledError:
                        print(f"ℹ️ [客户端断开] 真流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                        raise
                    except Exception as e_stream_call:
                        error_str = str(e_stream_call).lower()
                        is_retryable = (
                            "429" in error_str or "503" in error_str or "too many requests" in error_str
                            or "quota" in error_str or "resource exhausted" in error_str
                        )

                        if (fallback_model and model_to_call != fallback_model
                                and not has_yielded and is_location_pin_failure(e_stream_call)):
                            _, pin_err_msg = extract_upstream_error(e_stream_call)
                            print(f"↩️ [上游端点回退] 钉定路径 {model_to_call} 调用失败（{pin_err_msg[:180]}），"
                                  f"已自动退回默认路由 {fallback_model} 重试一次。")
                            model_to_call = fallback_model
                            current_client = _get_fallback_client()
                            continue

                        if is_retryable and not has_yielded and attempt < max_retries:
                            if await _client_gone():
                                print(f"ℹ️ [客户端断开] 重试前检测到客户端已断开，停止调用模型 {model_to_call}。")
                                return
                            wait_time = backoff_sec
                            stats.add_retry()
                            err_code, err_summary = extract_upstream_error(e_stream_call)
                            print(f"⚠️ [自动重试] Express 流式模型={request_obj.model} | HTTP {err_code} | "
                                  f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {wait_time}s | {err_summary[:240]}")
                            _waited = 0.0
                            _hb_acc = 0.0
                            while _waited < wait_time:
                                step = min(0.5, wait_time - _waited)
                                await asyncio.sleep(step)
                                _waited += step
                                _hb_acc += step
                                if await _client_gone():
                                    print(f"ℹ️ [客户端断开] 重试等待期间检测到客户端已断开，停止调用模型 {model_to_call}。")
                                    return
                                if _hb_acc >= 3.0:
                                    _hb_acc = 0.0
                                    yield ": keep-alive\n\n"
                            continue

                        stats.add_error()
                        err_code, err_formatted = extract_upstream_error(e_stream_call)
                        print(f"❌ [Express 流式报错] 模型={request_obj.model} (上游路径={model_to_call}) | "
                              f"HTTP {err_code} | {err_formatted[:500]}")
                        if is_auto_attempt:
                            raise e_stream_call
                        if has_yielded:
                            yield "data: [DONE]\n\n"
                        else:
                            err_resp = create_openai_error_response(err_code, err_formatted[:800], "upstream_error")
                            yield f"data: {json.dumps(err_resp, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                        return
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else:  # Non-streaming
        max_retries, backoff_sec = get_retry_settings()
        response_obj_call = None
        for attempt in range(max_retries + 1):
            if await _client_gone():
                print(f"ℹ️ [客户端断开] 非流式请求前检测到客户端已断开，停止调用模型 {model_to_call}。")
                return JSONResponse(
                    status_code=499,
                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                )
            call_task = asyncio.create_task(
                current_client.aio.models.generate_content(
                    model=model_to_call,
                    contents=actual_prompt_for_call,
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
                break
            except asyncio.CancelledError:
                if not call_task.done():
                    call_task.cancel()
                print(f"ℹ️ [客户端断开] 非流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                raise
            except Exception as e_call:
                if (fallback_model and model_to_call != fallback_model
                        and is_location_pin_failure(e_call)):
                    _, pin_err_msg = extract_upstream_error(e_call)
                    print(f"↩️ [上游端点回退] 钉定路径 {model_to_call} 调用失败（{pin_err_msg[:180]}），"
                          f"已自动退回默认路由 {fallback_model} 重试一次。")
                    model_to_call = fallback_model
                    current_client = _get_fallback_client()
                    continue
                if is_retryable_exception(e_call) and attempt < max_retries:
                    if await _client_gone():
                        print(f"ℹ️ [客户端断开] 非流式异常后检测到客户端已断开，停止重试模型 {model_to_call}。")
                        return JSONResponse(
                            status_code=499,
                            content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                        )
                    stats.add_retry()
                    wait_time = backoff_sec
                    err_code, err_summary = extract_upstream_error(e_call)
                    print(f"⚠️ [自动重试] Express 非流式模型={request_obj.model} | HTTP {err_code} | "
                          f"第 {attempt + 1}/{max_retries} 次退避重试，等待 {wait_time}s | {err_summary[:240]}")
                    waited = 0.0
                    while waited < wait_time:
                        step = min(0.5, wait_time - waited)
                        await asyncio.sleep(step)
                        waited += step
                        if await _client_gone():
                            print(f"ℹ️ [客户端断开] 非流式退避期间客户端已断开，停止重试模型 {model_to_call}。")
                            return JSONResponse(
                                status_code=499,
                                content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                            )
                    continue
                raise

        if response_obj_call is None:
            print(f"❌ [Express 上游无响应] 模型={request_obj.model} (上游路径={model_to_call}) 在 {max_retries + 1} 次尝试后仍未返回任何响应。")
            return JSONResponse(
                status_code=502,
                content=create_openai_error_response(
                    502, f"模型 {request_obj.model} 上游未返回任何响应（重试已耗尽）。", "upstream_error"),
            )

        if hasattr(response_obj_call, "prompt_feedback") and \
           hasattr(response_obj_call.prompt_feedback, "block_reason") and \
           response_obj_call.prompt_feedback.block_reason:
            raise ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))

        if not is_gemini_response_valid(response_obj_call):
            raise ValueError(_describe_empty_gemini_response(response_obj_call, model_to_call))

        _record_usage(response_obj_call)

        openai_response_content = convert_to_openai_format(response_obj_call, request_obj.model)
        _prepend_prefill(openai_response_content, prefill_text)
        return JSONResponse(content=openai_response_content)