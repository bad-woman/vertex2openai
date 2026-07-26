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

# 引入报错重试统计器
from logger import stats


def _safety_score_enabled() -> bool:
    try:
        return bool(app_state.get_setting("safety_score", app_config.SAFETY_SCORE))
    except Exception:
        return bool(app_config.SAFETY_SCORE)



def get_retry_settings() -> tuple[int, float]:
    """读取重试配置。

    **语义（全项目统一）**：`retry_max` 是「失败后的重试次数」，
    总请求次数 = retry_max + 1。所以循环一律写 `range(retry_max + 1)`。
    旧实现在 Express 通道写成 `range(retry_max)`，retry_max=0 时一次请求都不发（P0-2）。
    """
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
                                if new_reasoning: current_reasoning_chunks.append(new_reasoning)
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
                    if final_reasoning_chunk: current_reasoning_chunks.append(final_reasoning_chunk)
                    
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
                    
        return "".join(processed_content_chunks), "".join(current_reasoning_chunks)
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content_chunks = []
        if self.partial_tag_buffer:
            remaining_content_chunks.append(self.partial_tag_buffer)
            self.partial_tag_buffer = ""
            
        if not self.inside_tag:
            if self.tag_buffer: remaining_content_chunks.append(self.tag_buffer)
        else:
            if self.tag_buffer: self._reasoning_chunks.append(self.tag_buffer)
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

def extract_upstream_error(e: Exception) -> tuple[int, str]:
    """从上游异常里尽力提取 (HTTP 状态码, 简明消息)。

    google-genai 的 ClientError/ServerError 带 .code 与结构化 message；
    其余异常回退 500 + 类名+摘要。用于把 404/403/400 等如实透传给客户端，
    避免笼统的 500 Internal Server Error。
    """
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e)
    # 从形如 "{'error': {'code':404,'message':...}}" 的 message 里提取更干净的说明
    try:
        m = re.search(r"'message':\s*'([^']+)'", msg) or re.search(r'"message":\s*"([^"]+)"', msg)
        if m:
            msg = m.group(1)
    except Exception:
        pass
    if not isinstance(code, int) or not (400 <= code <= 599):
        low = str(e).lower()
        if "not found" in low or "404" in low:
            code = 404
        elif "permission" in low or "403" in low or "denied" in low:
            code = 403
        elif "invalid" in low or "400" in low:
            code = 400
        else:
            code = 500
    return code, msg


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
                        
    # P1-9：旧代码无条件剥离生图模型的 system_instruction，且没有注释理由。
    # 官方并未禁止生图模型使用系统指令，但为避免直接改变既有行为，
    # 改成控制台开关 image_system_instruction（默认关，保持旧行为）。
    if system_texts:
        _is_image_name = "image" in request.model.lower()
        _allow_sys = (not _is_image_name) or bool(
            app_state.get_setting("image_system_instruction", False))
        if _allow_sys:
            config["system_instruction"] = "\n".join(system_texts)
    
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: 
        config["max_output_tokens"] = request.max_tokens
    elif getattr(request, "max_completion_tokens", None) is not None:
        config["max_output_tokens"] = request.max_completion_tokens
        
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    # P1-3：OpenAI 允许 stop 为字符串，统一规范成数组
    if request.stop is not None:
        config["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else list(request.stop)
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    if getattr(request, "presence_penalty", None) is not None: config["presence_penalty"] = request.presence_penalty
    if getattr(request, "frequency_penalty", None) is not None: config["frequency_penalty"] = request.frequency_penalty
        
    # P1-3：OpenAI 语义 logprobs=bool + top_logprobs=int；Gemini 语义 response_logprobs=bool + logprobs=int。
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
            # OpenAI 结构化输出：{"type":"json_schema","json_schema":{"name":...,"schema":{...}}}
            config["response_mime_type"] = "application/json"
            json_schema_obj = fmt.get("json_schema") if isinstance(fmt, dict) else getattr(fmt, "json_schema", None)
            schema = None
            if isinstance(json_schema_obj, dict):
                schema = json_schema_obj.get("schema")
            if isinstance(schema, dict):
                schema = {k: v for k, v in schema.items() if k != "$schema"}
                config["response_schema"] = schema
    
    # 官方 2026 最新基准配置
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

    # 读取控制台设置与模型能力档案（优先级：单次请求 > 模型专属 > 全局 > 内置默认）
    settings = app_state.get_effective_settings(request.model)
    # 控制台的「采样参数处理」可覆盖版本自动判定（新模型版本号更小却已废弃采样时用）
    profile = mc.apply_sampling_policy(mc.get_profile(request.model), settings)
    is_image_model = profile["is_image"]

    if is_image_model:
        config["response_modalities"] = ["TEXT", "IMAGE"]

        # 宽高比：两通道共用解析（请求额外字段 > OpenAI size 映射 > 提示词 > 控制台默认，按模型白名单校验）
        target_ar = mc.resolve_aspect_ratio(request.model, request, settings)

        # 分辨率：请求 > 控制台默认，按模型白名单校验并回退
        image_size = mc.resolve_image_size(request.model, request, settings)
        image_config_args = {"image_size": image_size}
        if target_ar:
            image_config_args["aspect_ratio"] = target_ar

        config["image_config"] = types.ImageConfig(**image_config_args)
        # 生图模型不支持函数调用（官方明确）：丢弃 function_declarations，仅保留搜索
        tools_list = [{"google_search": {}}]

        # 生图不支持的键（采样类由 sanitize 统一剥离，这里清理其余）
        for key in ["response_mime_type", "response_schema", "response_logprobs", "logprobs"]:
            config.pop(key, None)
    else:
        # 文本/多模态：客户端未显式传采样值时，应用控制台默认（仅注入该模型支持的键）
        if config.get("temperature") is None and settings.get("default_temperature") is not None:
            config["temperature"] = settings["default_temperature"]
        if config.get("top_p") is None and settings.get("default_top_p") is not None:
            config["top_p"] = settings["default_top_p"]
        if config.get("max_output_tokens") is None and settings.get("default_max_tokens") is not None:
            config["max_output_tokens"] = settings["default_max_tokens"]

    # 按模型家族剥离不支持的采样参数（例如 Gemini 3.x 弃用 temperature/top_p/top_k、不支持 candidate_count）
    mc.sanitize_sampling(config, profile)

    if tools_list:
        config["tools"] = tools_list

    tool_config = None
    if request.tool_choice and not is_image_model:
        choice = request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none": mode = "NONE"
            elif choice == "auto": mode = "AUTO"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"
                allowed_functions = [func_name]
        if mode:
            config_dict = {"mode": mode}
            if allowed_functions: config_dict["allowed_function_names"] = allowed_functions
            tool_config = {"function_calling_config": config_dict}
    
    if tool_config: config["tool_config"] = tool_config
        
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
    """记录 token 用量并打印。

    P1-5：直接调 stats.add_tokens()，不再让 logger 用正则从日志文本里反解——
    那种做法与中文文案强耦合，改一个字就静默失效。
    """
    p_tk, c_tk, t_tk = _extract_usage(resp)
    if p_tk or c_tk:
        stats.add_tokens(p_tk, c_tk)
        print(f"💰 [算力消耗统计] 提示词: {p_tk} | 思考与生成: {c_tk} | 总计: {t_tk} Tokens")
    return {"prompt_tokens": p_tk, "completion_tokens": c_tk, "total_tokens": t_tk}


def wants_usage(request_obj: Any) -> bool:
    """客户端是否通过 stream_options.include_usage 要求在流末尾附带用量（P1-8）。"""
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
    if response is None: return False
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "text") and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "function_call", None) is not None: return True
                    if getattr(part, "inline_data", None) is not None: return True
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip(): return True
    return False


class ToolCallIndexer:
    """一次流式响应内，为 tool_calls 分配稳定递增的 index（P0-3）。

    OpenAI 客户端按 delta.tool_calls[].index 累积函数调用。旧实现把 index 硬编码成 0，
    多个并行调用会被前端合并成一个；同时旧实现遇到第一个 function_call 就 break，
    直接丢掉其余并行调用——而官方要求并行调用必须完整按
    FC1,FC2,FR1,FR2 的顺序回传，缺一个就 400。

    序号必须由调用方持有：并行调用可能分布在不同 chunk 里。
    每次重试都要新建一个实例（重试会重发整轮，序号需要归零）。
    """

    def __init__(self):
        self._next: Dict[int, int] = {}

    def next_index(self, candidate_index: int = 0) -> int:
        i = self._next.get(candidate_index, 0)
        self._next[candidate_index] = i + 1
        return i


def convert_chunk_to_openai(chunk: Any, model_name: str, response_id: str, candidate_index: int = 0,
                            indexer: Optional[ToolCallIndexer] = None) -> str:
    from message_processing import parse_gemini_response_for_reasoning_and_content, build_tool_call_id
    delta_payload = {}
    openai_finish_reason = None

    if hasattr(chunk, "candidates") and chunk.candidates and len(chunk.candidates) > candidate_index:
        candidate = chunk.candidates[candidate_index]
        raw_gemini_finish_reason = getattr(candidate, "finish_reason", None)
        if raw_gemini_finish_reason:
            if hasattr(raw_gemini_finish_reason, "name"): raw_gemini_finish_reason_str = raw_gemini_finish_reason.name.upper()
            else: raw_gemini_finish_reason_str = str(raw_gemini_finish_reason).upper()

            if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
            elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
            elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"

        # 遍历**全部** parts 收集并行函数调用，不再遇到第一个就 break
        tool_call_deltas = []
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                tc_index = indexer.next_index(candidate_index) if indexer else len(tool_call_deltas)
                tool_call_deltas.append({
                    "index": tc_index,
                    "id": build_tool_call_id(fc, part),
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args) if fc.args is not None else "",
                    },
                })

        if tool_call_deltas:
            delta_payload["tool_calls"] = tool_call_deltas
            delta_payload["content"] = None

        if not tool_call_deltas:
            reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)

            # 只在**最后一个** chunk 附安全分：上游每个 chunk 都带 safetyRatings，
            # 逐块追加会把同一份评分重复插进正文里。同样一律进正文，不进思考字段。
            # openai_finish_reason 只在**最后一块**才有值——上游每个 chunk 都带
            # safetyRatings，逐块追加会把同一份评分重复插进正文里。
            if (openai_finish_reason and _safety_score_enabled()
                    and hasattr(candidate, "safety_ratings") and candidate.safety_ratings):
                normal_text += _create_safety_ratings_html(candidate.safety_ratings)

            if reasoning_text: delta_payload["reasoning_content"] = reasoning_text
            if normal_text: 
                delta_payload["content"] = normal_text
            elif not reasoning_text and not delta_payload.get("tool_calls") and openai_finish_reason is None:
                delta_payload["content"] = ""
    
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

        if message.get("tool_calls"):
            tool_calls_list = message.get("tool_calls", [])
            for tc_item_idx, tool_call_item in enumerate(tool_calls_list):
                delta_tc_start = {
                    "tool_calls": [{
                        "index": tc_item_idx, 
                        "id": tool_call_item["id"],
                        "type": "function",
                        "function": {"name": tool_call_item["function"]["name"], "arguments": ""}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_start, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01) 

                delta_tc_args = {
                    "tool_calls": [{
                        "index": tc_item_idx,
                        "id": tool_call_item["id"], 
                        "function": {"arguments": tool_call_item["function"]["arguments"]}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_args, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)
        
        elif message.get("content") is not None or message.get("reasoning_content") is not None : 
            reasoning_content = message.get("reasoning_content", "")
            actual_content = message.get("content") 

            if reasoning_content:
                delta_reasoning = {"reasoning_content": reasoning_content}
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_reasoning, 'finish_reason': None}]})}\n\n"
                if actual_content is not None: await asyncio.sleep(0.01)

            content_to_chunk = actual_content if actual_content is not None else ""
            if actual_content is not None:
                # 【回滚】：恢复原版图片传输方案（一次性全量发送），彻底拯救前端解析器不卡死
                if "![Image](data:image/" in content_to_chunk:
                    chunk_size = max(1, len(content_to_chunk))
                else:
                    chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1

                if not content_to_chunk and not reasoning_content : 
                    yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': ''}, 'finish_reason': None}]})}\n\n"
                else:
                    for i in range(0, len(content_to_chunk), chunk_size):
                        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': content_to_chunk[i:i+chunk_size]}, 'finish_reason': None}]})}\n\n"
                        if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.01)
        
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {}, 'finish_reason': final_finish_reason}]})}\n\n"

    yield "data: [DONE]\n\n"

def _prepend_prefill(openai_dict: Dict[str, Any], prefill_text: str) -> Dict[str, Any]:
    """把预填充文本拼回到最终输出开头（预填充智能兼容用，带重叠去重）。"""
    if not prefill_text:
        return openai_dict
    try:
        # P2-3：n>1 时每个 choice 都要拼回预填充，旧实现处理完第一个就 break
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
    """真流式预填充去重：改写单条 SSE chunk 的 delta.content。

    - 去重器工作期间，正文会先被攒下（返回 None 表示该 chunk 可整条跳过）；
      判定完成后原样透传，零额外延迟。
    - force_flush=True 或 chunk 带 finish_reason 时，把攒着的文本一并放出，
      避免正文落在 finish 之后（部分客户端在 finish 后停止读取）。
    """
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

        # 没有可放出的正文：若 chunk 还有其他信息（角色/思考/finish 等）则去掉 content 保留其余
        if content is not None:
            delta.pop("content", None)
        if delta or has_finish:
            choice["delta"] = delta
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return None  # 纯 content 且被暂存 → 整条跳过
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

    # P1-6：不再使用 tenacity 的硬编码 20 次，改为与真流式/非流式一致的手写退避，
    # 读取控制台的 retry_max / retry_backoff_seconds，并在等待期间检测客户端断开。
    max_retries, backoff_sec = get_retry_settings()

    async def _client_gone() -> bool:
        if fastapi_request is None:
            return False
        try:
            return await fastapi_request.is_disconnected()
        except Exception:
            return False

    outer_keep_alive_interval = app_state.get_setting(
        "fake_streaming_interval", app_config.FAKE_STREAMING_INTERVAL_SECONDS)

    api_call_task = None
    raw_gemini_response = None
    last_error = None

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

            # 等待期间持续吐 keep-alive，避免前端因长时间无字节而超时
            while not api_call_task.done():
                if outer_keep_alive_interval > 0:
                    keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk",
                                       "created": int(time.time()), "model": request_obj.model,
                                       "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
                    yield f"data: {json.dumps(keep_alive_data)}\n\n"
                    await asyncio.sleep(outer_keep_alive_interval)
                else:
                    await asyncio.sleep(0.2)
                if await _client_gone():
                    print("ℹ️ [客户端断开] 假流式等待期间客户端已断开，正在取消上游任务。")
                    api_call_task.cancel()
                    return

            try:
                raw_gemini_response = await api_call_task
                break
            except asyncio.CancelledError:
                raise
            except Exception as e_call:
                last_error = e_call
                if is_retryable_exception(e_call) and attempt < max_retries:
                    stats.add_retry()
                    print(f"⚠️ [自动重试] 假流式上游繁忙（{e_call.__class__.__name__}），"
                          f"第 {attempt + 1} 次退避重试，等待 {backoff_sec} 秒。")
                    waited = 0.0
                    while waited < backoff_sec:
                        step = min(max(0.5, outer_keep_alive_interval or 1.0), backoff_sec - waited)
                        await asyncio.sleep(step)
                        waited += step
                        if await _client_gone():
                            print("ℹ️ [客户端断开] 假流式退避期间客户端已断开，停止重试。")
                            return
                        if outer_keep_alive_interval > 0:
                            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk",
                                               "created": int(time.time()), "model": request_obj.model,
                                               "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(keep_alive_data)}\n\n"
                    continue
                raise

        if raw_gemini_response is None:
            raise last_error or ValueError("上游未返回任何响应（重试已耗尽）。")
        
        _record_usage(raw_gemini_response)

        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        _prepend_prefill(openai_response_dict, prefill_text)

        if hasattr(raw_gemini_response, "prompt_feedback") and \
           hasattr(raw_gemini_response.prompt_feedback, "block_reason") and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, "block_reason_message") and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict
        ):
            yield chunk_sse

    except asyncio.CancelledError:
        print(f"ℹ️ [客户端断开] 假流式响应期间客户端已断开，正在清理模型 {request_obj.model} 的后台任务。")
        if "api_call_task" in locals() and not api_call_task.done():
            api_call_task.cancel()
        raise
    except Exception as e_outer_gemini:
        err_msg_detail = f"Gemini 假流式生成器异常（模型：{request_obj.model}）：{type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"❌ [API 错误响应] 假流发生器运行崩溃 (Model: {request_obj.model})。错误详情: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise
            
async def execute_gemini_call(
    current_client: Any,
    model_to_call: str,
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]],
    gen_config_dict: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool = False,
    fastapi_request: Optional[Any] = None,
    prefill_text: str = "",
):
    # P1-2：prompt 构建内部有远程图片下载与 PIL 压缩（同步阻塞），
    # 放到线程里执行，避免卡住整个事件循环。
    actual_prompt_for_call = await asyncio.to_thread(prompt_func, request_obj.messages)
    print(f"🚀 [上游请求] 正在调用 Agent Platform Express Mode 模型 {model_to_call}，客户端请求模型名为 {request_obj.model}。")

    async def _client_gone() -> bool:
        """检测客户端是否已断开连接（用于在重试前及时止损）。"""
        if fastapi_request is None:
            return False
        try:
            return await fastapi_request.is_disconnected()
        except Exception:
            return False

    if request_obj.stream:
        is_image_request = "image" in request_obj.model.lower()

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
        else: # True Streaming
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                max_retries, backoff_sec = get_retry_settings()
                has_yielded = False  # 是否已向客户端输出过内容
                # 立即吐一个 SSE 心跳，尽快建立连接（429 重试期间也保活，防前端超时中断）
                yield ": keep-alive\n\n"
                # 总尝试次数 = retry_max + 1，retry_max=0 时仍会请求一次
                for attempt in range(max_retries + 1):
                    # 客户端断开则停止重试，避免无谓的上游调用
                    if await _client_gone():
                        print(f"ℹ️ [客户端断开] 真流式请求前检测到客户端已断开，停止调用模型 {model_to_call}。")
                        return
                    try:
                        stream_gen_obj = await current_client.aio.models.generate_content_stream(
                            model=model_to_call,
                            contents=actual_prompt_for_call,
                            config=gen_config_dict
                        )

                        # 预填充智能兼容：把预填充文本作为回复开头先发出（仅一次）；
                        # 同时启用流式去重器，模型若复述预填充开头会被自动裁掉（n>1 时不启用）。
                        # 每次 attempt 都重置 tool_calls 序号（重试会重发整轮）
                        tool_indexer = ToolCallIndexer()
                        deduper = PrefillDeduper(prefill_text) if (prefill_text and (request_obj.n or 1) == 1) else None
                        if prefill_text and not has_yielded:
                            has_yielded = True
                            _pf = {"id": response_id_for_stream, "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": prefill_text}, "finish_reason": None}]}
                            yield f"data: {json.dumps(_pf)}\n\n"

                        final_p_tk, final_c_tk, final_t_tk = 0, 0, 0

                        async for chunk_item_call in stream_gen_obj:
                            if getattr(chunk_item_call, "usage_metadata", None):
                                final_p_tk, final_c_tk, final_t_tk = _extract_usage(chunk_item_call)

                            # 支持 n>1：按候选序号逐个输出
                            num_candidates = len(chunk_item_call.candidates) if getattr(chunk_item_call, "candidates", None) else 1
                            for ci in range(num_candidates):
                                has_yielded = True
                                sse_chunk = convert_chunk_to_openai(
                                    chunk_item_call, request_obj.model, response_id_for_stream, ci,
                                    indexer=tool_indexer)
                                if deduper is not None:
                                    sse_chunk = _dedup_sse_chunk_content(sse_chunk, deduper)
                                    if sse_chunk is None:
                                        continue  # 正文暂存于去重器，跳过空 chunk
                                yield sse_chunk

                        # 去重器可能还攒着开头文本（上游没发 finish chunk 的场景）
                        if deduper is not None and not deduper.done:
                            tail = deduper.flush()
                            if tail:
                                _tail = {"id": response_id_for_stream, "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}]}
                                yield f"data: {json.dumps(_tail, ensure_ascii=False)}\n\n"

                        if final_p_tk > 0 or final_c_tk > 0:
                            stats.add_tokens(final_p_tk, final_c_tk)
                            print(f"💰 [算力消耗统计] 提示词: {final_p_tk} | 思考与生成: {final_c_tk} | 总计: {final_t_tk} Tokens")

                        # P1-8：Express 真流式此前从不发 usage 块，客户端只能显示 0
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

                        # 关键修复：只有在“尚未向客户端输出任何内容”时才重试；
                        # 否则重试会导致整段答案重复输出（前半段 + 完整重发）。
                        if is_retryable and not has_yielded and attempt < max_retries:
                            # F-5：退避时长改读控制台的 retry_backoff_seconds，
                            # 原先硬编码 2**(attempt%4)，控制台设置对真流式完全没作用。
                            wait_time = backoff_sec
                            stats.add_retry() # 核心：手动重试计入大盘
                            print(f"⚠️ [自动重试] Agent Platform Express Mode 流式请求返回 429/503 或配额繁忙。第 {attempt + 1} 次退避重试，等待 {wait_time} 秒。")
                            if await _client_gone():
                                print(f"ℹ️ [客户端断开] 重试前检测到客户端已断开，停止调用模型 {model_to_call}。")
                                return
                            # 退避等待期间持续吐 SSE 心跳，保活前端连接
                            _waited = 0.0
                            while _waited < wait_time:
                                await asyncio.sleep(min(3.0, wait_time - _waited))
                                _waited += 3.0
                                if await _client_gone():
                                    print(f"ℹ️ [客户端断开] 重试等待期间检测到客户端已断开，停止调用模型 {model_to_call}。")
                                    return
                                yield ": keep-alive\n\n"
                            continue

                        err_msg_detail_stream = f"Gemini 流式请求异常（模型：{model_to_call}）：{type(e_stream_call).__name__} - {str(e_stream_call)}"
                        print(f"❌ [API 错误响应] 流式连接异常中断 (Model: {model_to_call})。错误详情: {err_msg_detail_stream}")
                        s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                        if is_auto_attempt:
                            raise e_stream_call
                        # 已经输出过内容：不再重发错误体，只补结束标记，避免污染已有输出
                        if has_yielded:
                            yield "data: [DONE]\n\n"
                        else:
                            err_resp = create_openai_error_response(500, s_err, "server_error")
                            yield f"data: {json.dumps(err_resp)}\n\n"
                            yield "data: [DONE]\n\n"
                        return

            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: # Non-streaming
        # 手动退避重试循环（替代 tenacity），以便在每次重试前检测客户端断开
        max_retries, backoff_sec = get_retry_settings()
        response_obj_call = None
        # 总尝试次数 = retry_max + 1，retry_max=0 时仍会请求一次
        for attempt in range(max_retries + 1):
            if await _client_gone():
                print(f"ℹ️ [客户端断开] 非流式请求前检测到客户端已断开，停止调用模型 {model_to_call}。")
                return JSONResponse(
                    status_code=499,
                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                )
            try:
                response_obj_call = await current_client.aio.models.generate_content(
                    model=model_to_call,
                    contents=actual_prompt_for_call,
                    config=gen_config_dict,
                )
                break
            except asyncio.CancelledError:
                print(f"ℹ️ [客户端断开] 非流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                raise
            except Exception as e_call:
                if is_retryable_exception(e_call) and attempt < max_retries:
                    stats.add_retry()
                    wait_time = backoff_sec   # F-5：同上，统一用控制台配置的退避
                    print(f"⚠️ [自动重试] 上游繁忙或触发配额限制（{e_call.__class__.__name__}）。第 {attempt + 1} 次退避重试，等待 {wait_time} 秒。")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        # 兜底：绝不让 None 流到下游的有效性检查里变成一条误导性的“无有效内容”
        if response_obj_call is None:
            print(f"❌ [上游无响应] 模型 {model_to_call} 在 {max_retries + 1} 次尝试后仍未返回任何响应。")
            return JSONResponse(
                status_code=502,
                content=create_openai_error_response(
                    502, "上游未返回任何响应（重试已耗尽）。", "upstream_error"),
            )

        if hasattr(response_obj_call, "prompt_feedback") and \
           hasattr(response_obj_call.prompt_feedback, "block_reason") and \
           response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Agent Platform 安全策略拦截了请求：{response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,"block_reason_message") and \
               response_obj_call.prompt_feedback.block_reason_message:
                block_msg+=f"（{response_obj_call.prompt_feedback.block_reason_message}）"
            raise ValueError(block_msg)

        if not is_gemini_response_valid(response_obj_call):
            error_details = f"Agent Platform 非流式响应无有效内容，模型：{model_to_call}。"
            if hasattr(response_obj_call, "candidates"):
                candidates = response_obj_call.candidates or []
                error_details += f"Candidates: {len(candidates)}. "
                if candidates:
                    candidate = candidates[0]
                    if hasattr(candidate, "content"):
                        error_details += "Has content. "
                        parts = getattr(candidate.content, "parts", None) or []
                        if hasattr(candidate.content, "parts"):
                            error_details += f"Parts: {len(parts)}. "
                            if parts:
                                part = parts[0]
                                if getattr(part, "function_call", None) is not None:
                                    error_details += f"First part is function_call: {part.function_call.name}"
                                elif hasattr(part, "text"):
                                    text_preview = str(getattr(part, "text", ""))[:100]
                                    error_details += f"First part text: '{text_preview}'"
            else:
                error_details += f"Response type: {type(response_obj_call).__name__}"
            raise ValueError(error_details)

        _record_usage(response_obj_call)

        openai_response_content = convert_to_openai_format(response_obj_call, request_obj.model)
        _prepend_prefill(openai_response_content, prefill_text)
        return JSONResponse(content=openai_response_content)