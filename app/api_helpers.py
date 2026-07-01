import json
import time
import math
import asyncio
import httpx
import re
import random 
import base64
from typing import List, Dict, Any, Callable, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    convert_to_openai_format,
    extract_reasoning_by_tags,
    _create_safety_ratings_html
)
import config as app_config
from config import VERTEX_REASONING_TAG

# 引入报错重试统计器
from logger import stats

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

def is_retryable_exception(e):
    error_str = str(e).lower()
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [429, 503, 502]:
        return True
    if hasattr(e, "code") and e.code in [429, 503, 502]:
        return True
    if "429" in error_str or "503" in error_str or "too many requests" in error_str or "quota" in error_str:
        return True
    return False

def log_retry_attempt(retry_state):
    attempt = retry_state.attempt_number
    e = retry_state.outcome.exception()
    stats.add_retry()
    print(f"⚠️ [自动重试] 上游暂时繁忙或触发 Express Mode 配额限制（{e.__class__.__name__}）。正在进行第 {attempt} 次退避重试。")

@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(is_retryable_exception),
    before_sleep=log_retry_attempt
)
async def execute_with_retry(func, *args, **kwargs):
    return await func(*args, **kwargs)
    
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
                        
    if system_texts and "image" not in request.model.lower():
        config["system_instruction"] = "\n".join(system_texts)
    
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: 
        config["max_output_tokens"] = request.max_tokens
    elif getattr(request, "max_completion_tokens", None) is not None:
        config["max_output_tokens"] = request.max_completion_tokens
        
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    if getattr(request, "presence_penalty", None) is not None: config["presence_penalty"] = request.presence_penalty
    if getattr(request, "frequency_penalty", None) is not None: config["frequency_penalty"] = request.frequency_penalty
        
    if getattr(request, "response_logprobs", None) is not None: config["response_logprobs"] = request.response_logprobs
    if getattr(request, "logprobs", None) is not None: config["logprobs"] = request.logprobs

    if getattr(request, "response_format", None) is not None:
        fmt = request.response_format
        fmt_type = fmt.get("type", "") if isinstance(fmt, dict) else getattr(fmt, "type", "")
        if fmt_type == "json_object":
            config["response_mime_type"] = "application/json"
    
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

    is_image_model = "image" in request.model.lower()
    
    if is_image_model:
        config["response_modalities"] = ["TEXT", "IMAGE"]
        target_ar = None
        req_dict = request.model_dump()
        size_param = req_dict.get("size")
        if size_param:
            if size_param in ["1024x1024", "1:1"]: target_ar = "1:1"
            elif size_param in ["1024x768", "4:3"]: target_ar = "4:3"
            elif size_param in ["768x1024", "3:4"]: target_ar = "3:4"
            elif size_param in ["16:9", "9:16", "21:9", "4:5", "5:4", "3:2", "2:3"]: target_ar = size_param
            
        for msg in reversed(request.messages):
            if msg.role == "user":
                content = ""
                if isinstance(msg.content, str): content = msg.content
                elif isinstance(msg.content, list): content = " ".join([p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"])
                
                ar_match = re.search(r"(?i)--ar\s*(\d+[:：]\d+)", content)
                if not ar_match:
                    ar_match = re.search(r"\b(\d+[:：]\d+)\b", content)
                    
                if ar_match:
                    parsed_ar = ar_match.group(1).replace("：", ":")
                    pro_supported = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
                    flash_supported = pro_supported | {"1:4", "4:1", "1:8", "8:1"}
                    is_flash_img = "3.1-flash" in request.model.lower()
                    allowed_set = flash_supported if is_flash_img else pro_supported
                    
                    if parsed_ar in allowed_set:
                        target_ar = parsed_ar
                    else:
                        print(f"⚠️ [生图配置] 模型 {request.model} 不支持比例 {parsed_ar}，已交由 Gemini 自动选择比例。")
                        target_ar = None
                break
                
        image_config_args = {"image_size": "4K"}
        if target_ar:
            image_config_args["aspect_ratio"] = target_ar
            
        config["image_config"] = types.ImageConfig(**image_config_args)
        tools_list.append({"google_search": {}})

        unsupported_keys = [
            "temperature", "top_p", "top_k", "stop_sequences", "seed", 
            "candidate_count", "presence_penalty", "frequency_penalty", 
            "response_mime_type", "response_logprobs", "logprobs"
        ]
        for key in unsupported_keys:
            config.pop(key, None)

    if tools_list:
        config["tools"] = tools_list

    tool_config = None
    if request.tool_choice:
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


def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "text") and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                for part in cand.content.parts:
                    if hasattr(part, "function_call"): return True 
                    if hasattr(part, "text") and isinstance(getattr(part, "text", None), str) and getattr(part, "text", "").strip(): return True
    return False


def convert_chunk_to_openai(chunk: Any, model_name: str, response_id: str, candidate_index: int = 0) -> str:
    from message_processing import parse_gemini_response_for_reasoning_and_content
    delta_payload = {}
    openai_finish_reason = None

    if hasattr(chunk, "candidates") and chunk.candidates:
        candidate = chunk.candidates[0] 
        raw_gemini_finish_reason = getattr(candidate, "finish_reason", None)
        if raw_gemini_finish_reason:
            if hasattr(raw_gemini_finish_reason, "name"): raw_gemini_finish_reason_str = raw_gemini_finish_reason.name.upper()
            else: raw_gemini_finish_reason_str = str(raw_gemini_finish_reason).upper()

            if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
            elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
            elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"

        function_call_detected_in_chunk = False
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call is not None: 
                    fc = part.function_call
                    real_id = getattr(fc, "id", None)
                    if not real_id: real_id = getattr(fc, "thought_signature", None)
                    thought_sig = getattr(part, "thought_signature", None)
                    thought_sig_b64 = ""
                    if thought_sig:
                        if isinstance(thought_sig, bytes):
                            thought_sig_b64 = base64.b64encode(thought_sig).decode("utf-8")
                        elif isinstance(thought_sig, str):
                            thought_sig_b64 = thought_sig
                    
                    safe_name = fc.name.replace(" ", "_")
                    rand_num = int(time.time() * 10000 + random.randint(0, 9999))
                    if real_id:
                        if thought_sig_b64: tool_call_id = f"{real_id}__thought__{thought_sig_b64}"
                        else: tool_call_id = real_id
                    else:
                        if thought_sig_b64: tool_call_id = f"call_{response_id}_{candidate_index}_{safe_name}__thought__{thought_sig_b64}"
                        else: tool_call_id = f"call_{response_id}_{candidate_index}_{safe_name}_{rand_num}"
                    
                    current_tool_call_delta = {
                        "index": 0, "id": tool_call_id, "type": "function", "function": {"name": fc.name}
                    }
                    if fc.args is not None: current_tool_call_delta["function"]["arguments"] = json.dumps(fc.args)
                    else: current_tool_call_delta["function"]["arguments"] = "" 

                    if "tool_calls" not in delta_payload: delta_payload["tool_calls"] = []
                    delta_payload["tool_calls"].append(current_tool_call_delta)
                    delta_payload["content"] = None 
                    function_call_detected_in_chunk = True
                    break 

        if not function_call_detected_in_chunk:
            reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)

            if app_config.SAFETY_SCORE and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                safety_html = _create_safety_ratings_html(candidate.safety_ratings)
                if reasoning_text: reasoning_text += safety_html
                else: normal_text += safety_html

            if reasoning_text: delta_payload["reasoning_content"] = reasoning_text
            if normal_text: delta_payload["content"] = normal_text
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

async def gemini_fake_stream_generator( 
    gemini_client_instance: Any, 
    model_for_api_call: str, 
    prompt_for_api_call: List[types.Content],
    gen_config_dict_for_api_call: Dict[str, Any], 
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
):
    print(f"🌊 [假流式] 已开始调用 Gemini 模型 {model_for_api_call}，客户端请求模型名为 {request_obj.model}。")
    
    api_call_task = asyncio.create_task(
        execute_with_retry(
            gemini_client_instance.aio.models.generate_content,
            model=model_for_api_call, 
            contents=prompt_for_api_call, 
            config=gen_config_dict_for_api_call
        )
    )

    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_gemini_response = await api_call_task 
        
        if hasattr(raw_gemini_response, "usage_metadata") and raw_gemini_response.usage_metadata:
            um = raw_gemini_response.usage_metadata
            p_tk = getattr(um, "prompt_token_count", 0) or 0
            c_tk = getattr(um, "candidates_token_count", 0) or 0
            t_tk = getattr(um, "total_token_count", p_tk + c_tk) or (p_tk + c_tk)
            print(f"💰 [算力消耗统计] 提示词: {p_tk} | 思考与生成: {c_tk} | 总计: {t_tk} Tokens")

        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        
        if hasattr(raw_gemini_response, "prompt_feedback") and \
           hasattr(raw_gemini_response.prompt_feedback, "block_reason") and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, "block_reason_message") and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(openai_response_dict):
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
        if not is_auto_attempt:
            yield f"data: {json.dumps(err_resp_sse)}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise
            
async def execute_gemini_call(
    current_client: Any, 
    model_to_call: str,  
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]], 
    gen_config_dict: Dict[str, Any], 
    request_obj: OpenAIRequest, 
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    print(f"🚀 [上游请求] 正在调用 Gemini Express Mode 模型 {model_to_call}，客户端请求模型名为 {request_obj.model}。")
    
    if request_obj.stream:
        is_image_request = "image" in request_obj.model.lower()
        if app_config.FAKE_STREAMING_ENABLED or is_image_request:
            if is_image_request: print("🖼️ [生图保护] 图片模型请求已自动切换为假流式输出，以避免上游流式限制。")
            return StreamingResponse(
                gemini_fake_stream_generator(
                    current_client, model_to_call, actual_prompt_for_call,
                    gen_config_dict, request_obj, is_auto_attempt
                ), media_type="text/event-stream"
            )
        else:
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                max_retries = 20
                for attempt in range(max_retries):
                    try:
                        stream_gen_obj = await current_client.aio.models.generate_content_stream(
                            model=model_to_call, 
                            contents=actual_prompt_for_call,
                            config=gen_config_dict
                        )
                        final_p_tk, final_c_tk, final_t_tk = 0, 0, 0
                        async for chunk_item_call in stream_gen_obj:
                            if hasattr(chunk_item_call, "usage_metadata") and chunk_item_call.usage_metadata:
                                um = chunk_item_call.usage_metadata
                                final_p_tk = getattr(um, "prompt_token_count", 0) or 0
                                final_c_tk = getattr(um, "candidates_token_count", 0) or 0
                                final_t_tk = getattr(um, "total_token_count", final_p_tk + final_c_tk) or (final_p_tk + final_c_tk)
                            yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                        
                        if final_p_tk > 0 or final_c_tk > 0:
                            print(f"💰 [算力消耗统计] 提示词: {final_p_tk} | 思考与生成: {final_c_tk} | 总计: {final_t_tk} Tokens")
                            
                        yield "data: [DONE]\n\n"
                        break 
                    except asyncio.CancelledError:
                        print(f"ℹ️ [客户端断开] 真流式响应期间客户端已断开，模型 {model_to_call} 的请求已安全终止。")
                        raise
                    except Exception as e_stream_call:
                        error_str = str(e_stream_call).lower()
                        is_retryable = False
                        if "429" in error_str or "503" in error_str or "too many requests" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                            is_retryable = True
                        if is_retryable and attempt < max_retries - 1:
                            wave_index = attempt % 4
                            round_num = (attempt // 4) + 1
                            wait_time = 2 ** wave_index
                            stats.add_retry()
                            print(f"⚠️ [自动重试] Gemini Express Mode 流式请求返回 429/503 或配额繁忙。第 {round_num} 轮第 {wave_index + 1} 次重试，等待 {wait_time} 秒。")
                            await asyncio.sleep(wait_time)
                            continue 
                            
                        err_msg_detail_stream = f"Gemini 流式请求异常（模型：{model_to_call}）：{type(e_stream_call).__name__} - {str(e_stream_call)}"
                        print(f"❌ [API 错误响应] 流式连接异常中断 (Model: {model_to_call})。错误详情: {err_msg_detail_stream}")
                        s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                        if not is_auto_attempt: 
                            yield f"data: {json.dumps(create_openai_error_response(500,s_err,'server_error'))}\n\n"
                            yield "data: [DONE]\n\n"
                        else: raise e_stream_call
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else:
        response_obj_call = await execute_with_retry(
            current_client.aio.models.generate_content,
            model=model_to_call, 
            contents=actual_prompt_for_call,
            config=gen_config_dict
        )
        if hasattr(response_obj_call, "prompt_feedback") and hasattr(response_obj_call.prompt_feedback, "block_reason") and response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Gemini 安全策略拦截了请求：{response_obj_call.prompt_feedback.block_reason}"
            raise ValueError(block_msg)
        if not is_gemini_response_valid(response_obj_call):
            raise ValueError(f"Gemini 非流式响应无有效内容，模型：{model_to_call}。")
        
        if hasattr(response_obj_call, "usage_metadata") and response_obj_call.usage_metadata:
            um = response_obj_call.usage_metadata
            p_tk = getattr(um, "prompt_token_count", 0) or 0
            c_tk = getattr(um, "candidates_token_count", 0) or 0
            print(f"💰 [算力消耗统计] 提示词: {p_tk} | 思考与生成: {c_tk} | 总计: {p_tk + c_tk} Tokens")

        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))


# =========================================================================
# 🎬 Gemini Interactions API (如 Omni 视频模型) 专属解析与调用通道
# =========================================================================

def _build_omni_config(request: OpenAIRequest) -> dict | None:
    """智能拦截并提取用户传入的宽高比和视频时长参数"""
    video_config = {}
    
    # 1. 从 OpenAI API 顶层参数提取 (例如支持传 size)
    req_dict = request.model_dump()
    size_param = req_dict.get("size")
    if size_param in ["16:9", "9:16"]:
        video_config["aspect_ratio"] = size_param
        
    # 2. 从用户的自然语言 Prompt 中提取魔法指令
    for msg in reversed(request.messages):
        if msg.role == "user":
            content = ""
            if isinstance(msg.content, str): 
                content = msg.content
            elif isinstance(msg.content, list): 
                content = " ".join([p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"])
            
            # 匹配 --ar 16:9 或 9:16
            ar_match = re.search(r"(?i)--ar\s*(16:9|9:16)", content)
            if ar_match:
                video_config["aspect_ratio"] = ar_match.group(1)
                
            # 匹配 --duration 5 或 --时长 8
            duration_match = re.search(r"(?i)--(?:duration|time|时长)\s*(\d+)", content)
            if duration_match:
                sec = int(duration_match.group(1))
                if 3 <= sec <= 10:
                    video_config["duration_seconds"] = sec
            break
            
    if video_config:
        print(f"⚙️ [视频配置] 提取到视频控制参数: {video_config}")
        return {"video_config": video_config}
    return None


def _parse_interaction_to_openai(interaction: Any, model_name: str) -> dict:
    text_content = ""
    steps = getattr(interaction, 'steps', [])
    if not steps and isinstance(interaction, dict):
        steps = interaction.get('steps', [])
        
    for step in steps:
        step_type = getattr(step, 'type', None) or (step.get('type') if isinstance(step, dict) else None)
        if step_type == 'model_output':
            content = getattr(step, 'content', []) or (step.get('content') if isinstance(step, dict) else [])
            for part in content:
                part_type = getattr(part, 'type', None) or (part.get('type') if isinstance(part, dict) else None)
                if part_type == 'text':
                    text_val = getattr(part, 'text', '') or (part.get('text', '') if isinstance(part, dict) else '')
                    text_content += text_val + "\n"
                elif part_type == 'video':
                    mime_type = getattr(part, 'mime_type', 'video/mp4') or (part.get('mime_type', 'video/mp4') if isinstance(part, dict) else 'video/mp4')
                    video_b64 = getattr(part, 'data', None) or (part.get('data') if isinstance(part, dict) else None)
                    uri = getattr(part, 'uri', None) or (part.get('uri') if isinstance(part, dict) else None)
                    
                    if video_b64:
                        text_content += f'\n<br><video controls autoplay loop width="100%" style="max-width: 640px; border-radius: 8px;"><source src="data:{mime_type};base64,{video_b64}" type="{mime_type}"></video><br>\n'
                    elif uri:
                        text_content += f'\n> 💡 **视频生成完毕！**\n> 但由于 Express Mode 暂无 Cloud Storage 读取权限，视频无法直接返回。它被托管在了：\n> `GCS URI: {uri}`\n> *（如果你看到了这条消息，说明你的 API Key 支持调用该模型，只是受到了储存桶权限限制）*\n'
                        
    response_timestamp = int(time.time())
    return {
        "id": f"chatcmpl-interaction-{response_timestamp}-{random.randint(1000,9999)}",
        "object": "chat.completion",
        "created": response_timestamp,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text_content.strip()},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

async def _interaction_fake_stream_generator(current_client, model_to_call, actual_prompt, request_obj):
    kwargs = {"model": model_to_call, "input": actual_prompt}
    omni_config = _build_omni_config(request_obj)
    if omni_config:
        kwargs["config"] = omni_config

    api_call_task = asyncio.create_task(
        execute_with_retry(current_client.aio.interactions.create, **kwargs)
    )
    
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
            
    try:
        interaction_response = await api_call_task
        openai_dict = _parse_interaction_to_openai(interaction_response, request_obj.model)
        
        async for chunk_sse in _chunk_openai_response_dict_for_sse(openai_dict):
            yield chunk_sse
    except Exception as e:
        print(f"❌ [Interactions API] Omni 模型生成异常: {e}")
        err_resp = create_openai_error_response(500, str(e), "server_error")
        yield f"data: {json.dumps(err_resp)}\n\n"
        yield "data: [DONE]\n\n"

async def execute_interaction_call(current_client, model_to_call, prompt_func, request_obj):
    actual_prompt = prompt_func(request_obj.messages)
    print(f"🚀 [上游请求] 正在通过 Interactions API 专属通道调用 Omni 模型 {model_to_call}...")

    if request_obj.stream:
        return StreamingResponse(
            _interaction_fake_stream_generator(current_client, model_to_call, actual_prompt, request_obj), 
            media_type="text/event-stream"
        )
    else:
        kwargs = {"model": model_to_call, "input": actual_prompt}
        omni_config = _build_omni_config(request_obj)
        if omni_config:
            kwargs["config"] = omni_config
            
        response = await execute_with_retry(current_client.aio.interactions.create, **kwargs)
        return JSONResponse(content=_parse_interaction_to_openai(response, request_obj.model))