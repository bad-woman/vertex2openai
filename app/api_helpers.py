import json
import time
import math
import asyncio
import httpx
import re
from typing import List, Dict, Any, Callable, Union, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
from openai import AsyncOpenAI 

from models import OpenAIRequest, OpenAIMessage
from message_processing import (
    convert_to_openai_format,
    convert_chunk_to_openai,
    extract_reasoning_by_tags,
    _create_safety_ratings_html
)
import config as app_config
from config import VERTEX_REASONING_TAG

class StreamingReasoningProcessor:
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self.reasoning_buffer = ""
        self.partial_tag_buffer = "" 

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content
        processed_content = ""
        current_reasoning = ""
        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content += self.tag_buffer[:-i]
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content += self.tag_buffer
                        self.tag_buffer = ""
                    break
                else:
                    processed_content += self.tag_buffer[:open_pos]
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
                                self.reasoning_buffer += new_reasoning
                                if new_reasoning: current_reasoning = new_reasoning
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self.reasoning_buffer += self.tag_buffer
                            current_reasoning = self.tag_buffer
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self.reasoning_buffer += final_reasoning_chunk
                    if final_reasoning_chunk: current_reasoning = final_reasoning_chunk
                    self.reasoning_buffer = "" 
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
        return processed_content, current_reasoning
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content, remaining_reasoning = "", ""
        if self.partial_tag_buffer:
            remaining_content += self.partial_tag_buffer
            self.partial_tag_buffer = ""
        if not self.inside_tag:
            if self.tag_buffer: remaining_content += self.tag_buffer
        else:
            if self.reasoning_buffer: remaining_reasoning = self.reasoning_buffer
            if self.tag_buffer: remaining_content += self.tag_buffer
            self.inside_tag = False
        self.tag_buffer, self.reasoning_buffer = "", ""
        return remaining_content, remaining_reasoning


def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    safe_message = re.sub(r'([?&]key=)[^&\s\'"]+', r'\1***HIDDEN_API_KEY***', message)
    return {
        "error": {
            "message": safe_message,
            "type": error_type,
            "code": status_code,
            "param": None
        }
    }

async def execute_with_retry(func, *args, max_retries=20, **kwargs):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            is_retryable = False
            error_str = str(e).lower()
            
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [429, 503, 502]:
                is_retryable = True
            elif hasattr(e, 'code') and e.code in [429, 503, 502]:
                is_retryable = True
            elif "429" in error_str or "503" in error_str or "too many requests" in error_str or "quota" in error_str:
                is_retryable = True

            if is_retryable and attempt < max_retries - 1:
                wave_index = attempt % 4
                round_num = (attempt // 4) + 1
                wait_time = 2 ** wave_index
                
                print(f"⚠️ 触发神性护盾 (API {e.__class__.__name__}): 遇到速率限制。正在进行第 {round_num} 轮第 {wave_index + 1} 次重试，等待 {wait_time} 秒...")
                await asyncio.sleep(wait_time)
            else:
                if is_retryable:
                    print(f"❌ 容错极限到达: 已穷尽 {max_retries} 次重试，额度已彻底枯竭。")
                raise e
    raise last_exception
    
def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    
    system_texts = []
    for msg in request.messages:
        if msg.role == "system" and msg.content:
            if isinstance(msg.content, str):
                system_texts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        system_texts.append(part.get('text', ''))
                    elif hasattr(part, 'text') and isinstance(part.text, str):
                        system_texts.append(part.text)
    if system_texts:
        config["system_instruction"] = "\n".join(system_texts)
    
    if "image" in request.model.lower():
        config["response_modalities"] = ["TEXT", "IMAGE"]
    
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    safety_threshold = "BLOCK_NONE"
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_UNSPECIFIED", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]

    function_declarations = []
    if request.tools:
        for tool in request.tools:
            if tool.get("type") == "function":
                func_def = tool
                if func_def:
                    declaration = {
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                    }
                    parameters = func_def.get("parameters")
                    if isinstance(parameters, dict) and "$schema" in parameters:
                        parameters = parameters.copy()
                        del parameters["$schema"]
                    if parameters is not None:
                        declaration["parameters"] = parameters
                    declaration = {k: v for k, v in declaration.items() if v is not None}
                    if declaration.get("name"): 
                        function_declarations.append(declaration)

    if function_declarations:
        config["tools"] = [{"function_declarations": function_declarations}]

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
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, 'candidates') and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, 'text') and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, 'content') and hasattr(cand.content, 'parts') and cand.content.parts:
                for part in cand.content.parts:
                    if hasattr(part, 'function_call'): return True 
                    if hasattr(part, 'text') and isinstance(getattr(part, 'text', None), str) and getattr(part, 'text', '').strip(): return True
    return False

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
                if actual_content is not None: await asyncio.sleep(0.05)

            content_to_chunk = actual_content if actual_content is not None else ""
            if actual_content is not None:
                chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1
                if not content_to_chunk and not reasoning_content : 
                    yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': ''}, 'finish_reason': None}]})}\n\n"
                else:
                    for i in range(0, len(content_to_chunk), chunk_size):
                        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': content_to_chunk[i:i+chunk_size]}, 'finish_reason': None}]})}\n\n"
                        if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)
        
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
    model_name_for_log = getattr(gemini_client_instance, 'model_name', 'unknown_gemini_model_object')
    print(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', client obj: '{model_name_for_log}')")
    
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
        
        # --- 新增开始：拦截假流式的 Token 消耗 ---
        if hasattr(raw_gemini_response, 'usage_metadata') and raw_gemini_response.usage_metadata:
            um = raw_gemini_response.usage_metadata
            p_tk = getattr(um, 'prompt_token_count', 0) or 0
            c_tk = getattr(um, 'candidates_token_count', 0) or 0
            t_tk = getattr(um, 'total_token_count', p_tk + c_tk) or (p_tk + c_tk)
            print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")
        # --- 新增结束 ---

        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        
        if hasattr(raw_gemini_response, 'prompt_feedback') and \
           hasattr(raw_gemini_response.prompt_feedback, 'block_reason') and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, 'block_reason_message') and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict
        ):
            yield chunk_sse

    except asyncio.CancelledError:
        print(f"INFO: Client disconnected during Fake Stream (Gemini: {request_obj.model}). Cleaning up.")
        raise
    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(500, sse_err_msg_display, "server_error")
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"
        if is_auto_attempt: raise

async def openai_fake_stream_generator( 
    openai_client: Union[AsyncOpenAI, Any], 
    openai_params: Dict[str, Any],
    openai_extra_body: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool
):
    api_model_name = openai_params.get("model", "unknown-openai-model")
    print(f"FAKE STREAMING (OpenAI Direct): Prep for '{request_obj.model}' (API model: '{api_model_name}')")
    response_id = f"chatcmpl-openaidirectfake-{int(time.time())}"
    
    async def _openai_api_call_task():
        params_for_call = openai_params.copy()
        params_for_call['stream'] = False 
        return await execute_with_retry(
            openai_client.chat.completions.create,
            **params_for_call, extra_body=openai_extra_body
        )
    api_call_task = asyncio.create_task(_openai_api_call_task())
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        raw_response_obj = await api_call_task 
        openai_response_dict = raw_response_obj.model_dump(exclude_unset=True, exclude_none=True)

        if app_config.SAFETY_SCORE and hasattr(raw_response_obj, "choices") and raw_response_obj.choices:
            for i, choice_obj in enumerate(raw_response_obj.choices):
                if hasattr(choice_obj, "safety_ratings") and choice_obj.safety_ratings:
                    safety_html = _create_safety_ratings_html(choice_obj.safety_ratings)
                    if i < len(openai_response_dict.get("choices", [])):
                        choice_dict = openai_response_dict["choices"][i]
                        message_dict = choice_dict.get("message")
                        if message_dict:
                            current_content = message_dict.get("content") or ""
                            message_dict["content"] = current_content + safety_html

        if openai_response_dict.get("choices") and \
           isinstance(openai_response_dict["choices"], list) and \
           len(openai_response_dict["choices"]) > 0:
            
            first_choice_dict_item = openai_response_dict["choices"][0]
            if first_choice_dict_item and isinstance(first_choice_dict_item, dict) :
                choice_message_ref = first_choice_dict_item.get("message", {})
                original_content = choice_message_ref.get("content")
                if isinstance(original_content, str):
                    reasoning_text, actual_content = extract_reasoning_by_tags(original_content, VERTEX_REASONING_TAG)
                    choice_message_ref["content"] = actual_content
                    if reasoning_text:
                        choice_message_ref["reasoning_content"] = reasoning_text
        
        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict,
            response_id_override=response_id, 
            model_name_override=request_obj.model
        ):
            yield chunk_sse
            
    except asyncio.CancelledError:
        print(f"INFO: Client disconnected during Fake Stream (OpenAI: {request_obj.model}). Cleaning up.")
        raise
    except Exception as e_outer:
        err_msg_detail = f"Error in openai_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer).__name__} - {str(e_outer)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer)
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
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    client_model_name_for_log = getattr(current_client, 'model_name', 'unknown_direct_client_object')
    print(f"INFO: execute_gemini_call for requested API model '{model_to_call}', using client object with internal name '{client_model_name_for_log}'. Original request model: '{request_obj.model}'")
    
    if request_obj.stream:
        is_image_request = "image" in request_obj.model.lower()
        
        if app_config.FAKE_STREAMING_ENABLED or is_image_request:
            if is_image_request:
                 print("INFO: 触发生图保护机制 —— 已强制切换为假流式输出以避开 Google 报错限制！")
            return StreamingResponse(
                gemini_fake_stream_generator(
                    current_client, model_to_call, actual_prompt_for_call,
                    gen_config_dict, request_obj, is_auto_attempt
                ), media_type="text/event-stream"
            )
        else: # True Streaming
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                # [核心修复]: 为原生 SDK 注入全包裹式的 20 次波浪退避护盾！
                max_retries = 20
                for attempt in range(max_retries):
                    try:
                        # 惰性调用：此时可能不报错
                        stream_gen_obj = await current_client.aio.models.generate_content_stream(
                            model=model_to_call, 
                            contents=actual_prompt_for_call,
                            config=gen_config_dict
                        )
                        # 真正的深水区：在这里迭代时随时会引爆 429
                        async for chunk_item_call in stream_gen_obj:
                            # --- 新增开始：嗅探流式最后一个 Chunk 的算力消耗 ---
                            if hasattr(chunk_item_call, 'usage_metadata') and chunk_item_call.usage_metadata:
                                um = chunk_item_call.usage_metadata
                                p_tk = getattr(um, 'prompt_token_count', 0) or 0
                                c_tk = getattr(um, 'candidates_token_count', 0) or 0
                                t_tk = getattr(um, 'total_token_count', p_tk + c_tk) or (p_tk + c_tk)
                                if p_tk > 0 or c_tk > 0:
                                    print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")
                            # --- 新增结束 ---
                            
                            yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                        
                        # 顺利迭代完毕，安全跳出重试循环
                        yield "data: [DONE]\n\n"
                        break 
                        
                    except asyncio.CancelledError:
                            print(f"INFO: Client disconnected during Real Stream ({model_to_call}). Clean abort.")
                            raise
                        except Exception as e_stream_call:
                            error_str = str(e_stream_call).lower()
                        is_retryable = False
                        
                        # 极速嗅探网络堵塞与算力拒绝
                        if "429" in error_str or "503" in error_str or "too many requests" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                            is_retryable = True
                            
                        if is_retryable and attempt < max_retries - 1:
                            wave_index = attempt % 4
                            round_num = (attempt // 4) + 1
                            wait_time = 2 ** wave_index
                            print(f"⚠️ [Gemini SDK Stream] 遭遇算力天花板/速率限制. 第 {round_num} 轮/第 {wave_index + 1} 次护盾激活，等待 {wait_time}s 后重试...")
                            await asyncio.sleep(wait_time)
                            continue # 进入深层休眠，随后重试
                            
                        # 如果不是 429 或者是死局，向下传递死亡通知
                        err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                        print(f"ERROR: {err_msg_detail_stream}")
                        s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                        err_resp = create_openai_error_response(500,s_err,"server_error")
                        j_err = json.dumps(err_resp)
                        if not is_auto_attempt: 
                            yield f"data: {j_err}\n\n"
                            yield "data: [DONE]\n\n"
                        else:
                            raise e_stream_call
            
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: # Non-streaming
        response_obj_call = await execute_with_retry(
            current_client.aio.models.generate_content,
            model=model_to_call, 
            contents=actual_prompt_for_call,
            config=gen_config_dict
        )
        if hasattr(response_obj_call, 'prompt_feedback') and \
           hasattr(response_obj_call.prompt_feedback, 'block_reason') and \
           response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and \
               response_obj_call.prompt_feedback.block_reason_message: 
                block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        
        if not is_gemini_response_valid(response_obj_call):
            error_details = f"Invalid non-streaming Gemini response for model string '{model_to_call}'. "
            if hasattr(response_obj_call, 'candidates'):
                error_details += f"Candidates: {len(response_obj_call.candidates) if response_obj_call.candidates else 0}. "
                if response_obj_call.candidates and len(response_obj_call.candidates) > 0:
                    candidate = response_obj_call.candidates if isinstance(response_obj_call.candidates, list) else response_obj_call.candidates
                    if hasattr(candidate, 'content'):
                        error_details += "Has content. "
                        if hasattr(candidate.content, 'parts'):
                            error_details += f"Parts: {len(candidate.content.parts) if candidate.content.parts else 0}. "
                            if candidate.content.parts and len(candidate.content.parts) > 0:
                                part = candidate.content.parts if isinstance(candidate.content.parts, list) else candidate.content.parts
                                if hasattr(part, 'text'):
                                    text_preview = str(getattr(part, 'text', ''))[:100]
                                    error_details += f"First part text: '{text_preview}'"
                                elif hasattr(part, 'function_call'):
                                    error_details += f"First part is function_call: {part.function_call.name}"
            else:
                error_details += f"Response type: {type(response_obj_call).__name__}"
            raise ValueError(error_details)
                # --- 新增开始：拦截非流式的 Token 消耗 ---
        if hasattr(response_obj_call, 'usage_metadata') and response_obj_call.usage_metadata:
            um = response_obj_call.usage_metadata
            p_tk = getattr(um, 'prompt_token_count', 0) or 0
            c_tk = getattr(um, 'candidates_token_count', 0) or 0
            t_tk = getattr(um, 'total_token_count', p_tk + c_tk) or (p_tk + c_tk)
            print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")
        # --- 新增结束 ---

        openai_response_content = convert_to_openai_format(response_obj_call, request_obj.model)
        return JSONResponse(content=openai_response_content)
