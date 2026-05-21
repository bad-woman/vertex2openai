"""
OpenAI handler module for creating clients and processing OpenAI Direct mode responses.
This module encapsulates all OpenAI-specific logic that was previously in chat_api.py.
"""
import asyncio
import json
import time
import httpx
import re
from typing import Dict, Any, AsyncGenerator

from fastapi.responses import JSONResponse, StreamingResponse
import openai

from models import OpenAIRequest
from config import VERTEX_REASONING_TAG
import config as app_config
from api_helpers import (
    create_openai_error_response,
    openai_fake_stream_generator,
    StreamingReasoningProcessor,
    execute_with_retry
)
from message_processing import extract_reasoning_by_tags
from credentials_manager import _refresh_auth
from project_id_discovery import discover_project_id


class FakeChatCompletionChunk:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class FakeChatCompletion:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class ExpressClientWrapper:
    def __init__(self, project_id: str, api_key: str, location: str = "global"):
        self.project_id = project_id
        self.api_key = api_key
        self.location = location
        self.base_url = f"https://aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi"
        
        self.chat = self
        self.completions = self

    async def _stream_generator(self, response: httpx.Response) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        final_p_tk, final_c_tk, final_t_tk = 0, 0, 0
        async for line in response.aiter_lines():
            if not line:
                continue
            
            if line.startswith("data:"):
                json_str = line[len("data: "):].strip()
                
                if json_str == "[DONE]":
                    break
                    
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "usage" in data and data["usage"]:
                        usage = data["usage"]
                        final_p_tk = usage.get("prompt_tokens", 0)
                        final_c_tk = usage.get("completion_tokens", 0)
                        final_t_tk = usage.get("total_tokens", final_p_tk + final_c_tk)
                    
                    yield FakeChatCompletionChunk(data)
                    
                except json.JSONDecodeError:
                    continue
        
        if final_p_tk > 0 or final_c_tk > 0:
            print(f"💰 [算力消耗] 提示词: {final_p_tk} | 模型思考与生成: {final_c_tk} | 总计: {final_t_tk} Tokens")
            
    async def _streaming_create(self, **kwargs) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        payload["stream_options"] = {"include_usage": True}

        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {'timeout': 300}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
            
        async with httpx.AsyncClient(**client_args) as client:
            max_retries = 20
            for attempt in range(max_retries):
                try:
                    async with client.stream("POST", endpoint, headers=headers, params=params, json=payload, timeout=None) as response:
                        response.raise_for_status() 
                        async for chunk in self._stream_generator(response):
                            yield chunk
                    break 
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [429, 503, 502] and attempt < max_retries - 1:
                        wave_index = attempt % 4
                        round_num = (attempt // 4) + 1
                        wait_time = 2 ** wave_index
                        print(f"⚠️ [Express Stream] 遭遇 HTTP {e.response.status_code}. 第 {round_num} 轮/第 {wave_index + 1} 次护盾激活，等待 {wait_time}s 后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise e

    async def create(self, **kwargs) -> Any:
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            return self._streaming_create(**kwargs)
        
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {'timeout': 300}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
            
        async with httpx.AsyncClient(**client_args) as client:
            max_retries = 20
            for attempt in range(max_retries):
                try:
                    response = await client.post(endpoint, headers=headers, params=params, json=payload, timeout=None)
                    response.raise_for_status()
                    resp_json = response.json()
                    if isinstance(resp_json, dict) and "usage" in resp_json and resp_json["usage"]:
                        usage = resp_json["usage"]
                        p_tk = usage.get("prompt_tokens", 0)
                        c_tk = usage.get("completion_tokens", 0)
                        t_tk = usage.get("total_tokens", p_tk + c_tk)
                        print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")
                    return FakeChatCompletion(resp_json)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [429, 503, 502] and attempt < max_retries - 1:
                        wave_index = attempt % 4
                        round_num = (attempt // 4) + 1
                        wait_time = 2 ** wave_index
                        print(f"⚠️ [Express Non-Stream] 遭遇 HTTP {e.response.status_code}. 第 {round_num} 轮/第 {wave_index + 1} 次护盾激活，等待 {wait_time}s 后重试...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise e
    
class OpenAIDirectHandler:
    
    def __init__(self, credential_manager=None, express_key_manager=None):
        self.credential_manager = credential_manager
        self.express_key_manager = express_key_manager
        
        safety_threshold = "BLOCK_NONE"
        safety_method = "PROBABILITY"
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": safety_threshold, "method": safety_method},
            {"category": "HARM_CATEGORY_JAILBREAK", "threshold": safety_threshold, "method": safety_method}
        ]

    def create_openai_client(self, project_id: str, gcp_token: str, location: str = "global") -> openai.AsyncOpenAI:
        endpoint_url = (
            f"https://aiplatform.googleapis.com/v1beta1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )
        
        proxies = None
        if app_config.PROXY_URL:
            if app_config.PROXY_URL.startswith("socks"):
                proxies = {"all://": app_config.PROXY_URL}
            else:
                proxies = {"https://": app_config.PROXY_URL}

        client_args = {}
        if proxies:
            client_args['proxies'] = proxies
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        
        http_client = httpx.AsyncClient(**client_args) if client_args else None
        return openai.AsyncOpenAI(
            base_url=endpoint_url,
            api_key=gcp_token, 
            http_client=http_client,
        )
    
    def prepare_openai_params(self, request: OpenAIRequest, model_id: str, is_openai_search: bool = False) -> Dict[str, Any]:
        params = request.model_dump(exclude_unset=True)
        params['model'] = model_id
        
        if is_openai_search:
            params['web_search_options'] = {}
            
        # 兼容最新版 OpenAI 客户端传入的 max_completion_tokens，防止直连云端网关抛出不支持参数异常
        if "max_completion_tokens" in params:
            if "max_tokens" not in params:
                params["max_tokens"] = params["max_completion_tokens"]
            del params["max_completion_tokens"]
            
        openai_params = {k: v for k, v in params.items() if v is not None}
        if "reasoning_effort" in openai_params and openai_params["reasoning_effort"] not in ["low", "medium", "high"]:
            del openai_params["reasoning_effort"]
        return openai_params
    
    def prepare_extra_body(self, base_model_name: str, reasoning_effort: str = None) -> Dict[str, Any]:
        google_config = {
            "safetySettings": self.safety_settings
        }
        
        is_thinking_capable = False
        is_gemini_2_5 = False
        is_gemini_3_or_above = False
        
        version_match = re.search(r'gemini-(\d+)\.(\d+)|gemini-(\d+)', base_model_name.lower())
        if version_match:
            groups = version_match.groups()
            major = 0
            minor_val = 0.0
            
            if groups[2]:
                major = int(groups[2])
            elif groups[0] and groups[1]:
                major = int(groups[0])
                try:
                    minor_val = float(groups[1])
                except ValueError:
                    pass
            
            if major > 2 or (major == 2 and minor_val >= 5.0):
                is_thinking_capable = True
            if major == 2 and minor_val == 5.0:
                is_gemini_2_5 = True
            elif major >= 3:
                is_gemini_3_or_above = True

        if is_thinking_capable and "image" not in base_model_name.lower():
            thinking_config = {"includeThoughts": True}
            
            if is_gemini_3_or_above:
                if reasoning_effort == "low":
                    thinking_config["thinkingLevel"] = "LOW"
                elif reasoning_effort == "medium":
                    thinking_config["thinkingLevel"] = "MEDIUM"
                else:
                    thinking_config["thinkingLevel"] = "HIGH"
            elif is_gemini_2_5:
                if reasoning_effort == "low":
                    thinking_config["thinkingBudget"] = 1024
                else:
                    thinking_config["thinkingBudget"] = -1
                    
            google_config["thinkingConfig"] = thinking_config
            
        return {
            "extra_body": {
                "google": google_config
            }
        }
    
    async def handle_streaming_response(
        self,
        openai_client: Any, 
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> StreamingResponse:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                openai_fake_stream_generator(
                    openai_client=openai_client,
                    openai_params=openai_params,
                    openai_extra_body=openai_extra_body,
                    request_obj=request,
                    is_auto_attempt=False
                ),
                media_type="text/event-stream"
            )
        else:
            return StreamingResponse(
                self._true_stream_generator(openai_client, openai_params, openai_extra_body, request),
                media_type="text/event-stream"
            )
    
    async def _true_stream_generator(
        self,
        openai_client: Any,
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> AsyncGenerator[str, None]:
        try:
            openai_params_for_stream = {
                **openai_params, 
                "stream": True,
                "stream_options": {"include_usage": True}
            }
            
            stream_response = await execute_with_retry(
                openai_client.chat.completions.create,
                **openai_params_for_stream,
                extra_body=openai_extra_body
            )
            
            reasoning_processor = StreamingReasoningProcessor()
            
            async for chunk in stream_response:
                try:
                    chunk_as_dict = chunk.model_dump(exclude_unset=True, exclude_none=True)
                    
                    if not isinstance(chunk_as_dict, dict):
                        continue
                    
                    usage = chunk_as_dict.get('usage')
                    if usage:
                        p_tk = usage.get("prompt_tokens", 0)
                        c_tk = usage.get("completion_tokens", 0)
                        t_tk = usage.get("total_tokens", p_tk + c_tk)
                        print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")

                    choices = chunk_as_dict.get('choices')
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        delta = choices[0].get('delta')
                        if delta and isinstance(delta, dict):
                            if 'extra_content' in delta:
                                del delta['extra_content']
                            
                            content = delta.get('content', '')
                            original_choice = chunk_as_dict['choices'][0]
                            if content:
                                processed_content, current_reasoning = reasoning_processor.process_chunk(content)
                                
                                original_finish_reason = original_choice.get('finish_reason')
                                original_usage = original_choice.get('usage')

                                if current_reasoning:
                                    reasoning_delta = {'reasoning_content': current_reasoning}
                                    reasoning_payload = {
                                        "id": chunk_as_dict["id"], "object": chunk_as_dict["object"],
                                        "created": chunk_as_dict["created"], "model": chunk_as_dict["model"],
                                        "choices": [{"index": 0, "delta": reasoning_delta, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(reasoning_payload)}\n\n"
                                
                                if processed_content:
                                    content_delta = {'content': processed_content}
                                    finish_reason_for_this_content_delta = None
                                    usage_for_this_content_delta = None

                                    if original_finish_reason and not reasoning_processor.inside_tag:
                                        finish_reason_for_this_content_delta = original_finish_reason
                                        if original_usage:
                                            usage_for_this_content_delta = original_usage
                                    
                                    content_payload = {
                                        "id": chunk_as_dict["id"], "object": chunk_as_dict["object"],
                                        "created": chunk_as_dict["created"], "model": chunk_as_dict["model"],
                                        "choices": [{"index": 0, "delta": content_delta, "finish_reason": finish_reason_for_this_content_delta}]
                                    }
                                    if usage_for_this_content_delta:
                                        content_payload['choices'][0]['usage'] = usage_for_this_content_delta
                                    
                                    yield f"data: {json.dumps(content_payload)}\n\n"
                                
                            elif original_choice.get('finish_reason'): 
                                yield f"data: {json.dumps(chunk_as_dict)}\n\n"
                            elif not content and not original_choice.get('finish_reason') :
                                yield f"data: {json.dumps(chunk_as_dict)}\n\n"
                    else:
                        yield f"data: {json.dumps(chunk_as_dict)}\n\n"

                except asyncio.CancelledError:
                    raise
                except Exception as chunk_error:
                    error_msg = f"Error processing OpenAI chunk for {request.model}: {str(chunk_error)}"
                    print(f"ERROR: {error_msg}")
                    if len(error_msg) > 1024:
                        error_msg = error_msg[:1024] + "..."
                    error_response = create_openai_error_response(500, error_msg, "server_error")
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            
            remaining_content, remaining_reasoning = reasoning_processor.flush_remaining()
            
            if remaining_reasoning:
                reasoning_flush_payload = {
                    "id": f"chatcmpl-flush-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": remaining_reasoning}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(reasoning_flush_payload)}\n\n"
            
            if remaining_content:
                content_flush_payload = {
                    "id": f"chatcmpl-flush-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": remaining_content}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(content_flush_payload)}\n\n"
            
            finish_payload = {
                "id": f"chatcmpl-final-{int(time.time())}", 
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(finish_payload)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except asyncio.CancelledError:
            print(f"INFO: Client disconnected during OpenAI Direct Stream ({request.model}). Releasing resources.")
            raise
        except Exception as stream_error:
            error_msg = str(stream_error)
            if len(error_msg) > 1024:
                error_msg = error_msg[:1024] + "..."
            error_msg_full = f"Error during OpenAI streaming for {request.model}: {error_msg}"
            print(f"ERROR: {error_msg_full}")
            error_response = create_openai_error_response(500, error_msg_full, "server_error")
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"             
            return
    
    async def handle_non_streaming_response(
        self,
        openai_client: Any, 
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> JSONResponse:
        try:
            openai_params_non_stream = {**openai_params, "stream": False}
            
            response = await execute_with_retry(
                openai_client.chat.completions.create,
                **openai_params_non_stream,
                extra_body=openai_extra_body
            )
            response_dict = response.model_dump(exclude_unset=True, exclude_none=True)
            
            usage = response_dict.get('usage')
            if usage:
                p_tk = usage.get("prompt_tokens", 0)
                c_tk = usage.get("completion_tokens", 0)
                t_tk = usage.get("total_tokens", p_tk + c_tk)
                print(f"💰 [算力消耗] 提示词: {p_tk} | 模型思考与生成: {c_tk} | 总计: {t_tk} Tokens")

            try:
                choices = response_dict.get('choices')
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message_dict = choices[0].get('message')
                    if message_dict and isinstance(message_dict, dict):
                        if 'extra_content' in message_dict:
                            del message_dict['extra_content']
                        
                        full_content = message_dict.get('content')
                        actual_content = full_content if isinstance(full_content, str) else ""
                        
                        if actual_content:
                            reasoning_text, actual_content = extract_reasoning_by_tags(actual_content, "think")
                            message_dict['content'] = actual_content
                            if reasoning_text:
                                message_dict['reasoning_content'] = reasoning_text
                        else:
                            message_dict['content'] = ""
                            
            except Exception as e_reasoning:
                print(f"WARNING: Error during non-streaming reasoning processing for model {request.model}: {e_reasoning}")
            
            return JSONResponse(content=response_dict)
            
        except Exception as e:
            error_msg = f"Error calling OpenAI client for {request.model}: {str(e)}"
            print(f"ERROR: {error_msg}")
            return JSONResponse(
                status_code=500, 
                content=create_openai_error_response(500, error_msg, "server_error")
            )
    
    async def process_request(self, request: OpenAIRequest, base_model_name: str, is_express: bool = False, is_openai_search: bool = False):
        print(f"INFO: Using OpenAI Direct Path for model: {request.model} (Express: {is_express})")
        
        client: Any = None 

        try:
            if is_express:
                if not self.express_key_manager:
                    raise Exception("Express mode requires an ExpressKeyManager, but it was not provided.")
                
                key_tuple = self.express_key_manager.get_express_api_key()
                if not key_tuple:
                    raise Exception("OpenAI Express Mode requires an API key, but none were available.")
                
                _, express_api_key = key_tuple
                project_id = await discover_project_id(express_api_key)
                
                client = ExpressClientWrapper(project_id=project_id, api_key=express_api_key)
                print(f"INFO: [OpenAI Express Path] Using ExpressClientWrapper for project: {project_id}")

            else: 
                if not self.credential_manager:
                    raise Exception("Standard OpenAI Direct mode requires a CredentialManager.")

                rotated_credentials, rotated_project_id = self.credential_manager.get_credentials()
                if not rotated_credentials or not rotated_project_id:
                    raise Exception("OpenAI Direct Mode requires GCP credentials, but none were available.")

                print(f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
                gcp_token = _refresh_auth(rotated_credentials)
                if not gcp_token:
                    raise Exception(f"Failed to obtain valid GCP token for OpenAI client (Project: {rotated_project_id}).")
                client = self.create_openai_client(rotated_project_id, gcp_token)

            model_id = f"google/{base_model_name}"
            
            reasoning_effort = getattr(request, "reasoning_effort", None)
            if not reasoning_effort and hasattr(request, "model_extra") and request.model_extra:
                reasoning_effort = request.model_extra.get("reasoning_effort")
                
            openai_params = self.prepare_openai_params(request, model_id, is_openai_search)
            openai_extra_body = self.prepare_extra_body(base_model_name, reasoning_effort)
            
            if request.stream:
                return await self.handle_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
            else:
                return await self.handle_non_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
        except Exception as e:
            error_msg = f"Error in process_request for {request.model}: {e}"
            print(f"ERROR: {error_msg}")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))