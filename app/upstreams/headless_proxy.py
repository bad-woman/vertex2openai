"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
支持完整的 Function Calling (工具调用) 与 Google Search。
内置终极防弹 Schema 清洗器，完美解决所有第三方格式兼容报错。
"""

import json
import time
import uuid
import asyncio
import httpx
import traceback
from typing import Any, Optional, List, Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)

# ========== 重试配置 ==========
MAX_RETRIES = 10
RETRY_BACKOFF = [5] * 10

RETRYABLE_KEYWORDS = [
    "resource exhausted", "try again later", "429", "quota", 
    "rate limit", "overloaded", "temporarily unavailable", "internal error"
]

COOKIE_EXPIRED_KEYWORDS = [
    "permission", "denied", "aiplatform.endpoints.predict", 
    "not authorized", "unauthenticated", "login required", 
    "session expired", "invalid credentials"
]

COOKIE_REFRESH_HINT = (
    "\n\n💡 Cookie 可能已过期（PSIDTS 约 1-2 小时有效）。"
    "请重新获取：在电脑浏览器打开 console.cloud.google.com，"
    "按 F12 打开控制台，复制 Request Headers 中的 Cookie，然后到大盘粘贴新 Cookie。"
)

def _is_retryable_error(error_msg: str) -> bool:
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)

def _is_cookie_expired_error(error_msg: str) -> bool:
    lower = error_msg.lower()
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS)


# ========== 🛡️ 终极防弹 Schema 清洗器 ==========
ALLOWED_SCHEMA_FIELDS = {"type", "description", "properties", "required", "items", "enum", "format", "nullable"}

def _clean_and_convert_schema(schema: dict) -> dict:
    """递归清洗 Schema，防范第三方客户端发送的不规范数据导致 Google gRPC 崩溃"""
    if not isinstance(schema, dict):
        return schema
    
    new_schema = {}
    for k, v in schema.items():
        if k not in ALLOWED_SCHEMA_FIELDS:
            continue
        if v is None:  # 防御 1: 剔除任何 null 值
            continue

        if k == "type":
            if isinstance(v, str):
                new_schema[k] = v.upper()
            elif isinstance(v, list) and len(v) > 0:
                new_schema[k] = str(v[0]).upper()
        elif k == "properties":
            if isinstance(v, dict):
                cleaned_props = {}
                for pk, pv in v.items():
                    # 防御 2: 严防 null 键、空字符串键以及 null 属性值
                    if pk and str(pk).strip() and pv is not None:
                        cleaned_props[str(pk)] = _clean_and_convert_schema(pv)
                if cleaned_props:
                    new_schema[k] = cleaned_props
        elif k == "items":
            if isinstance(v, dict):
                cleaned_items = _clean_and_convert_schema(v)
                if cleaned_items:
                    new_schema[k] = cleaned_items
        elif k == "required":
            if isinstance(v, list):
                reqs = [str(x) for x in v if x and str(x).strip()]
                if reqs:
                    new_schema[k] = reqs
        elif k == "enum":
            if isinstance(v, list):
                enums = [str(x) for x in v if x is not None]
                if enums:
                    new_schema[k] = enums
        else:
            new_schema[k] = v
            
    # 防御 3: 强制兜底类型补全（Gemini 必须要有 type）
    if "type" not in new_schema:
        if "properties" in new_schema:
            new_schema["type"] = "OBJECT"
        elif "items" in new_schema:
            new_schema["type"] = "ARRAY"
        else:
            new_schema["type"] = "STRING"
            
    # 防御 4: 防止 required 引用了不存在的属性导致校验失败
    if "required" in new_schema:
        if "properties" in new_schema:
            valid_reqs = [r for r in new_schema["required"] if r in new_schema["properties"]]
            if valid_reqs:
                new_schema["required"] = valid_reqs
            else:
                del new_schema["required"]
        else:
            del new_schema["required"]
            
    return new_schema


# ========== requestContext 模板 ==========
def _get_experiment_flags() -> str:
    if app_config.EXPERIMENT_FLAGS:
        return app_config.EXPERIMENT_FLAGS
    bundle = app_state.get_auth_bundle()
    if bundle and "body" in bundle and isinstance(bundle["body"], dict):
        req_ctx = bundle["body"].get("requestContext")
        if req_ctx and isinstance(req_ctx, dict):
            flags = req_ctx.get("experimentFlagsBinary")
            if flags:
                return flags
    return ""

def _build_request_context(project_id: str) -> dict:
    return {
        "clientVersion": "boq_cloud-boq-clientweb-vertexaistudio_20260609.06_p0",
        "pagePath": "/agent-platform/studio/multimodal",
        "pageViewId": int(time.time() * 1000) % (10**15),
        "trackingId": str(int(time.time() * 1000000) % (10**17)),
        "backendOverrides": {},
        "clientSessionId": str(uuid.uuid4()).upper(),
        "projectId": project_id,
        "selectedPurview": {"projectId": project_id},
        "jurisdiction": "global",
        "experimentFlagsBinary": _get_experiment_flags(),
        "localizationData": {"locale": "zh_CN", "timezone": "Asia/Hong_Kong"}
    }

# ========== OpenAI → batchGraphql 消息格式转换 ==========
def _convert_messages_to_contents(messages: list) -> tuple:
    contents = []
    system_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue
        
        parts = []
        
        if role == "tool":
            gemini_role = "user" 
            tool_output_data = {}
            try:
                if isinstance(content, str) and (content.strip().startswith("{") or content.strip().startswith("[")):
                    tool_output_data = json.loads(content)
                else:
                    tool_output_data = {"result": content}
            except json.JSONDecodeError:
                tool_output_data = {"result": str(content)}
            
            parts.append({
                "functionResponse": {
                    "name": msg.name or "unknown",
                    "response": tool_output_data
                }
            })
            
        elif role == "assistant" and getattr(msg, "tool_calls", None):
            gemini_role = "model"
            for tool_call in msg.tool_calls:
                func_data = tool_call.get("function", {})
                args_str = func_data.get("arguments", "{}")
                try:
                    args_dict = json.loads(args_str)
                except json.JSONDecodeError:
                    args_dict = {}
                parts.append({
                    "functionCall": {
                        "name": func_data.get("name", "unknown"),
                        "args": args_dict
                    }
                })
            if content:
                parts.append({"text": str(content)})
                
        else:
            gemini_role = "user" if role == "user" else "model"
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if hasattr(item, 'model_dump'):
                        item = item.model_dump()
                    
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item_type == "image_url":
                            url = item.get("image_url", {})
                            if isinstance(url, dict): url = url.get("url", "")
                            if url.startswith("data:"):
                                try:
                                    header, encoded = url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
                                except Exception:
                                    parts.append({"text": "[图片解析失败]"})
                    elif isinstance(item, str):
                        parts.append({"text": item})
        
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    
    system_text = "\n".join(system_parts) if system_parts else None
    return contents, system_text


def _build_batch_graphql_body(project_id: str, model_name: str, request: OpenAIRequest) -> dict:
    contents, system_text = _convert_messages_to_contents(request.messages)
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"
    
    gen_config = {
        "temperature": request.temperature if request.temperature is not None else 1,
        "topP": request.top_p if request.top_p is not None else 0.95,
        "maxOutputTokens": request.max_tokens if request.max_tokens is not None else 65535,
    }
    
    if any(kw in model_name for kw in ("gemini-3", "gemini-2.5")):
        gen_config["thinkingConfig"] = {"thinkingLevel": "HIGH", "includeThoughts": True}
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ]
    
    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }
    
    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}
    if request.stop:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
    
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
                        parameters = _clean_and_convert_schema(parameters)
                        if parameters:
                            parameters["type"] = "OBJECT"
                            declaration["parameters"] = parameters
                    function_declarations.append(declaration)
        if function_declarations:
            tools_list.append({"functionDeclarations": function_declarations})

    if hasattr(request, 'model') and request.model.endswith("-search"):
        tools_list.append({"googleSearch": {}})
        
    if tools_list:
        variables["tools"] = tools_list

    if request.tool_choice:
        choice = request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none": mode = "NONE"
            elif choice in ["auto", "auto_call"]: mode = "AUTO"
            elif choice == "required": mode = "ANY"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"
                allowed_functions = [func_name]
        if mode:
            tool_config = {"functionCallingConfig": {"mode": mode}}
            if allowed_functions:
                tool_config["functionCallingConfig"]["allowedFunctionNames"] = allowed_functions
            variables["toolConfig"] = tool_config
    
    body = {
        "requestContext": _build_request_context(project_id),
        "querySignature": STREAM_GENERATE_QUERY_SIGNATURE,
        "operationName": STREAM_GENERATE_OPERATION_NAME,
        "variables": variables,
    }
    
    return body

# ========== batchGraphql 流式响应解析 ==========
async def _iter_json_objects(response) -> AsyncGenerator[dict, None]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk: continue
        buffer += chunk
        
        while True:
            start = buffer.find('{')
            if start == -1:
                buffer = ""
                break
            
            brace_count = 0
            in_string = False
            escape = False
            end = -1
            
            for i in range(start, len(buffer)):
                c = buffer[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{': brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break
            
            if end == -1:
                buffer = buffer[start:]
                break
            
            json_str = buffer[start:end + 1]
            buffer = buffer[end + 1:]
            
            try: yield json.loads(json_str)
            except json.JSONDecodeError: pass


def _extract_from_results(obj: dict):
    if "error" in obj:
        yield ("error", obj["error"])
        return
    
    results = obj.get("results", [])
    for result in results:
        if "errors" in result:
            for err in result["errors"]: yield ("error", err)
            continue
        
        data = result.get("data")
        if not data: continue
        
        candidates = data.get("candidates", [])
        for candidate in candidates:
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts") or []
            
            for part in parts:
                func_call = part.get("functionCall")
                if func_call:
                    yield ("function_call", func_call)
                    continue
                
                text = part.get("text", "")
                if text:
                    if part.get("thought", False): yield ("thought", text)
                    else: yield ("text", text)
                
                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)
            
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason in ("STOP", "MAX_TOKENS", "SAFETY"):
                yield ("finish", finish_reason)


def _make_openai_chunk(response_id: str, model: str, content: str = None, reasoning_content: str = None, finish_reason: str = None, role: str = None, tool_calls: list = None) -> str:
    delta = {}
    if role: delta["role"] = role
    if content is not None: delta["content"] = content
    if reasoning_content is not None: delta["reasoning_content"] = reasoning_content
    if tool_calls is not None: delta["tool_calls"] = tool_calls
    
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _get_cookie_string() -> str: return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""
def _get_project_id() -> str: return app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or ""

async def _execute_stream_request(client, headers, body, model_display, response_id, attempt):
    events = []
    has_content = False
    has_tool_call = False
    
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL, headers=headers, json=body) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode('utf-8', errors='replace')[:1000]
                if response.status_code in (401, 403) or _is_cookie_expired_error(error_msg):
                    print(f"🔑 [Studio] HTTP {response.status_code} Cookie 过期/权限错误 (尝试 {attempt+1})")
                    return False, [], error_msg + COOKIE_REFRESH_HINT, False
                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(error_msg)
                print(f"{'⚠️' if is_retryable else '❌'} [Studio] HTTP {response.status_code} (尝试 {attempt+1}): {error_msg[:150]}")
                return False, [], error_msg, is_retryable
            
            async for obj in _iter_json_objects(response):
                for event_type, data in _extract_from_results(obj):
                    if event_type == "text":
                        events.append(_make_openai_chunk(response_id, model_display, content=data))
                        has_content = True
                    elif event_type == "thought":
                        events.append(_make_openai_chunk(response_id, model_display, reasoning_content=data))
                        has_content = True
                    elif event_type == "image":
                        events.append(_make_openai_chunk(response_id, model_display, content=data))
                        has_content = True
                    elif event_type == "function_call":
                        func_name = data.get("name", "unknown")
                        args_dict = data.get("args", {})
                        tool_call_id = f"call_{response_id}_{int(time.time()*1000)}_{func_name}"
                        tc = [{"index": 0, "id": tool_call_id, "type": "function", "function": {"name": func_name, "arguments": json.dumps(args_dict, ensure_ascii=False)}}]
                        events.append(_make_openai_chunk(response_id, model_display, tool_calls=tc))
                        has_content = True
                        has_tool_call = True
                    elif event_type == "finish":
                        fr = "tool_calls" if has_tool_call else ("stop" if data == "STOP" else "length" if data == "MAX_TOKENS" else "stop")
                        events.append(_make_openai_chunk(response_id, model_display, finish_reason=fr))
                    elif event_type == "error":
                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        if _is_cookie_expired_error(err_msg) and not has_content:
                            print(f"🔑 [Studio] Cookie 过期/权限错误: {err_msg[:150]}")
                            return False, [], err_msg + COOKIE_REFRESH_HINT, False
                        if _is_retryable_error(err_msg) and not has_content:
                            print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {err_msg[:150]}")
                            return False, [], err_msg, True
                        print(f"❌ [Studio] API 错误: {err_msg[:200]}")
                        events.append(_make_openai_chunk(response_id, model_display, content=f"\n[Studio API 错误] {err_msg}"))
            
            return True, events, None, False
    except Exception as e:
        err_msg = str(e)
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        print(f"{'⚠️' if is_retryable else '❌'} [Studio] 异常 (尝试 {attempt+1}): {err_msg[:150]}")
        return False, [], err_msg, is_retryable

class HeadlessProxyUpstream(BaseUpstream):
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        cookie_str = _get_cookie_string()
        if not cookie_str: return JSONResponse(status_code=401, content={"error": {"message": "未配置 Google Cookie。", "type": "auth_error"}})
        project_id = _get_project_id()
        if not project_id: return JSONResponse(status_code=400, content={"error": {"message": "未配置 Google Cloud Project ID。", "type": "config_error"}})
        headers = build_headers(cookie_str)
        if not headers: return JSONResponse(status_code=401, content={"error": {"message": "Cookie 中未找到 SAPISID，无法计算认证头。", "type": "auth_error"}})
        
        model_display = request_obj.model
        base_model_name = model_display
        if base_model_name.endswith("-search"): base_model_name = base_model_name[:-len("-search")]
        
        client_kwargs = {"timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0), "follow_redirects": True}
        if app_config.PROXY_URL: client_kwargs["proxy"] = app_config.PROXY_URL
        
        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}"
        start_time = time.time()
        
        msg_count = len(request_obj.messages)
        tool_count = len(request_obj.tools) if request_obj.tools else 0
        print(f"→ [Studio] {base_model_name} | {msg_count} 条消息 | {tool_count} 个工具 | {'流式' if is_stream else '非流式'}")
        
        if is_stream:
            async def stream_generator():
                nonlocal start_time
                for attempt in range(MAX_RETRIES + 1):
                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        success, events, error_msg, is_retryable = await _execute_stream_request(client, req_headers, body, model_display, response_id, attempt)
                    
                    if success:
                        elapsed = time.time() - start_time
                        print(f"✅ [Studio] {base_model_name} | {len(events)} 块 | {elapsed:.1f}s")
                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                        for event in events: yield event
                        
                        has_finish = any('"finish_reason":' in e and ('"stop"' in e or '"length"' in e or '"tool_calls"' in e) for e in events)
                        if not has_finish: yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
                    elif is_retryable and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else RETRY_BACKOFF[-1]
                        print(f"🔄 [Studio] {wait_sec}s 后重试 ({attempt+2}/{MAX_RETRIES+1})...")
                        await asyncio.sleep(wait_sec)
                        start_time = time.time()
                        continue
                    else:
                        elapsed = time.time() - start_time
                        if attempt >= MAX_RETRIES: print(f"❌ [Studio] {base_model_name} | 重试 {MAX_RETRIES} 次后仍失败 | {elapsed:.1f}s")
                        else: print(f"❌ [Studio] {base_model_name} | 不可重试错误 | {elapsed:.1f}s")
                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                        yield _make_openai_chunk(response_id, model_display, content=f"[Studio 错误] {error_msg[:500]}")
                        yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                        return
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        response = await client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)
                    
                    if response.status_code in (429, 503, 500) and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt]
                        print(f"⚠️ [Studio] HTTP {response.status_code} (尝试 {attempt+1}), {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue
                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        print(f"❌ [Studio] {base_model_name} | HTTP {response.status_code} | {elapsed:.1f}s")
                        return JSONResponse(status_code=response.status_code, content={"error": {"message": response.text[:500], "type": "upstream_error"}})
                    
                    full_text = ""
                    reasoning_text = ""
                    tool_calls_list = []
                    finish_reason = "stop"
                    api_error = None
                    
                    class _FakeResponse:
                        def __init__(self, text): self._text = text
                        async def aiter_text(self): yield self._text
                    
                    fake_resp = _FakeResponse(response.text)
                    async for obj in _iter_json_objects(fake_resp):
                        for event_type, data in _extract_from_results(obj):
                            if event_type == "text" or event_type == "image": full_text += data
                            elif event_type == "thought": reasoning_text += data
                            elif event_type == "function_call":
                                func_name = data.get("name", "unknown")
                                args_dict = data.get("args", {})
                                tool_call_id = f"call_{response_id}_{int(time.time()*1000)}_{func_name}"
                                tool_calls_list.append({"id": tool_call_id, "type": "function", "function": {"name": func_name, "arguments": json.dumps(args_dict, ensure_ascii=False)}})
                            elif event_type == "finish":
                                if data == "MAX_TOKENS": finish_reason = "length"
                            elif event_type == "error":
                                err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                if _is_retryable_error(err_msg) and attempt < MAX_RETRIES: api_error = err_msg; break
                                full_text += f"\n[错误] {err_msg}"
                    
                    if api_error and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt]
                        print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {api_error[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue
                    
                    elapsed = time.time() - start_time
                    print(f"✅ [Studio] {base_model_name} | {len(full_text)} 字符 | {len(tool_calls_list)} 个工具调用 | {elapsed:.1f}s")
                    
                    if tool_calls_list: finish_reason = "tool_calls"
                    
                    message_obj = {"role": "assistant"}
                    if full_text.strip(): message_obj["content"] = full_text
                    else: message_obj["content"] = None
                    if reasoning_text: message_obj["reasoning_content"] = reasoning_text
                    if tool_calls_list: message_obj["tool_calls"] = tool_calls_list
                    
                    return JSONResponse(content={
                        "id": response_id, "object": "chat.completion", "created": int(time.time()),
                        "model": model_display, "choices": [{"index": 0, "message": message_obj, "finish_reason": finish_reason}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    })
                except Exception as e:
                    err_msg = str(e)
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
                    if is_retryable and attempt < MAX_RETRIES:
                        wait_sec = RETRY_BACKOFF[attempt]
                        print(f"⚠️ [Studio] 异常 (尝试 {attempt+1}): {err_msg[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue
                    elapsed = time.time() - start_time
                    print(f"❌ [Studio] {base_model_name} | 异常 | {elapsed:.1f}s: {err_msg[:150]}")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={"error": {"message": f"batchGraphql proxy error: {err_msg}", "type": "proxy_error"}})
            
            elapsed = time.time() - start_time
            print(f"❌ [Studio] {base_model_name} | 重试 {MAX_RETRIES} 次后仍失败 | {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={"error": {"message": "请求被限流，已重试多次仍失败。请稍后再试。", "type": "rate_limit_error"}})