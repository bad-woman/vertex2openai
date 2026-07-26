"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
无需任何浏览器，直接通过 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。

支持真正的实时流式响应（防 60s 超时）、429 智能退避重试，
并在客户端断开连接时立即停止上游调用。
"""

import re
import json
import time
import uuid
import asyncio
import httpx
import traceback
from typing import Any, Optional, List, Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest, normalize_content_part
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config
import model_capabilities as mc
from message_processing import (
    apply_prefill_compat,
    strip_prefill_overlap,
    PrefillDeduper,
    openai_content_to_wire_parts,
)
from logger import stats
from api_helpers import get_retry_settings

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)

# ========== 重试关键词 ==========
# 可重试的错误关键词（429 限流类）
RETRYABLE_KEYWORDS = [
    "resource exhausted",
    "try again later",
    "429",
    "quota",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "internal error",
]

# Cookie 过期/权限失效的错误关键词（不可重试，需要刷新 Cookie）
COOKIE_EXPIRED_KEYWORDS = [
    "permission",
    "denied",
    "aiplatform.endpoints.predict",
    "not authorized",
    "unauthenticated",
    "login required",
    "session expired",
    "invalid credentials",
]

COOKIE_REFRESH_HINT = (
    "\n\n💡 Cookie 通常较为持久（只要不退出登录/改密码/被 Google 主动失效，可维持数周甚至更久）；"
    "仅当确实出现权限错误时才需更新。"
    "重新获取：电脑浏览器打开 console.cloud.google.com，F12 → Network，"
    "复制任意请求的 Cookie 头（或用 Cookie-Editor 导出），到大盘粘贴保存。"
)

# “只有思考没有正文”的可操作提示：多为原生思考在高强度下跑飞/被截断（尤其酒馆预设 + 前端恒发 xhigh）。
_THINKING_RUNAWAY_HINT = (
    "这通常是原生思考在高强度下跑飞或被截断（前端如 SillyTavern 常发 reasoning_effort=xhigh，"
    "会覆盖控制台档位）。解决：控制台“模型参数→思考强度→原生思考控制”选“关闭原生思考”"
    "（可用“保存为该模型专属”只对本模型生效），即忽略前端强度、压到 minimal 并剥离原生思考。"
)
_NO_BODY_HINT = (
    "可能被安全策略拦截或接口行为变化；原始响应样本已写入运行日志，可到大盘“运行日志”页查看。"
)


def _is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试（429 限流类）"""
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)


def _is_cookie_expired_error(error_msg: str) -> bool:
    """判断是否为 Cookie 过期/权限失效错误"""
    lower = error_msg.lower()
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS)


# ========== requestContext 模板 ==========

def _get_experiment_flags() -> str:
    """获取 experimentFlagsBinary（Express Mode 权限的关键标识，可选，从配置读取）"""
    return app_config.EXPERIMENT_FLAGS or ""


def _build_request_context(project_id: str) -> dict:
    """
    构建 batchGraphql 的 requestContext
    包含 experimentFlagsBinary，这是 Express Mode 权限的关键标识。
    """
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


# ========== 思考配置（委托中心能力模块，转 batchGraphql 的 camelCase） ==========

def _build_thinking_config(model_name: str, request: OpenAIRequest,
                           prefill_active: bool = False) -> Optional[dict]:
    """
    按模型能力档案 + 控制台设置 + 单次请求构建 thinkingConfig：
    - Gemini 3 及以上：thinkingLevel（MINIMAL/LOW/MEDIUM/HIGH）
    - Gemini 2.5：thinkingBudget（-1 动态；flash 可 0 关闭）
    - 其它/生图：None
    prefill_active=True 时按控制台开关压制思考（见 mc.resolve_thinking）。
    """
    settings = app_state.get_effective_settings(model_name)
    t = mc.resolve_thinking(model_name, request, settings, prefill_active=prefill_active)
    if t.get("mode") == "level":
        return {"thinkingLevel": t["level"].upper(), "includeThoughts": t.get("include_thoughts", True)}
    if t.get("mode") == "budget":
        return {"thinkingBudget": t["budget"], "includeThoughts": t.get("include_thoughts", True)}
    return None


# ========== OpenAI → batchGraphql 消息格式转换 ==========

def has_tool_traffic(messages: list, tools: Any = None) -> bool:
    """请求里是否含函数调用相关内容（P1-1）。

    Cookie 通道不下发 functionDeclarations，也无法表达 functionCall/functionResponse。
    旧实现把 role="tool" 一律折成 model 轮次、assistant.tool_calls 直接丢弃，
    发出去的是一段语义错乱的历史——静默出错比明确报错糟糕得多。
    """
    if tools:
        return True
    for m in messages or []:
        if getattr(m, "role", None) == "tool":
            return True
        if getattr(m, "tool_calls", None):
            return True
    return False


def _convert_messages_to_contents(messages: list) -> tuple:
    """OpenAI messages → batchGraphql contents。

    注意：内部会做远程图片抓取与 PIL 压缩（同步阻塞），
    调用方须用 asyncio.to_thread 包住（见 chat_completions）。
    """
    contents = []
    system_parts = []

    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                # 分段 system 也要收下，否则酒馆预设的多段 system 会被整段丢弃。
                # F-1：必须先归一，pydantic 已把标准 text part 变成 ContentPartText 实例。
                for p in (content or []):
                    p = normalize_content_part(p)
                    if isinstance(p, dict) and p.get("type") == "text":
                        system_parts.append(p.get("text", ""))
            continue

        # 已在入口用 has_tool_traffic() 拒绝了工具流量，这里只可能是 user / assistant
        gemini_role = "user" if role == "user" else "model"
        parts = openai_content_to_wire_parts(content)

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    # 合并连续同角色轮次（与标准通道 create_gemini_prompt 行为对齐）：
    # batchGraphql 按网页端习惯期望 user/model 交替，连续同角色可能引发异常行为。
    merged: list = []
    for c in contents:
        if merged and merged[-1]["role"] == c["role"]:
            merged[-1]["parts"].append({"text": "\n\n"})
            merged[-1]["parts"].extend(c["parts"])
        else:
            merged.append(c)

    system_text = "\n".join(t for t in system_parts if t) if system_parts else None
    return merged, system_text


async def build_batch_graphql_body_async(project_id: str, model_name: str,
                                         request: OpenAIRequest,
                                         prefill_active: bool = False) -> dict:
    """在线程里构建请求体：内部有远程图片下载与 PIL 压缩，不能阻塞事件循环（P1-2）。"""
    return await asyncio.to_thread(_build_batch_graphql_body, project_id, model_name,
                                   request, prefill_active)


def _build_batch_graphql_body(
    project_id: str,
    model_name: str,
    request: OpenAIRequest,
    prefill_active: bool = False,
) -> dict:
    contents, system_text = _convert_messages_to_contents(request.messages)
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"

    profile = mc.get_profile(model_name)
    settings = app_state.get_effective_settings(model_name)
    allowed = profile["allowed_sampling"]

    gen_config = {}

    if profile["is_image"]:
        # 生图：设 responseModalities + imageConfig，不发采样/思考参数
        gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        img_cfg = {}
        size = mc.resolve_image_size(model_name, request, settings)
        if size:
            img_cfg["imageSize"] = size
        ar = mc.resolve_aspect_ratio(model_name, request, settings)
        if ar:
            img_cfg["aspectRatio"] = ar
        if img_cfg:
            gen_config["imageConfig"] = img_cfg
    else:
        # 文本/多模态：仅注入该模型支持、且被显式设置（请求或控制台）的采样参数。
        # 与标准（Express）通道对齐：都未设置时**不发**该参数，交给模型默认——
        # 旧版会编造 temperature=1 / topP=0.95 / maxOutputTokens=65535 兜底值发出。
        if "temperature" in allowed:
            tv = request.temperature if request.temperature is not None else settings.get("default_temperature")
            if tv is not None:
                gen_config["temperature"] = tv
        if "top_p" in allowed:
            pv = request.top_p if request.top_p is not None else settings.get("default_top_p")
            if pv is not None:
                gen_config["topP"] = pv
        if "max_output_tokens" in allowed:
            mv = request.max_tokens if request.max_tokens is not None else getattr(request, "max_completion_tokens", None)
            if mv is None:
                mv = settings.get("default_max_tokens")
            if mv is not None:
                gen_config["maxOutputTokens"] = mv

        thinking_config = _build_thinking_config(model_name, request, prefill_active=prefill_active)
        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        # 与标准（Express）通道对齐：不发这个分类时，越狱式提示词（roleplay 预设常见）
        # 可能被默认越狱过滤拦掉“正文”而思考照常流出 → 表现为只有思考没有正文。
        # 已真机验证 batchGraphql 接受该分类（OFF / BLOCK_NONE 均可）。
        {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
    ]

    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }

    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}

    if request.stop and "stop_sequences" in allowed:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if hasattr(request, 'model') and request.model.endswith("-search") and profile["supports_search"]:
        variables["tools"] = [{"googleSearch": {}}]

    return {
        "requestContext": _build_request_context(project_id),
        "querySignature": STREAM_GENERATE_QUERY_SIGNATURE,
        "operationName": STREAM_GENERATE_OPERATION_NAME,
        "variables": variables,
    }


# ========== batchGraphql 流式响应解析 ==========

async def _iter_json_objects(response) -> AsyncGenerator[dict, None]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue
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
                    if c == '{':
                        brace_count += 1
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

            try:
                yield json.loads(json_str)
            except json.JSONDecodeError:
                pass


# 会被安全/合规拦截的 finishReason（映射为 OpenAI content_filter）
_FINISH_FILTERED = {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


def _map_finish_reason(raw: str | None) -> str:
    """batchGraphql finishReason → OpenAI finish_reason。"""
    r = (raw or "").upper()
    if r == "MAX_TOKENS":
        return "length"
    if r in _FINISH_FILTERED:
        return "content_filter"
    return "stop"


def _is_thought_part(part: dict) -> bool:
    """健壮的思考 part 判定：兼容 true / "true" / "True" 等表示，
    避免上游把布尔编码成字符串时（"false" 为真值）把正文误判成思考。"""
    t = part.get("thought")
    if isinstance(t, str):
        return t.strip().lower() == "true"
    return bool(t)


def _summ_obj(obj: dict, width: int = 700) -> str:
    """把一个响应对象压成截断的单行 JSON（诊断日志用，图片 base64 会掐掉）。"""
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = re.sub(r'"data":\s*"[A-Za-z0-9+/=]{64,}"', '"data": "<base64 已省略>"', s)
    return s[:width] + ("…" if len(s) > width else "")


class _RawSampler:
    """收集原始响应对象的首尾样本，供“无正文/空响应”时输出诊断日志。"""

    def __init__(self, keep: int = 2):
        self.keep = keep
        self.head: list = []
        self.tail: list = []
        self.count = 0

    def add(self, obj: dict):
        self.count += 1
        if len(self.head) < self.keep:
            self.head.append(_summ_obj(obj))
        else:
            self.tail.append(_summ_obj(obj))
            if len(self.tail) > self.keep:
                self.tail.pop(0)

    def dump(self) -> str:
        if self.count == 0:
            return "（上游未返回任何可解析的 JSON 对象）"
        lines = [f"共 {self.count} 个对象；首尾样本："]
        lines += [f"  ▸ {s}" for s in self.head]
        if self.tail:
            lines.append(f"  …")
            lines += [f"  ▸ {s}" for s in self.tail]
        return "\n".join(lines)


def _extract_from_results(obj: dict):
    if "error" in obj:
        yield ("error", obj["error"])
        return

    results = obj.get("results", [])
    for result in results:
        if "errors" in result:
            for err in result["errors"]:
                yield ("error", err)
            continue

        data = result.get("data")
        if not data:
            continue

        # 提示词级拦截（promptFeedback.blockReason）：无候选、直接被挡。
        # 注意：batchGraphql 每个流式块都会带 blockReason=BLOCKED_REASON_UNSPECIFIED
        # （proto 枚举默认值 = 未拦截），必须过滤掉，只透出真实拦截。
        pf = data.get("promptFeedback")
        if isinstance(pf, dict):
            br = str(pf.get("blockReason") or "")
            if br and not br.upper().endswith("UNSPECIFIED"):
                msg = br
                if pf.get("blockReasonMessage"):
                    msg += f"（{pf['blockReasonMessage']}）"
                yield ("blocked", msg)

        candidates = data.get("candidates", [])
        for candidate in candidates:
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts") or []

            for part in parts:
                text = part.get("text", "")
                if text:
                    if _is_thought_part(part):
                        yield ("thought", text)
                    else:
                        yield ("text", text)

                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)

            # finishReason 全量透出（含 SAFETY/PROHIBITED_CONTENT 等），由消费方统一映射；
            # 旧实现只认 STOP/MAX_TOKENS/SAFETY 白名单，其余被静默吞掉，导致“无正文”无从排查。
            # FINISH_REASON_UNSPECIFIED 是流式中间块的枚举默认值（非真实结束），需过滤。
            finish_reason = candidate.get("finishReason")
            if finish_reason and not str(finish_reason).upper().endswith("UNSPECIFIED"):
                yield ("finish", str(finish_reason))

        # 尽力解析 token 用量（私有接口通常不回传；保留解析以防未来补上）
        usage = data.get("usageMetadata")
        if isinstance(usage, dict) and usage:
            yield ("usage", usage)


# ========== token 用量映射 ==========

def _map_usage(usage_meta: dict | None) -> dict:
    """把 batchGraphql 的 usageMetadata 转成 OpenAI usage 字典。

    注意：私有 batchGraphql 接口通常不回传用量（恒为 0），因此 Cookie 通道
    不再打印 💰 统计行——大盘的 token 统计仅由标准（Express）通道计入；
    Cookie 通道的成功数改由 stats.add_success() 单独计入。
    """
    usage_meta = usage_meta or {}
    p = int(usage_meta.get("promptTokenCount", 0) or 0)
    c = int(usage_meta.get("candidatesTokenCount", 0) or 0)
    t = int(usage_meta.get("totalTokenCount", p + c) or (p + c))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


# ========== OpenAI SSE 格式化 ==========

def _make_openai_chunk(
    response_id: str,
    model: str,
    content: str = None,
    reasoning_content: str = None,
    finish_reason: str = None,
    role: str = None,
) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content

    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _sse_heartbeat() -> str:
    """SSE 注释行心跳：符合 SSE 规范、被客户端忽略、不注入任何内容。
    用于 429 退避等待期间保活连接，避免前端因长时间无字节而超时中断。"""
    return ": keep-alive\n\n"


async def _sleep_with_heartbeat(total_sec: float, hb_interval: float = 3.0):
    """边等待边吐心跳的异步生成器：每 hb_interval 秒 yield 一次心跳，直到累计 total_sec。"""
    waited = 0.0
    step = max(0.5, min(hb_interval, total_sec)) if total_sec > 0 else 0
    while waited < total_sec:
        await asyncio.sleep(min(step, total_sec - waited))
        waited += step
        yield _sse_heartbeat()


def _make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
    """OpenAI 风格的用量尾块（choices 为空，仅携带 usage）"""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ========== 认证解析 ==========

def _get_cookie_string() -> str:
    return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""

def _get_project_id() -> str:
    return app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or ""


def _wants_usage(request_obj: OpenAIRequest) -> bool:
    opts = getattr(request_obj, "stream_options", None)
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


# ========== 单次流式请求执行器（重构为真正的异步实时生成器） ==========

async def _execute_stream_request_generator(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    sampler: "_RawSampler | None" = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """
    异步生成器：真正实时地拉取数据，抛出**原始事件**（不做 SSE 格式化）：
    ("text"|"thought"|"image", str)、("finish", 原始finishReason)、("usage", dict)、
    ("blocked", str)、("api_error_text", str)、("cookie_error"|"retryable_error"|"fatal_error", str)。
    由消费方负责格式化成 OpenAI chunk（便于预填充去重与无正文诊断）。
    """
    has_content = False
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL, headers=headers, json=body) as response:

            # 1. 拦截 HTTP 状态码错误
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode('utf-8', errors='replace')[:1000]

                if response.status_code in (401, 403) or _is_cookie_expired_error(error_msg):
                    yield "cookie_error", error_msg + COOKIE_REFRESH_HINT
                    return

                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(error_msg)
                yield "retryable_error" if is_retryable else "fatal_error", error_msg
                return

            # 2. 实时遍历并抛出流式 JSON 事件块
            async for obj in _iter_json_objects(response):
                if sampler is not None:
                    sampler.add(obj)
                for event_type, data in _extract_from_results(obj):
                    if event_type in ("text", "thought", "image"):
                        has_content = True
                        yield event_type, data

                    elif event_type in ("finish", "usage", "blocked"):
                        yield event_type, data

                    elif event_type == "error":
                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)

                        # 在还没发送任何有效数据前遇到错误，尝试走顶层重试
                        if _is_cookie_expired_error(err_msg) and not has_content:
                            yield "cookie_error", err_msg + COOKIE_REFRESH_HINT
                            return
                        if _is_retryable_error(err_msg) and not has_content:
                            yield "retryable_error", err_msg
                            return

                        # 如果流已经开始输出才发生错误，直接作为文本信息告知前端
                        yield "api_error_text", err_msg

    except Exception as e:
        err_msg = str(e)
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        if not has_content:
            yield "retryable_error" if is_retryable else "fatal_error", err_msg
        else:
            # 数据传输中途断开
            yield "api_error_text", f"连接中断: {err_msg}"


async def _collect_full_response(project_id, base_model_name, request_obj, headers, client_kwargs,
                                 retry_max, backoff_sec, fastapi_request) -> dict:
    """非流式地完整取回一次响应（供生图假流式复用）。返回 dict。"""
    for attempt in range(retry_max + 1):
        if await fastapi_request.is_disconnected():
            return {"kind": "error", "message": "客户端已断开连接。"}
        try:
            body = await build_batch_graphql_body_async(
                project_id, base_model_name, request_obj)
            req_headers = build_headers(_get_cookie_string()) or headers
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)

            if response.status_code in (429, 503, 500) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] HTTP {response.status_code}（生图，尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            if response.status_code != 200:
                return {"kind": "error", "message": f"HTTP {response.status_code}: {response.text[:300]}"}

            full_text, reasoning_text, api_error, usage_meta = "", "", None, None
            finish_raw, blocked_msg = None, None
            sampler = _RawSampler()

            class _F:
                def __init__(self, t): self._t = t
                async def aiter_text(self): yield self._t

            async for obj in _iter_json_objects(_F(response.text)):
                sampler.add(obj)
                for et, data in _extract_from_results(obj):
                    if et == "text":
                        full_text += data
                    elif et == "thought":
                        reasoning_text += data
                    elif et == "image":
                        full_text += data
                    elif et == "finish":
                        finish_raw = data
                    elif et == "blocked":
                        blocked_msg = data
                    elif et == "usage":
                        usage_meta = data
                    elif et == "error":
                        em = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        if _is_retryable_error(em) and attempt < retry_max:
                            api_error = em
                            break
                        full_text += f"\n[错误] {em}"

            if api_error and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图 429/限流（尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue

            if not full_text.strip():
                # 无正文：给出可见的诊断信息，而不是静默空回复
                detail = f"finishReason={finish_raw or '无'}"
                if blocked_msg:
                    detail += f"；promptFeedback 拦截：{blocked_msg}"
                print(f"🔎 [Studio 诊断] 生图无正文（{detail}）。原始响应样本：\n{sampler.dump()}")
                return {"kind": "error", "message": f"上游未返回图片/正文（{detail}）。已在日志记录原始响应样本。"}

            return {"kind": "ok", "full_text": full_text, "reasoning_text": reasoning_text,
                    "finish_reason": _map_finish_reason(finish_raw), "usage_meta": usage_meta}
        except Exception as e:
            em = str(e)
            if (_is_retryable_error(em) or "timeout" in em.lower()) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图异常（尝试 {attempt+1}）：{em[:80]}，{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            return {"kind": "error", "message": f"batchGraphql proxy error: {em}"}
    return {"kind": "error", "message": "重试多次仍失败。"}


# ========== 主代理类 ==========

class CookieProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    使用 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。
    """

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        # ===== 1. 验证认证 =====
        cookie_str = _get_cookie_string()
        if not cookie_str:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "未配置 Google Cookie。\n"
                "请在大盘控制台中粘贴 Cookie 和 Project ID，\n"
                "或设置环境变量 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID。"
            ), "type": "auth_error"}})

        project_id = _get_project_id()
        if not project_id:
            return JSONResponse(status_code=400, content={"error": {"message": (
                "未配置 Google Cloud Project ID。\n"
                "请在大盘中填写，或设置环境变量 GOOGLE_PROJECT_ID。\n"
                "可从 Studio URL 中获取：...?project=YOUR_PROJECT_ID"
            ), "type": "config_error"}})

        # ===== 2. 构建请求头 =====
        headers = build_headers(cookie_str)
        if not headers:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "Cookie 中未找到 SAPISID，无法计算认证头。\n"
                "请确保 Cookie 来自已登录的 console.cloud.google.com 页面。"
            ), "type": "auth_error"}})

        # ===== 2.5 拒绝该通道无法表达的工具流量（P1-1）=====
        if has_tool_traffic(request_obj.messages, request_obj.tools):
            print("⛔ [Studio] 请求含函数调用内容，Cookie 直连通道不支持，已明确拒绝。")
            return JSONResponse(status_code=400, content={"error": {
                "message": ("Cookie 直连通道暂不支持函数调用（不下发 functionDeclarations，"
                            "也无法表达 functionCall / functionResponse）。\n"
                            "请在控制台切换到「标准模式（Express API Key）」后重试。"),
                "type": "unsupported_feature",
            }})

        # ===== 3. 解析模型名 =====
        model_display = request_obj.model
        base_model_name = model_display
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]

        # ===== 3.5 预填充智能兼容（按控制台模式 + 模型能力；新模型自动生效）=====
        _profile = mc.get_profile(base_model_name)
        prefill_text = ""
        prefill_active = False
        _prefill_mode = app_state.get_setting("prefill_mode", "smart")
        if _prefill_mode != "off":
            _new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not _profile["requires_user_last_turn"],
                instruction_template=app_state.get_setting("prefill_instruction", ""),
            )
            if _new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _new_msgs})
                if prefill_text:
                    print(f"🩹 [预填充兼容] 已将末尾 assistant 预填充转为续写指令（{len(prefill_text)} 字），并将拼回输出开头。")
            elif prefill_text:
                print(f"🩹 [预填充兼容] 该模型支持 model 结尾，预填充原生透传（{len(prefill_text)} 字），模型将直接续写。")
            if prefill_active and app_state.get_setting("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        # ===== 4. HTTP 客户端配置 =====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0),
            "follow_redirects": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL

        # 重试配置（控制台可调）；语义与 Express 通道统一：总尝试次数 = retry_max + 1
        retry_max, backoff_sec = get_retry_settings()

        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}"
        start_time = time.time()
        want_usage = _wants_usage(request_obj)
        cookie_debug = bool(app_state.get_setting("debug_outbound", False)
                            or app_state.get_setting("cookie_debug", False))

        # batchGraphql(Studio) 会忽略 includeThoughts=false（真机验证），因此当解析出的思考配置
        # 要求不回传思考时，由本通道在响应侧主动剥离思考块。生图无思考，strip 恒 False。
        _eff_settings = app_state.get_effective_settings(base_model_name)
        _tk = mc.resolve_thinking(base_model_name, request_obj, _eff_settings, prefill_active=prefill_active)
        strip_thoughts = bool(_tk.get("mode") and not _tk.get("include_thoughts", True))
        if strip_thoughts and cookie_debug:
            print("🔎 [Studio 调试] 已启用响应侧思考剥离（batchGraphql 不支持 includeThoughts=false）。")

        # 打印请求日志
        msg_count = len(request_obj.messages)
        print(f"→ [Studio] {base_model_name} | {msg_count} 条消息 | {'流式' if is_stream else '非流式'}")

        is_image = _profile["is_image"]

        # ========== 生图 + 流式：强制假流式 ==========
        # 生图输出是超大 base64，若按流式分块传输会卡死前端解析器；
        # 因此先完整取回，再把整张图作为“单个 chunk”一次性发出（与官方 SDK 通道一致）。
        if is_stream and is_image:
            async def image_fake_stream():
                if await fastapi_request.is_disconnected():
                    return
                print(f"🖼️ [生图保护] 图片模型 {base_model_name} 已自动切换为假流式输出（Cookie 通道），避免分块 base64 卡死前端。")
                res = await _collect_full_response(
                    project_id, base_model_name, request_obj, headers, client_kwargs,
                    retry_max, backoff_sec, fastapi_request,
                )
                yield _make_openai_chunk(response_id, model_display, role="assistant")
                if res.get("kind") != "ok":
                    stats.add_error()
                    print(f"❌ [Studio] 生图失败 | {res.get('message', '')[:150]}")
                    yield _make_openai_chunk(response_id, model_display, content=f"[Studio 错误] {res.get('message', '生图失败')}")
                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                full_text = res.get("full_text") or ""
                if prefill_text:
                    full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                if res.get("reasoning_text"):
                    yield _make_openai_chunk(response_id, model_display, reasoning_content=res["reasoning_text"])
                # 关键：整张图作为单个 chunk 发出，绝不分块
                yield _make_openai_chunk(response_id, model_display, content=full_text or " ")
                stats.add_success()
                usage = _map_usage(res.get("usage_meta"))
                if want_usage:
                    yield _make_usage_chunk(response_id, model_display, usage)
                yield _make_openai_chunk(response_id, model_display, finish_reason=res.get("finish_reason", "stop"))
                yield "data: [DONE]\n\n"
                print(f"✅ [Studio] {base_model_name} | 生图假流式完成")
            return StreamingResponse(image_fake_stream(), media_type="text/event-stream")

        # ========== 流式处理（彻底解决 60s 超时的真·流式机制） ==========
        if is_stream:
            async def stream_generator():
                nonlocal start_time

                # 立即吐一个心跳，尽快建立连接（避免首个上游调用较慢/429 时前端久等无字节）
                yield _sse_heartbeat()

                for attempt in range(retry_max + 1):
                    # 客户端已断开则立即停止，避免无谓的上游调用与重试
                    if await fastapi_request.is_disconnected():
                        print("ℹ️ [Studio] 客户端已断开连接，停止流式重试。")
                        return

                    body = await build_batch_graphql_body_async(
                        project_id, base_model_name, request_obj, prefill_active=prefill_active)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        has_yielded_to_client = False
                        got_text = False       # 是否收到过“正文”（text/image）
                        got_thought = False    # 是否收到过思考
                        should_retry = False
                        error_to_raise = None
                        usage_meta = None
                        finish_raw = None
                        blocked_msg = None
                        sampler = _RawSampler()
                        deduper = PrefillDeduper(prefill_text)

                        # 消费实时生成器（原始事件 → 此处格式化为 OpenAI chunk）
                        async for status, data in _execute_stream_request_generator(
                            client, req_headers, body, sampler=sampler
                        ):
                            # 思考剥离：batchGraphql 忽略 includeThoughts，这里按解析出的配置主动丢弃思考块
                            if status == "thought":
                                got_thought = True
                                if strip_thoughts:
                                    continue  # 不建连、不输出，等真正的正文

                            if status in ("text", "thought", "image", "api_error_text"):
                                if not has_yielded_to_client:
                                    print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                    has_yielded_to_client = True
                                    # 预填充智能兼容：先把预填充文本作为回复开头发出
                                    if prefill_text:
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)

                                if status == "thought":
                                    yield _make_openai_chunk(response_id, model_display, reasoning_content=data)
                                elif status == "api_error_text":
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio API 错误] {data}")
                                elif status == "image":
                                    got_text = True
                                    yield _make_openai_chunk(response_id, model_display, content=data)
                                else:  # text（经过预填充去重器）
                                    got_text = True
                                    out = deduper.feed(data)
                                    if out:
                                        yield _make_openai_chunk(response_id, model_display, content=out)

                            elif status == "finish":
                                finish_raw = data
                            elif status == "usage":
                                usage_meta = data
                            elif status == "blocked":
                                blocked_msg = data

                            # 如果属于 Cookie 权限错误
                            elif status == "cookie_error":
                                if not has_yielded_to_client:
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                stats.add_error()
                                print(f"🔑 [Studio] 权限错误: {data[:150]}")
                                yield _make_openai_chunk(response_id, model_display, content=f"[Studio 权限错误] {data}")
                                yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                yield "data: [DONE]\n\n"
                                return

                            # 如果属于其他网络故障或限流错误
                            elif status in ("retryable_error", "fatal_error"):
                                # 未发送任何有效数据前发生可重试错误 -> 触发安全退避重试
                                if not has_yielded_to_client and status == "retryable_error" and attempt < retry_max:
                                    should_retry = True
                                    error_to_raise = data
                                    break  # 跳出当前 async for，进入外部循环的 sleep 阶段
                                else:
                                    # 如果已经开始了输出，或者错误不可重试，直接抛给前端结束
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")

                                    err_prefix = "不可重试错误" if status == "fatal_error" else "重试耗尽"
                                    stats.add_error()
                                    print(f"❌ [Studio] {err_prefix} | {data[:150]}")
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio 错误] {data}")
                                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                    yield "data: [DONE]\n\n"
                                    return

                        # 如果标记了需要重试，在当前 attempt 结束时等待并开启下一次循环
                        if should_retry:
                            # 重试前再次确认客户端仍在
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                return
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] 遇到可重试拥堵/限流: {error_to_raise[:80]}... {wait_sec}s 后进行第 {attempt+2} 次退避重试")
                            # 退避等待期间持续吐心跳，保活前端连接（issue4：3.1-pro 频繁 429 长等待易被前端断开）
                            async for _hb in _sleep_with_heartbeat(wait_sec):
                                if await fastapi_request.is_disconnected():
                                    print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                    return
                                yield _hb
                            start_time = time.time()
                            continue

                        # ===== 正常收尾（修复旧版“空响应静默关流”，并显式暴露“只有思考没有正文”）=====
                        if not has_yielded_to_client:
                            yield _make_openai_chunk(response_id, model_display, role="assistant")
                            has_yielded_to_client = True
                            if prefill_text:
                                yield _make_openai_chunk(response_id, model_display, content=prefill_text)

                        # 预填充去重器里可能还攒着开头的一小段（短回复场景）
                        tail = deduper.flush()
                        if tail:
                            yield _make_openai_chunk(response_id, model_display, content=tail)

                        if got_text:
                            stats.add_success()
                            yield _make_openai_chunk(response_id, model_display,
                                                     finish_reason=_map_finish_reason(finish_raw))
                        else:
                            # 无正文：明确告知客户端 + 落诊断日志（旧版此处一个字节都不发就关流）
                            detail = f"finishReason={finish_raw or '无'}"
                            if blocked_msg:
                                detail += f"；promptFeedback 拦截：{blocked_msg}"
                            desc = "只返回了思考、未返回正文" if got_thought else "未返回任何内容"
                            hint = (_THINKING_RUNAWAY_HINT if (got_thought and not strip_thoughts) else _NO_BODY_HINT)
                            stats.add_error()
                            print(f"❌ [Studio] {base_model_name} | 上游{desc}（{detail}）")
                            print(f"🔎 [Studio 诊断] 原始响应样本：\n{sampler.dump()}")
                            yield _make_openai_chunk(
                                response_id, model_display,
                                content=f"\n[Studio 提示] 上游{desc}（{detail}）。{hint}")
                            filtered = bool(blocked_msg) or ((finish_raw or "").upper() in _FINISH_FILTERED)
                            yield _make_openai_chunk(response_id, model_display,
                                                     finish_reason="content_filter" if filtered else "stop")

                        usage = _map_usage(usage_meta)
                        if want_usage:
                            yield _make_usage_chunk(response_id, model_display, usage)

                        yield "data: [DONE]\n\n"

                        elapsed = time.time() - start_time
                        if got_text:
                            print(f"✅ [Studio] {base_model_name} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                        return

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # ========== 非流式处理 ==========
        else:
            for attempt in range(retry_max + 1):
                # 客户端已断开则立即停止，避免无谓的上游调用与重试
                if await fastapi_request.is_disconnected():
                    print("ℹ️ [Studio] 客户端已断开连接，停止非流式重试。")
                    return JSONResponse(status_code=499, content={
                        "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                    })
                try:
                    body = await build_batch_graphql_body_async(
                        project_id, base_model_name, request_obj, prefill_active=prefill_active)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        response = await client.post(
                            BATCH_GRAPHQL_URL, headers=req_headers, json=body
                        )

                    if response.status_code in (429, 503, 500):
                        if attempt < retry_max:
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] HTTP {response.status_code} (尝试 {attempt+1}), {wait_sec}s 后重试...")
                            await asyncio.sleep(wait_sec)
                            continue

                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        print(f"❌ [Studio] {base_model_name} | HTTP {response.status_code} | {elapsed:.1f}s")
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": response.text[:500], "type": "upstream_error"}
                        })

                    full_text = ""
                    reasoning_text = ""
                    api_error = None
                    usage_meta = None
                    finish_raw = None
                    blocked_msg = None
                    sampler = _RawSampler()

                    class _FakeResponse:
                        def __init__(self, text):
                            self._text = text
                        async def aiter_text(self):
                            yield self._text

                    fake_resp = _FakeResponse(response.text)
                    async for obj in _iter_json_objects(fake_resp):
                        sampler.add(obj)
                        for event_type, data in _extract_from_results(obj):
                            if event_type == "text":
                                full_text += data
                            elif event_type == "thought":
                                reasoning_text += data   # 先收集（供诊断判断），输出时按 strip 决定是否附带
                            elif event_type == "image":
                                full_text += data
                            elif event_type == "finish":
                                finish_raw = data
                            elif event_type == "blocked":
                                blocked_msg = data
                            elif event_type == "usage":
                                usage_meta = data
                            elif event_type == "error":
                                err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                if _is_retryable_error(err_msg) and attempt < retry_max:
                                    api_error = err_msg
                                    break
                                full_text += f"\n[错误] {err_msg}"

                    if api_error and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {api_error[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    got_text = bool(full_text.strip())
                    elapsed = time.time() - start_time

                    if got_text:
                        # 预填充智能兼容：去重后把预填充文本拼回输出开头
                        if prefill_text:
                            full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                        finish_reason = _map_finish_reason(finish_raw)
                        stats.add_success()
                        print(f"✅ [Studio] {base_model_name} | {len(full_text)} 字符 | {elapsed:.1f}s")
                    else:
                        # 无正文（只有思考 / 完全空）：返回明确诊断而不是一个空格
                        detail = f"finishReason={finish_raw or '无'}"
                        if blocked_msg:
                            detail += f"；promptFeedback 拦截：{blocked_msg}"
                        desc = "只返回了思考、未返回正文" if reasoning_text else "未返回任何内容"
                        hint = (_THINKING_RUNAWAY_HINT if (reasoning_text and not strip_thoughts) else _NO_BODY_HINT)
                        stats.add_error()
                        print(f"❌ [Studio] {base_model_name} | 上游{desc}（{detail}）| {elapsed:.1f}s")
                        print(f"🔎 [Studio 诊断] 原始响应样本：\n{sampler.dump()}")
                        full_text = f"[Studio 提示] 上游{desc}（{detail}）。{hint}"
                        filtered = bool(blocked_msg) or ((finish_raw or "").upper() in _FINISH_FILTERED)
                        finish_reason = "content_filter" if filtered else "stop"

                    usage = _map_usage(usage_meta)

                    message_obj = {"role": "assistant", "content": full_text}
                    if reasoning_text and not strip_thoughts:   # strip：batchGraphql 忽略 includeThoughts，输出侧剥离
                        message_obj["reasoning_content"] = reasoning_text

                    return JSONResponse(content={
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_display,
                        "choices": [{
                            "index": 0,
                            "message": message_obj,
                            "finish_reason": finish_reason,
                        }],
                        "usage": usage,
                    })

                except Exception as e:
                    err_msg = str(e)
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()

                    if is_retryable and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 异常 (尝试 {attempt+1}): {err_msg[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    elapsed = time.time() - start_time
                    print(f"❌ [Studio] {base_model_name} | 异常 | {elapsed:.1f}s: {err_msg[:150]}")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={
                        "error": {"message": f"batchGraphql proxy error: {err_msg}", "type": "proxy_error"}
                    })

            elapsed = time.time() - start_time
            print(f"❌ [Studio] {base_model_name} | 重试 {retry_max} 次后仍失败 | {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={
                "error": {"message": "请求被限流，已重试多次仍失败。请稍后再试。", "type": "rate_limit_error"}
            })
