"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
无需任何浏览器，直接通过 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。

支持真正的实时流式响应、长思考保活（防 60s 超时）、429 智能退避重试，
并在客户端断开连接时立即停止上游调用与重试。
"""

import re
import json
import time
import uuid
import base64
import hashlib
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
    DEFAULT_IMAGE_PREFILL_NUDGE,
    _create_safety_ratings_html,
    apply_console_injection,
    apply_prefill_compat,
    prepare_anti_truncation_request,
    strip_prefill_overlap,
    PrefillDeduper,
    FunctionCallStreamAssembler,
    openai_content_to_wire_parts,
    _coerce_tool_response,
    resolve_tool_call_signature,
    signature_from_extra,
    thought_signature_extra,
    ordinary_part_metadata,
    _requires_signature,
)
from logger import stats
from api_helpers import get_retry_settings
from signature_store import SignatureRecord, SignatureState, signature_store
from schema_validation import SchemaValidationError, validate_request_schemas

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)

# ========== 重试与错误分类关键词 ==========
RETRYABLE_KEYWORDS = [
    "resource exhausted",
    "resource_exhausted",
    "try again later",
    "429",
    "quota",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "internal error",
    "unavailable",
]

COOKIE_EXPIRED_KEYWORDS = [
    "unauthenticated",
    "login required",
    "session expired",
    "invalid credentials",
    "invalid authentication",
    "autherror",
    "not logged in",
    "cookies",
]

PROJECT_BILLING_KEYWORDS = [
    "requires billing",
    "billing to be enabled",
    "billing account",
    "billing_disabled",
    "has not been used in project",
    "service_disabled",
    "api has not been enabled",
    "is not found and cannot be used",
    "project not found",
    "consumer_invalid",
]

PROJECT_ERROR_HINT = (
    "\n\n💡 【诊断：GCP 项目/计费配置问题】\n"
    "1) 请检查该项目是否已绑定并开启计费（Billing），未开启计费会报 requires billing；\n"
    "2) 请确认控制台填写的 Project ID 是否拼写正确，且已在 Google Cloud 启用了 Vertex AI (Agent Platform) API。"
)

COOKIE_REFRESH_HINT = (
    "\n\n💡 【诊断：Google Cookie 登录态已失效】\n"
    "您的 Google 会话凭证已过期或退出登录。请重新获取 Cookie：\n"
    "电脑端打开 console.cloud.google.com → 按 F12 → Network → 刷新页面 → 复制任意请求的 Cookie 请求头（或用 Cookie-Editor 导出），到大盘「通道与凭证」页粘贴并保存激活。"
)

PREDICT_DENIED_HINT = (
    "\n\n💡 【诊断：Cookie 登录态失效（最常见）或账号无该项目权限】\n"
    "出现 Permission 'aiplatform.endpoints.predict' denied 通常有两种原因：\n"
    "1) 👉 90% 以上情况：Google Cookie 已过期或会话失效（系统将您当成了未登录匿名访问），请重新从浏览器复制最新 Cookie 到大盘保存激活；\n"
    "2) 若您刚刚更换过最新 Cookie 仍报此错：说明当前登录的 Google 账号对所填 Project ID 没有访问权限，请核对 Project ID 或切换对应账号。"
)

_THINKING_RUNAWAY_HINT = (
    "这通常是原生思考在高强度下跑飞或被截断（前端如 SillyTavern 常发 reasoning_effort=xhigh，"
    "会覆盖控制台档位）。解决：控制台“模型参数→思考强度→原生思考控制”选“关闭原生思考”"
    "（可用“保存为该模型专属”只对本模型生效），即忽略前端强度、压到 minimal 并剥离原生思考。"
)
_NO_BODY_HINT = (
    "可能被安全策略拦截或接口行为变化；原始响应样本已写入运行日志，可到大盘“运行日志”页查看。"
)


def _is_project_error(error_msg: str) -> bool:
    """是否为明确的项目计费/API未启用/项目不存在错误。"""
    lower = (error_msg or "").lower()
    return any(kw in lower for kw in PROJECT_BILLING_KEYWORDS)


def _is_predict_permission_denied(error_msg: str) -> bool:
    """是否为 predict 权限拒绝（最常见于 Cookie 失效导致匿名化，其次为项目越权）。"""
    lower = (error_msg or "").lower()
    if _is_project_error(lower):
        return False
    return (
        "aiplatform.endpoints.predict" in lower
        or "permission_denied" in lower
        or ("permission" in lower and "denied" in lower)
        or "not authorized" in lower
    )


def _is_cookie_expired_error(error_msg: str) -> bool:
    """判断是否为 Cookie 过期/会话失效或权限拒绝错误。"""
    lower = (error_msg or "").lower()
    if _is_project_error(lower):
        return False
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS) or _is_predict_permission_denied(lower)


def _get_auth_or_project_hint(error_msg: str, http_status: int = 200) -> str:
    """根据错误内容精准返回中文排查指引（解决把 Cookie 过期误报为计费问题的 Bug）。"""
    if _is_project_error(error_msg):
        return PROJECT_ERROR_HINT
    if http_status == 401 or any(kw in (error_msg or "").lower() for kw in COOKIE_EXPIRED_KEYWORDS):
        return COOKIE_REFRESH_HINT
    if _is_predict_permission_denied(error_msg) or http_status == 403:
        return PREDICT_DENIED_HINT
    return COOKIE_REFRESH_HINT


def _extract_clean_error_message(raw: Any) -> str:
    """从 batchGraphql 嵌套 JSON 或原始响应文本中提取干净、完整的错误原因（含 details / extensions）。"""
    if isinstance(raw, dict):
        err_dict = raw.get("error") if isinstance(raw.get("error"), dict) else raw
        msg = err_dict.get("message") or raw.get("message")
        code = err_dict.get("code") or (raw.get("extensions") or {}).get("code")
        status = err_dict.get("status") or (raw.get("extensions") or {}).get("status")
        details = err_dict.get("details") or (raw.get("extensions") or {}).get("details") or raw.get("extensions")
        if msg:
            prefix = " / ".join(str(x) for x in (code, status) if x)
            base = f"[{prefix}] {msg}" if prefix else str(msg)
            if details:
                try:
                    det_s = json.dumps(details, ensure_ascii=False)
                except Exception:
                    det_s = str(details)
                if det_s and det_s not in base and det_s != "{}":
                    base += f" | 详情: {det_s[:1200]}"
            return base
        return json.dumps(raw, ensure_ascii=False)[:2000]

    text = str(raw or "").strip()
    if not text:
        return "未知上游错误（响应为空）"

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
            extracted = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("error"), dict):
                    extracted.append(_extract_clean_error_message(item["error"]))
                for res in (item.get("results") or []):
                    for err in (res.get("errors") or []):
                        if isinstance(err, dict):
                            extracted.append(_extract_clean_error_message(err))
            if extracted:
                return "；".join(extracted)[:2000]
        except Exception:
            pass

    msgs = re.findall(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if msgs:
        cleaned = []
        for raw_m in msgs:
            try:
                cm = json.loads(f'"{raw_m}"')
            except Exception:
                cm = raw_m
            if cm and cm not in cleaned:
                cleaned.append(cm)
        if cleaned:
            return "；".join(cleaned)[:2000]

    if "<html" in text.lower() or "<!doctype" in text.lower():
        title_m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        body_clean = re.sub(r"<[^>]+>", " ", text)
        body_clean = re.sub(r"\s+", " ", body_clean).strip()
        if title_m:
            return f"HTML 错误页 ({title_m.group(1).strip()}): {body_clean[:800]}"
        return f"HTML 错误页: {body_clean[:800]}"

    return text[:2000]

def _diagnose_studio_error(err_msg: str, http_status: int = 200) -> str:
    """为 Studio (Cookie) 通道常见错误生成一目了然的中文原因摘要。"""
    low = (err_msg or "").lower()
    if "query_signature_not_found" in low:
        return "Google 已轮换控制台私有接口签名（QUERY_SIGNATURE_NOT_FOUND），需抓包更新 cookie_auth.py 中的 STREAM_GENERATE_QUERY_SIGNATURE"
    if _is_project_error(low):
        return "GCP 项目未开启计费（Billing）或未启用接口"
    if http_status == 401 or any(kw in low for kw in COOKIE_EXPIRED_KEYWORDS):
        return "Google Cookie 已过期或未登录"
    if _is_predict_permission_denied(low) or http_status == 403:
        return "Cookie 登录态失效（最常见）或当前账号对该项目无权限"
    if http_status == 429 or _is_retryable_error(low):
        return "触发上游配额或频率限制（429 Resource Exhausted / 服务繁忙）"
    if "not found" in low or http_status == 404:
        return "请求的模型不存在或在当前项目不可用（404 Not Found）"
    if "invalid argument" in low or http_status == 400:
        return "请求参数不被该模型接受（400 Invalid Argument）"
    return f"上游返回异常（HTTP {http_status}）"


def _safety_html_if_enabled(ratings: Any) -> str:
    """开关打开时把 batchGraphql 的 safetyRatings 渲染成附加块，否则空串。"""
    if not ratings:
        return ""
    if not app_state.get_setting("safety_score", app_config.SAFETY_SCORE):
        return ""
    return _create_safety_ratings_html(ratings)


def _prefill_tpl(user_template: str, is_image_model: bool) -> str:
    """选用续写指令：生图模型在用户未自定义时改用要图片的那句。"""
    tpl = (user_template or "").strip()
    if tpl:
        return tpl
    return DEFAULT_IMAGE_PREFILL_NUDGE if is_image_model else ""


def _prefill_log(mode: str, prefill_text: str) -> str:
    """按模式说明预填充被怎么处理了。"""
    n = len(prefill_text)
    if mode == "keep_turn":
        return (f"🩹 [预填充兼容] 预填充（{n} 字）保留为 model 轮次，其后补一句续写推动语；"
                "输出会把它拼回开头。")
    if mode == "minimal":
        return ("🩹 [预填充兼容] 仅补占位 user 保证不报错；预填充**不会**拼回输出开头"
                "（预设的思考开标签可能因此缺失，酒馆正则会抓不到）。")
    return (f"🩹 [预填充兼容] 预填充（{n} 字）已并入末尾 user 消息作为续写指令，并将拼回输出开头。"
            "若预填充停在半截词/半截标签且模型接不上，可试试「保留模型轮次」。")


def _is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试（429 限流类）"""
    lower = (error_msg or "").lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)


# ========== requestContext 模板 ==========

def _get_experiment_flags() -> str:
    return app_config.EXPERIMENT_FLAGS or ""


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


# ========== 思考配置 ==========

def _build_thinking_config(model_name: str, request: OpenAIRequest,
                           prefill_active: bool = False) -> Optional[dict]:
    settings = app_state.get_effective_settings(model_name)
    t = mc.resolve_thinking(model_name, request, settings, prefill_active=prefill_active)
    if t.get("mode") == "level":
        return {"thinkingLevel": t["level"].upper(), "includeThoughts": t.get("include_thoughts", True)}
    if t.get("mode") == "budget":
        return {"thinkingBudget": t["budget"], "includeThoughts": t.get("include_thoughts", True)}
    return None


# ========== OpenAI → batchGraphql 消息格式转换 ==========

_BUILTIN_SEARCH_NAMES = {
    "google_search", "googlesearch", "google_search_retrieval",
    "web_search", "websearch", "search_web", "browse", "search",
}


def classify_tool_traffic(messages: list, tools: Any = None) -> dict:
    declared = bool(tools)
    builtin_search = False
    custom_names: list = []
    for t in (tools or []):
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = str(fn.get("name") or t.get("name") or t.get("type") or "").strip()
        if name.lower().replace("-", "_") in _BUILTIN_SEARCH_NAMES:
            builtin_search = True
        elif name:
            custom_names.append(name)

    history = False
    for m in messages or []:
        if getattr(m, "role", None) == "tool" or getattr(m, "tool_calls", None):
            history = True
            break

    return {"declared": declared, "history": history,
            "builtin_search": builtin_search, "custom_names": custom_names}


def has_tool_traffic(messages: list, tools: Any = None) -> bool:
    tt = classify_tool_traffic(messages, tools)
    return bool(tt["declared"] or tt["history"])


_SCHEMA_TYPES = {
    "object": "OBJECT", "array": "ARRAY", "string": "STRING",
    "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN",
    "null": "NULL",
}


def _cookie_ui_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    out = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if str(t).lower() != "null"]
        raw_type = non_null[0] if non_null else "null"
        if len(non_null) != len(schema.get("type") or []):
            out["nullable"] = True
    if not raw_type:
        if isinstance(schema.get("properties"), dict):
            raw_type = "object"
        elif "items" in schema:
            raw_type = "array"
    if raw_type:
        out["type"] = _SCHEMA_TYPES.get(str(raw_type).lower(), str(raw_type).upper())

    for key, value in schema.items():
        if key == "type":
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = [{"key": name, "value": _cookie_ui_schema(child)}
                        for name, child in value.items()]
        elif key in {"items", "additionalProperties", "not"} and isinstance(value, dict):
            out[key] = _cookie_ui_schema(value)
        elif key in {"oneOf", "anyOf", "allOf", "prefixItems"} and isinstance(value, list):
            out[key] = [_cookie_ui_schema(item) for item in value]
        else:
            out[key] = value
    return out


def _tool_function(tool: Any) -> dict:
    return tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else {}


def _normalized_tool_name(tool: Any) -> str:
    fn = _tool_function(tool)
    return str(fn.get("name") or (tool.get("name") if isinstance(tool, dict) else "") or "").strip()


def _is_builtin_search_name(name: str) -> bool:
    return name.lower().replace("-", "_") in _BUILTIN_SEARCH_NAMES


def _build_function_declarations(tools: Any) -> tuple[list[dict], bool]:
    declarations = []
    builtin_search = False
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = _normalized_tool_name(tool)
        if not name:
            continue
        if _is_builtin_search_name(name):
            builtin_search = True
            continue
        fn = _tool_function(tool)
        declaration = {"name": name}
        if fn.get("description") is not None:
            declaration["description"] = fn["description"]
        declaration["parameters"] = _cookie_ui_schema(fn.get("parameters") or {"type": "object"})
        declarations.append(declaration)
    return declarations, builtin_search


def _build_tool_config(tool_choice: Any, function_names: list[str], stream_args: bool = False) -> Optional[dict]:
    if tool_choice is None:
        mode, allowed = "AUTO", None
    elif isinstance(tool_choice, str):
        choice = tool_choice.lower()
        if choice == "none":
            mode, allowed = "NONE", None
        elif choice == "required":
            mode, allowed = "ANY", function_names or None
        else:
            mode, allowed = "AUTO", None
    elif isinstance(tool_choice, dict):
        name = _normalized_tool_name(tool_choice)
        mode = "ANY"
        allowed = None if (name and _is_builtin_search_name(name)) else (
            [name] if name else (function_names or None))
    else:
        mode, allowed = "AUTO", None

    config: Dict[str, Any] = {"mode": mode}
    if allowed:
        config["allowedFunctionNames"] = allowed
    if stream_args:
        config["streamFunctionCallArguments"] = True
    return {"functionCallingConfig": config}


def _native_tool_error(error_msg: str) -> bool:
    lower = (error_msg or "").lower()
    tool_terms = (
        "functiondeclaration", "function declaration", "functioncallingconfig",
        "function calling", "toolconfig", "tool config", "variables.tools",
        "streamfunctioncallarguments", "allowedfunctionnames",
    )
    reject_terms = (
        "unsupported", "not supported", "unknown field", "unrecognized",
        "invalid argument", "invalid value", "cannot use", "not enabled",
        "error in $toolconfig", "error in $tools",
    )
    return any(term in lower for term in tool_terms) and any(term in lower for term in reject_terms)

def _native_tool_error_response(status_code: int, payload: str) -> bool:
    if status_code != 200:
        return _native_tool_error(payload)
    has_error_envelope = bool(re.search(r'"errors?"\s*:', payload or ""))
    return has_error_envelope and _native_tool_error(payload)


def _stable_cookie_tool_call_id(response_id: str, call_index: int, name: str, args: Any) -> str:
    canonical = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{response_id}\0{call_index}\0{name}\0{canonical}".encode()).hexdigest()[:20]
    return f"call_{digest}"


def _signature_bytes_from_wire(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value or None
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value, validate=True) or None
    except Exception:
        return value.encode("utf-8")


def _openai_tool_call(response_id: str, call_index: int, call: dict) -> dict:
    name, args = call["name"], call.get("args") or {}
    call_id = call.get("id") or _stable_cookie_tool_call_id(response_id, call_index, name, args)
    signature = _signature_bytes_from_wire(call.get("thought_signature"))
    if signature:
        signature_store.put_record(call_id, SignatureRecord(SignatureState.SIGNED, signature))
    elif call_index > 0:
        signature_store.put_unsigned_follower(call_id)
    else:
        signature_store.put_unknown(call_id)
    payload = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
    extra = thought_signature_extra(signature)
    if extra:
        payload["extra_content"] = extra
    return payload


def _plain_text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in (content or []):
        p = normalize_content_part(p)
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
    if out:
        return "\n".join(out)
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _encode_wire_signature(signature: Optional[bytes]) -> Optional[str]:
    return base64.b64encode(signature).decode("ascii") if signature else None


def _parse_function_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _ordinary_wire_parts_from_extra(message: Any) -> Optional[list[dict]]:
    google = ((getattr(message, "extra_content", None) or {}).get("google") or {})
    raw = google.get("ordinary_parts")
    if not isinstance(raw, list):
        return None
    parts = []
    for item in raw:
        if not isinstance(item, dict) or item.get("kind") not in {"text", "thought", "signature_only"}:
            return None
        text = item.get("text")
        if not isinstance(text, str) or (item["kind"] == "text" and "data:image/" in text):
            return None
        part = {"text": text}
        if item["kind"] == "thought":
            part["thought"] = True
        if item.get("thought_signature"):
            part["thoughtSignature"] = item["thought_signature"]
        parts.append(part)
    return parts


def _attach_message_signature(parts: list[dict], message: Any) -> None:
    signature, kind = signature_from_extra(message)
    encoded = _encode_wire_signature(signature)
    if not encoded:
        return
    target = None
    if kind == "thought":
        target = next((p for p in reversed(parts) if _is_thought_part(p)), None)
    elif kind == "text":
        target = next((p for p in reversed(parts)
                       if "text" in p and not _is_thought_part(p)), None)
    if target is None and kind != "signature_only":
        target = parts[-1] if parts else None
    if target is None:
        target = {"text": ""}
        parts.append(target)
    target["thoughtSignature"] = encoded


def _convert_messages_to_contents(messages: list, model_name: str = "", native_tools: bool = True) -> tuple:
    contents = []
    system_parts = []
    require_sig = _requires_signature(model_name)
    call_names = {}
    for message in messages or []:
        for tc in (getattr(message, "tool_calls", None) or []):
            tc_id = str((tc or {}).get("id") or "")
            fn_name = str(((tc or {}).get("function") or {}).get("name") or "")
            if tc_id and fn_name:
                call_names[tc_id] = fn_name

    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                for p in (content or []):
                    p = normalize_content_part(p)
                    if isinstance(p, dict) and p.get("type") == "text":
                        system_parts.append(p.get("text", ""))
            continue

        if role == "tool":
            name = getattr(msg, "name", None) or call_names.get(getattr(msg, "tool_call_id", "") or "")
            if native_tools and name:
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {"name": name, "response": _coerce_tool_response(content)}
                }]})
            else:
                body = _plain_text_of(content)
                contents.append({"role": "user",
                                 "parts": [{"text": f"[工具执行结果 · {name or 'tool'}]\n{body}"}]})
            continue

        tool_calls = getattr(msg, "tool_calls", None)
        if role in ("assistant", "model") and tool_calls:
            if not native_tools:
                lines = []
                for tc in tool_calls:
                    fn = (tc or {}).get("function") or {}
                    lines.append(f"[请求调用工具 · {fn.get('name', 'unknown')}] 参数：{fn.get('arguments', '{}')}")
                said = _plain_text_of(content)
                if said:
                    lines.insert(0, said)
                contents.append({"role": "model", "parts": [{"text": "\n".join(lines)}]})
                continue

            call_parts = []
            for tool_index, tc in enumerate(tool_calls):
                fn = (tc or {}).get("function") or {}
                explicit_sig, _ = signature_from_extra(tc)
                _, signature = resolve_tool_call_signature(
                    str((tc or {}).get("id") or ""),
                    require_signature=require_sig and tool_index == 0,
                    explicit_signature=explicit_sig,
                    explicit_unsigned=tool_index > 0,
                )
                part = {"functionCall": {
                    "name": str(fn.get("name") or "unknown"),
                    "args": _parse_function_args(fn.get("arguments", {})),
                }}
                encoded = _encode_wire_signature(signature)
                if encoded:
                    part["thoughtSignature"] = encoded
                call_parts.append(part)

            ordinary_parts = _ordinary_wire_parts_from_extra(msg)
            if ordinary_parts is None:
                ordinary_parts = []
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    ordinary_parts.append({"text": reasoning, "thought": True})
                ordinary_parts.extend(openai_content_to_wire_parts(content))
                _attach_message_signature(ordinary_parts, msg)

            google = ((getattr(msg, "extra_content", None) or {}).get("google") or {})
            order = google.get("part_order")
            if isinstance(order, list):
                parts, used_calls, used_ordinary = [], set(), set()
                for item in order:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    if item.get("type") == "tool_call" and isinstance(idx, int) and 0 <= idx < len(call_parts):
                        parts.append(call_parts[idx])
                        used_calls.add(idx)
                    elif item.get("type") == "ordinary" and isinstance(idx, int) and 0 <= idx < len(ordinary_parts):
                        parts.append(ordinary_parts[idx])
                        used_ordinary.add(idx)
                parts.extend(p for i, p in enumerate(call_parts) if i not in used_calls)
                parts.extend(p for i, p in enumerate(ordinary_parts) if i not in used_ordinary)
            else:
                parts = call_parts + ordinary_parts
            contents.append({"role": "model", "parts": parts})
            continue

        gemini_role = "user" if role == "user" else "model"
        parts = (_ordinary_wire_parts_from_extra(msg)
                 if role in ("assistant", "model") else None)
        if parts is None:
            parts = openai_content_to_wire_parts(content)
            if role in ("assistant", "model"):
                _attach_message_signature(parts, msg)
        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    merged: list = []
    for c in contents:
        if merged and merged[-1]["role"] == c["role"]:
            previous_fr = bool(merged[-1]["parts"]) and all(
                "functionResponse" in p for p in merged[-1]["parts"])
            current_fr = bool(c["parts"]) and all("functionResponse" in p for p in c["parts"])
            if previous_fr != current_fr:
                merged.append(c)
                continue
            if not (previous_fr and current_fr):
                merged[-1]["parts"].append({"text": "\n\n"})
            merged[-1]["parts"].extend(c["parts"])
        else:
            merged.append(c)

    system_text = "\n".join(t for t in system_parts if t) if system_parts else None
    return merged, system_text


async def build_batch_graphql_body_async(project_id: str, model_name: str,
                                         request: OpenAIRequest,
                                         prefill_active: bool = False,
                                         force_search: bool = False,
                                         native_tools: bool = True,
                                         stream_fc_args: bool = False) -> dict:
    return await asyncio.to_thread(_build_batch_graphql_body, project_id, model_name,
                                   request, prefill_active, force_search, native_tools, stream_fc_args)


def _build_batch_graphql_body(
    project_id: str,
    model_name: str,
    request: OpenAIRequest,
    prefill_active: bool = False,
    force_search: bool = False,
    native_tools: bool = True,
    stream_fc_args: bool = False,
) -> dict:
    contents, system_text = _convert_messages_to_contents(
        request.messages, model_name=model_name, native_tools=native_tools)
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"

    settings = app_state.get_effective_settings(model_name)
    profile = mc.apply_sampling_policy(mc.get_profile(model_name), settings)
    allowed = profile["allowed_sampling"]

    gen_config = {}

    if profile["is_image"]:
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
        thinking_config = _build_thinking_config(model_name, request, prefill_active=prefill_active)
        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config
        if "nano-banana" in (model_name or "").lower():
            gen_config["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"
    else:
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

    _want_scores = bool(app_state.get_setting("safety_score", app_config.SAFETY_SCORE))
    _threshold = "BLOCK_NONE" if _want_scores else "OFF"
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": _threshold},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": _threshold},
        {"category": "HARM_CATEGORY_JAILBREAK", "threshold": _threshold},
    ]
    if profile["is_image"] and "nano-banana" in (model_name or "").lower():
        safety_settings.extend([
            {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        ])

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

    declarations, declared_search = _build_function_declarations(request.tools) if native_tools else ([], False)
    if profile["is_image"]:
        declarations = []

    choice_name = _normalized_tool_name(request.tool_choice) if isinstance(request.tool_choice, dict) else ""
    forced_search = bool(choice_name and _is_builtin_search_name(choice_name))
    tools_disabled = isinstance(request.tool_choice, str) and request.tool_choice.lower() == "none"
    _wants_search = (force_search or declared_search or forced_search
                     or (hasattr(request, 'model') and request.model.endswith("-search")))

    selected_declarations = []
    use_google_search = False
    if not tools_disabled:
        if forced_search:
            use_google_search = bool(profile["supports_search"])
        elif declarations:
            selected_declarations = declarations
        elif _wants_search and profile["supports_search"]:
            use_google_search = True

    if selected_declarations:
        variables["tools"] = [{"functionDeclarations": selected_declarations}]
        function_names = [item["name"] for item in selected_declarations]
        variables["toolConfig"] = _build_tool_config(request.tool_choice, function_names, stream_args=stream_fc_args)
    elif use_google_search:
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


_FINISH_FILTERED = {"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


def _map_finish_reason(raw: str | None) -> str:
    r = (raw or "").upper()
    if r == "MAX_TOKENS":
        return "length"
    if r in _FINISH_FILTERED:
        return "content_filter"
    return "stop"


def _is_thought_part(part: dict) -> bool:
    t = part.get("thought")
    if isinstance(t, str):
        return t.strip().lower() == "true"
    return bool(t)


def _summ_obj(obj: dict, width: int = 700) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = re.sub(r'"data":\s*"[A-Za-z0-9+/=]{64,}"', '"data": "<base64 已省略>"', s)
    return s[:width] + ("…" if len(s) > width else "")


class _RawSampler:
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
            lines.append("  …")
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

                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    fc_name = str(function_call.get("name") or "").strip()
                    fc_args = function_call.get("args") if isinstance(function_call.get("args"), dict) else {}
                    fc_partial = function_call.get("partialArgs") if isinstance(function_call.get("partialArgs"), list) else []
                    fc_wc = bool(function_call.get("willContinue"))
                    if fc_name or fc_args or fc_partial or fc_wc:
                        yield ("function_call", {
                            "id": str(function_call.get("id") or "").strip() or None,
                            "name": fc_name,
                            "args": fc_args,
                            "partialArgs": fc_partial,
                            "willContinue": fc_wc,
                            "thought_signature": part.get("thoughtSignature") or None,
                        })
                        continue

                if part.get("thoughtSignature"):
                    yield ("thought_signature", {
                        "value": part["thoughtSignature"],
                        "part_kind": "thought" if _is_thought_part(part) else ("text" if text else "signature_only"),
                    })

                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)

            finish_reason = candidate.get("finishReason")
            if finish_reason and not str(finish_reason).upper().endswith("UNSPECIFIED"):
                ratings = candidate.get("safetyRatings")
                if ratings:
                    yield ("safety", ratings)
                yield ("finish", str(finish_reason))

        usage = data.get("usageMetadata")
        if isinstance(usage, dict) and usage:
            yield ("usage", usage)


# ========== token 用量映射 ==========

def _map_usage(usage_meta: dict | None) -> dict:
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
    tool_calls: Optional[list[dict]] = None,
    extra_content: Optional[dict] = None,
) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if extra_content is not None:
        delta["extra_content"] = extra_content

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
    return ": keep-alive\n\n"


async def _sleep_with_heartbeat(total_sec: float, hb_interval: float = 3.0):
    """边等待边吐心跳的异步生成器：以 0.5s 细粒度步进（便于快速响应客户端取消），每 hb_interval 秒 yield 一次心跳。"""
    waited = 0.0
    hb_acc = 0.0
    while waited < total_sec:
        step = min(0.5, total_sec - waited)
        await asyncio.sleep(step)
        waited += step
        hb_acc += step
        if hb_acc >= max(0.5, hb_interval):
            hb_acc = 0.0
            yield _sse_heartbeat()


def _make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
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


# ========== 单次流式请求执行器 ==========

async def _execute_stream_request_generator_once(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    sampler: "_RawSampler | None" = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    has_content = False
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL, headers=headers, json=body) as response:

            if response.status_code != 200:
                error_text = await response.aread()
                raw_msg = error_text.decode('utf-8', errors='replace')
                clean_msg = _extract_clean_error_message(raw_msg)
                diag = _diagnose_studio_error(clean_msg, response.status_code)
                formatted_err = f"HTTP {response.status_code}（{diag}）：{clean_msg}"

                if (
                    response.status_code in (401, 403)
                    or _is_cookie_expired_error(clean_msg)
                    or _is_project_error(clean_msg)
                ):
                    hint = _get_auth_or_project_hint(clean_msg, response.status_code)
                    yield "cookie_error", formatted_err + hint
                    return

                if _native_tool_error(clean_msg):
                    yield "native_tool_unsupported", formatted_err
                    return
                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(clean_msg)
                yield "retryable_error" if is_retryable else "fatal_error", formatted_err
                return

            async for obj in _iter_json_objects(response):
                if sampler is not None:
                    sampler.add(obj)
                for event_type, data in _extract_from_results(obj):
                    if event_type in ("text", "thought", "image", "function_call", "thought_signature"):
                        has_content = True
                        yield event_type, data

                    elif event_type in ("finish", "usage", "blocked", "safety"):
                        yield event_type, data

                    elif event_type == "error":
                        clean_msg = _extract_clean_error_message(data)
                        diag = _diagnose_studio_error(clean_msg, 200)
                        formatted_err = f"{diag} — {clean_msg}"

                        if _native_tool_error(clean_msg) and not has_content:
                            yield "native_tool_unsupported", formatted_err
                            return
                        if (_is_cookie_expired_error(clean_msg) or _is_project_error(clean_msg)) and not has_content:
                            hint = _get_auth_or_project_hint(clean_msg, 200)
                            yield "cookie_error", formatted_err + hint
                            return
                        if _is_retryable_error(clean_msg) and not has_content:
                            yield "retryable_error", formatted_err
                            return
                        if not has_content:
                            yield "fatal_error", formatted_err
                            return

                        yield "api_error_text", formatted_err

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        if not has_content:
            yield "retryable_error" if is_retryable else "fatal_error", err_msg
        else:
            yield "api_error_text", f"连接中断 ({err_msg})"


async def _execute_stream_request_generator(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    sampler: "_RawSampler | None" = None,
    fallback_body: Any = None,
    fallback_state: Optional[dict] = None,
    plain_fallback_body: Any = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    async for status, data in _execute_stream_request_generator_once(client, headers, body, sampler):
        if status != "native_tool_unsupported":
            yield status, data
            continue
        if fallback_body is None:
            yield "fatal_error", data
            return
        if fallback_state is not None:
            fallback_state["latched"] = True
        resolved_fallback = fallback_body() if callable(fallback_body) else fallback_body
        if hasattr(resolved_fallback, "__await__"):
            resolved_fallback = await resolved_fallback
        print("⚠️ [Studio 自适应回退] 当前模型或 batchGraphql 协议不支持当前工具参数形态，已自动降级重试。")
        async for fb_status, fb_data in _execute_stream_request_generator_once(
                client, headers, resolved_fallback, sampler):
            if fb_status != "native_tool_unsupported":
                yield fb_status, fb_data
                continue
            if plain_fallback_body is None:
                yield "fatal_error", fb_data
                return
            resolved_plain = plain_fallback_body() if callable(plain_fallback_body) else plain_fallback_body
            if hasattr(resolved_plain, "__await__"):
                resolved_plain = await resolved_plain
            print("⚠️ [Studio 自适应回退] batchGraphql 拒绝工具声明，已自动退回纯文本模式重试。")
            async for plain_status, plain_data in _execute_stream_request_generator_once(
                    client, headers, resolved_plain, sampler):
                yield plain_status, plain_data
            return
        return


async def _collect_full_response(project_id, base_model_name, request_obj, headers, client_kwargs,
                                 retry_max, backoff_sec, fastapi_request,
                                 prefill_active: bool = False,
                                 force_search: bool = False,
                                 strip_thoughts: bool = False) -> dict:
    """完整取回一次 batchGraphql 响应（供生图假流式与 Cookie 模式假流式共用）。"""
    for attempt in range(retry_max + 1):
        if await fastapi_request.is_disconnected():
            return {"kind": "cancelled", "message": "客户端已断开连接。"}
        try:
            body = await build_batch_graphql_body_async(
                project_id, base_model_name, request_obj,
                prefill_active=prefill_active, force_search=force_search)
            req_headers = build_headers(_get_cookie_string()) or headers
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)

            if response.status_code in (429, 503, 500) and attempt < retry_max:
                if await fastapi_request.is_disconnected():
                    return {"kind": "cancelled", "message": "客户端已断开连接。"}
                stats.add_retry()
                clean_http_err = _extract_clean_error_message(response.text)
                print(f"⚠️ [Studio 假流式重试] 模型={base_model_name} | HTTP {response.status_code} | "
                      f"第 {attempt+1}/{retry_max} 次重试，等待 {backoff_sec}s | 原因: {clean_http_err[:2000]}")
                waited = 0.0
                while waited < backoff_sec:
                    step = min(0.5, backoff_sec - waited)
                    await asyncio.sleep(step)
                    waited += step
                    if await fastapi_request.is_disconnected():
                        return {"kind": "cancelled", "message": "客户端已断开连接。"}
                continue

            if response.status_code != 200:
                clean_msg = _extract_clean_error_message(response.text)
                diag = _diagnose_studio_error(clean_msg, response.status_code)
                hint = ""
                if (
                    response.status_code in (401, 403)
                    or _is_cookie_expired_error(clean_msg)
                    or _is_project_error(clean_msg)
                ):
                    hint = _get_auth_or_project_hint(clean_msg, response.status_code)
                return {
                    "kind": "error",
                    "message": f"HTTP {response.status_code}（{diag}）：{clean_msg}{hint}",
                }

            full_text, reasoning_text, retry_error, fatal_error, usage_meta = "", "", None, None, None
            finish_raw, blocked_msg = None, None
            safety_html = ""
            sampler = _RawSampler()
            fc_assembler = FunctionCallStreamAssembler()

            class _F:
                def __init__(self, t):
                    self._t = t
                async def aiter_text(self):
                    yield self._t

            async for obj in _iter_json_objects(_F(response.text)):
                sampler.add(obj)
                for et, data in _extract_from_results(obj):
                    if et == "text":
                        full_text += data
                    elif et == "thought":
                        reasoning_text += data
                    elif et == "image":
                        full_text += data
                    elif et == "function_call":
                        _, restored_txt = fc_assembler.feed_with_text(data)
                        if restored_txt:
                            full_text += restored_txt
                    elif et == "finish":
                        finish_raw = data
                    elif et == "safety":
                        safety_html = _safety_html_if_enabled(data)
                    elif et == "blocked":
                        blocked_msg = data
                    elif et == "usage":
                        usage_meta = data
                    elif et == "error":
                        clean_em = _extract_clean_error_message(data)
                        diag = _diagnose_studio_error(clean_em, 200)
                        if _is_retryable_error(clean_em) and attempt < retry_max and not full_text.strip():
                            retry_error = f"{diag} — {clean_em}"
                            break
                        hint = ""
                        if _is_cookie_expired_error(clean_em) or _is_project_error(clean_em):
                            hint = _get_auth_or_project_hint(clean_em, 200)
                        fatal_error = f"{diag} — {clean_em}{hint}"

            if retry_error and attempt < retry_max:
                if await fastapi_request.is_disconnected():
                    return {"kind": "cancelled", "message": "客户端已断开连接。"}
                stats.add_retry()
                print(f"⚠️ [Studio 假流式限流重试] 模型={base_model_name} | 第 {attempt+1}/{retry_max} 次重试，"
                      f"等待 {backoff_sec}s | 详情: {retry_error[:2000]}")
                waited = 0.0
                while waited < backoff_sec:
                    step = min(0.5, backoff_sec - waited)
                    await asyncio.sleep(step)
                    waited += step
                    if await fastapi_request.is_disconnected():
                        return {"kind": "cancelled", "message": "客户端已断开连接。"}
                continue

            if fatal_error and not full_text.strip():
                return {"kind": "error", "message": fatal_error}

            if not full_text.strip():
                detail = f"finishReason={finish_raw or '无'}"
                if blocked_msg:
                    detail += f"；promptFeedback 拦截：{blocked_msg}"
                print(f"🔎 [Studio 诊断] 模型={base_model_name} 假流式无正文（{detail}）。原始响应样本：\n{sampler.dump()}")
                return {"kind": "error", "message": f"上游未返回图片或正文（{detail}）。已在运行日志记录原始响应样本。"}

            if fatal_error:
                full_text += f"\n\n[Studio 中途报错] {fatal_error}"

            return {
                "kind": "ok",
                "full_text": full_text + safety_html,
                "reasoning_text": "" if strip_thoughts else reasoning_text,
                "finish_reason": _map_finish_reason(finish_raw),
                "usage_meta": usage_meta,
            }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            em = f"{type(e).__name__}: {e}"
            if (_is_retryable_error(em) or "timeout" in em.lower()) and attempt < retry_max:
                if await fastapi_request.is_disconnected():
                    return {"kind": "cancelled", "message": "客户端已断开连接。"}
                stats.add_retry()
                print(f"⚠️ [Studio 假流式异常重试] 模型={base_model_name} | 第 {attempt+1}/{retry_max} 次重试，"
                      f"等待 {backoff_sec}s | 异常: {em[:2000]}")
                waited = 0.0
                while waited < backoff_sec:
                    step = min(0.5, backoff_sec - waited)
                    await asyncio.sleep(step)
                    waited += step
                    if await fastapi_request.is_disconnected():
                        return {"kind": "cancelled", "message": "客户端已断开连接。"}
                continue
            return {"kind": "error", "message": f"Studio 代理请求异常（{em}）"}
    return {"kind": "error", "message": f"模型 {base_model_name} 重试 {retry_max} 次后仍失败，请检查配额或网络。"}


# ========== 主代理类 ==========

class CookieProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    使用 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。
    """

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        try:
            validate_request_schemas(request_obj)
        except SchemaValidationError as exc:
            print(f"❌ [Studio 参数校验失败] 模型={request_obj.model} | JSON Schema 不合法: {exc}")
            return JSONResponse(status_code=400, content={
                "error": {"message": str(exc), "type": "invalid_request_error", "code": 400}
            })

        # ===== 1. 验证认证 =====
        cookie_str = _get_cookie_string()
        if not cookie_str:
            err_text = (
                "未配置 Google Cookie。请在大盘「通道与凭证」页粘贴 Cookie 和 Project ID，"
                "或设置环境变量 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID。"
            )
            print(f"❌ [Studio 鉴权失败] 模型={request_obj.model} | {err_text}")
            return JSONResponse(status_code=401, content={"error": {"message": err_text, "type": "auth_error"}})

        project_id = _get_project_id()
        if not project_id:
            err_text = (
                "未配置 Google Cloud Project ID。请在大盘「通道与凭证」页填写，"
                "或直接粘贴含 ?project=xxx 的控制台 URL 自动提取。"
            )
            print(f"❌ [Studio 配置缺失] 模型={request_obj.model} | {err_text}")
            return JSONResponse(status_code=400, content={"error": {"message": err_text, "type": "config_error"}})

        # ===== 2. 构建请求头 =====
        headers = build_headers(cookie_str)
        if not headers:
            err_text = (
                "当前配置的 Cookie 中缺少 SAPISID / __Secure-3PAPISID 字段，无法计算 SAPISIDHASH 认证头。"
                "请确保从已登录的 console.cloud.google.com 复制完整 Request Cookie 头。"
            )
            print(f"❌ [Studio 鉴权失败] 模型={request_obj.model} | {err_text}")
            return JSONResponse(status_code=401, content={"error": {"message": err_text, "type": "auth_error"}})

        # ===== 2.5 工具流量处理：原生函数调用 =====
        _tool_info = classify_tool_traffic(request_obj.messages, request_obj.tools)
        force_builtin_search = bool(_tool_info["builtin_search"])
        _choice_name = _normalized_tool_name(request_obj.tool_choice) if isinstance(request_obj.tool_choice, dict) else ""
        _forced_search = bool(_choice_name and _is_builtin_search_name(_choice_name))
        native_custom_tools = bool(_tool_info["custom_names"] or _tool_info["history"])
        if force_builtin_search and _tool_info["custom_names"]:
            if _forced_search:
                print("🔎 [Studio] batchGraphql 不支持内建搜索与自定义函数混用；本轮按强制选择仅启用 googleSearch。")
            else:
                print("⚠️ [Studio] batchGraphql 不支持内建搜索与自定义函数混用；本轮优先启用自定义函数，忽略搜索声明。")
        elif force_builtin_search:
            print("🔎 [Studio] 请求声明了搜索类工具，已映射为 Studio 内建 googleSearch。")
        if _tool_info["custom_names"] and not _forced_search:
            print(f"🛠️ [Studio] 原生函数调用：{'、'.join(_tool_info['custom_names'][:5])}"
                  f"{' 等' if len(_tool_info['custom_names']) > 5 else ''}。")

        # ===== 3. 解析模型名 =====
        model_display = request_obj.model
        base_model_name = model_display
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]

        # ===== 3.5 预填充智能兼容（支持按模型专属覆盖）=====
        _profile = mc.get_profile(base_model_name)
        prefill_text = ""
        prefill_active = False
        _inj_settings = app_state.get_effective_settings(base_model_name)
        _injected, _inj_notes = apply_console_injection(
            request_obj.messages,
            system_text=_inj_settings.get("inject_system_instruction", ""),
            prefill_text=_inj_settings.get("inject_prefill", ""),
            has_tools=bool(_tool_info["declared"] or _tool_info["history"]),
            is_image_model=_profile["is_image"],
            allow_image_prefill=bool(_inj_settings.get("inject_prefill_for_image", False)),
        )
        for _n in _inj_notes:
            print(_n)
        if _injected is not request_obj.messages:
            request_obj = request_obj.model_copy(update={"messages": _injected})

        _prefill_mode = _inj_settings.get("prefill_mode", app_config.DEFAULT_SETTINGS["prefill_mode"])
        if _prefill_mode != "off":
            _new_msgs, prefill_text, prefill_active = apply_prefill_compat(
                request_obj.messages, _prefill_mode,
                allow_model_last=not _profile["requires_user_last_turn"],
                instruction_template=_prefill_tpl(_inj_settings.get("prefill_instruction", ""), _profile["is_image"]),
                cot_guard=bool(_inj_settings.get("prefill_cot_guard", True)) and not _profile["is_image"],
            )
            if _new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _new_msgs})
                print(_prefill_log(_prefill_mode, prefill_text))
            elif prefill_text:
                print(f"🩹 [预填充兼容] 该模型支持 model 结尾，预填充原生透传（{len(prefill_text)} 字），模型将直接续写。")
            else:
                print("ℹ️ [预填充兼容] 未检测到预填充（请求最后一条不是 assistant 消息）。"
                      "预设里的思维链指令属于 system/user 条目，不是预填充；"
                      "若要用预设思维链顶掉原生思考，需在预设末尾放一条 assistant 条目（通常是思考块的开标签）。")
            if prefill_active and _inj_settings.get("prefill_suppress_thinking", True):
                print("🧠 [预填充兼容] 已按模型压制原生思考（可在控制台关闭），让预设思维链接管。")

        # ===== 3.6 合成工具抗截断包装（保留 plain_request_obj 供 Cookie 协议不支持时无感回退） =====
        plain_request_obj = request_obj
        _anti_mode = str(_inj_settings.get("anti_truncation_mode", "off") or "off").lower()
        anti_tool_name: Optional[str] = None
        anti_stream_args: bool = False
        if (
            _anti_mode in ("streaming", "buffered")
            and not _profile["is_image"]
            and not force_builtin_search
            and not model_display.endswith("-search")
        ):
            request_obj, anti_tool_name, anti_stream_args, anti_note = prepare_anti_truncation_request(
                request_obj,
                mode=_anti_mode,
                prefill_text=prefill_text,
            )
            if anti_note:
                print(anti_note)
            if anti_tool_name:
                native_custom_tools = True

        # ===== 4. HTTP 客户端配置（read 600s，适配 gemini-4-argon 长思考）=====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=10.0),
            "follow_redirects": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL

        retry_max, backoff_sec = get_retry_settings()

        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        want_usage = _wants_usage(request_obj)
        cookie_debug = bool(app_state.get_setting("debug_outbound", False)
                            or app_state.get_setting("cookie_debug", False))

        _eff_settings = app_state.get_effective_settings(base_model_name)
        _tk = mc.resolve_thinking(base_model_name, request_obj, _eff_settings, prefill_active=prefill_active)
        strip_thoughts = bool(_tk.get("mode") and not _tk.get("include_thoughts", True))
        if strip_thoughts and cookie_debug:
            print("🔎 [Studio 调试] 已启用响应侧思考剥离（batchGraphql 不支持 includeThoughts=false）。")

        msg_count = len(request_obj.messages)
        print(f"→ [Studio] {base_model_name} (项目={project_id}) | {msg_count} 条消息 | {'流式' if is_stream else '非流式'}")

        is_image = _profile["is_image"]
        fake_stream_enabled = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))

        # ========== 生图流式 或 显式开启假流式（且无原生工具调用需求） ==========
        has_real_client_tools = bool(_tool_info["custom_names"] or _tool_info["history"])
        if is_stream and (is_image or (fake_stream_enabled and not has_real_client_tools)):
            async def studio_fake_stream():
                if await fastapi_request.is_disconnected():
                    return
                if is_image:
                    print(f"🖼️ [生图保护] 图片模型 {base_model_name} 已自动切换为假流式输出（Cookie 通道），等待期间持续发送心跳。")
                else:
                    print(f"🌊 [Studio 假流式] 模型 {base_model_name} 已启用假流式保活模式，等待上游流式汇总期间持续发送心跳。")

                try:
                    hb_interval = float(app_state.get_setting(
                        "fake_streaming_interval", app_config.FAKE_STREAMING_INTERVAL_SECONDS) or 1.0)
                except (TypeError, ValueError):
                    hb_interval = 1.0
                hb_interval = max(0.5, hb_interval)

                collect_task = asyncio.create_task(
                    _collect_full_response(
                        project_id, base_model_name, request_obj, headers, client_kwargs,
                        retry_max, backoff_sec, fastapi_request,
                        prefill_active=prefill_active,
                        force_search=force_builtin_search,
                        strip_thoughts=strip_thoughts,
                    )
                )

                try:
                    elapsed_hb = 0.0
                    while not collect_task.done():
                        await asyncio.sleep(0.5)
                        elapsed_hb += 0.5
                        if await fastapi_request.is_disconnected():
                            print(f"ℹ️ [Studio] 假流式等待期间客户端已断开，正在取消 {base_model_name} 上游任务。")
                            collect_task.cancel()
                            try:
                                await collect_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            return
                        if not collect_task.done() and elapsed_hb >= hb_interval:
                            elapsed_hb = 0.0
                            keep_alive_data = {
                                "id": "chatcmpl-keepalive",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_display,
                                "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(keep_alive_data, ensure_ascii=False)}\n\n"

                    res = await collect_task
                except asyncio.CancelledError:
                    if not collect_task.done():
                        collect_task.cancel()
                    raise

                if res.get("kind") == "cancelled":
                    return

                yield _make_openai_chunk(response_id, model_display, role="assistant")
                if res.get("kind") != "ok":
                    stats.add_error()
                    err_detail = res.get("message", "生成失败")
                    print(f"❌ [Studio 假流式报错] 模型={base_model_name} (项目={project_id}) | 详情: {err_detail[:2000]}")
                    yield _make_openai_chunk(response_id, model_display, content=f"[Studio 错误] {err_detail}")
                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                full_text = res.get("full_text") or ""
                if prefill_text:
                    full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                if res.get("reasoning_text"):
                    yield _make_openai_chunk(response_id, model_display, reasoning_content=res["reasoning_text"])
                yield _make_openai_chunk(response_id, model_display, content=full_text or " ")
                stats.add_success()
                usage = _map_usage(res.get("usage_meta"))
                if want_usage:
                    yield _make_usage_chunk(response_id, model_display, usage)
                yield _make_openai_chunk(response_id, model_display, finish_reason=res.get("finish_reason", "stop"))
                yield "data: [DONE]\n\n"
                print(f"✅ [Studio] {base_model_name} | 假流式完成")
            return StreamingResponse(studio_fake_stream(), media_type="text/event-stream")

        # ========== 真流式处理 ==========
        if is_stream:
            async def stream_generator():
                nonlocal start_time

                yield _sse_heartbeat()
                last_hb_ts = time.time()
                native_fallback_state = {"latched": False, "no_stream_args": False}

                for attempt in range(retry_max + 1):
                    if await fastapi_request.is_disconnected():
                        print("ℹ️ [Studio] 客户端已断开连接，停止流式重试。")
                        return

                    use_native_tools = not native_fallback_state["latched"]
                    use_stream_args = anti_stream_args and not native_fallback_state["no_stream_args"]
                    active_req = request_obj if use_native_tools else plain_request_obj

                    body = await build_batch_graphql_body_async(
                        project_id, base_model_name, active_req, prefill_active=prefill_active,
                        force_search=force_builtin_search,
                        native_tools=use_native_tools,
                        stream_fc_args=use_stream_args,
                    )

                    fallback_body = None
                    plain_fallback_body = None
                    if native_custom_tools and use_native_tools:
                        if use_stream_args:
                            async def fallback_body():
                                native_fallback_state["no_stream_args"] = True
                                return await build_batch_graphql_body_async(
                                    project_id, base_model_name, request_obj,
                                    prefill_active=prefill_active,
                                    force_search=force_builtin_search,
                                    native_tools=True, stream_fc_args=False)

                            async def plain_fallback_body():
                                native_fallback_state["latched"] = True
                                return await build_batch_graphql_body_async(
                                    project_id, base_model_name, plain_request_obj,
                                    prefill_active=prefill_active,
                                    force_search=force_builtin_search,
                                    native_tools=False, stream_fc_args=False)
                        else:
                            async def fallback_body():
                                native_fallback_state["latched"] = True
                                return await build_batch_graphql_body_async(
                                    project_id, base_model_name, plain_request_obj,
                                    prefill_active=prefill_active,
                                    force_search=force_builtin_search,
                                    native_tools=False, stream_fc_args=False)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        has_yielded_to_client = False
                        prefill_emitted = not bool(prefill_text)
                        got_text = False
                        got_thought = False
                        got_tool_call = False
                        tool_call_index = 0
                        stream_ordinary_metadata = []
                        stream_part_order = []
                        should_retry = False
                        error_to_raise = None
                        usage_meta = None
                        finish_raw = None
                        blocked_msg = None
                        sampler = _RawSampler()
                        deduper = PrefillDeduper(prefill_text)
                        fc_assembler = FunctionCallStreamAssembler(anti_trunc_tool_name=anti_tool_name)

                        async for status, data in _execute_stream_request_generator(
                            client, req_headers, body, sampler=sampler,
                            fallback_body=fallback_body,
                            plain_fallback_body=plain_fallback_body,
                        ):
                            if not has_yielded_to_client and await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 首包产出前客户端已断开，立即终止流式请求。")
                                return

                            if status == "thought":
                                got_thought = True
                                if strip_thoughts:
                                    now_ts = time.time()
                                    if now_ts - last_hb_ts >= 2.0:
                                        last_hb_ts = now_ts
                                        yield _sse_heartbeat()
                                    continue

                            if status == "function_call":
                                completed_calls, restored_txt = fc_assembler.feed_with_text(
                                    data,
                                    thought_signature=_signature_bytes_from_wire(data.get("thought_signature")),
                                )
                                if restored_txt:
                                    if not has_yielded_to_client:
                                        print(f"⚡ [Studio] {base_model_name} | 合成工具流式解包连接建立，正在实时输出...")
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                                        has_yielded_to_client = True
                                    if prefill_text and not prefill_emitted:
                                        prefill_emitted = True
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)
                                    got_text = True
                                    out = deduper.feed(restored_txt)
                                    if out:
                                        stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                        stream_ordinary_metadata.append(ordinary_part_metadata("text", out, None))
                                        yield _make_openai_chunk(response_id, model_display, content=out)
                                for c_item in completed_calls:
                                    if not has_yielded_to_client:
                                        print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                                        has_yielded_to_client = True
                                    tool_payload = _openai_tool_call(response_id, tool_call_index, c_item)
                                    tool_payload["index"] = tool_call_index
                                    stream_part_order.append({"type": "tool_call", "index": tool_call_index})
                                    yield _make_openai_chunk(response_id, model_display, tool_calls=[tool_payload])
                                    tool_call_index += 1
                                    got_tool_call = True

                            elif status == "thought_signature":
                                signature = _signature_bytes_from_wire(data.get("value"))
                                extra = thought_signature_extra(signature, data.get("part_kind"))
                                if extra:
                                    if stream_ordinary_metadata:
                                        stream_ordinary_metadata[-1]["thought_signature"] = _encode_wire_signature(signature)
                                    elif data.get("part_kind") == "signature_only":
                                        stream_part_order.append({"type": "ordinary", "index": 0})
                                        stream_ordinary_metadata.append(
                                            ordinary_part_metadata("signature_only", "", signature))
                                    google = extra.setdefault("google", {})
                                    if stream_ordinary_metadata:
                                        google["ordinary_parts"] = list(stream_ordinary_metadata)
                                    if got_tool_call:
                                        google["part_order"] = list(stream_part_order)
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")
                                        has_yielded_to_client = True
                                    yield _make_openai_chunk(response_id, model_display, extra_content=extra)

                            elif status in ("text", "thought", "image", "api_error_text"):
                                if not has_yielded_to_client:
                                    print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                    has_yielded_to_client = True

                                if status == "thought":
                                    stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                    stream_ordinary_metadata.append(ordinary_part_metadata("thought", data, None))
                                    yield _make_openai_chunk(response_id, model_display, reasoning_content=data)
                                elif status == "api_error_text":
                                    print(f"❌ [Studio 流中断报错] 模型={base_model_name} | {data[:2000]}")
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio API 错误] {data}")
                                elif status == "image":
                                    if prefill_text and not prefill_emitted:
                                        prefill_emitted = True
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)
                                    got_text = True
                                    stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                    stream_ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                                    yield _make_openai_chunk(response_id, model_display, content=data)
                                else:
                                    if prefill_text and not prefill_emitted:
                                        prefill_emitted = True
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)
                                    got_text = True
                                    out = deduper.feed(data)
                                    if out:
                                        stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                                        stream_ordinary_metadata.append(ordinary_part_metadata("text", out, None))
                                        yield _make_openai_chunk(response_id, model_display, content=out)

                            elif status == "finish":
                                finish_raw = data
                            elif status == "safety":
                                _sh = _safety_html_if_enabled(data)
                                if _sh:
                                    yield _make_openai_chunk(response_id, model_display, content=_sh)
                            elif status == "usage":
                                usage_meta = data
                            elif status == "blocked":
                                blocked_msg = data

                            elif status == "cookie_error":
                                if not has_yielded_to_client:
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                stats.add_error()
                                print(f"❌ [Studio 鉴权/项目错误] 模型={base_model_name} (项目={project_id}) | {data[:2000]}")
                                yield _make_openai_chunk(response_id, model_display, content=f"[Studio 权限错误] {data}")
                                yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                yield "data: [DONE]\n\n"
                                return

                            elif status in ("retryable_error", "fatal_error"):
                                if not has_yielded_to_client and status == "retryable_error" and attempt < retry_max:
                                    should_retry = True
                                    error_to_raise = data
                                    break
                                else:
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")

                                    err_prefix = "不可重试错误" if status == "fatal_error" else f"重试 {retry_max} 次后仍失败"
                                    stats.add_error()
                                    print(f"❌ [Studio {err_prefix}] 模型={base_model_name} (项目={project_id}) | {data[:2000]}")
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio 错误] {data}")
                                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                    yield "data: [DONE]\n\n"
                                    return

                        if should_retry:
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                return
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio 限流退避] 模型={base_model_name} | 第 {attempt+1}/{retry_max} 次重试，"
                                  f"等待 {wait_sec}s | 原因: {error_to_raise[:2000]}")
                            async for _hb in _sleep_with_heartbeat(wait_sec):
                                if await fastapi_request.is_disconnected():
                                    print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                    return
                                yield _hb
                            start_time = time.time()
                            continue

                        if not has_yielded_to_client:
                            yield _make_openai_chunk(response_id, model_display, role="assistant")
                            has_yielded_to_client = True
                        if prefill_text and not prefill_emitted:
                            prefill_emitted = True
                            yield _make_openai_chunk(response_id, model_display, content=prefill_text)
                        tail = deduper.flush()
                        if tail:
                            stream_part_order.append({"type": "ordinary", "index": len(stream_ordinary_metadata)})
                            stream_ordinary_metadata.append(ordinary_part_metadata("text", tail, None))
                            yield _make_openai_chunk(response_id, model_display, content=tail)

                        for c_item in fc_assembler.flush():
                            tool_payload = _openai_tool_call(response_id, tool_call_index, c_item)
                            tool_payload["index"] = tool_call_index
                            stream_part_order.append({"type": "tool_call", "index": tool_call_index})
                            yield _make_openai_chunk(response_id, model_display, tool_calls=[tool_payload])
                            tool_call_index += 1
                            got_tool_call = True

                        if got_text or got_tool_call:
                            stats.add_success()
                            if got_tool_call and stream_ordinary_metadata:
                                google = {
                                    "ordinary_parts": list(stream_ordinary_metadata),
                                    "part_order": list(stream_part_order),
                                }
                                last_signed = next((item for item in reversed(stream_ordinary_metadata)
                                                    if item.get("thought_signature")), None)
                                if last_signed:
                                    google["thought_signature"] = last_signed["thought_signature"]
                                    google["thought_signature_part"] = last_signed["kind"]
                                yield _make_openai_chunk(
                                    response_id, model_display, extra_content={"google": google})
                            yield _make_openai_chunk(
                                response_id, model_display,
                                finish_reason="tool_calls" if got_tool_call else _map_finish_reason(finish_raw))
                        else:
                            detail = f"finishReason={finish_raw or '无'}"
                            if blocked_msg:
                                detail += f"；promptFeedback 拦截：{blocked_msg}"
                            desc = "只返回了思考、未返回正文" if got_thought else "未返回任何内容"
                            hint = (_THINKING_RUNAWAY_HINT if (got_thought and not strip_thoughts) else _NO_BODY_HINT)
                            stats.add_error()
                            print(f"❌ [Studio 空正文报错] 模型={base_model_name} (项目={project_id}) | 上游{desc}（{detail}）| 建议：{hint}")
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
                        if got_text or got_tool_call:
                            print(f"✅ [Studio] {base_model_name} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                        return

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # ========== 非流式处理 ==========
        else:
            native_fallback_state = {"latched": False}
            for attempt in range(retry_max + 1):
                if await fastapi_request.is_disconnected():
                    print("ℹ️ [Studio] 客户端已断开连接，停止非流式重试。")
                    return JSONResponse(status_code=499, content={
                        "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                    })
                try:
                    use_native_tools = not native_fallback_state["latched"]
                    if use_native_tools:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search)
                    else:
                        body = await build_batch_graphql_body_async(
                            project_id, base_model_name, plain_request_obj, prefill_active=prefill_active,
                            force_search=force_builtin_search, native_tools=False)
                    fallback_body = None
                    if native_custom_tools and use_native_tools:
                        async def fallback_body():
                            native_fallback_state["latched"] = True
                            return await build_batch_graphql_body_async(
                                project_id, base_model_name, plain_request_obj,
                                prefill_active=prefill_active,
                                force_search=force_builtin_search, native_tools=False)
                    req_headers = build_headers(_get_cookie_string()) or headers
                    if cookie_debug:
                        print(f"🔎 [Studio 调试] 出站 generationConfig: "
                              f"{json.dumps(body['variables'].get('generationConfig', {}), ensure_ascii=False)}")

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        post_task = asyncio.create_task(
                            client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)
                        )
                        while not post_task.done():
                            await asyncio.sleep(0.5)
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 非流式等待期间客户端已断开，取消上游请求。")
                                post_task.cancel()
                                try:
                                    await post_task
                                except (asyncio.CancelledError, Exception):
                                    pass
                                return JSONResponse(status_code=499, content={
                                    "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                                })
                        response = await post_task

                        if fallback_body is not None and _native_tool_error_response(
                                response.status_code, response.text):
                            native_fallback_state["latched"] = True
                            resolved_fallback = fallback_body()
                            if hasattr(resolved_fallback, "__await__"):
                                resolved_fallback = await resolved_fallback
                            print("⚠️ [Studio] 当前模型/协议拒绝原生函数 Schema；本次及后续重试固定降级为文本工具观测。")
                            response = await client.post(
                                BATCH_GRAPHQL_URL, headers=req_headers, json=resolved_fallback
                            )
                            fallback_body = None

                    if response.status_code in (429, 503, 500):
                        if attempt < retry_max:
                            if await fastapi_request.is_disconnected():
                                return JSONResponse(status_code=499, content={
                                    "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                                })
                            wait_sec = backoff_sec
                            stats.add_retry()
                            clean_http_err = _extract_clean_error_message(response.text)
                            print(f"⚠️ [Studio 非流式重试] 模型={base_model_name} | HTTP {response.status_code} | "
                                  f"第 {attempt+1}/{retry_max} 次重试，等待 {wait_sec}s | 原因: {clean_http_err[:2000]}")
                            waited = 0.0
                            while waited < wait_sec:
                                step = min(0.5, wait_sec - waited)
                                await asyncio.sleep(step)
                                waited += step
                                if await fastapi_request.is_disconnected():
                                    print("ℹ️ [Studio] 非流式退避等待期间客户端已断开，停止重试。")
                                    return JSONResponse(status_code=499, content={
                                        "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                                    })
                            continue

                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        clean_msg = _extract_clean_error_message(response.text)
                        diag = _diagnose_studio_error(clean_msg, response.status_code)
                        hint = ""
                        if (
                            response.status_code in (401, 403)
                            or _is_cookie_expired_error(clean_msg)
                            or _is_project_error(clean_msg)
                        ):
                            hint = _get_auth_or_project_hint(clean_msg, response.status_code)
                        full_err_msg = f"HTTP {response.status_code}（{diag}）：{clean_msg}{hint}"
                        print(f"❌ [Studio 非流式报错] 模型={base_model_name} (项目={project_id}) | 耗时 {elapsed:.1f}s | {full_err_msg[:2000]}")
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": full_err_msg, "type": "upstream_error", "code": response.status_code}
                        })

                    full_text = ""
                    reasoning_text = ""
                    safety_html = ""
                    tool_calls = []
                    ordinary_metadata = []
                    part_order = []
                    message_extra = None
                    retry_error = None
                    fatal_error = None
                    usage_meta = None
                    finish_raw = None
                    blocked_msg = None
                    sampler = _RawSampler()
                    fc_assembler = FunctionCallStreamAssembler(anti_trunc_tool_name=anti_tool_name)

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
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                            elif event_type == "thought":
                                reasoning_text += data
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("thought", data, None))
                            elif event_type == "image":
                                full_text += data
                                part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                ordinary_metadata.append(ordinary_part_metadata("text", data, None))
                            elif event_type == "function_call":
                                completed_calls, restored_txt = fc_assembler.feed_with_text(
                                    data,
                                    thought_signature=_signature_bytes_from_wire(data.get("thought_signature")),
                                )
                                if restored_txt:
                                    full_text += restored_txt
                                    part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                    ordinary_metadata.append(ordinary_part_metadata("text", restored_txt, None))
                                for c_item in completed_calls:
                                    part_order.append({"type": "tool_call", "index": len(tool_calls)})
                                    tool_calls.append(_openai_tool_call(response_id, len(tool_calls), c_item))
                            elif event_type == "thought_signature":
                                sig = _signature_bytes_from_wire(data.get("value"))
                                if sig:
                                    message_extra = thought_signature_extra(sig, data.get("part_kind"))
                                    if ordinary_metadata:
                                        ordinary_metadata[-1]["thought_signature"] = _encode_wire_signature(sig)
                                    elif data.get("part_kind") == "signature_only":
                                        part_order.append({"type": "ordinary", "index": len(ordinary_metadata)})
                                        ordinary_metadata.append(ordinary_part_metadata("signature_only", "", sig))
                            elif event_type == "finish":
                                finish_raw = data
                            elif event_type == "safety":
                                safety_html = _safety_html_if_enabled(data)
                            elif event_type == "blocked":
                                blocked_msg = data
                            elif event_type == "usage":
                                usage_meta = data
                            elif event_type == "error":
                                clean_em = _extract_clean_error_message(data)
                                diag = _diagnose_studio_error(clean_em, 200)
                                if _is_retryable_error(clean_em) and attempt < retry_max and not full_text.strip() and not tool_calls:
                                    retry_error = f"{diag} — {clean_em}"
                                    break
                                hint = ""
                                if _is_cookie_expired_error(clean_em) or _is_project_error(clean_em):
                                    hint = _get_auth_or_project_hint(clean_em, 200)
                                fatal_error = f"{diag} — {clean_em}{hint}"

                    for c_item in fc_assembler.flush():
                        part_order.append({"type": "tool_call", "index": len(tool_calls)})
                        tool_calls.append(_openai_tool_call(response_id, len(tool_calls), c_item))

                    if retry_error and attempt < retry_max:
                        if await fastapi_request.is_disconnected():
                            return JSONResponse(status_code=499, content={
                                "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                            })
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio 非流式限流重试] 模型={base_model_name} | 第 {attempt+1}/{retry_max} 次重试，"
                              f"等待 {wait_sec}s | 详情: {retry_error[:2000]}")
                        waited = 0.0
                        while waited < wait_sec:
                            step = min(0.5, wait_sec - waited)
                            await asyncio.sleep(step)
                            waited += step
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 非流式退避等待期间客户端已断开，停止重试。")
                                return JSONResponse(status_code=499, content={
                                    "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                                })
                        continue

                    got_text = bool(full_text.strip())
                    got_output = got_text or bool(tool_calls)
                    elapsed = time.time() - start_time

                    if not got_output and fatal_error:
                        err_status = 403 if (_is_cookie_expired_error(fatal_error) or _is_project_error(fatal_error)) else (
                            429 if _is_retryable_error(fatal_error) else 400
                        )
                        print(f"❌ [Studio 非流式报错] 模型={base_model_name} (项目={project_id}) | HTTP {err_status} | "
                              f"耗时 {elapsed:.1f}s | {fatal_error[:2000]}")
                        return JSONResponse(status_code=err_status, content={
                            "error": {"message": fatal_error, "type": "upstream_error", "code": err_status}
                        })

                    if got_output:
                        if fatal_error:
                            full_text += f"\n\n[Studio 中途报错] {fatal_error}"
                        if prefill_text and got_text:
                            full_text = prefill_text + strip_prefill_overlap(prefill_text, full_text)
                        finish_reason = "tool_calls" if tool_calls else _map_finish_reason(finish_raw)
                        stats.add_success()
                        print(f"✅ [Studio] {base_model_name} | {len(full_text)} 字符 | {elapsed:.1f}s")
                    else:
                        detail = f"finishReason={finish_raw or '无'}"
                        if blocked_msg:
                            detail += f"；promptFeedback 拦截：{blocked_msg}"
                        desc = "只返回了思考、未返回正文" if reasoning_text else "未返回任何内容"
                        hint = (_THINKING_RUNAWAY_HINT if (reasoning_text and not strip_thoughts) else _NO_BODY_HINT)
                        stats.add_error()
                        print(f"❌ [Studio 空正文报错] 模型={base_model_name} (项目={project_id}) | 上游{desc}（{detail}）| 耗时 {elapsed:.1f}s | 建议：{hint}")
                        print(f"🔎 [Studio 诊断] 原始响应样本：\n{sampler.dump()}")
                        full_text = f"[Studio 提示] 上游{desc}（{detail}）。{hint}"
                        filtered = bool(blocked_msg) or ((finish_raw or "").upper() in _FINISH_FILTERED)
                        finish_reason = "content_filter" if filtered else "stop"

                    usage = _map_usage(usage_meta)

                    content_value = full_text + safety_html
                    message_obj = {"role": "assistant", "content": content_value if content_value else None}
                    if reasoning_text and not strip_thoughts:
                        message_obj["reasoning_content"] = reasoning_text
                    if tool_calls:
                        message_obj["tool_calls"] = tool_calls
                    if ordinary_metadata:
                        if message_extra is None:
                            message_extra = {"google": {}}
                        google = message_extra.setdefault("google", {})
                        google["ordinary_parts"] = ordinary_metadata
                        if tool_calls:
                            google["part_order"] = part_order
                    if message_extra:
                        message_obj["extra_content"] = message_extra

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

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()

                    if is_retryable and attempt < retry_max:
                        if await fastapi_request.is_disconnected():
                            return JSONResponse(status_code=499, content={
                                "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                            })
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio 非流式异常重试] 模型={base_model_name} | 第 {attempt+1}/{retry_max} 次重试，"
                              f"等待 {wait_sec}s | 异常: {err_msg[:2000]}")
                        waited = 0.0
                        while waited < wait_sec:
                            step = min(0.5, wait_sec - waited)
                            await asyncio.sleep(step)
                            waited += step
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 非流式退避等待期间客户端已断开，停止重试。")
                                return JSONResponse(status_code=499, content={
                                    "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                                })
                        continue

                    elapsed = time.time() - start_time
                    print(f"❌ [Studio 非流式异常] 模型={base_model_name} (项目={project_id}) | 耗时 {elapsed:.1f}s | {err_msg[:2000]}")
                    print(traceback.format_exc())
                    return JSONResponse(status_code=500, content={
                        "error": {"message": f"Studio 代理异常: {err_msg}", "type": "proxy_error"}
                    })

            elapsed = time.time() - start_time
            print(f"❌ [Studio 重试耗尽] 模型={base_model_name} (项目={project_id}) | 重试 {retry_max} 次后仍被限流 | 耗时 {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={
                "error": {"message": f"模型 {base_model_name} 请求被限流（429），已重试 {retry_max} 次仍失败，请稍后再试。", "type": "rate_limit_error"}
            })