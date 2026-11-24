"""
上游错误诊断、HTTP 状态码映射、重试退避提取与坏凭证识别模块。

从原 api_helpers.py 精准拆离，api_helpers.py 保留同名兼容导出以确保所有既有调用方零破坏。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx

import config as app_config
from runtime_state import app_state


def get_retry_settings() -> tuple[int, float]:
    """读取重试配置。总请求次数 = retry_max + 1。"""
    try:
        retry_max = int(app_state.get_setting("retry_max", app_config.DEFAULT_SETTINGS["retry_max"]))
    except (TypeError, ValueError):
        retry_max = app_config.DEFAULT_SETTINGS["retry_max"]
    try:
        backoff = float(
            app_state.get_setting(
                "retry_backoff_seconds", app_config.DEFAULT_SETTINGS["retry_backoff_seconds"]
            )
        )
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
            "param": None,
        }
    }


def _diagnose_express_error(code: int, msg: str, status_str: str = "") -> str:
    """为 Express SDK 通道的常见报错生成清晰的中文诊断说明。"""
    low = f"{status_str} {msg}".lower()
    if any(
        k in low
        for k in (
            "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
            "finish_reason=safety", "image_safety", "recitation", "blocklist",
        )
    ):
        return "请求或回复被 Google 安全策略拦截（不可重试），请检查提示词内容"
    if any(
        k in low
        for k in (
            "只返回了思考过程", "返回了空内容", "返回了空对象", "未返回任何有效内容",
            "解码格式异常", "malformed_response", "malformed_function_call",
        )
    ):
        return "模型未输出有效正文（思考耗尽额度、返回空候选或解码格式异常，不可重试）"
    if any(
        k in low
        for k in (
            "remoteprotocolerror", "connecterror", "readerror", "writeerror",
            "proxyerror", "pooltimeout", "server disconnected", "connection reset",
        )
    ):
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

    clean_msg = re.sub(
        r"([?&]key=)[^&\s'\"]+", r"\1***HIDDEN_API_KEY***", str(clean_msg).strip() + extra_detail_str
    )

    if not isinstance(code, int) or not (400 <= code <= 599):
        low = f"{type(e).__name__} {status_str} {clean_msg}".lower()
        if any(
            k in low
            for k in (
                "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
                "finish_reason=safety", "image_safety", "recitation", "blocklist",
            )
        ):
            code = 400
        elif any(
            k in low
            for k in (
                "只返回了思考过程", "返回了空内容", "返回了空对象", "未返回任何有效内容",
                "解码格式异常", "malformed_response", "malformed_function_call", "unexpected_tool_call",
            )
        ):
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
        elif any(
            k in low
            for k in (
                "remoteprotocolerror", "connecterror", "readerror", "writeerror",
                "proxyerror", "pooltimeout", "server disconnected", "connection reset", "502", "504",
            )
        ):
            code = 502
        else:
            code = 500

    diag = _diagnose_express_error(code, clean_msg, status_str)
    status_tag = f" [{status_str}]" if status_str and status_str not in clean_msg else ""
    formatted = f"【{diag}】{status_tag} {clean_msg}"
    return code, formatted


def extract_retry_after_seconds(err: Any, msg: str = "") -> Optional[float]:
    """从上游 429 / 503 异常响应头或 RetryInfo 中提取 Google 明确返回的短建议秒数。"""
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
    """判断异常是否应按用户设置的最大次数进行退避重试。"""
    if getattr(e, "is_retryable", None) is True:
        return True
    if getattr(e, "is_retryable", None) is False and getattr(e, "is_auth_or_billing", None) is not None:
        return False

    if isinstance(e, ValueError):
        return False

    error_str = f"{type(e).__name__} {getattr(e, 'status', '')} {e}".lower()
    if any(
        k in error_str
        for k in (
            "提示词阶段拦截", "block_reason", "静默过滤", "prohibited_content",
            "finish_reason=safety", "image_safety", "blocklist",
            "只返回了思考过程", "返回了空内容", "返回了空对象",
        )
    ):
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
    """判断是否为单个 Express Key 的鉴权失效或所属项目欠费/无权限错误。"""
    if getattr(e, "is_auth_or_billing", None) is True:
        return True
    if isinstance(e, ValueError):
        return False
    low = f"{getattr(e, 'status', '')} {e}".lower()
    if err_code in (401, 403):
        return True
    return any(
        k in low
        for k in (
            "api_key_invalid", "api key not valid", "unauthenticated",
            "permission_denied", "requires billing", "billing to be enabled",
            "account_suspended", "consumer_invalid",
        )
    )


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
