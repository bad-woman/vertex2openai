"""
控制台后台管理路由模块（鉴权、静态入口、持久化统计、凭证池、参数覆写、实况脉冲与试验场）。
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import config
import express_key_probe
import model_capabilities as mc
from api_helpers import create_openai_error_response, execute_unified_completion, extract_upstream_error
from cookie_auth import validate_cookie
from live_tracker import live_tracker
from logger import call_recorder, rt_logger, stats
from metrics_store import key_fingerprint, metrics_store
from model_loader import get_express_models, get_model_categories
from models import OpenAIMessage, OpenAIRequest
from runtime_state import app_state
from synthetic_wrapper import create_sandbox_spec, supports_stream_fc_args
from upstreams.express_sdk import _normalize_model_name

router = APIRouter()

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

AUTH_COOKIE = "ap_session"
DEFAULT_API_KEY = "123456"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30

_sessions: dict = {}
_sessions_lock = threading.Lock()

_login_failures: dict = {}
_login_lock = threading.Lock()
LOGIN_LOCK_BASE_SECONDS = 2
LOGIN_LOCK_MAX_SECONDS = 300


def _issue_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _sessions_lock:
        for t, exp in list(_sessions.items()):
            if exp < now:
                _sessions.pop(t, None)
        _sessions[token] = now + SESSION_TTL_SECONDS
    return token


def _revoke_session(token: str) -> None:
    with _sessions_lock:
        _sessions.pop(token, None)


def _is_authed(request: Request) -> bool:
    token = request.cookies.get(AUTH_COOKIE, "")
    if not token:
        return False
    with _sessions_lock:
        exp = _sessions.get(token)
        if exp is None:
            return False
        if exp < time.time():
            _sessions.pop(token, None)
            return False
    return True


def _login_retry_after(ip: str) -> int:
    with _login_lock:
        rec = _login_failures.get(ip)
        if not rec:
            return 0
        count, last = rec
        if count < 3:
            return 0
        wait = min(LOGIN_LOCK_BASE_SECONDS * (2 ** (count - 3)), LOGIN_LOCK_MAX_SECONDS)
        remain = int(last + wait - time.time())
        return max(0, remain)


def _record_login_failure(ip: str) -> None:
    with _login_lock:
        count, _ = _login_failures.get(ip, (0, 0.0))
        _login_failures[ip] = (count + 1, time.time())


def _clear_login_failure(ip: str) -> None:
    with _login_lock:
        _login_failures.pop(ip, None)


def mask_cookie(cookie_str: str) -> str:
    if not cookie_str:
        return ""
    names = []
    for seg in cookie_str.split(";"):
        name = seg.strip().split("=", 1)[0].strip()
        if name:
            names.append(name)
    return f"已配置（共 {len(names)} 个 cookie 字段，{len(cookie_str)} 字符）"


async def require_auth(request: Request):
    if not _is_authed(request):
        raise HTTPException(status_code=401, detail="未登录")
    return True


def _read_static_html(filename: str) -> str:
    fp = os.path.join(STATIC_DIR, filename)
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# 页面入口与静态图标
# ============================================================

# 429 娘 128x128 WebP Base64 头像（纯文本存仓，无需向 GitHub 上传二进制图片文件）
AVATAR_WEBP_B64 = "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"
AVATAR_WEBP_BYTES = base64.b64decode(AVATAR_WEBP_B64)


@router.get("/favicon.ico")
@router.get("/avatar.webp")
async def favicon_endpoint():
    return Response(
        content=AVATAR_WEBP_BYTES,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard_ui(request: Request):
    filename = "index.html" if _is_authed(request) else "login.html"
    return HTMLResponse(
        content=_read_static_html(filename),
        headers={"Cache-Control": "no-store"},
    )


# ============================================================
# 登录与登出
# ============================================================

class LoginBody(BaseModel):
    password: str


@router.post("/api/login")
async def login(body: LoginBody, request: Request):
    ip = request.client.host if request.client else "unknown"

    remain = _login_retry_after(ip)
    if remain > 0:
        return JSONResponse(status_code=429, content={"error": f"尝试过于频繁，请 {remain} 秒后再试"})

    if config.API_KEY and secrets.compare_digest(body.password, config.API_KEY):
        _clear_login_failure(ip)
        resp = JSONResponse(content={"ok": True})
        resp.set_cookie(
            AUTH_COOKIE,
            _issue_session(),
            httponly=True,
            samesite="lax",
            max_age=SESSION_TTL_SECONDS,
            path="/",
            secure=(
                request.url.scheme == "https"
                or request.headers.get("x-forwarded-proto", "") == "https"
            ),
        )
        return resp

    _record_login_failure(ip)
    print(f"🔐 [登录失败] 来自 {ip} 的密码尝试失败。")
    return JSONResponse(status_code=401, content={"error": "密码错误"})


@router.post("/api/logout")
async def logout(request: Request):
    _revoke_session(request.cookies.get(AUTH_COOKIE, ""))
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(AUTH_COOKIE, path="/")
    return resp


# ============================================================
# 持久化统计、调用记录与实况脉冲
# ============================================================

@router.get("/api/stats")
async def get_stats_api(
    range_: str = Query("today", alias="range"),
    _auth: bool = Depends(require_auth),
):
    rk = range_ if range_ in ("today", "7d", "all") else "all"
    return JSONResponse(content=stats.get_json_stats(range_key=rk))


@router.get("/api/records")
async def get_call_records_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=call_recorder.get_payload())


@router.post("/api/records/clear")
async def clear_call_records_api(_auth: bool = Depends(require_auth)):
    call_recorder.clear()
    return JSONResponse(content=call_recorder.get_payload())


@router.get("/api/live")
async def get_live_requests_api(_auth: bool = Depends(require_auth)):
    items = live_tracker.snapshot()
    return JSONResponse(content={"active": items, "count": len(items)})


@router.post("/api/live/{req_id}/abort")
async def abort_live_request_api(req_id: str, _auth: bool = Depends(require_auth)):
    ok = live_tracker.abort(req_id)
    if not ok:
        return JSONResponse(status_code=404, content={"ok": False, "error": "该请求已完成或不存在"})
    return JSONResponse(content={"ok": True, "req_id": req_id})


# ============================================================
# 通道与运行时设置
# ============================================================

class ModeSetting(BaseModel):
    mode: str


@router.get("/api/settings/runtime")
async def get_runtime_settings(_auth: bool = Depends(require_auth)):
    cookie = app_state.get_google_cookie()
    return JSONResponse(
        content={
            "use_web_proxy": app_state.is_web_proxy_enabled(),
            "google_cookie": mask_cookie(cookie),
            "google_cookie_configured": bool(cookie),
            "google_project_id": app_state.get_project_id(),
            "cookie_hits": metrics_store.get_cookie_hits(),
        }
    )


@router.post("/api/settings/mode")
async def set_settings_mode(setting: ModeSetting, _auth: bool = Depends(require_auth)):
    app_state.enable_web_proxy(setting.mode == "web_proxy")
    return JSONResponse(content={"status": "success"})


@router.get("/api/settings")
async def get_settings_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_settings())


@router.post("/api/settings")
async def update_settings_api(request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    updated = app_state.update_settings(patch)
    return JSONResponse(content=updated)


@router.get("/api/capabilities")
async def get_capabilities_api(_auth: bool = Depends(require_auth)):
    try:
        models = await get_express_models()
        categories = await get_model_categories()
    except Exception:
        models = []
        categories = []
    caps = {m: mc.capabilities_summary(m, app_state.get_effective_settings(m)) for m in models}
    overrides = app_state.get_model_overrides()
    return JSONResponse(
        content={"models": models, "categories": categories, "capabilities": caps, "overrides": overrides}
    )


@router.get("/api/model-overrides")
async def list_model_overrides(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_model_overrides())


@router.post("/api/model-overrides/{model_name}")
async def save_model_override(model_name: str, request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    saved = app_state.set_model_override(model_name, patch)
    return JSONResponse(content={"status": "success", "model": model_name, "override": saved})


@router.delete("/api/model-overrides/{model_name}")
async def delete_model_override(model_name: str, _auth: bool = Depends(require_auth)):
    ok = app_state.clear_model_override(model_name)
    return JSONResponse(content={"status": "success" if ok else "not_found", "model": model_name})


# ============================================================
# Express API Key 凭证池（附带指纹持久化命中次数）与 Cookie
# ============================================================

class ExpressKeysBody(BaseModel):
    keys: list = []


class ProbeBody(BaseModel):
    key: str = ""
    all: bool = False
    force: bool = False
    keys: Optional[list] = None


class KeyProjectBody(BaseModel):
    key: str
    project_id: str = ""


def _express_keys_payload() -> dict:
    records = app_state.get_express_keys()
    hits_map = metrics_store.get_key_hits_map()
    enriched = []
    total_hits = 0
    for rec in records:
        item = dict(rec)
        fp = key_fingerprint(item.get("key", ""))
        hits = int(hits_map.get(fp, 0)) if fp else 0
        item["key_fp"] = fp
        item["hits"] = hits
        total_hits += hits
        enriched.append(item)
    return {
        "keys": enriched,
        "total": len(enriched),
        "enabled": sum(1 for r in enriched if r.get("enabled", True)),
        "total_hits": total_hits,
        "roundrobin": bool(app_state.get_setting("roundrobin", config.ROUNDROBIN)),
        "env_import_done": True,
    }


def _refresh_key_manager(request: Request) -> None:
    mgr = getattr(request.app.state, "express_key_manager", None)
    if mgr is not None:
        mgr.refresh_keys()


@router.get("/api/express-keys")
async def list_express_keys(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=_express_keys_payload())


@router.post("/api/express-keys")
async def save_express_keys(body: ExpressKeysBody, request: Request, _auth: bool = Depends(require_auth)):
    if not isinstance(body.keys, list):
        return JSONResponse(status_code=400, content={"error": "keys 必须是数组。"})
    saved = app_state.set_express_keys(body.keys)
    _refresh_key_manager(request)
    pending = [r["key"] for r in saved if not r.get("project_id")]
    if pending:
        asyncio.create_task(express_key_probe.probe_all(keys=pending))
    payload = _express_keys_payload()
    payload["status"] = "success"
    return JSONResponse(content=payload)


@router.post("/api/express-keys/probe")
async def probe_express_keys(body: ProbeBody, request: Request, _auth: bool = Depends(require_auth)):
    if isinstance(body.keys, list) and len(body.keys) > 0:
        app_state.set_express_keys(body.keys)
        _refresh_key_manager(request)

    if body.all:
        results = await express_key_probe.probe_all(force=body.force)
    else:
        key = (body.key or "").strip()
        if not key:
            return JSONResponse(status_code=400, content={"error": "请提供 key，或设 all=true。"})
        results = [await express_key_probe.probe_and_store(key, force=body.force)]
        _refresh_key_manager(request)
    payload = _express_keys_payload()
    payload["results"] = results
    return JSONResponse(content=payload)


@router.post("/api/express-keys/project")
async def set_express_key_project(body: KeyProjectBody, _auth: bool = Depends(require_auth)):
    ok = app_state.update_express_key_project(
        body.key, body.project_id, source=("manual" if body.project_id.strip() else "")
    )
    if not ok:
        return JSONResponse(status_code=404, content={"error": "该 Key 不在列表中。"})
    payload = _express_keys_payload()
    payload["status"] = "success"
    return JSONResponse(content=payload)


class CookieSetting(BaseModel):
    cookie: str = ""
    project_id: str = ""


@router.post("/api/cookie")
async def set_google_cookie(setting: CookieSetting, _auth: bool = Depends(require_auth)):
    new_cookie = (setting.cookie or "").strip()
    project_id = (setting.project_id or "").strip()

    if new_cookie:
        validation = validate_cookie(new_cookie)
        if not validation["valid"]:
            return JSONResponse(status_code=400, content={"error": validation["message"]})
        app_state.set_google_cookie(new_cookie)
        message = validation["message"]
    else:
        if not app_state.get_google_cookie():
            return JSONResponse(status_code=400, content={"error": "尚未配置 Cookie，请粘贴完整的 Google Cookie。"})
        message = "✅ 已保留原有 Cookie，仅更新 Project ID。"

    if project_id:
        app_state.set_project_id(project_id)
    return JSONResponse(content={"status": "success", "message": message})


# ============================================================
# 连通实测与解剖（Live Lab — 合并于通道与凭证，支持定向单凭证直测与生图）
# ============================================================

class LabRequestBody(BaseModel):
    model: str = "gemini-4-carbon"
    stream: bool = True
    prompt: str = ""
    system: str = ""
    prefill: str = ""
    target: str = "auto"  # "auto" | "cookie" | "key:<16-char-sha256-fp>"


def _resolve_lab_target(target_str: str) -> tuple[Optional[dict], bool, Optional[str], str]:
    """解析定向测试目标。
    返回 (forced_target, disable_retries, error_msg, human_label)。
    - auto：跟随全局主通道，保留正常轮询与容灾；
    - cookie：仅测 Studio Cookie 直连（单发直测，不切换全局主通道，不触发跨通道兜底）；
    - key:<fp>：按 SHA-256 指纹定位具体某把 Express Key（单发直测，即使该 Key 处于禁用态也允许单独验活）。
    """
    t = (target_str or "auto").strip()
    if not t or t == "auto":
        return None, False, None, "跟随当前全局路由"
    if t == "cookie":
        return {"channel": "cookie"}, True, None, "仅测 Studio Cookie 直连"
    if t.startswith("key:"):
        fp = t[4:].strip().lower()
        records = app_state.get_express_keys()
        for idx, rec in enumerate(records):
            if key_fingerprint(rec.get("key", "")) == fp:
                pid = (rec.get("project_id") or "").strip() or "默认"
                dis = " · 已停用" if not rec.get("enabled", True) else ""
                krec = dict(rec)
                krec["index"] = idx
                return (
                    {"channel": "express", "key_record": krec},
                    True,
                    None,
                    f"仅测 Express Key #{idx + 1}（项目 {pid}{dis}）",
                )
        return None, True, "指定的 Express Key 已不存在或已被删除，请刷新凭证列表后重试。", "未知 Key"
    return None, True, f"无效的测试目标：{t}", "无效目标"


def _diagnose_lab_error(code: int, message: str, channel: str = "") -> Dict[str, Any]:
    """生成结构化中文错误诊断卡片数据。"""
    c = int(code or 500)
    low = (message or "").lower()
    if c == 401:
        title = "凭证无效或未配置 (HTTP 401)"
        hint = "请确认目标通道已填入有效的 Express API Key，或已在「凭证配置」页启用至少一把 Key。"
    elif c == 403 or "permission_denied" in low or "predict denied" in low:
        title = "权限被拒绝 / Cookie 或项目计费异常 (HTTP 403)"
        if channel == "cookie" or "cookie" in low or "predict denied" in low:
            hint = "Studio Cookie 会话已过期或无当前项目权限，请从已登录的 console.cloud.google.com 重新复制 Request Cookie 并保存。"
        else:
            hint = "该 Express API Key 可能已被吊销、权限不足或所属项目计费异常，请点击「探测」或更换凭证。"
    elif c == 429 or "resource_exhausted" in low or "quota" in low:
        title = "触发上游配额限流 (HTTP 429)"
        hint = "当前凭证或公共池瞬时并发/配额达到上限，可等待几秒重测、切换其他 Express Key，或开启多 Key 轮询。"
    elif c == 404 or "not found" in low:
        title = "模型或区域端点不可用 (HTTP 404)"
        hint = "该模型在当前区域或项目路径下不可访问，建议将「全局基础设施 → 目标部署区域」设为 global 后再试。"
    elif c == 400:
        title = "请求参数或特性不兼容 (HTTP 400)"
        hint = "请检查当前模型的思考档位、预填充或生图宽高比设置；可查看右侧「出站透视」核对实际下发参数。"
    elif c in (500, 502, 503, 504):
        title = f"上游服务繁忙或空响应 (HTTP {c})"
        hint = "Google Vertex AI 上游出现瞬时抖动、超时或空正文返回，通常稍等片刻再次实测即可恢复。"
    else:
        title = f"上游返回异常 (HTTP {c})"
        hint = "请查看下方原始错误详情，或切换至「运行日志 → 实时终端」排查完整上下文。"
    return {"code": c, "title": title, "hint": hint, "message": message or f"HTTP {c}"}


def _build_lab_openai_request(body: LabRequestBody) -> tuple[OpenAIRequest, bool, bool]:
    model_str = (body.model or "gemini-4-carbon").strip()
    base_model, _ = _normalize_model_name(model_str)
    is_image = bool(mc.get_profile(base_model).get("is_image", False))

    prompt_txt = (body.prompt or "").strip()
    is_ping = not prompt_txt
    if is_ping:
        if is_image:
            prompt_txt = "Draw a minimal cute sky-blue cloud icon on a clean white background."
        else:
            prompt_txt = "请用简短的一句话回复：连通测试正常。"

    msgs: List[OpenAIMessage] = []
    sys_txt = (body.system or "").strip()
    if sys_txt:
        msgs.append(OpenAIMessage(role="system", content=sys_txt))
    msgs.append(OpenAIMessage(role="user", content=prompt_txt))
    pre_txt = (body.prefill or "").strip()
    if pre_txt:
        msgs.append(OpenAIMessage(role="assistant", content=pre_txt))

    req_kwargs: Dict[str, Any] = {
        "model": model_str,
        "messages": msgs,
        "stream": bool(body.stream),
    }
    if body.stream:
        req_kwargs["stream_options"] = {"include_usage": True}
    return OpenAIRequest(**req_kwargs), is_ping, is_image


def _build_lab_inspect_summary(body: LabRequestBody) -> Dict[str, Any]:
    req_obj, is_ping, is_image = _build_lab_openai_request(body)
    base_model, is_search = _normalize_model_name(req_obj.model)
    eff = app_state.get_effective_settings(base_model)
    prof = mc.get_profile(base_model)
    prefill_txt = (body.prefill or "").strip()
    prefill_mode = str(eff.get("prefill_mode") or "smart")
    prefill_active = bool(prefill_txt and prefill_mode != "off")

    tk = mc.resolve_thinking(req_obj.model, req_obj, eff, prefill_active=prefill_active)
    sandbox = create_sandbox_spec(base_model, req_obj, eff)
    fake_on = bool(app_state.get_setting("fake_streaming", config.FAKE_STREAMING_ENABLED))
    real_stream = bool(req_obj.stream and not fake_on and not is_image)
    use_stream_fc = bool(sandbox.enabled and real_stream and supports_stream_fc_args(base_model))

    forced_target, disable_retries, target_err, target_human = _resolve_lab_target(body.target)
    loc = str(eff.get("express_location") or "global").strip() or "默认"

    if forced_target is not None:
        use_cookie = forced_target["channel"] == "cookie"
        if use_cookie:
            channel_desc = "仅测 Studio Cookie 直连（单发直测·不切主通道）"
        else:
            krec = forced_target["key_record"]
            pid = (krec.get("project_id") or "").strip() or "默认"
            channel_desc = f"仅测 Key #{krec.get('index', 0) + 1} · 项目 {pid} · {loc}（单发直测）"
    else:
        use_cookie = app_state.is_web_proxy_enabled()
        enabled_keys = sum(1 for r in app_state.get_express_keys() if r.get("enabled", True))
        channel_desc = (
            "跟随主通道 · Studio Cookie 直连"
            if use_cookie
            else f"跟随主通道 · Express SDK（{enabled_keys} 把可用 Key · {loc}）"
        )
        if target_err:
            channel_desc = f"目标异常：{target_err}"

    if tk.get("mode") is None:
        tk_desc = "无思考调节"
    elif not tk.get("include_thoughts", True):
        tk_desc = f"压制关闭（下发 {str(tk.get('level') or 'low').upper()} 且隐藏思考块）"
    else:
        tk_desc = f"{eff.get('native_thinking_mode', 'request')} · 档位 {str(tk.get('level') or '官方默认').upper()}"

    img_size = mc.resolve_image_size(base_model, req_obj, eff) if is_image else None
    img_ar = (mc.resolve_aspect_ratio(base_model, req_obj, eff) or "自动") if is_image else None

    if is_image:
        card5 = {"k": "生图配置", "v": f"分辨率 {img_size} · 宽高比 {img_ar}（生图自动旁路抗截断）"}
    elif sandbox.enabled:
        card5 = {"k": "抗截断守护", "v": f"单字段 {sandbox.emit_tool_name} + 滑动尾窗" + ("（流式分片）" if use_stream_fc else "")}
    else:
        card5 = {"k": "抗截断守护", "v": f"未启用（{sandbox.bypass_reason or '原生输出'}）"}

    stream_desc = (
        "非流式整包 (stream=false)"
        if not req_obj.stream
        else ("假流式保活转发（生图保护）" if is_image else ("假流式保活转发" if fake_on else "真流式 SSE 透传"))
    )

    sampling_stripped = bool(prof.get("sampling_advice") == "deprecated" or is_image)
    pre_desc = (
        f"附带 {len(prefill_txt)} 字 · 模式 {prefill_mode}"
        if prefill_txt
        else (f"未附带预填充（模式 {prefill_mode}）")
    )

    payload_dict: Dict[str, Any] = {
        "model": base_model,
        "is_image": is_image,
        "target": body.target or "auto",
        "target_label": target_human,
        "targeted_single_shot": bool(disable_retries),
        "channel": "cookie" if use_cookie else "express",
        "express_location": loc if not use_cookie else None,
        "stream": bool(req_obj.stream),
        "fake_streaming": fake_on,
        "thinking_config": tk,
        "anti_truncation": {
            "enabled": bool(sandbox.enabled),
            "tool": sandbox.emit_tool_name if sandbox.enabled else None,
            "stream_fc_args": use_stream_fc,
        },
        "prefill": {
            "active": prefill_active,
            "mode": prefill_mode,
            "suppress_thinking": bool(eff.get("prefill_suppress_thinking", True)),
            "chars": len(prefill_txt),
        },
        "has_system_instruction": bool((body.system or "").strip() or (eff.get("inject_system_instruction") or "").strip()),
        "deprecated_sampling_stripped": sampling_stripped,
        "resumable_enabled": bool(app_state.get_setting("resumable_enabled", True)),
    }
    if is_image:
        payload_dict["image_config"] = {"image_size": img_size, "aspect_ratio": img_ar}

    return {
        "is_ping": is_ping,
        "is_image": is_image,
        "targeted_single_shot": bool(disable_retries),
        "cards": [
            {"k": "测试目标 & 路由", "v": channel_desc},
            {"k": "上游目标模型", "v": base_model + (" [生图]" if is_image else "") + (" + Google Search" if is_search else "")},
            {"k": "传输模式", "v": stream_desc},
            {"k": "思考配置", "v": tk_desc},
            card5,
            {"k": "预填充 & 采样清理", "v": f"{pre_desc} · {'废弃采样自动剥离' if sampling_stripped else '默认透传'}"},
        ],
        "payload": payload_dict,
    }


@router.post("/api/lab/inspect")
async def lab_inspect_api(body: LabRequestBody, _auth: bool = Depends(require_auth)):
    return JSONResponse(content=_build_lab_inspect_summary(body))


@router.post("/api/lab/run")
async def lab_run_api(body: LabRequestBody, request: Request, _auth: bool = Depends(require_auth)):
    """执行一次真实的连通实测请求，以统一 SSE 解剖事件流返回（支持定向单凭证直测、生图直显、真流式与非流式及实时中止）。"""
    inspect_info = _build_lab_inspect_summary(body)
    req_obj, is_ping, is_image = _build_lab_openai_request(body)
    forced_target, disable_retries, target_err, target_human = _resolve_lab_target(body.target)
    primary_ch = (
        forced_target["channel"]
        if isinstance(forced_target, dict)
        else ("cookie" if app_state.is_web_proxy_enabled() else "express")
    )

    async def _lab_stream():
        t0 = time.monotonic()
        first_thought_t: Optional[float] = None
        first_body_t: Optional[float] = None
        last_thought_t: Optional[float] = None
        thought_chars = 0
        body_chars = 0
        usage_info: Dict[str, Any] = {}
        finish_reason = "stop"
        run_meta: Dict[str, Any] = {
            "channel": primary_ch,
            "targeted_single_shot": bool(disable_retries),
            "target_label": target_human,
            "is_image": is_image,
            "sandbox_active": bool((inspect_info.get("payload") or {}).get("anti_truncation", {}).get("enabled")),
            "prefill_active": bool((inspect_info.get("payload") or {}).get("prefill", {}).get("active")),
        }
        if is_image:
            run_meta["image_config"] = (inspect_info.get("payload") or {}).get("image_config") or {}

        def _ev(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        yield _ev({"type": "inspect", "inspect": inspect_info, "is_ping": is_ping, "is_image": is_image})

        if target_err:
            diag = _diagnose_lab_error(400, target_err, primary_ch)
            yield _ev({"type": "error", "code": 400, "message": target_err, "diagnosis": diag, "meta": run_meta})
            return

        try:
            call_kw: Dict[str, Any] = {
                "request_obj": req_obj,
                "fastapi_request": request,
                "primary_channel": primary_ch,
            }
            try:
                sig_params = inspect.signature(execute_unified_completion).parameters
                if "forced_target" in sig_params:
                    call_kw["forced_target"] = forced_target
                if "disable_retries" in sig_params:
                    call_kw["disable_retries"] = disable_retries
                if "result_meta" in sig_params:
                    call_kw["result_meta"] = run_meta
            except Exception:
                pass
            resp = await execute_unified_completion(**call_kw)
        except Exception as e:
            code, msg = extract_upstream_error(e)
            diag = _diagnose_lab_error(code, msg, str(run_meta.get("channel") or primary_ch))
            yield _ev({"type": "error", "code": code, "message": msg, "diagnosis": diag, "meta": run_meta})
            return

        if isinstance(resp, JSONResponse):
            status = int(getattr(resp, "status_code", 200))
            raw_b = getattr(resp, "body", b"") or b""
            try:
                data = json.loads(raw_b.decode("utf-8", errors="replace")) if raw_b else {}
            except Exception:
                data = {}
            if status >= 400:
                err_msg = (data.get("error") or {}).get("message") if isinstance(data.get("error"), dict) else str(data)
                diag = _diagnose_lab_error(status, err_msg or f"HTTP {status}", str(run_meta.get("channel") or primary_ch))
                yield _ev({"type": "error", "code": status, "message": err_msg or f"HTTP {status}", "diagnosis": diag, "meta": run_meta})
                return

            ch0 = (data.get("choices") or [{}])[0]
            msg0 = ch0.get("message") or {}
            finish_reason = str(ch0.get("finish_reason") or "stop")
            rc = msg0.get("reasoning_content") or ""
            ct = msg0.get("content") or ""
            if rc:
                thought_chars = len(rc)
                yield _ev({"type": "thought", "delta": rc})
            if ct:
                body_chars = len(ct)
                yield _ev({"type": "content", "delta": ct})
            u = data.get("usage") or {}
            p_tk = int(u.get("prompt_tokens") or 0)
            c_tk = int(u.get("completion_tokens") or 0)
            cached_tk = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
            th_tk = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            out_tk = max(0, c_tk - th_tk)
            total_s = round(max(0.01, time.monotonic() - t0), 2)
            tps = round(c_tk / total_s, 1) if (c_tk > 0 and total_s > 0.05) else None
            yield _ev({
                "type": "done",
                "finish_reason": finish_reason,
                "meta": run_meta,
                "timing": {"conn": round(total_s * 0.15, 2), "think": round(total_s * 0.35, 2) if th_tk > 0 else 0.0, "gen": round(total_s * (0.5 if th_tk > 0 else 0.85), 2), "total": total_s, "tps": tps},
                "usage": {"prompt_tokens": p_tk, "cached_tokens": cached_tk, "thoughts_tokens": th_tk, "output_tokens": out_tk, "total_tokens": p_tk + th_tk + out_tk},
                "chars": {"thought": thought_chars, "body": body_chars},
            })
            return

        # StreamingResponse from execute_unified_completion
        body_iter = resp.body_iterator
        buf = ""
        try:
            async for raw_chunk in body_iter:
                text_chunk = raw_chunk.decode("utf-8", errors="replace") if isinstance(raw_chunk, bytes) else str(raw_chunk)
                buf += text_chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if not payload_str or payload_str == "[DONE]":
                            continue
                        try:
                            obj = json.loads(payload_str)
                        except Exception:
                            continue
                        if isinstance(obj, dict) and obj.get("error"):
                            err_o = obj["error"]
                            e_code = int(err_o.get("code") or 500) if isinstance(err_o, dict) else 500
                            e_msg = err_o.get("message") if isinstance(err_o, dict) else str(err_o)
                            diag = _diagnose_lab_error(e_code, e_msg, str(run_meta.get("channel") or primary_ch))
                            yield _ev({
                                "type": "error",
                                "code": e_code,
                                "message": e_msg,
                                "diagnosis": diag,
                                "meta": run_meta,
                            })
                            return
                        if obj.get("id") == "chatcmpl-keepalive":
                            continue
                        if isinstance(obj.get("usage"), dict):
                            u = obj["usage"]
                            p_tk = int(u.get("prompt_tokens") or 0)
                            c_tk = int(u.get("completion_tokens") or 0)
                            cached_tk = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
                            th_tk = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
                            out_tk = max(0, c_tk - th_tk)
                            usage_info = {
                                "prompt_tokens": p_tk,
                                "cached_tokens": cached_tk,
                                "thoughts_tokens": th_tk,
                                "output_tokens": out_tk,
                                "total_tokens": p_tk + th_tk + out_tk,
                            }
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        c0 = choices[0]
                        if c0.get("finish_reason"):
                            finish_reason = str(c0["finish_reason"])
                        delta = c0.get("delta") or {}
                        rc = delta.get("reasoning_content")
                        if rc:
                            now_m = time.monotonic()
                            if first_thought_t is None:
                                first_thought_t = now_m
                            last_thought_t = now_m
                            thought_chars += len(rc)
                            yield _ev({"type": "thought", "delta": rc})
                        ct = delta.get("content")
                        if ct:
                            now_m = time.monotonic()
                            if first_body_t is None:
                                first_body_t = now_m
                            body_chars += len(ct)
                            yield _ev({"type": "content", "delta": ct})
        finally:
            if hasattr(body_iter, "aclose"):
                try:
                    await body_iter.aclose()
                except Exception:
                    pass

        end_m = time.monotonic()
        total_s = round(max(0.01, end_m - t0), 2)
        first_any = min([x for x in (first_thought_t, first_body_t) if x is not None], default=end_m)
        conn_s = round(max(0.01, first_any - t0), 2)
        if first_thought_t is not None:
            th_end = first_body_t if first_body_t is not None else (last_thought_t or end_m)
            think_s = round(max(0.0, th_end - first_thought_t), 2)
        else:
            think_s = 0.0
        gen_s = round(max(0.0, end_m - (first_body_t if first_body_t is not None else end_m)), 2)
        out_tot_tk = int(usage_info.get("output_tokens") or 0) + int(usage_info.get("thoughts_tokens") or 0)
        denom = max(0.1, end_m - first_any) if first_any < end_m else total_s
        tps = round(out_tot_tk / denom, 1) if out_tot_tk > 0 else (round((body_chars / 1.65) / max(0.1, gen_s), 1) if body_chars > 0 else None)

        yield _ev({
            "type": "done",
            "finish_reason": finish_reason,
            "meta": run_meta,
            "timing": {"conn": conn_s, "think": think_s, "gen": gen_s, "total": total_s, "tps": tps},
            "usage": usage_info or {"prompt_tokens": 0, "cached_tokens": 0, "thoughts_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "chars": {"thought": thought_chars, "body": body_chars},
        })

    return StreamingResponse(_lab_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


# ============================================================
# 实时运行日志 SSE 流
# ============================================================

@router.get("/stream-logs")
async def stream_logs_endpoint(request: Request, _auth: bool = Depends(require_auth)):
    def _event_chunks(msg: str):
        for line in msg.splitlines() or [""]:
            yield f"data: {line}\n"
        yield "\n"

    async def log_generator():
        q = rt_logger.subscribe()
        try:
            for msg in rt_logger.snapshot_history():
                for chunk in _event_chunks(msg):
                    yield chunk
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    for chunk in _event_chunks(msg):
                        yield chunk
                except asyncio.TimeoutError:
                    yield ": keep-alive heartbeat\n\n"
        finally:
            rt_logger.unsubscribe(q)

    return StreamingResponse(log_generator(), media_type="text/event-stream")
