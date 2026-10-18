import asyncio
import os
import secrets
import threading
import time
from typing import Optional
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from auth import get_api_key
from express_key_manager import ExpressKeyManager
from routes import models_api, chat_api

from logger import rt_logger, stats
import config
from runtime_state import app_state
import model_capabilities as mc
from model_loader import get_express_models

from cookie_auth import validate_cookie
import express_key_probe
from resumable import resumable_store

express_key_manager = ExpressKeyManager()


async def _probe_express_projects_on_startup() -> None:
    """启动后异步探测尚无 Project ID 的 Express Key（失败不影响服务）。"""
    try:
        pending = [r["key"] for r in app_state.get_express_keys() if not r.get("project_id")]
        if not pending:
            return
        print(f"🔎 [密钥探测] 启动探测：{len(pending)} 个 Key 尚无 Project ID，正在用不存在的模型名调 generateContent 探测。")
        await express_key_probe.probe_all(keys=pending)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"⚠️ [密钥探测] 启动探测失败（不影响服务）：{type(e).__name__} - {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_loader import refresh_models_config_cache
    print("🚀 [服务启动] agentplatform2api 适配器已启动（Express API Key / Cookie 直连 双通道）。")

    if config.API_KEY == DEFAULT_API_KEY:
        public_host = any(os.environ.get(k) for k in ("SPACE_ID", "SPACE_HOST", "HF_SPACE_ID"))
        if public_host and os.environ.get("ALLOW_DEFAULT_KEY", "").lower() not in ("1", "true", "yes"):
            raise RuntimeError(
                "检测到公开托管环境（HuggingFace Space 等）且 API_KEY 仍为默认值 123456。\n"
                "该口令同时是控制台登录密码，而控制台可以读写完整的 Google 会话 Cookie。\n"
                "请设置一个强 API_KEY 后重启；确需临时放行可设 ALLOW_DEFAULT_KEY=true。"
            )
        print("🔴 [安全警告] API_KEY 仍是默认值 123456！它既是本代理的 Key，也是控制台登录口令，"
              "请立刻改成强口令。")

    app_state.import_env_express_keys_once(config.VERTEX_EXPRESS_API_KEY_VAL)
    if express_key_manager.get_total_keys() > 0:
        print(f"✅ [密钥配置] 已加载 {express_key_manager.get_total_keys()} 个 Express API Key。")
    else:
        print("⚠️ [密钥配置] 未配置任何 Express API Key（控制台「通道与凭证」页可随时添加）。"
              "若不启用 Cookie 直连模式，聊天请求将会报错。")
    await refresh_models_config_cache()

    _probe_task = asyncio.create_task(_probe_express_projects_on_startup())
    _sweeper_task = asyncio.create_task(resumable_store.sweeper_loop())
    try:
        yield
    finally:
        for _t in (_probe_task, _sweeper_task):
            if not _t.done():
                _t.cancel()
        resumable_store.cancel_all_pumps()


app = FastAPI(title="agentplatform2api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.express_key_manager = express_key_manager


@app.middleware("http")
async def stats_tracker_middleware(request: Request, call_next):
    if "chat/completions" in request.url.path:
        stats.increment_total()
        try:
            response = await call_next(request)
            if response.status_code >= 400 and response.status_code != 499:
                stats.add_error()
            return response
        except Exception as e:
            stats.add_error()
            raise e
    return await call_next(request)


# ====== 仅密码登录（Cookie 会话，免输账号）======
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


LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>登录 · agentplatform2api</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
body{
  background:radial-gradient(circle at 15% 20%, #f5eeff 0%, transparent 45%),
             radial-gradient(circle at 85% 80%, #e8fbf6 0%, transparent 45%),
             #fdfbf9;
  font-family:'Inter',system-ui,sans-serif;color:#3d354b;
}
.card{
  background:rgba(255,255,255,.92);backdrop-filter:blur(10px);
  border:1px solid #ede7f6;border-radius:20px;
  box-shadow:0 10px 30px -10px rgba(139,124,246,.14);
}
.inp{border:1px solid #e6dff2;border-radius:12px;padding:11px 14px;width:100%;outline:none;font-size:15px;background:#fcfaff;color:#3d354b;transition:.2s}
.inp:focus{border-color:#8b7cf6;background:#fff;box-shadow:0 0 0 3px rgba(139,124,246,.16)}
.btn{background:linear-gradient(135deg,#8b7cf6 0%,#f48fb1 100%);color:#fff;border-radius:12px;font-weight:600;padding:11px;width:100%;transition:.2s;box-shadow:0 4px 12px rgba(139,124,246,.28)}
.btn:hover{opacity:.94;transform:translateY(-1px)}
</style>
</head><body class="min-h-screen flex items-center justify-center px-4">
  <div class="w-full max-w-sm">
    <div class="flex items-center gap-3 mb-6 justify-center">
      <svg width="42" height="42" viewBox="0 0 40 40" fill="none"><rect width="40" height="40" rx="12" fill="url(#lg)"/>
      <path d="M11 16 H27" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/><path d="M23 12 L28.5 16 L23 20" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M29 24 H13" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-opacity=".92"/><path d="M17 20 L11.5 24 L17 28" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".92"/>
      <defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse"><stop stop-color="#8b7cf6"/><stop offset=".55" stop-color="#f48fb1"/><stop offset="1" stop-color="#5fd3bc"/></linearGradient></defs></svg>
      <span class="text-xl font-bold tracking-tight">agentplatform<span style="color:#8b7cf6">2api</span></span>
    </div>
    <div class="card p-6">
      <p class="text-sm text-[#7c728d] mb-4">请输入访问密码（即 API_KEY）</p>
      <form id="f" onsubmit="return doLogin(event)">
        <input id="pw" type="password" class="inp mb-3.5" placeholder="密码" autofocus autocomplete="current-password">
        <button class="btn" type="submit">进入控制台</button>
      </form>
      <p id="err" class="text-xs text-[#e05670] mt-3 h-4"></p>
    </div>
  </div>
<script>
async function doLogin(e){
  e.preventDefault();
  const pw=document.getElementById('pw').value;
  const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
  if(r.ok){ location.href='/'; } else {
    const d=await r.json().catch(()=>({}));
    document.getElementById('err').textContent=d.error||'密码错误，请重试';
  }
  return false;
}
</script>
</body></html>
"""


# ==========================================
# 控制台（马卡龙柔和配色，单文件，免构建）
# ==========================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agentplatform2api · 控制台</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --border:#ece5f5;
    --muted:#7d738e;
    --fg:#3b334a;
    --bg:#fdfbf9;
    --accent:#8b7cf6;
    --accent2:#f48fb1;
    --accent-hover:#7665e8;
    --mint:#5fd3bc;
    --mint-text:#239480;
    --apricot:#f8c27a;
    --apricot-text:#c97b1e;
    --coral:#f68697;
    --coral-text:#d64860;
    --sky:#74b9ff;
  }
  * { -webkit-font-smoothing:antialiased; }
  body {
    background:
      radial-gradient(circle at 8% 12%, rgba(243,232,255,.65) 0%, transparent 40%),
      radial-gradient(circle at 92% 18%, rgba(230,252,245,.65) 0%, transparent 40%),
      radial-gradient(circle at 50% 90%, rgba(255,238,244,.55) 0%, transparent 45%),
      var(--bg);
    color:var(--fg);
    font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  }
  .card {
    background:rgba(255,255,255,.94);
    border:1px solid var(--border);
    border-radius:16px;
    box-shadow:0 4px 18px -4px rgba(139,124,246,.07);
  }
  .stat-total { background:linear-gradient(145deg,#ffffff 0%,#f6f2ff 100%); border-color:#e6defa; }
  .stat-ok    { background:linear-gradient(145deg,#ffffff 0%,#eefcf8 100%); border-color:#d5f5ec; }
  .stat-retry { background:linear-gradient(145deg,#ffffff 0%,#fff9ef 100%); border-color:#fae7c8; }
  .stat-err   { background:linear-gradient(145deg,#ffffff 0%,#fff2f5 100%); border-color:#fbd5dc; }
  .lbl { font-size:12px; color:var(--muted); font-weight:500; }
  .val { font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .btn {
    background:linear-gradient(135deg, #8b7cf6 0%, #a288f8 100%);
    color:#fff; border-radius:10px; font-weight:600; transition:.18s;
    box-shadow:0 3px 10px rgba(139,124,246,.24);
  }
  .btn:hover { background:linear-gradient(135deg, #7a6ae8 0%, #9175f0 100%); transform:translateY(-1px); }
  .btn:disabled { opacity:.5; cursor:not-allowed; transform:none; }
  .btn-soft {
    background:#f8f5ff; color:#6b5bd6; border:1px solid #e4dcfa;
    border-radius:10px; font-weight:500; transition:.15s;
  }
  .btn-soft:hover { background:#f0ebff; border-color:#d2c5f8; }
  .inp {
    border:1px solid var(--border); border-radius:10px; background:#fff;
    font-size:14px; padding:8px 11px; width:100%; outline:none; transition:.15s; color:var(--fg);
  }
  .inp:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(139,124,246,.14); }
  .inp:disabled { background:#f7f5fa; color:#b0a8be; }
  .tab {
    padding:13px 6px; font-size:14px; font-weight:600; color:var(--muted);
    border-bottom:2.5px solid transparent; cursor:pointer; white-space:nowrap; transition:.15s;
  }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab:hover:not(.active){ color:var(--fg); }
  .pill {
    display:inline-flex; align-items:center; gap:6px; font-size:12px;
    padding:3.5px 11px; border-radius:999px; border:1px solid var(--border);
    background:#fff; color:#655b77;
  }
  .pill-accent {
    border-color:#e2d9fc;
    background:linear-gradient(135deg,rgba(139,124,246,.12),rgba(244,143,177,.12));
    color:#6b5bd6; font-weight:600;
  }
  .pill-mint { border-color:#cdeee5; background:#eefcf8; color:var(--mint-text); font-weight:600; }
  .pill-apricot { border-color:#f9e2be; background:#fff8ee; color:var(--apricot-text); font-weight:600; }
  .log { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; word-break:break-all; }
  ::-webkit-scrollbar{width:8px;height:8px}
  ::-webkit-scrollbar-thumb{background:#dfd8ec;border-radius:8px}
  ::-webkit-scrollbar-thumb:hover{background:#c9bfe0}
  .toast {
    position:fixed; bottom:24px; left:50%; transform:translateX(-50%);
    background:linear-gradient(135deg,#4a3f63,#635380); color:#fff;
    padding:10px 20px; border-radius:12px; font-size:14px; opacity:0;
    transition:.25s; pointer-events:none; z-index:50;
    box-shadow:0 8px 24px rgba(74,63,99,.22);
  }
  .toast.show { opacity:1; }
  .switch { position:relative; width:40px; height:22px; display:inline-block; }
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#e6e0f0;border-radius:999px;transition:.2s;cursor:pointer}
  .slider:before{content:"";position:absolute;height:16px;width:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s;box-shadow:0 1px 3px rgba(0,0,0,.15)}
  input:checked + .slider{background:linear-gradient(135deg,#8b7cf6,#5fd3bc)}
  input:checked + .slider:before{transform:translateX(18px)}
  .hero {
    background:linear-gradient(135deg,rgba(255,255,255,.96) 0%,rgba(249,245,255,.94) 55%,rgba(239,252,248,.94) 100%);
    border:1px solid var(--border); border-radius:18px;
    box-shadow:0 6px 20px -6px rgba(139,124,246,.09);
  }
  .helpq {
    display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px;
    border-radius:50%; border:1px solid #d5cce6; color:#9286a8; font-size:10px; line-height:1;
    cursor:pointer; user-select:none; vertical-align:middle; margin-left:5px; transition:.15s;
    background:#faf8ff;
  }
  .helpq:hover { border-color:var(--accent); color:var(--accent); background:#f2edff; }
  .helpq.on { border-color:var(--accent); color:#fff; background:var(--accent); }
  .helpbox {
    display:none; margin-top:7px; padding:10px 12px; border-radius:10px; background:#faf8fd;
    border:1px solid #ece5f7; font-size:12px; line-height:1.75; color:#5e5470;
  }
  .helpbox.show { display:block; }
  .helpbox, .helpbox code { overflow-wrap:anywhere; word-break:break-word; }
  .helpbox code { display:inline; white-space:normal; background:#ede6fa; color:#5a48b8; padding:1px 5px; border-radius:5px; font-size:11px; }
  .helpbox b { color:var(--fg); }
</style>
</head>
<body class="min-h-screen">
<div class="max-w-5xl mx-auto px-5 md:px-8 py-6">

  <!-- Header -->
  <header class="hero px-5 py-4 mb-6 flex items-center justify-between flex-wrap gap-3">
    <div class="flex items-center gap-3.5">
      <svg width="42" height="42" viewBox="0 0 40 40" fill="none" aria-hidden="true">
        <rect width="40" height="40" rx="12" fill="url(#lg)"/>
        <path d="M11 16 H27" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M23 12 L28.5 16 L23 20" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M29 24 H13" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-opacity=".92"/>
        <path d="M17 20 L11.5 24 L17 28" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".92"/>
        <defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse"><stop stop-color="#8b7cf6"/><stop offset=".55" stop-color="#f48fb1"/><stop offset="1" stop-color="#5fd3bc"/></linearGradient></defs>
      </svg>
      <div>
        <h1 class="text-xl font-bold tracking-tight leading-none">agentplatform<span style="color:var(--accent)">2api</span></h1>
        <p class="text-xs text-[#7d738e] mt-1.5">OpenAI 兼容代理 · Gemini Agent Platform 双通道</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <span class="pill pill-accent" id="mode-pill">通道 —</span>
      <span class="pill pill-mint"><span class="relative flex h-2 w-2"><span class="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style="background:#5fd3bc"></span><span class="relative inline-flex rounded-full h-2 w-2" style="background:#2eb89e"></span></span><span id="uptime">运行中</span></span>
      <button onclick="logout()" class="pill hover:bg-[#f7f4fc]" style="cursor:pointer" title="退出登录">退出</button>
    </div>
  </header>

  <!-- Tabs -->
  <nav class="flex gap-6 border-b border-[#ece5f5] mb-6 overflow-x-auto">
    <div class="tab active" data-tab="overview" onclick="switchTab('overview')">数据概览</div>
    <div class="tab" data-tab="channel" onclick="switchTab('channel')">通道与凭证</div>
    <div class="tab" data-tab="params" onclick="switchTab('params')">模型参数</div>
    <div class="tab" data-tab="logs" onclick="switchTab('logs')">运行日志</div>
  </nav>

  <!-- Overview -->
  <section id="view-overview">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3.5 mb-4">
      <div class="card stat-total p-4"><div class="lbl">总请求</div><div id="s-total" class="val text-2xl font-bold mt-1" style="color:#6b5bd6">0</div></div>
      <div class="card stat-ok p-4"><div class="lbl">成功响应</div><div id="s-success" class="val text-2xl font-bold mt-1" style="color:var(--mint-text)">0</div></div>
      <div class="card stat-retry p-4"><div class="lbl">拥堵重试</div><div id="s-retries" class="val text-2xl font-bold mt-1" style="color:var(--apricot-text)">0</div></div>
      <div class="card stat-err p-4"><div class="lbl">错误 / 拦截</div><div id="s-error" class="val text-2xl font-bold mt-1" style="color:var(--coral-text)">0</div></div>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5">
      <div class="card p-5 flex flex-col items-center justify-between">
        <div class="flex items-center justify-between w-full mb-2">
          <span class="lbl">服务健康度</span>
          <span id="health-rate" class="pill pill-mint" style="font-size:11px;padding:2px 8px">100% 正常</span>
        </div>
        <div class="w-36 h-36 my-1"><canvas id="donut"></canvas></div>
        <div class="flex items-center justify-center gap-4 text-xs text-[#6e6480] mt-2 w-full">
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#5fd3bc"></span>成功</span>
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#f8c27a"></span>重试</span>
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#f68697"></span>错误</span>
        </div>
      </div>
      <div class="card p-5 md:col-span-2 flex flex-col justify-between">
        <div class="lbl mb-4">Token 算力消耗 <span class="text-[#9a90ab]" style="text-transform:none;font-weight:400">· 仅标准 Express 通道计入（Cookie 直连接口不回传用量）</span></div>
        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-sm mb-1.5"><span class="text-[#5e5470]">Prompt（输入）</span><span id="t-prompt" class="val font-semibold" style="color:#6b5bd6">0</span></div>
            <div class="w-full bg-[#f3effa] rounded-full h-2 overflow-hidden"><div id="bar-prompt" class="h-2 rounded-full transition-all duration-300" style="width:0%;background:linear-gradient(90deg,#8b7cf6,#74b9ff)"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1.5"><span class="text-[#5e5470]">Completion（思考与输出）</span><span id="t-comp" class="val font-semibold" style="color:var(--mint-text)">0</span></div>
            <div class="w-full bg-[#f3effa] rounded-full h-2 overflow-hidden"><div id="bar-comp" class="h-2 rounded-full transition-all duration-300" style="width:0%;background:linear-gradient(90deg,#5fd3bc,#f48fb1)"></div></div>
          </div>
          <div class="pt-4 border-t border-[#f0eaf8] flex justify-between items-center"><span class="lbl">累计总消耗</span><span id="t-total" class="val text-xl font-bold" style="color:#5344a8">0</span></div>
        </div>
      </div>
    </div>
  </section>

  <!-- Channel -->
  <section id="view-channel" class="hidden">
    <div class="card p-5 mb-4">
      <div class="lbl mb-3">上游调用通道</div>
      <div class="flex flex-col sm:flex-row gap-3">
        <label class="flex-1 border border-[#e6dff2] rounded-xl p-3.5 cursor-pointer flex items-start gap-3 hover:bg-[#faf8ff] transition" id="opt-api">
          <input type="radio" name="mode" value="api_key" class="mt-1 accent-[#8b7cf6]" onchange="updateMode('api_key')">
          <div><div class="font-semibold text-sm">Express API Key（标准模式）</div><div class="text-xs text-[#7d738e] mt-0.5">使用官方 google-genai SDK + 多账号 Express Key 轮询调用</div></div>
        </label>
        <label class="flex-1 border border-[#e6dff2] rounded-xl p-3.5 cursor-pointer flex items-start gap-3 hover:bg-[#faf8ff] transition" id="opt-web">
          <input type="radio" name="mode" value="web_proxy" class="mt-1 accent-[#8b7cf6]" onchange="updateMode('web_proxy')">
          <div><div class="font-semibold text-sm">Cookie 直连反代（Studio 模式）</div><div class="text-xs text-[#7d738e] mt-0.5">使用控制台 Cookie + SAPISIDHASH 直连 batchGraphql</div></div>
        </label>
      </div>
    </div>

    <!-- Express API Key 列表（多账号，热编辑，即添即探） -->
    <div class="card p-5 mb-4">
      <div class="flex items-center justify-between flex-wrap gap-2 mb-3">
        <div>
          <div class="text-sm font-semibold">Express API Key 列表<span class="helpq" onclick="hlp(this,'h_keys')">?</span></div>
          <div class="text-xs text-[#7d738e] mt-0.5">支持添加多个谷歌账号的 Key；添加后直接点「探测」会自动保存并探测归属项目</div>
        </div>
        <div class="flex items-center gap-2 flex-wrap">
          <span id="keys-count" class="pill pill-accent" style="text-transform:none">0 个</span>
          <button class="btn-soft px-3 py-1.5 text-sm" onclick="probeAllKeys()">🔎 探测全部项目</button>
          <button class="btn-soft px-3 py-1.5 text-sm" onclick="addKeyRow()">＋ 添加 Key</button>
          <button class="btn px-4 py-1.5 text-sm" onclick="saveExpressKeys()">保存 Key 列表</button>
        </div>
      </div>
      <div id="h_keys" class="helpbox">
        <b>即添即探 &amp; 热生效</b>：点击「＋ 添加 Key」填入新 Key 后，无论直接点单行的「探测」、「🔎 探测全部项目」还是「保存 Key 列表」，都会<b>自动将新 Key 保存并立即热生效</b>，不会丢失已填内容，也无需重启服务。<br>
        <b>Project ID 绑定</b>：每个 Key 对应各自谷歌账号的 GCP 项目，标准模式钉定 location 时优先使用该 Key 自己的 Project ID。<br>
        <b>环境变量</b>：<code>VERTEX_EXPRESS_API_KEY</code> 仅在首次启动时做一次性导入；清空列表并保存后，重启不会复种。
      </div>
      <div id="keys-list" class="space-y-2.5"></div>
      <p id="keys-empty" class="text-xs text-[#7d738e] mt-2 hidden">当前没有任何 Express Key。点「＋ 添加 Key」粘贴一个，直接点「探测」或「保存 Key 列表」即可。</p>
    </div>

    <div id="cookie-box" class="card p-5 hidden">
      <div class="lbl mb-2">Google Cookie（含 HttpOnly 字段）</div>
      <textarea id="cookie-input" rows="3" class="inp log mb-3" placeholder="粘贴 console.cloud.google.com 的完整 Cookie（支持 Cookie-Editor 导出的 JSON / Header String，自动解析）"></textarea>
      <div class="lbl mb-2">Google Cloud Project ID</div>
      <input id="project-input" class="inp mb-3" placeholder="可直接粘贴含 ?project=xxx 的整条 URL，自动提取">
      <div class="flex items-center justify-between">
        <span class="text-xs text-[#7d738e]">保存后自动校验是否包含 SAPISID</span>
        <button class="btn px-4 py-2 text-sm" onclick="saveCookie()">保存并激活</button>
      </div>
      <p class="text-xs text-[#7d738e] mt-3 leading-relaxed">💡 Cookie 通常较为持久（可维持数周甚至更久）；当出现 <code>aiplatform.endpoints.predict denied</code> 权限错误时，90% 以上是因为 Cookie 会话已过期，重新从浏览器复制粘贴保存即可。</p>
    </div>
  </section>

  <!-- Params -->
  <section id="view-params" class="hidden">
    <div class="card p-5 mb-4">
      <div class="flex items-center justify-between flex-wrap gap-3">
        <div>
          <div class="lbl mb-1">选择编辑作用域（全局默认 / 指定模型专属） <span id="ov-badge" class="pill pill-accent hidden" style="text-transform:none">★ 已有专属参数</span></div>
          <select id="model-sel" class="inp" style="min-width:260px" onchange="renderCaps()"></select>
        </div>
        <div id="caps-summary" class="text-xs flex flex-wrap gap-2 max-w-xl"></div>
      </div>
      <p class="text-xs text-[#7d738e] mt-3 leading-relaxed">
        • <b>① 按模型参数</b>（思考 / 生图 / 采样 / 预填充与注入）完全跟随上方下拉框：选「＊ 全局默认」时修改的是所有模型的默认值；选具体模型时修改的是该模型的专属值（带 ★），通过本卡片的按钮保存。<br>
        • <b>② 全局设置</b>（location 区域 / 图片压缩 / 重试退避 / 假流式与弱网续传 / 调试开关）全局唯一，通过页面底部的「保存全局设置」按钮保存，<b>两个分区完全独立、互不覆盖</b>。
      </p>
      <div class="flex items-center gap-2.5 mt-3.5 flex-wrap">
        <button id="btn-scope-save" class="btn px-4 py-2 text-sm" onclick="saveModelOverride()">💾 保存为该模型专属</button>
        <button id="btn-scope-clear" class="btn-soft px-4 py-2 text-sm" onclick="clearModelOverride()">清除该模型专属</button>
        <span id="ov-hint" class="text-xs text-[#7d738e]"></span>
      </div>
    </div>

    <div class="flex items-center gap-2 mb-2.5 mt-5">
      <span class="pill pill-accent" style="text-transform:none">① 按模型参数（跟随上方下拉框）</span>
      <span class="text-xs text-[#7d738e]">本区所有设置均支持按模型单独保存，使用上方按钮写入。</span>
    </div>

    <div class="grid md:grid-cols-2 gap-4">
      <!-- 思考 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">思考强度与控制</div>
        <div class="mb-3" id="wrap-g3level">
          <div class="lbl mb-1">思考档位（thinking_level）</div>
          <select id="thinking_g3_level" class="inp"><option value="">自动（按模型默认）</option><option value="minimal">minimal</option><option value="low">low</option><option value="medium">medium</option><option value="high">high</option></select>
        </div>
        <p id="think-note" class="text-xs text-[#7d738e] mt-2"></p>
        <div class="mt-3 pt-3 border-t border-[#f0eaf8]">
          <div class="lbl mb-1">原生思考控制（native_thinking_mode）</div>
          <select id="native_thinking_mode" class="inp">
            <option value="request">跟随请求（默认）</option>
            <option value="off">关闭原生思考（角色扮演推荐）</option>
            <option value="console">强制用上方档位（忽略前端）</option>
          </select>
          <p class="text-xs text-[#7d738e] mt-2 leading-relaxed">🎭 跑酒馆角色扮演预设时推荐选<b>「关闭原生思考」</b>：自动忽略前端传来的 <code>reasoning_effort=xhigh</code>，将档位压到该模型最低合法档并剥离原生思考块，让预设自带的思维链接管。</p>
        </div>
      </div>

      <!-- 生图 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">生图参数</div>
        <div class="mb-3">
          <div class="lbl mb-1">默认分辨率（image_size）</div>
          <select id="image_size" class="inp"><option value="512">512</option><option value="1K">1K</option><option value="2K">2K</option><option value="4K">4K</option></select>
        </div>
        <div>
          <div class="lbl mb-1">默认宽高比（aspect_ratio）</div>
          <select id="image_aspect_ratio" class="inp"></select>
        </div>
        <p id="image-note" class="text-xs text-[#7d738e] mt-2"></p>
        <div class="space-y-3 mt-3 pt-3 border-t border-[#f0eaf8]">
          <div class="text-xs text-[#9a90ab]">以下两项可针对生图模型单独开启保存</div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">生图下发 system 指令<span class="helpq" onclick="hlp(this,'h_isi')">?</span></span><label class="switch"><input type="checkbox" id="image_system_instruction"><span class="slider"></span></label></div>
            <div id="h_isi" class="helpbox">开启后，system 提示词中的画风、构图要求会随生图请求一并下发给生图模型。</div>
          </div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">生图也注入预填充<span class="helpq" onclick="hlp(this,'h_ipf')">?</span></span><label class="switch"><input type="checkbox" id="inject_prefill_for_image"><span class="slider"></span></label></div>
            <div id="h_ipf" class="helpbox">默认关，避免角色扮演的思考标签误注入给生图模型。若想用预填充强力引导画风，可对生图模型单独打开并配专属预填充。</div>
          </div>
        </div>
      </div>

      <!-- 采样默认 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">采样默认值 <span class="text-xs font-normal text-[#9a90ab]">（留空=不注入，交给模型默认）</span></div>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          <div><div class="lbl mb-1.5">temperature</div><input id="default_temperature" type="number" step="0.1" class="inp" placeholder="—"></div>
          <div><div class="lbl mb-1.5">top_p</div><input id="default_top_p" type="number" step="0.05" class="inp" placeholder="—"></div>
          <div><div class="lbl mb-1.5">max_tokens</div><input id="default_max_tokens" type="number" class="inp" placeholder="—"></div>
        </div>
        <div class="flex items-center justify-between gap-3 mt-5 pt-4 border-t border-[#f0eaf8]">
          <span class="text-sm">采样参数处理<span class="helpq" onclick="hlp(this,'h_sp')">?</span></span>
          <select id="sampling_policy" class="inp" style="max-width:170px"><option value="auto">自动判定</option><option value="deprecated">强制剥离</option><option value="allowed">强制保留</option></select>
        </div>
        <div id="h_sp" class="helpbox">决定是否把 <code>temperature / top_p / top_k</code> 发给模型。自 Gemini 3.6 / 3.5-flash-lite / 4.x 起官方已废弃采样参数，自动判定会自动剥离以避免 400 报错。</div>
        <p id="sampling-note" class="text-xs text-[#7d738e] mt-2"></p>
      </div>

      <!-- 预填充与注入（全部归入 ① 按模型参数区） -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">预填充兼容与注入 <span class="text-xs font-normal text-[#9a90ab]">（支持按模型单独配置）</span></div>
        <div class="space-y-3">
          <div>
            <div class="flex items-center justify-between gap-3"><span class="text-sm">预填充兼容模式<span class="helpq" onclick="hlp(this,'h_pfm')">?</span></span>
              <select id="prefill_mode" class="inp" style="width:155px"><option value="smart">智能（默认）</option><option value="keep_turn">保留模型轮次</option><option value="minimal">最小</option><option value="off">关闭</option></select>
            </div>
            <div id="h_pfm" class="helpbox">
              Gemini 3.x / 4.x 拒绝以 assistant 轮次结尾的请求。<br>
              • <b>智能（默认）</b>：将末尾预填充并入最后一条 user 末尾（严格以预填充原文收尾），并在输出侧自动拼回与智能去重，最稳健。<br>
              • <b>保留模型轮次</b>：保留 assistant 预填充轮次，在其后补一句简短 user 推动语，输出侧自动剥离重复开标签。<br>
              • <b>最小</b>：仅补占位 user 保证不报错，不拼回预填充。<br>
              • <b>关闭</b>：原样下发。
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">预填充时压制原生思考<span class="helpq" onclick="hlp(this,'h_pst')">?</span></span><label class="switch"><input type="checkbox" id="prefill_suppress_thinking"><span class="slider"></span></label></div>
            <div id="h_pst" class="helpbox">检测到请求末尾带预填充时，自动将模型的原生思考压到最低并不回传，让预设自带的思维链接管。</div>
          </div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">思维链守卫（防跳过思考）<span class="helpq" onclick="hlp(this,'h_cotg')">?</span></span><label class="switch"><input type="checkbox" id="prefill_cot_guard"><span class="slider"></span></label></div>
            <div id="h_cotg" class="helpbox">自动识别预填充中未闭合的思维链开标签（如 <code>&lt;thinking&gt;</code>、<code>&lt;cot&gt;</code>），前置注入格式硬性要求并在输出侧去重双开标签，防止模型跳过思考直接写正文。</div>
          </div>
          <div class="pt-2 border-t border-[#f0eaf8]">
            <div class="lbl mb-1">附加 system 指令<span class="helpq" onclick="hlp(this,'h_injs')">?</span></div>
            <textarea id="inject_system_instruction" rows="2" class="inp log" placeholder="留空 = 不注入"></textarea>
            <div id="h_injs" class="helpbox">追加到客户端 system 之后。适合 RikkaHub 等轻量前端；酒馆用户已有预设，请留空。</div>
          </div>
          <div>
            <div class="lbl mb-1">注入预填充<span class="helpq" onclick="hlp(this,'h_injp')">?</span></div>
            <textarea id="inject_prefill" rows="2" class="inp log" placeholder="留空 = 不注入"></textarea>
            <div id="h_injp" class="helpbox">当客户端请求未带预填充时，自动补一条 assistant 预填充；若客户端已自带预填充则自动跳过。</div>
          </div>
          <div>
            <div class="lbl mb-1">续写指令模板<span class="helpq" onclick="hlp(this,'h_pfi')">?</span></div>
            <textarea id="prefill_instruction" rows="2" class="inp log" placeholder="留空 = 使用内置默认"></textarea>
            <div id="h_pfi" class="helpbox">留空即使用内置默认续写引导语（生图模型留空时会自动换用要求输出图片的专用指令）。</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ② 全局设置 -->
    <div class="flex items-center gap-2 mb-2.5 mt-7">
      <span class="pill pill-apricot" style="text-transform:none">② 全局基础设施设置</span>
      <span class="text-xs text-[#7d738e]">对所有模型统一生效；修改后点击页面底部「保存全局设置」，不会影响上方 ① 区的任何参数。</span>
    </div>

    <div class="grid md:grid-cols-2 gap-4">
      <div class="card p-5">
        <div class="flex items-center justify-between gap-3">
          <span class="text-sm font-semibold">标准模式 location<span class="helpq" onclick="hlp(this,'h_loc')">?</span></span>
          <select id="express_location" class="inp" style="max-width:190px">
            <option value="global">global（默认·推荐）</option>
            <option value="">默认（后端自选·旧行为）</option>
            <optgroup label="美国">
              <option value="us-central1">us-central1（爱荷华）</option>
              <option value="us-east1">us-east1（南卡）</option>
              <option value="us-east4">us-east4（北弗吉尼亚）</option>
              <option value="us-east5">us-east5（哥伦布）</option>
              <option value="us-south1">us-south1（达拉斯）</option>
              <option value="us-west1">us-west1（俄勒冈）</option>
              <option value="us-west4">us-west4（拉斯维加斯）</option>
            </optgroup>
            <optgroup label="欧洲">
              <option value="europe-west1">europe-west1（比利时）</option>
              <option value="europe-west2">europe-west2（伦敦）</option>
              <option value="europe-west3">europe-west3（法兰克福）</option>
              <option value="europe-west4">europe-west4（荷兰）</option>
              <option value="europe-west8">europe-west8（米兰）</option>
              <option value="europe-west9">europe-west9（巴黎）</option>
              <option value="europe-southwest1">europe-southwest1（马德里）</option>
              <option value="europe-central2">europe-central2（华沙）</option>
              <option value="europe-north1">europe-north1（芬兰）</option>
            </optgroup>
            <optgroup label="亚太">
              <option value="asia-east1">asia-east1（台湾）</option>
              <option value="asia-east2">asia-east2（香港）</option>
              <option value="asia-northeast1">asia-northeast1（东京）</option>
              <option value="asia-northeast2">asia-northeast2（大阪）</option>
              <option value="asia-northeast3">asia-northeast3（首尔）</option>
              <option value="asia-south1">asia-south1（孟买）</option>
              <option value="asia-southeast1">asia-southeast1（新加坡）</option>
              <option value="asia-southeast2">asia-southeast2（雅加达）</option>
              <option value="australia-southeast1">australia-southeast1（悉尼）</option>
            </optgroup>
            <optgroup label="其它">
              <option value="northamerica-northeast1">northamerica-northeast1（蒙特利尔）</option>
              <option value="southamerica-east1">southamerica-east1（圣保罗）</option>
              <option value="me-central1">me-central1（多哈）</option>
              <option value="me-west1">me-west1（特拉维夫）</option>
            </optgroup>
          </select>
        </div>
        <div id="h_loc" class="helpbox">
          决定标准（Express）模式请求钉定到哪个区域，优先使用当前选中 Key 自有的 Project ID 拼装完整资源路径；若钉定路径失败会自动回退裸模型名重试一次。
        </div>
      </div>

      <!-- 输入图压缩 -->
      <div class="card p-5">
        <div class="flex items-center justify-between mb-3">
          <div class="text-sm font-semibold">输入图片压缩</div>
          <label class="switch"><input type="checkbox" id="img_compress_enabled"><span class="slider"></span></label>
        </div>
        <div class="grid grid-cols-3 gap-3">
          <div><div class="lbl mb-1">最长边(px)</div><input id="img_compress_max_dim" type="number" class="inp"></div>
          <div><div class="lbl mb-1">阈值(MB)</div><input id="img_compress_max_mb" type="number" step="0.1" class="inp"></div>
          <div><div class="lbl mb-1">JPEG质量</div><input id="img_compress_quality" type="number" class="inp"></div>
        </div>
      </div>

      <!-- 重试 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">重试与退避</div>
        <div class="grid grid-cols-2 gap-3">
          <div><div class="lbl mb-1">最大重试次数</div><input id="retry_max" type="number" class="inp"></div>
          <div><div class="lbl mb-1">退避间隔(秒)</div><input id="retry_backoff_seconds" type="number" step="0.5" class="inp"></div>
        </div>
      </div>

      <!-- 全局开关 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">流式、弱网续传 &amp; 调试开关</div>
        <div class="space-y-3">
          <div class="flex items-center justify-between"><span class="text-sm">假流式（fake streaming）<span class="helpq" onclick="hlp(this,'h_fks')">?</span></span><label class="switch"><input type="checkbox" id="fake_streaming"><span class="slider"></span></label></div>
          <div id="h_fks" class="helpbox">开启后在等待长思考模型（如 gemini-4-argon）完整生成期间，按下方间隔持续向客户端发送保活数据帧，防止前端超时断连。两条通道均生效。</div>
          <div class="flex items-center justify-between"><span class="text-sm">假流式心跳间隔(秒)</span><input id="fake_streaming_interval" type="number" step="0.5" class="inp" style="width:90px"></div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">弱网续传（服务端缓冲）<span class="helpq" onclick="hlp(this,'h_rsm')">?</span></span><label class="switch"><input type="checkbox" id="resumable_enabled"><span class="slider"></span></label></div>
            <div id="h_rsm" class="helpbox">反代将上游回复放入带序号缓冲区，前端断网重连可凭 <code>Last-Event-ID</code> 或 <code>?resume_from=</code> 续传；首包产出前若前端主动取消，会自动停止重试。</div>
          </div>
          <div class="flex items-center justify-between"><span class="text-sm">续传缓冲保留时长(秒)</span><input id="resumable_ttl_seconds" type="number" step="60" class="inp" style="width:90px"></div>
          <div class="flex items-center justify-between"><span class="text-sm">续传缓冲数量上限</span><input id="resumable_max_streams" type="number" step="10" class="inp" style="width:90px"></div>
          <div class="flex items-center justify-between"><span class="text-sm">多 Key 轮询（round-robin）</span><label class="switch"><input type="checkbox" id="roundrobin"><span class="slider"></span></label></div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">输出附加安全分<span class="helpq" onclick="hlp(this,'h_ss')">?</span></span><label class="switch"><input type="checkbox" id="safety_score"><span class="slider"></span></label></div>
            <div id="h_ss" class="helpbox">在回复末尾附加可折叠的安全分类评分块。</div>
          </div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">出站参数调试日志<span class="helpq" onclick="hlp(this,'h_dbg')">?</span></span><label class="switch"><input type="checkbox" id="debug_outbound"><span class="slider"></span></label></div>
            <div id="h_dbg" class="helpbox">在运行日志里打印实际发出的思考档位与采样参数。</div>
          </div>
          <div>
            <div class="flex items-center justify-between"><span class="text-sm">Cookie 通道额外诊断<span class="helpq" onclick="hlp(this,'h_ckd')">?</span></span><label class="switch"><input type="checkbox" id="cookie_debug"><span class="slider"></span></label></div>
            <div id="h_ckd" class="helpbox">仅对 Cookie 直连通道生效，打印出站 <code>generationConfig</code>。</div>
          </div>
        </div>
      </div>
    </div>

    <div class="flex items-center justify-end gap-3 mt-4">
      <span class="text-xs text-[#7d738e]">仅保存「② 全局基础设施设置」；① 区请用顶部按钮保存</span>
      <button class="btn px-5 py-2.5 text-sm" onclick="saveSettings()">保存全局设置</button>
    </div>
  </section>

  <!-- Logs -->
  <section id="view-logs" class="hidden">
    <div class="card overflow-hidden">
      <div class="px-4 py-2.5 border-b border-[#ece5f5] flex items-center gap-2 bg-[#faf8ff]">
        <div class="flex gap-1.5"><div class="w-3 h-3 rounded-full" style="background:#f68697"></div><div class="w-3 h-3 rounded-full" style="background:#f8c27a"></div><div class="w-3 h-3 rounded-full" style="background:#5fd3bc"></div></div>
        <span class="ml-2 text-xs text-[#8a7fa0] log">terminal · 实时监控</span>
      </div>
      <div id="logwin" class="log p-4 space-y-1.5 overflow-y-auto bg-[#fdfcff]" style="height:60vh"></div>
    </div>
  </section>
</div>
<div id="toast" class="toast"></div>

<script>
const $ = id => document.getElementById(id);
let CAPS = {}, chart = null, curAR = "";
let GLOBAL_SETTINGS = {}, OVERRIDES = {};
const PER_MODEL_KEYS = [
  'native_thinking_mode','thinking_g3_level','image_size','image_aspect_ratio',
  'default_temperature','default_top_p','default_max_tokens',
  'inject_system_instruction','inject_prefill','prefill_instruction',
  'image_system_instruction','inject_prefill_for_image','sampling_policy',
  'prefill_cot_guard','prefill_mode','prefill_suppress_thinking'
];
const COMMON_ARS = ["1:1","3:2","2:3","3:4","4:3","4:5","5:4","9:16","16:9","21:9","1:4","4:1","1:8","8:1","9:21"];

function toast(m){ const t=$('toast'); t.textContent=m; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2200); }
function fmt(n){ return (n||0).toLocaleString('en-US'); }
async function logout(){ try{ await fetch('/api/logout',{method:'POST'}); }catch(e){} location.href='/'; }

function switchTab(t){
  document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active', e.dataset.tab===t));
  ['overview','channel','params','logs'].forEach(v=>$('view-'+v).classList.toggle('hidden', v!==t));
}

/* ---------- Overview (Macaron Donut & Dynamic Token Bars) ---------- */
function renderChart(s,e,r){
  const ctx=$('donut').getContext('2d');
  const isEmpty = (s===0 && e===0 && r===0);
  const labels = isEmpty ? ['暂无请求'] : ['成功响应','拥堵重试','错误 / 拦截'];
  const data   = isEmpty ? [1] : [s, r, e];
  const colors = isEmpty ? ['#ede7f8'] : ['#5fd3bc', '#f8c27a', '#f68697'];

  const totalDone = s + e;
  const rateEl = $('health-rate');
  if(rateEl){
    if(totalDone === 0){
      rateEl.textContent = '待命';
      rateEl.className = 'pill pill-accent';
    } else {
      const pct = ((s / totalDone) * 100).toFixed(1).replace(/\.0$/, '');
      rateEl.textContent = `成功率 ${pct}%`;
      rateEl.className = pct >= 90 ? 'pill pill-mint' : 'pill pill-apricot';
    }
  }

  if(chart){
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].backgroundColor = colors;
    chart.update();
    return;
  }
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderWidth: 3,
        borderColor: '#ffffff',
        hoverOffset: 4
      }]
    },
    options: {
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#4a3f63',
          cornerRadius: 8,
          padding: 9
        }
      }
    }
  });
}

async function fetchStats(){
  try{
    const d=await (await fetch('/api/stats')).json();
    $('s-total').textContent=fmt(d.total);
    $('s-success').textContent=fmt(d.success);
    $('s-error').textContent=fmt(d.error);
    $('s-retries').textContent=fmt(d.retries);
    const pTk = d.prompt_tokens || 0;
    const cTk = d.completion_tokens || 0;
    const totTk = pTk + cTk;
    $('t-prompt').textContent=fmt(pTk);
    $('t-comp').textContent=fmt(cTk);
    $('t-total').textContent=fmt(totTk);
    $('bar-prompt').style.width = totTk > 0 ? Math.max(4, Math.round((pTk / totTk) * 100)) + '%' : '0%';
    $('bar-comp').style.width   = totTk > 0 ? Math.max(4, Math.round((cTk / totTk) * 100)) + '%' : '0%';
    $('uptime').textContent='已运行 '+(d.uptime/3600).toFixed(1)+' h';
    renderChart(d.success||0, d.error||0, d.retries||0);
  }catch(e){}
}

/* ---------- Channel ---------- */
async function updateMode(m){
  $('cookie-box').classList.toggle('hidden', m!=='web_proxy');
  $('mode-pill').textContent = m==='web_proxy' ? '通道 Cookie 直连' : '通道 Express API';
  await fetch('/api/settings/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
}
function parseCookies(str){
  str=(str||'').trim();
  if(str.startsWith('[')&&str.endsWith(']')){ try{ const a=JSON.parse(str); if(Array.isArray(a)) return a.map(c=>{const n=c.name||c.key,v=c.value; return (n&&v)?`${n}=${v}`:'';}).filter(Boolean).join('; ');}catch(e){} }
  return str;
}
async function saveCookie(){
  let ck=parseCookies($('cookie-input').value); let pid=$('project-input').value.trim();
  const m=pid.match(/[?&]project=([^&]+)/)||pid.match(/\/projects\/([^\/]+)/); if(m) pid=m[1];
  $('cookie-input').value=ck; $('project-input').value=pid;
  if(!pid){ toast('请填写 Project ID'); return; }
  try{
    const r=await fetch('/api/cookie',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cookie:ck,project_id:pid})});
    const d=await r.json(); toast(r.ok?(d.message||'已保存并激活'):('❌ '+(d.error||'保存失败')));
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function loadRuntime(){
  try{
    const s=await (await fetch('/api/settings/runtime')).json();
    const m=s.use_web_proxy?'web_proxy':'api_key';
    document.querySelector(`input[name=mode][value=${m}]`).checked=true;
    $('mode-pill').textContent = m==='web_proxy' ? '通道 Cookie 直连' : '通道 Express API';
    $('cookie-box').classList.toggle('hidden', m!=='web_proxy');
    const ci=$('cookie-input');
    if(ci){
      ci.value='';
      ci.placeholder = s.google_cookie_configured
        ? ('已配置：'+s.google_cookie+'　（留空则保持不变，需更新时粘贴新 Cookie）')
        : '粘贴完整 Cookie 头或 Cookie-Editor 导出内容';
    }
    if(s.google_project_id) $('project-input').value=s.google_project_id;
  }catch(e){}
}

/* ---------- Express Key 列表（多账号，热编辑，即添即探） ---------- */
let KEYS = [];
function keySrcPill(rec){
  if(rec.project_source==='auto') return '<span class="pill pill-mint" style="text-transform:none">来源 auto</span>';
  if(rec.project_source==='manual') return '<span class="pill pill-accent" style="text-transform:none">来源 manual</span>';
  return '<span class="pill pill-apricot" style="text-transform:none">未探测</span>';
}
function renderKeys(){
  const box=$('keys-list');
  box.innerHTML = KEYS.map((rec,i)=>`
    <div class="border border-[#ebe4f7] bg-[#fcfbff] rounded-xl p-3.5" data-i="${i}">
      <div class="flex items-center gap-2 mb-2">
        <span class="pill pill-accent" style="text-transform:none">#${i+1}</span>
        <input class="inp log k-key" style="flex:1" placeholder="粘贴 Express API Key（明文显示）" value="${(rec.key||'').replace(/"/g,'&quot;')}">
        <label class="flex items-center gap-1 text-xs text-[#6e6480] whitespace-nowrap"><input type="checkbox" class="k-enabled accent-[#8b7cf6]" ${rec.enabled===false?'':'checked'}>启用</label>
        <button class="btn-soft px-2.5 py-1 text-xs" onclick="probeOneKey(${i})">探测</button>
        <button class="px-2.5 py-1 text-xs rounded-lg border border-[#fbd2da] text-[#d64860] bg-[#fff5f7] hover:bg-[#ffecef]" onclick="removeKeyRow(${i})">删除</button>
      </div>
      <div class="flex items-center gap-2 flex-wrap">
        <span class="lbl whitespace-nowrap">Project ID</span>
        <input class="inp log k-project" style="flex:1;min-width:160px" placeholder="留空 = 点击「探测」自动获取" value="${(rec.project_id||'').replace(/"/g,'&quot;')}">
        ${keySrcPill(rec)}
        <span class="text-xs text-[#9a90ab]">${rec.detected_at?('探测于 '+new Date(rec.detected_at*1000).toLocaleString()):''}</span>
      </div>
      <div class="text-xs mt-1 ${(rec._msg_ok===false)?'text-[#d64860]':'text-[#239480]'}">${rec._msg||''}</div>
    </div>`).join('');
  $('keys-count').textContent = KEYS.filter(r=>r.key).length + ' 个';
  $('keys-empty').classList.toggle('hidden', KEYS.length>0);
}
function syncKeysFromDom(){
  [...document.querySelectorAll('#keys-list [data-i]')].forEach(row=>{
    const i=Number(row.dataset.i);
    if(!KEYS[i]) return;
    KEYS[i].key = row.querySelector('.k-key').value.trim();
    const pid = row.querySelector('.k-project').value.trim();
    if(pid !== (KEYS[i].project_id||'')){ KEYS[i].project_id = pid; KEYS[i].project_source = pid?'manual':''; }
    KEYS[i].enabled = row.querySelector('.k-enabled').checked;
  });
}
function collectKeys(){
  syncKeysFromDom();
  return KEYS.filter(r=>r.key);
}
function addKeyRow(){ syncKeysFromDom(); KEYS.push({key:'',project_id:'',project_source:'',enabled:true}); renderKeys(); }
function removeKeyRow(i){ syncKeysFromDom(); KEYS.splice(i,1); renderKeys(); }
async function loadExpressKeys(){
  try{
    const d=await (await fetch('/api/express-keys')).json();
    KEYS=(d.keys||[]).map(r=>Object.assign({},r));
    renderKeys();
  }catch(e){}
}
async function saveExpressKeys(){
  const payload=collectKeys();
  if(payload.length===0 && !confirm('将保存为“空 Key 列表”。\n保存后即使环境变量 VERTEX_EXPRESS_API_KEY 有值，重启也不会再导入。\n确定吗？')) { renderKeys(); return; }
  try{
    const r=await fetch('/api/express-keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:payload})});
    const d=await r.json();
    if(r.ok){ KEYS=(d.keys||[]).map(x=>Object.assign({},x)); renderKeys(); toast('已保存 '+KEYS.length+' 个 Key（已热生效）'); }
    else toast('❌ '+(d.error||'保存失败'));
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function probeOneKey(i){
  syncKeysFromDom();
  const rec=KEYS[i];
  if(!rec||!rec.key){ toast('请先在该行填入 Express API Key'); return; }
  const targetKey = rec.key;
  const emptyRows = KEYS.filter(r=>!r.key);
  const payload = KEYS.filter(r=>r.key);
  toast('正在自动保存并探测归属项目…');
  try{
    // 将当前界面上的全部非空 Key 一并传给后端先保存入库再探测，避免未保存的新 Key 被覆盖丢失
    const r=await fetch('/api/express-keys/probe',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:targetKey, force:true, keys:payload})
    });
    const d=await r.json();
    if(!r.ok){ toast('❌ '+(d.error||'探测失败')); return; }
    const res=(d.results||[])[0]||{};
    KEYS=(d.keys||[]).map(x=>Object.assign({},x)).concat(emptyRows);
    const hit=KEYS.find(x=>x.key===targetKey);
    if(hit){ hit._msg=res.message||''; hit._msg_ok=!!res.ok; }
    renderKeys();
    toast(res.ok?('✅ 已保存并探测成功：'+(res.project_id||'—')):('⚠️ Key 已保存，但探测未果：'+(res.message||'请手动填写项目')));
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function probeAllKeys(){
  syncKeysFromDom();
  const emptyRows = KEYS.filter(r=>!r.key);
  const payload = KEYS.filter(r=>r.key);
  if(payload.length===0){ toast('请先至少添加一个 Express API Key'); return; }
  toast('正在自动保存并探测全部 Key…');
  try{
    const r=await fetch('/api/express-keys/probe',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({all:true, force:true, keys:payload})
    });
    const d=await r.json();
    if(!r.ok){ toast('❌ '+(d.error||'探测失败')); return; }
    const results=d.results||[];
    KEYS=(d.keys||[]).map(x=>Object.assign({},x)).concat(emptyRows);
    results.forEach(res=>{ const hit=KEYS.find(x=>x.key===res.key); if(hit){ hit._msg=res.message||''; hit._msg_ok=!!res.ok; } });
    renderKeys();
    toast('已保存并完成探测：成功 '+results.filter(x=>x.ok).length+' / '+results.length);
  }catch(e){ toast('❌ 网络请求失败'); }
}

/* ---------- Params (① 按模型参数 vs ② 全局基础设施 严格分离) ---------- */
function setV(id,v){ const el=$(id); if(!el) return; if(el.type==='checkbox') el.checked=!!v; else el.value=(v===null||v===undefined)?'':v; }
async function loadParams(){
  try{
    const s=await (await fetch('/api/settings')).json();
    GLOBAL_SETTINGS=s;
    curAR = s.image_aspect_ratio || "";
    [
      'native_thinking_mode','thinking_g3_level','image_size',
      'default_temperature','default_top_p','default_max_tokens',
      'img_compress_max_dim','img_compress_max_mb','img_compress_quality',
      'retry_max','retry_backoff_seconds','fake_streaming_interval',
      'prefill_mode','prefill_instruction','inject_system_instruction','inject_prefill',
      'sampling_policy','express_location','resumable_ttl_seconds','resumable_max_streams'
    ].forEach(k=>setV(k,s[k]));
    [
      'img_compress_enabled','fake_streaming','roundrobin','safety_score',
      'cookie_debug','debug_outbound','prefill_suppress_thinking',
      'image_system_instruction','inject_prefill_for_image','prefill_cot_guard','resumable_enabled'
    ].forEach(k=>setV(k,s[k]));
    if((!s.native_thinking_mode || s.native_thinking_mode==='request')){
      if(s.hide_thoughts) setV('native_thinking_mode','off');
      else if(s.thinking_force_console) setV('native_thinking_mode','console');
    }
  }catch(e){}
  try{
    const c=await (await fetch('/api/capabilities')).json();
    CAPS=c.capabilities||{};
    OVERRIDES=c.overrides||{};
    $('model-sel').innerHTML='<option value="__global__">＊ 全局默认（对所有模型）</option>'
      + (c.models||[]).map(m=>{const star=OVERRIDES[m]?' ★':''; return `<option value="${m}">${m}${star}</option>`;}).join('');
    renderCaps();
  }catch(e){}
}

function applyModelParamFields(model){
  const isGlobal = (model==='__global__');
  const ov = isGlobal ? {} : (OVERRIDES[model] || {});
  PER_MODEL_KEYS.forEach(k=>{
    const v = (k in ov) ? ov[k] : GLOBAL_SETTINGS[k];
    if(k==='image_aspect_ratio'){ curAR = v || ""; }
    else setV(k, v);
  });
  const has = !isGlobal && !!OVERRIDES[model] && Object.keys(OVERRIDES[model]).length>0;
  $('ov-badge').classList.toggle('hidden', !has);
  $('btn-scope-save').textContent = isGlobal ? '💾 保存 ① 区为全局默认' : '💾 保存 ① 区为该模型专属';
  $('btn-scope-clear').classList.toggle('hidden', isGlobal);
  $('ov-hint').textContent = isGlobal
    ? '当前编辑的是「① 按模型参数」的全局默认值，对所有未设 ★ 专属的模型生效。'
    : (has ? '当前显示该模型的 ★ 专属参数；修改后点左侧按钮更新。'
           : '当前继承全局默认值；修改后点左侧按钮即可存为该模型专属。');
}

function collectSectionOnePatch(){
  return {
    native_thinking_mode:$('native_thinking_mode').value,
    thinking_g3_level:$('thinking_g3_level').value,
    image_size:$('image_size').value,
    image_aspect_ratio:$('image_aspect_ratio').value,
    default_temperature:numOrNull('default_temperature'),
    default_top_p:numOrNull('default_top_p'),
    default_max_tokens:numOrNull('default_max_tokens'),
    inject_system_instruction:$('inject_system_instruction').value,
    inject_prefill:$('inject_prefill').value,
    prefill_instruction:$('prefill_instruction').value,
    image_system_instruction:$('image_system_instruction').checked,
    inject_prefill_for_image:$('inject_prefill_for_image').checked,
    sampling_policy:$('sampling_policy').value,
    prefill_cot_guard:$('prefill_cot_guard').checked,
    prefill_mode:$('prefill_mode').value,
    prefill_suppress_thinking:$('prefill_suppress_thinking').checked,
  };
}

async function saveModelOverride(){
  const m=$('model-sel').value; if(!m) return;
  const patch = collectSectionOnePatch();
  if(m==='__global__'){
    try{
      const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
      if(r.ok){ GLOBAL_SETTINGS=await r.json(); toast('✅ 已将 ① 区参数保存为全局默认'); }
      else toast('❌ 保存失败');
    }catch(e){ toast('❌ 保存失败'); }
    return;
  }
  try{
    const r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    const d=await r.json();
    if(r.ok){ OVERRIDES[m]=d.override||patch; toast('✅ 已保存 '+m+' 的专属参数'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('❌ 保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

async function clearModelOverride(){
  const m=$('model-sel').value; if(!m || m==='__global__') return;
  if(!OVERRIDES[m]){ toast('该模型当前没有专属参数'); return; }
  try{
    const r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'DELETE'});
    if(r.ok){ delete OVERRIDES[m]; toast('已清除 '+m+' 专属参数，回退全局默认'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('❌ 清除失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

function refreshModelSelStars(){
  const sel=$('model-sel'); const cur=sel.value;
  [...sel.options].forEach(o=>{
    const base=o.value;
    if(base==='__global__'){
      o.textContent = '＊ 全局默认（对所有模型）';
    } else {
      o.textContent = base + (OVERRIDES[base] ? ' ★' : '');
    }
  });
  sel.value=cur;
}

function chip(t, accent){ return `<span class="pill${accent?' pill-accent':''}">${t}</span>`; }
function fillARFor(cap){
  const sel=$('image_aspect_ratio');
  const list = (cap && cap.is_image && cap.image_aspect_ratios.length) ? cap.image_aspect_ratios : COMMON_ARS;
  sel.innerHTML='<option value="">自动</option>' + list.map(a=>`<option value="${a}">${a}</option>`).join('');
  sel.value = (curAR && list.includes(curAR)) ? curAR : "";
}
function renderCaps(){
  const m=$('model-sel').value;
  if(m==='__global__'){
    applyModelParamFields(m);
    fillARFor(null);
    $('caps-summary').innerHTML='<span class="text-xs text-[#7d738e]">全局默认作用域：下方 ① 区的值对所有未设 ★ 专属的模型生效。切到具体模型可查看该模型的能力白名单与专属值。</span>';
    ['thinking_g3_level','image_size','image_aspect_ratio'].forEach(id=>{const el=$(id); if(el) el.disabled=false;});
    $('wrap-g3level').style.opacity='1';
    $('think-note').textContent='全局默认：具体下发时仍按各模型能力自动裁剪（例如 Pro 上 minimal 会就近取 low）。';
    $('image-note').textContent='全局默认：仅生图模型会用到分辨率与宽高比。';
    $('sampling-note').textContent='全局默认：已废弃采样参数的模型会自动剥离。';
    return;
  }
  const cap=CAPS[m]; if(!cap) return;
  applyModelParamFields(m);
  const th=cap.thinking||{};
  let sum=[chip('家族 '+cap.family, true)];
  if(th.kind==='level') sum.push(chip('思考档位 '+(th.levels||[]).join(' / ')));
  else sum.push(chip('无思考调节'));
  if(cap.is_image){
    sum.push(chip('分辨率 '+cap.image_sizes.join('/')));
    sum.push(chip('比例 '+cap.image_aspect_ratios.length+' 种'));
    sum.push(chip('不支持函数调用'));
  } else {
    sum.push(chip('采样 '+(cap.sampling_advice==='deprecated'?'已废弃·自动移除':cap.sampling_advice==='recommend_default'?'可调·建议默认':'可调')));
    sum.push(chip('支持函数调用'));
  }
  if(cap.supports_search) sum.push(chip('支持搜索'));
  $('caps-summary').innerHTML=sum.join('');

  const isLevel=th.kind==='level';
  $('thinking_g3_level').disabled=!isLevel;
  $('wrap-g3level').style.opacity=isLevel?'1':'.4';
  let tn='';
  if(cap.is_image) tn='生图模型思考档位（默认 high，可在上方调整）。';
  else if(isLevel) tn='该模型用思考档位' + ((th.levels && !th.levels.includes('minimal')) ? '（此模型最低 low，无 minimal，且无法关闭）' : '（无法完全关闭，最省为 minimal）') + '。';
  else tn='该模型不支持思考调节。';
  $('think-note').textContent=tn;

  const imgOn=cap.is_image;
  $('image_size').disabled=!imgOn; $('image_aspect_ratio').disabled=!imgOn;
  fillARFor(cap);
  $('image-note').textContent = imgOn
    ? ('该模型支持比例：' + cap.image_aspect_ratios.join('、') + '；分辨率 ' + cap.image_sizes.join('/') + '。')
    : '仅生图模型使用此项。';

  $('sampling-note').textContent = cap.is_image
    ? '生图模型会自动剥离采样参数。'
    : (cap.sampling_advice==='deprecated'
        ? '⚠️ 该模型已废弃 temperature/top_p/top_k，代理会自动移除以避免 400。'
        : cap.sampling_advice==='recommend_default'
          ? '官方建议保持采样默认值；留空即用模型默认。'
          : '该模型支持采样参数；留空即用模型默认。');
}
function numOrNull(id){ const v=$(id).value.trim(); return v===''?null:Number(v); }
function numOr(id,d){ const v=$(id).value.trim(); return v===''?d:Number(v); }
function hlp(el,id){const b=document.getElementById(id);const on=b.classList.toggle('show');el.classList.toggle('on',on);}

async function saveSettings(){
  // 严格只保存「② 全局基础设施设置」，绝不触碰「① 按模型参数」
  const patch={
    express_location:$('express_location').value,
    img_compress_enabled:$('img_compress_enabled').checked,
    img_compress_max_dim:numOr('img_compress_max_dim',1536),
    img_compress_max_mb:numOr('img_compress_max_mb',1.5),
    img_compress_quality:numOr('img_compress_quality',85),
    retry_max:numOr('retry_max',10),
    retry_backoff_seconds:numOr('retry_backoff_seconds',5),
    fake_streaming:$('fake_streaming').checked,
    fake_streaming_interval:numOr('fake_streaming_interval',1),
    resumable_enabled:$('resumable_enabled').checked,
    resumable_ttl_seconds:numOr('resumable_ttl_seconds',1800),
    resumable_max_streams:numOr('resumable_max_streams',200),
    roundrobin:$('roundrobin').checked,
    safety_score:$('safety_score').checked,
    debug_outbound:$('debug_outbound').checked,
    cookie_debug:$('cookie_debug').checked,
  };
  try{
    const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    if(r.ok){
      GLOBAL_SETTINGS=await r.json();
      toast('✅ 全局基础设施设置已保存');
    } else toast('❌ 保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

/* ---------- Logs (Macaron Soft Log Highlighting) ---------- */
const logwin=$('logwin'); let autoscroll=true;
logwin.addEventListener('scroll',()=>{ autoscroll = logwin.scrollHeight-logwin.scrollTop-logwin.clientHeight<50; });
function logLine(t){
  let c='#4d4460',bg='transparent',bl='3px solid transparent';
  if(t.includes('✅')||t.includes('🎉')){c='#1f8270';bg='#f1fcf9';bl='3px solid #5fd3bc';}
  else if(t.includes('⚠️')||t.includes('WARN')||t.includes('🔄')||t.includes('重试')||t.includes('🛑')){c='#b56c14';bg='#fffaf2';bl='3px solid #f8c27a';}
  else if(t.includes('❌')||t.includes('ERROR')||t.includes('🔴')){c='#c43c54';bg='#fff4f6';bl='3px solid #f68697';}
  else if(t.includes('💰')){c='#5e4bc4';bg='#f7f4ff';bl='3px solid #8b7cf6';}
  let s=t.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/(gemini-[a-zA-Z0-9.\-]+)/g,'<span style="color:#239480;font-weight:600">$1</span>');
  return `<div style="color:${c};background:${bg};border-left:${bl};padding:6px 10px;border-radius:6px;white-space:pre-wrap;word-break:break-word">${s}</div>`;
}
try{
  const es=new EventSource('/stream-logs');
  es.onmessage=e=>{ if(e.data.includes('keep-alive')) return; logwin.insertAdjacentHTML('beforeend',logLine(e.data)); if(autoscroll) logwin.scrollTop=logwin.scrollHeight; };
}catch(e){}

/* ---------- Project ID auto-extract ---------- */
$('project-input').addEventListener('input',e=>{ const v=e.target.value.trim(); const m=v.match(/[?&]project=([^&]+)/)||v.match(/\/projects\/([^\/]+)/); if(m) e.target.value=m[1]; });
$('image_aspect_ratio').addEventListener('change', e=>{ curAR=e.target.value; });

/* init */
fetchStats(); setInterval(fetchStats,3000); loadRuntime(); loadParams(); loadExpressKeys();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(request: Request):
    if _is_authed(request):
        return HTMLResponse(DASHBOARD_HTML)
    return HTMLResponse(LOGIN_HTML)


class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
async def login(body: LoginBody, request: Request):
    ip = request.client.host if request.client else "unknown"

    remain = _login_retry_after(ip)
    if remain > 0:
        return JSONResponse(status_code=429,
                            content={"error": f"尝试过于频繁，请 {remain} 秒后再试"})

    if config.API_KEY and secrets.compare_digest(body.password, config.API_KEY):
        _clear_login_failure(ip)
        resp = JSONResponse(content={"ok": True})
        resp.set_cookie(
            AUTH_COOKIE, _issue_session(),
            httponly=True, samesite="lax", max_age=SESSION_TTL_SECONDS, path="/",
            secure=(request.url.scheme == "https"
                    or request.headers.get("x-forwarded-proto", "") == "https"),
        )
        return resp

    _record_login_failure(ip)
    print(f"🔐 [登录失败] 来自 {ip} 的密码尝试失败。")
    return JSONResponse(status_code=401, content={"error": "密码错误"})


@app.post("/api/logout")
async def logout(request: Request):
    _revoke_session(request.cookies.get(AUTH_COOKIE, ""))
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(AUTH_COOKIE, path="/")
    return resp


@app.get("/api/stats")
async def get_stats_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=stats.get_json_stats())


# ==========================================
# 设置与通道控制
# ==========================================
class ModeSetting(BaseModel):
    mode: str


@app.get("/api/settings/runtime")
async def get_runtime_settings(_auth: bool = Depends(require_auth)):
    cookie = app_state.get_google_cookie()
    return JSONResponse(content={
        "use_web_proxy": app_state.is_web_proxy_enabled(),
        "google_cookie": mask_cookie(cookie),
        "google_cookie_configured": bool(cookie),
        "google_project_id": app_state.get_project_id(),
    })


@app.post("/api/settings/mode")
async def set_settings_mode(setting: ModeSetting, _auth: bool = Depends(require_auth)):
    app_state.enable_web_proxy(setting.mode == "web_proxy")
    return JSONResponse(content={"status": "success"})


@app.get("/api/settings")
async def get_settings_api(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_settings())


@app.post("/api/settings")
async def update_settings_api(request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    updated = app_state.update_settings(patch)
    return JSONResponse(content=updated)


@app.get("/api/capabilities")
async def get_capabilities_api(_auth: bool = Depends(require_auth)):
    try:
        models = await get_express_models()
    except Exception:
        models = []
    caps = {m: mc.capabilities_summary(m, app_state.get_effective_settings(m)) for m in models}
    overrides = app_state.get_model_overrides()
    return JSONResponse(content={"models": models, "capabilities": caps, "overrides": overrides})


# ==========================================
# 按模型参数覆盖（per-model overrides）
# ==========================================
@app.get("/api/model-overrides")
async def list_model_overrides(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=app_state.get_model_overrides())


@app.post("/api/model-overrides/{model_name}")
async def save_model_override(model_name: str, request: Request, _auth: bool = Depends(require_auth)):
    try:
        patch = await request.json()
    except Exception:
        patch = {}
    if not isinstance(patch, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象。"})
    saved = app_state.set_model_override(model_name, patch)
    return JSONResponse(content={"status": "success", "model": model_name, "override": saved})


@app.delete("/api/model-overrides/{model_name}")
async def delete_model_override(model_name: str, _auth: bool = Depends(require_auth)):
    ok = app_state.clear_model_override(model_name)
    return JSONResponse(content={"status": "success" if ok else "not_found", "model": model_name})


# ==========================================
# Express API Key 列表（多账号，热编辑，支持未存先探自动保存）
# ==========================================
class ExpressKeysBody(BaseModel):
    keys: list = []


class ProbeBody(BaseModel):
    key: str = ""
    all: bool = False
    force: bool = False
    keys: Optional[list] = None  # 若前端传入当前编辑中的 keys 列表，探测前先自动保存入库


class KeyProjectBody(BaseModel):
    key: str
    project_id: str = ""


def _express_keys_payload() -> dict:
    records = app_state.get_express_keys()
    return {
        "keys": records,
        "total": len(records),
        "enabled": sum(1 for r in records if r.get("enabled", True)),
        "roundrobin": bool(app_state.get_setting("roundrobin", config.ROUNDROBIN)),
        "env_import_done": True,
    }


@app.get("/api/express-keys")
async def list_express_keys(_auth: bool = Depends(require_auth)):
    return JSONResponse(content=_express_keys_payload())


@app.post("/api/express-keys")
async def save_express_keys(body: ExpressKeysBody, _auth: bool = Depends(require_auth)):
    if not isinstance(body.keys, list):
        return JSONResponse(status_code=400, content={"error": "keys 必须是数组。"})
    saved = app_state.set_express_keys(body.keys)
    express_key_manager.refresh_keys()
    pending = [r["key"] for r in saved if not r.get("project_id")]
    if pending:
        asyncio.create_task(express_key_probe.probe_all(keys=pending))
    payload = _express_keys_payload()
    payload["status"] = "success"
    return JSONResponse(content=payload)


@app.post("/api/express-keys/probe")
async def probe_express_keys(body: ProbeBody, _auth: bool = Depends(require_auth)):
    """手动触发探测：若请求附带前端当前 keys 列表，先同步保存再探测，彻底避免新加 Key 消失。"""
    if isinstance(body.keys, list) and len(body.keys) > 0:
        app_state.set_express_keys(body.keys)
        express_key_manager.refresh_keys()

    if body.all:
        results = await express_key_probe.probe_all(force=body.force)
    else:
        key = (body.key or "").strip()
        if not key:
            return JSONResponse(status_code=400, content={"error": "请提供 key，或设 all=true。"})
        results = [await express_key_probe.probe_and_store(key, force=body.force)]
        express_key_manager.refresh_keys()
    payload = _express_keys_payload()
    payload["results"] = results
    return JSONResponse(content=payload)


@app.post("/api/express-keys/project")
async def set_express_key_project(body: KeyProjectBody, _auth: bool = Depends(require_auth)):
    ok = app_state.update_express_key_project(
        body.key, body.project_id, source=("manual" if body.project_id.strip() else ""))
    if not ok:
        return JSONResponse(status_code=404, content={"error": "该 Key 不在列表中。"})
    payload = _express_keys_payload()
    payload["status"] = "success"
    return JSONResponse(content=payload)


class CookieSetting(BaseModel):
    cookie: str = ""
    project_id: str = ""


@app.post("/api/cookie")
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
            return JSONResponse(status_code=400, content={
                "error": "尚未配置 Cookie，请粘贴完整的 Google Cookie。"})
        message = "✅ 已保留原有 Cookie，仅更新 Project ID。"

    if project_id:
        app_state.set_project_id(project_id)
    return JSONResponse(content={"status": "success", "message": message})


@app.get("/stream-logs")
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


app.include_router(models_api.router)
app.include_router(chat_api.router)