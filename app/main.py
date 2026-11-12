import asyncio
import base64
import os
import secrets
import threading
import time
from typing import Optional
from fastapi import FastAPI, Depends, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

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



# 极简 128x128 429 娘 WebP 头像 (6.3 KB)，经 1:1 下对齐视网膜重采样与高压缩比处理，零内存/网络负担
AVATAR_WEBP_B64 = "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"
AVATAR_DATA_URI = f"data:image/webp;base64,{AVATAR_WEBP_B64}"
AVATAR_BYTES = base64.b64decode(AVATAR_WEBP_B64)

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>登录 · agentplatform2api</title>
<link rel="icon" type="image/webp" href="__AVATAR_URI__">
<script>
(function(){
  try{
    var t=localStorage.getItem('ap_theme');
    if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){
      document.documentElement.classList.add('dark');
    }
  }catch(e){}
})();
</script>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
body{background:radial-gradient(circle at 15% 20%,#f5eeff 0%,transparent 45%),radial-gradient(circle at 85% 80%,#e8fbf6 0%,transparent 45%),#fdfbf9;font-family:'Inter',system-ui,sans-serif;color:#3d354b;transition:background .25s,color .25s}
.card{background:rgba(255,255,255,.92);backdrop-filter:blur(12px);border:1px solid #ede7f6;border-radius:22px;box-shadow:0 14px 36px -10px rgba(139,124,246,.15)}
.inp{border:1px solid #e6dff2;border-radius:12px;padding:11px 14px;width:100%;outline:none;font-size:14px;background:#fcfaff;color:#3d354b;transition:.2s}
.inp:focus{border-color:#8b7cf6;background:#fff;box-shadow:0 0 0 3px rgba(139,124,246,.16)}
.btn{background:linear-gradient(135deg,#8b7cf6 0%,#f48fb1 100%);color:#fff;border-radius:12px;font-weight:600;padding:11px;width:100%;transition:.2s;box-shadow:0 4px 14px rgba(139,124,246,.28);font-size:14px}
.btn:hover{opacity:.95;transform:translateY(-1px)}
.theme-seg{display:inline-flex;align-items:center;padding:2px;border-radius:999px;border:1px solid #e6dff2;background:rgba(255,255,255,.85);font-size:12px}
.theme-seg span{padding:3px 10px;border-radius:999px;cursor:pointer;transition:.18s;color:#7d738e;user-select:none}
.theme-seg span.on{background:linear-gradient(135deg,#8b7cf6,#f48fb1);color:#fff;font-weight:600;box-shadow:0 2px 6px rgba(139,124,246,.25)}
.muted-txt{color:#7c728d}
html.dark body{background:radial-gradient(circle at 15% 20%,rgba(139,124,246,.12) 0%,transparent 45%),radial-gradient(circle at 85% 80%,rgba(95,211,188,.08) 0%,transparent 45%),#15131d;color:#e6e1f2}
html.dark .card{background:rgba(30,27,41,.94);border-color:#2f2a40;box-shadow:0 14px 36px -10px rgba(0,0,0,.5)}
html.dark .inp{background:#171421;border-color:#352f47;color:#e6e1f2}
html.dark .inp:focus{background:#1c1828;border-color:#9d8df8;box-shadow:0 0 0 3px rgba(157,141,248,.2)}
html.dark .theme-seg{background:#1e1b29;border-color:#352f47}
html.dark .theme-seg span{color:#9a90b0}
html.dark .theme-seg span.on{background:linear-gradient(135deg,#8b7cf6,#5fd3bc);color:#fff}
html.dark .muted-txt{color:#9a90b0}
</style>
</head><body class="min-h-screen flex items-center justify-center px-4 relative">
  <div class="theme-seg absolute top-5 right-5" title="切换日间马卡龙 / 夜间护眼模式">
    <span id="t-light" onclick="setTheme('light')">☀️ 日间</span>
    <span id="t-dark" onclick="setTheme('dark')">🌙 夜间</span>
  </div>
  <div class="w-full max-w-sm">
    <div class="flex flex-col items-center gap-2 mb-6 text-center">
      <img src="__AVATAR_URI__" width="46" height="46" alt="429" class="w-12 h-12 rounded-2xl object-cover shadow-md border border-[rgba(255,255,255,.15)] flex-shrink-0" />
      <div class="text-xl font-bold tracking-tight mt-1">agentplatform<span style="color:#8b7cf6">2api</span></div>
      <div class="text-xs muted-txt">OpenAI 兼容网关 · 双通道管理控制台</div>
    </div>
    <div class="card p-6">
      <label class="block text-xs font-medium muted-txt mb-2">访问密码（即环境变量 API_KEY）</label>
      <form id="f" onsubmit="return doLogin(event)">
        <input id="pw" type="password" class="inp mb-3.5" placeholder="请输入访问密码..." autofocus autocomplete="current-password">
        <button class="btn" type="submit">进入控制台</button>
      </form>
      <p id="err" class="text-xs text-[#e05670] mt-2.5 min-h-[16px] text-center"></p>
    </div>
  </div>
<script>
function syncThemeSeg(){
  var isDark=document.documentElement.classList.contains('dark');
  var l=document.getElementById('t-light'), d=document.getElementById('t-dark');
  if(l) l.classList.toggle('on', !isDark);
  if(d) d.classList.toggle('on', isDark);
}
function setTheme(mode){
  var wantDark=(mode==='dark');
  document.documentElement.classList.toggle('dark', wantDark);
  try{localStorage.setItem('ap_theme',wantDark?'dark':'light');}catch(e){}
  syncThemeSeg();
}
syncThemeSeg();
async function doLogin(e){
  e.preventDefault();
  var pw=document.getElementById('pw').value;
  var r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
  if(r.ok){ location.href='/'; } else {
    var d=await r.json().catch(function(){return {};});
    document.getElementById('err').textContent=d.error||'密码错误，请重试';
  }
  return false;
}
</script>
</body></html>
"""


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agentplatform2api · 控制台</title>
<link rel="icon" type="image/webp" href="__AVATAR_URI__">
<script>
(function(){
  try{
    var t=localStorage.getItem('ap_theme');
    if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){
      document.documentElement.classList.add('dark');
    }
  }catch(e){}
})();
</script>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --border:#ebe4f5; --border-soft:#f2edf9; --muted:#7c728e; --muted-light:#9b90ad;
    --fg:#383046; --bg:#fdfbf9; --card-bg:rgba(255,255,255,.95); --surface-soft:#faf8fe;
    --inp-bg:#ffffff; --inp-dis:#f6f4fa; --accent:#8b7cf6; --accent2:#f48fb1;
    --mint:#5fd3bc; --mint-text:#1f8f7b; --apricot:#f8c27a; --apricot-text:#c47619;
    --coral:#f68697; --coral-text:#d4445c; --sky:#74b9ff;
    --select-arrow:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%237c728e' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  }
  html.dark {
    --border:#2e2940; --border-soft:#262235; --muted:#9c92b3; --muted-light:#7a7091;
    --fg:#e8e3f4; --bg:#14121c; --card-bg:rgba(29,26,40,.95); --surface-soft:#181522;
    --inp-bg:#161320; --inp-dis:#201c2b; --accent:#9d8df8; --accent2:#f48fb1;
    --mint:#5fd3bc; --mint-text:#68dfc8; --apricot:#f8c27a; --apricot-text:#fad096;
    --coral:#f68697; --coral-text:#ff9bab; --sky:#74b9ff;
    --select-arrow:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%239c92b3' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  }
  * { -webkit-font-smoothing:antialiased; box-sizing:border-box; }
  body {
    background:radial-gradient(circle at 8% 12%,rgba(243,232,255,.65) 0%,transparent 40%),
               radial-gradient(circle at 92% 18%,rgba(230,252,245,.65) 0%,transparent 40%),
               radial-gradient(circle at 50% 90%,rgba(255,238,244,.55) 0%,transparent 45%),var(--bg);
    color:var(--fg); font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
    transition:background-color .25s ease,color .25s ease; overflow-x:hidden;
  }
  html.dark body {
    background:radial-gradient(circle at 8% 12%,rgba(139,124,246,.10) 0%,transparent 42%),
               radial-gradient(circle at 92% 18%,rgba(95,211,188,.07) 0%,transparent 42%),
               radial-gradient(circle at 50% 90%,rgba(244,143,177,.06) 0%,transparent 45%),var(--bg);
  }
  @keyframes fadeSlideUp { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
  .tab-pane { animation:fadeSlideUp .18s cubic-bezier(.16,1,.3,1); }
  .text-muted { color:var(--muted); }
  .text-muted-light { color:var(--muted-light); }
  .divider-line { border-color:var(--border-soft) !important; }
  .card {
    background:var(--card-bg); border:1px solid var(--border); border-radius:16px;
    box-shadow:0 4px 18px -4px rgba(139,124,246,.06);
    transition:background-color .2s ease,border-color .2s ease,transform .18s ease,box-shadow .18s ease;
  }
  html.dark .card { box-shadow:0 6px 22px -6px rgba(0,0,0,.45); }
  .stat-card { position:relative; overflow:hidden; }
  .stat-card:hover { transform:translateY(-2px); }
  .stat-total { background:linear-gradient(145deg,#ffffff 0%,#f6f2ff 100%); border-color:#e6defa; }
  .stat-ok    { background:linear-gradient(145deg,#ffffff 0%,#eefcf8 100%); border-color:#d5f5ec; }
  .stat-retry { background:linear-gradient(145deg,#ffffff 0%,#fff9ef 100%); border-color:#fae7c8; }
  .stat-err   { background:linear-gradient(145deg,#ffffff 0%,#fff2f5 100%); border-color:#fbd5dc; }
  html.dark .stat-total { background:linear-gradient(145deg,#1d1a28 0%,#251f3d 100%); border-color:#382e5c; }
  html.dark .stat-ok    { background:linear-gradient(145deg,#1d1a28 0%,#172d2a 100%); border-color:#234740; }
  html.dark .stat-retry { background:linear-gradient(145deg,#1d1a28 0%,#2d2419 100%); border-color:#4c3b24; }
  html.dark .stat-err   { background:linear-gradient(145deg,#1d1a28 0%,#311c24 100%); border-color:#522b38; }
  .stat-icon { width:32px; height:32px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
  .card-hd { display:flex; align-items:center; justify-content:space-between; gap:8px; padding-bottom:11px; margin-bottom:14px; border-bottom:1px solid var(--border-soft); }
  .card-title { font-size:14px; font-weight:600; color:var(--fg); display:flex; align-items:center; gap:7px; line-height:1.3; }
  .card-dot { width:7px; height:7px; border-radius:999px; flex-shrink:0; background:linear-gradient(135deg,var(--accent),var(--accent2)); }
  .lbl { font-size:12px; color:var(--muted); font-weight:500; display:flex; align-items:center; flex-wrap:wrap; gap:4px; }
  .val { font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .code-tag { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px; font-weight:400; color:var(--muted); background:var(--surface-soft); border:1px solid var(--border); padding:0.5px 5px; border-radius:5px; line-height:1.4; }
  .note-box { margin-top:10px; padding:8px 11px; border-radius:9px; background:var(--surface-soft); border:1px solid var(--border-soft); border-left:3px solid var(--accent); font-size:12px; line-height:1.6; color:var(--muted); }
  .note-box:empty { display:none; }
  .set-row { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:7px 0; }
  .set-row + .set-row { border-top:1px dashed var(--border-soft); }
  .set-lbl { font-size:13px; color:var(--fg); font-weight:500; display:flex; align-items:center; flex-wrap:wrap; gap:4px; }
  .btn { background:linear-gradient(135deg,#8b7cf6 0%,#a288f8 100%); color:#fff; border-radius:10px; font-weight:600; transition:.18s; box-shadow:0 3px 10px rgba(139,124,246,.24); display:inline-flex; align-items:center; justify-content:center; gap:5px; }
  .btn:hover { background:linear-gradient(135deg,#7a6ae8 0%,#9175f0 100%); transform:translateY(-1px); }
  .btn:disabled { opacity:.5; cursor:not-allowed; transform:none; }
  .btn-soft { background:#f7f4ff; color:#6b5bd6; border:1px solid #e3dbfa; border-radius:10px; font-weight:500; transition:.15s; display:inline-flex; align-items:center; justify-content:center; gap:4px; }
  .btn-soft:hover { background:#efeaff; border-color:#d0c2f7; }
  html.dark .btn-soft { background:#241f35; color:#bcb0ff; border-color:#3a3252; }
  html.dark .btn-soft:hover { background:#2d2742; border-color:#4d426e; }
  .btn-danger-soft { background:#fff5f7; color:var(--coral-text); border:1px solid #fbd2da; border-radius:8px; font-weight:500; transition:.15s; }
  .btn-danger-soft:hover { background:#ffe8ed; border-color:#f8b4c2; }
  html.dark .btn-danger-soft { background:rgba(246,134,151,.10); color:#ff9bab; border-color:#522b38; }
  html.dark .btn-danger-soft:hover { background:rgba(246,134,151,.18); border-color:#6b3748; }
  .inp { border:1px solid var(--border); border-radius:10px; background:var(--inp-bg); font-size:13.5px; padding:7.5px 11px; width:100%; outline:none; transition:.15s; color:var(--fg); }
  select.inp { appearance:none; -webkit-appearance:none; -moz-appearance:none; background-image:var(--select-arrow); background-repeat:no-repeat; background-position:right 10px center; background-size:14px; padding-right:30px; cursor:pointer; }
  .inp:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(139,124,246,.15); }
  .inp:disabled { background:var(--inp-dis); color:var(--muted-light); cursor:not-allowed; opacity:.75; }
  .inp-sm { width:92px; text-align:center; padding:5px 8px; font-size:13px; }
  /* 1. 将 repeat(4,1fr) 改为 repeat(4,minmax(0,1fr))，锁死四等分不越界 */
  .main-nav { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; padding:5px; border-radius:14px; background:var(--card-bg); border:1px solid var(--border); box-shadow:0 3px 14px -4px rgba(139,124,246,.07); }
  .tab { display:flex; align-items:center; justify-content:center; gap:6px; min-width:0; padding:9px 8px; font-size:13.5px; font-weight:600; color:var(--muted); border-radius:10px; cursor:pointer; white-space:nowrap; transition:all .18s ease; user-select:none; }
  .tab:hover:not(.active) { color:var(--fg); background:rgba(139,124,246,.06); }
  .tab.active { color:#fff; background:linear-gradient(135deg,#8b7cf6 0%,#a288f8 100%); box-shadow:0 3px 10px rgba(139,124,246,.26); }
  html.dark .tab.active { background:linear-gradient(135deg,#8b7cf6 0%,#7362e6 100%); box-shadow:0 3px 12px rgba(0,0,0,.4); }
  .pill { display:inline-flex; align-items:center; gap:5px; font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--border); background:var(--inp-bg); color:var(--muted); transition:.15s; white-space:nowrap; }
  .pill-accent { border-color:#e2d9fc; background:linear-gradient(135deg,rgba(139,124,246,.12),rgba(244,143,177,.12)); color:#6b5bd6; font-weight:600; }
  html.dark .pill-accent { border-color:#3d335c; background:linear-gradient(135deg,rgba(139,124,246,.22),rgba(244,143,177,.16)); color:#c1b6ff; }
  .pill-mint { border-color:#cdeee5; background:#eefcf8; color:var(--mint-text); font-weight:600; }
  html.dark .pill-mint { border-color:#244a42; background:rgba(95,211,188,.14); color:#6de0c9; }
  .pill-apricot { border-color:#f9e2be; background:#fff8ee; color:var(--apricot-text); font-weight:600; }
  html.dark .pill-apricot { border-color:#4d3b24; background:rgba(248,194,122,.14); color:#fad096; }
  .theme-seg { display:inline-flex; align-items:center; padding:2px; border-radius:999px; border:1px solid var(--border); background:var(--inp-bg); font-size:12px; }
  .theme-seg span { padding:2.5px 9px; border-radius:999px; cursor:pointer; transition:.18s; color:var(--muted); user-select:none; white-space:nowrap; }
  .theme-seg span.on { background:linear-gradient(135deg,#8b7cf6,#f48fb1); color:#fff; font-weight:600; box-shadow:0 2px 6px rgba(139,124,246,.24); }
  html.dark .theme-seg span.on { background:linear-gradient(135deg,#8b7cf6,#5fd3bc); color:#fff; }

  /* 2. 二级导航同样改为 minmax(0,1fr) 防撑破 */
  .subnav-seg { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; width:100%; padding:4px; border-radius:13px; border:1px solid var(--border); background:var(--surface-soft); }
  .subnav-seg button { display:flex; align-items:center; justify-content:center; gap:6px; min-width:0; width:100%; padding:8.5px 10px; border-radius:9px; border:1px solid transparent; background:transparent; font-size:13px; font-weight:600; color:var(--muted); cursor:pointer; transition:all .18s ease; user-select:none; white-space:nowrap; }

  /* 3. 新增手机窄屏（<640px）专属适配：隐藏主导航冗余Emoji图标或缩小内边距，并限制右侧下拉框宽度 */
  @media (max-width: 639px) {
    .main-nav { gap:4px; padding:4px; }
    .tab { padding:8px 2px; font-size:12.5px; gap:0; }
    .tab > span:first-child { display:none; } /* 手机端仅展示纯文字，4个标签一行严丝合缝 */
    .subnav-seg button { font-size:12px; padding:8px 4px; gap:4px; }
    .set-row { gap:8px; }
    .set-lbl { flex:1; min-width:0; font-size:12.5px; }
    .set-row > select.inp { width:auto !important; max-width:46% !important; min-width:118px; flex-shrink:0; font-size:12.5px; padding-left:8px; }
  }
  .subnav-seg button:hover:not(.active) { color:var(--fg); }
  .subnav-seg button.active { background:var(--inp-bg); color:var(--accent); border-color:var(--border); box-shadow:0 2px 8px rgba(139,124,246,.10); }
  html.dark .subnav-seg button.active { background:#241f35; color:#c4b8ff; border-color:#3b3354; box-shadow:0 2px 10px rgba(0,0,0,.35); }
  .chan-card { border:1.5px solid var(--border); border-radius:14px; padding:14px 16px; cursor:pointer; transition:all .2s ease; position:relative; background:var(--surface-soft); }
  .chan-card:hover { border-color:#d4c8f8; }
  html.dark .chan-card:hover { border-color:#453c63; }
  .chan-card.active { border-color:var(--accent); background:linear-gradient(135deg,rgba(139,124,246,.08) 0%,rgba(95,211,188,.06) 100%); box-shadow:0 4px 16px -4px rgba(139,124,246,.18); }
  html.dark .chan-card.active { background:linear-gradient(135deg,rgba(139,124,246,.18) 0%,rgba(95,211,188,.10) 100%); box-shadow:0 4px 18px -4px rgba(0,0,0,.5); }
  .chan-radio { width:18px; height:18px; border-radius:50%; border:2px solid var(--muted-light); display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; transition:.18s; }
  .chan-card.active .chan-radio { border-color:var(--accent); background:var(--accent); box-shadow:0 0 0 3px rgba(139,124,246,.18); }
  .chan-radio::after { content:""; width:6px; height:6px; border-radius:50%; background:#fff; opacity:0; transform:scale(.5); transition:.18s; }
  .chan-card.active .chan-radio::after { opacity:1; transform:scale(1); }
  details.adv-box { border:1px solid var(--border-soft); border-radius:11px; background:var(--surface-soft); padding:8px 12px; transition:.2s; }
  details.adv-box[open] { border-color:var(--border); }
  details.adv-box > summary { list-style:none; cursor:pointer; user-select:none; display:flex; align-items:center; justify-content:space-between; font-size:12.5px; font-weight:600; color:var(--muted); padding:2px 0; }
  details.adv-box > summary::-webkit-details-marker { display:none; }
  details.adv-box > summary:hover { color:var(--fg); }
  .adv-arrow { display:inline-block; transition:transform .2s ease; font-size:11px; color:var(--accent); }
  details.adv-box[open] .adv-arrow { transform:rotate(90deg); }
  .log { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; word-break:break-all; }
  .log-item { cursor:pointer; transition:filter .12s ease; }
  .log-item:hover { filter:brightness(0.96); }
  html.dark .log-item:hover { filter:brightness(1.15); }
  .log-ts { display:inline-block; font-size:11px; opacity:.72; margin-right:6px; padding:0 5px; border-radius:4px; background:rgba(139,124,246,.10); font-variant-numeric:tabular-nums; }
  ::-webkit-scrollbar{width:7px;height:7px}
  ::-webkit-scrollbar-thumb{background:#dfd8ec;border-radius:8px}
  ::-webkit-scrollbar-thumb:hover{background:#c9bfe0}
  html.dark ::-webkit-scrollbar-thumb{background:#352f47}
  html.dark ::-webkit-scrollbar-thumb:hover{background:#473f5e}
  .toast { position:fixed; bottom:24px; left:50%; transform:translateX(-50%); background:linear-gradient(135deg,#4a3f63,#635380); color:#fff; padding:10px 20px; border-radius:12px; font-size:13.5px; font-weight:500; opacity:0; transition:.25s; pointer-events:none; z-index:50; max-width:92vw; text-align:center; box-shadow:0 8px 24px rgba(74,63,99,.25); }
  .toast.show { opacity:1; }
  .switch { position:relative; width:40px; height:22px; display:inline-block; flex-shrink:0; }
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#e4ddf0;border-radius:999px;transition:.2s;cursor:pointer}
  html.dark .slider{background:#342e46}
  .slider:before{content:"";position:absolute;height:16px;width:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s;box-shadow:0 1px 3px rgba(0,0,0,.15)}
  input:checked + .slider{background:linear-gradient(135deg,#8b7cf6,#5fd3bc)}
  input:checked + .slider:before{transform:translateX(18px)}
  .hero { background:linear-gradient(135deg,rgba(255,255,255,.96) 0%,rgba(249,245,255,.94) 55%,rgba(239,252,248,.94) 100%); border:1px solid var(--border); border-radius:18px; box-shadow:0 6px 20px -6px rgba(139,124,246,.09); }
  html.dark .hero { background:linear-gradient(135deg,rgba(32,28,44,.96) 0%,rgba(37,31,54,.94) 55%,rgba(26,38,41,.94) 100%); box-shadow:0 8px 24px -6px rgba(0,0,0,.45); }
  .helpq { display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; border-radius:50%; border:1px solid #d5cce6; color:#9286a8; font-size:10px; font-weight:600; line-height:1; cursor:pointer; user-select:none; vertical-align:middle; margin-left:3px; transition:.15s; background:#faf8ff; flex-shrink:0; }
  html.dark .helpq { background:#252036; border-color:#3d3554; color:#a89ec2; }
  .helpq:hover { border-color:var(--accent); color:var(--accent); background:#f2edff; }
  html.dark .helpq:hover { background:#2f2846; }
  .helpq.on { border-color:var(--accent); color:#fff; background:var(--accent); }
  .helpbox { display:none; margin-top:7px; margin-bottom:4px; padding:10px 12px; border-radius:10px; background:var(--surface-soft); border:1px solid var(--border); font-size:12px; line-height:1.7; color:var(--muted); animation:fadeSlideUp .16s ease-out; }
  .helpbox.show { display:block; }
  .helpbox, .helpbox code { overflow-wrap:anywhere; word-break:break-word; }
  .helpbox code { display:inline; white-space:normal; background:#ede6fa; color:#5a48b8; padding:1px 5px; border-radius:5px; font-size:11px; }
  html.dark .helpbox code { background:#2b2442; color:#c4b8ff; }
  .helpbox b { color:var(--fg); }
  .key-row { background:var(--surface-soft); border:1px solid var(--border); transition:.15s; }
  .key-row:hover { border-color:#d5c8f7; }
  html.dark .key-row:hover { border-color:#433a60; }
  #logwin { background:#f8f6fc; box-shadow:inset 0 2px 6px rgba(139,124,246,.05); }
  html.dark #logwin { background:#110f17 !important; box-shadow:inset 0 2px 8px rgba(0,0,0,.35); }
  .log-header { background:var(--surface-soft); border-color:var(--border); }
</style>
</head>
<body class="min-h-screen">
<div class="max-w-5xl mx-auto px-4 sm:px-6 md:px-8 py-5 sm:py-6">

  <!-- Header -->
  <header class="hero px-4 sm:px-5 py-3.5 mb-5 flex items-center justify-between flex-wrap gap-3">
    <div class="flex items-center gap-3">
      <img src="__AVATAR_URI__" width="40" height="40" alt="429" class="w-10 h-10 rounded-xl object-cover shadow-sm border border-[var(--border)] flex-shrink-0" />
      <div>
        <h1 class="text-lg sm:text-xl font-bold tracking-tight leading-none">agentplatform<span style="color:var(--accent)">2api</span></h1>
        <p class="text-xs text-muted mt-1">OpenAI 兼容代理 · Gemini Agent Platform 双通道</p>
      </div>
    </div>
    <div class="flex items-center gap-1.5 sm:gap-2 flex-wrap">
      <span class="pill pill-accent" id="mode-pill">通道 —</span>
      <span class="pill pill-mint"><span class="relative flex h-2 w-2"><span class="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style="background:#5fd3bc"></span><span class="relative inline-flex rounded-full h-2 w-2" style="background:#2eb89e"></span></span><span id="uptime">运行中</span></span>
      <button onclick="copyEndpoint()" class="pill hover:opacity-85" style="cursor:pointer" title="点击复制 OpenAI 兼容接口地址（/v1）">🔗 复制 /v1</button>
      <div class="theme-seg" title="切换日间马卡龙 / 夜间护眼模式">
        <span id="t-light" onclick="setTheme('light')">☀️ 日间</span>
        <span id="t-dark" onclick="setTheme('dark')">🌙 夜间</span>
      </div>
      <button onclick="logout()" class="pill hover:opacity-85" style="cursor:pointer" title="退出登录">退出</button>
    </div>
  </header>

  <!-- Primary Navigation -->
  <nav class="main-nav mb-5">
    <div class="tab active" data-tab="overview" onclick="switchTab('overview')"><span>📊</span><span>数据概览</span></div>
    <div class="tab" data-tab="channel" onclick="switchTab('channel')"><span>🔑</span><span>通道与凭证</span></div>
    <div class="tab" data-tab="params" onclick="switchTab('params')"><span>⚙️</span><span>模型参数</span></div>
    <div class="tab" data-tab="logs" onclick="switchTab('logs')"><span>🖥️</span><span>运行日志</span></div>
  </nav>

  <!-- Tab 1: Overview -->
  <section id="view-overview" class="tab-pane">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-3.5 mb-4">
      <div class="card stat-card stat-total p-4">
        <div class="flex items-center justify-between"><span class="lbl">总请求</span><span class="stat-icon" style="background:rgba(139,124,246,.14);color:var(--accent)">📨</span></div>
        <div id="s-total" class="val text-2xl font-bold mt-2" style="color:var(--accent)">0</div>
      </div>
      <div class="card stat-card stat-ok p-4">
        <div class="flex items-center justify-between"><span class="lbl">成功响应</span><span class="stat-icon" style="background:rgba(95,211,188,.16);color:var(--mint-text)">✨</span></div>
        <div id="s-success" class="val text-2xl font-bold mt-2" style="color:var(--mint-text)">0</div>
      </div>
      <div class="card stat-card stat-retry p-4">
        <div class="flex items-center justify-between"><span class="lbl">拥堵重试</span><span class="stat-icon" style="background:rgba(248,194,122,.18);color:var(--apricot-text)">🔄</span></div>
        <div id="s-retries" class="val text-2xl font-bold mt-2" style="color:var(--apricot-text)">0</div>
      </div>
      <div class="card stat-card stat-err p-4">
        <div class="flex items-center justify-between"><span class="lbl">错误 / 拦截</span><span class="stat-icon" style="background:rgba(246,134,151,.16);color:var(--coral-text)">🛡️</span></div>
        <div id="s-error" class="val text-2xl font-bold mt-2" style="color:var(--coral-text)">0</div>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5">
      <div class="card p-5 flex flex-col items-center justify-between">
        <div class="flex items-center justify-between w-full mb-1">
          <span class="card-title"><span class="card-dot"></span>服务健康度</span>
          <span id="health-rate" class="pill pill-mint" style="font-size:11px;padding:2px 8px">待命</span>
        </div>
        <div class="relative flex items-center justify-center my-2" style="width:144px;height:144px">
          <canvas id="donut"></canvas>
          <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none">
            <span id="donut-pct" class="val font-bold text-xl leading-none" style="color:var(--fg)">100%</span>
            <span id="donut-sub" class="text-[11px] text-muted mt-1">待命</span>
          </div>
        </div>
        <div class="flex items-center justify-center gap-4 text-xs text-muted mt-2 w-full">
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#5fd3bc"></span>成功</span>
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#f8c27a"></span>重试</span>
          <span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full" style="background:#f68697"></span>错误</span>
        </div>
      </div>

      <div class="card p-5 md:col-span-2 flex flex-col justify-between">
        <div class="flex items-center justify-between flex-wrap gap-2 mb-4">
          <span class="card-title"><span class="card-dot"></span>Token 算力消耗</span>
          <span class="text-xs text-muted">双通道实时累计（含输入与思考输出）</span>
        </div>
        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-sm mb-1.5"><span class="lbl" style="font-size:13px">Prompt（输入提示词）</span><span id="t-prompt" class="val font-semibold" style="color:var(--accent)">0</span></div>
            <div class="w-full rounded-full h-2.5 overflow-hidden" style="background:rgba(139,124,246,.12)"><div id="bar-prompt" class="h-2.5 rounded-full transition-all duration-300" style="width:0%;background:linear-gradient(90deg,#8b7cf6,#74b9ff)"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-sm mb-1.5"><span class="lbl" style="font-size:13px">Completion（思考与生成）</span><span id="t-comp" class="val font-semibold" style="color:var(--mint-text)">0</span></div>
            <div class="w-full rounded-full h-2.5 overflow-hidden" style="background:rgba(139,124,246,.12)"><div id="bar-comp" class="h-2.5 rounded-full transition-all duration-300" style="width:0%;background:linear-gradient(90deg,#5fd3bc,#f48fb1)"></div></div>
          </div>
          <div class="pt-3.5 border-t divider-line flex justify-between items-center">
            <span class="lbl">累计总消耗 Tokens</span>
            <span id="t-total" class="val text-xl font-bold" style="color:var(--accent)">0</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Tab 2: Channel -->
  <section id="view-channel" class="tab-pane hidden space-y-4">
    <div class="card p-5">
      <div class="card-hd">
        <div class="card-title"><span class="card-dot"></span>优先主通道选择</div>
        <span class="text-xs text-muted">开启「跨通道兜底」时另一通道自动作为热备</span>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label class="chan-card flex items-start gap-3" id="opt-api">
          <input type="radio" name="mode" value="api_key" class="hidden" onchange="updateMode('api_key')">
          <span class="chan-radio"></span>
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between gap-2 flex-wrap">
              <span class="font-semibold text-sm">Express API Key（标准模式）</span>
              <span id="badge-api" class="pill pill-mint hidden" style="font-size:11px;padding:1px 8px">当前主用</span>
            </div>
            <div class="text-xs text-muted mt-1 leading-relaxed">使用官方 google-genai SDK + 多账号 Key 池轮询调用</div>
          </div>
        </label>
        <label class="chan-card flex items-start gap-3" id="opt-web">
          <input type="radio" name="mode" value="web_proxy" class="hidden" onchange="updateMode('web_proxy')">
          <span class="chan-radio"></span>
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between gap-2 flex-wrap">
              <span class="font-semibold text-sm">Cookie 直连反代（Studio 模式）</span>
              <span id="badge-web" class="pill pill-mint hidden" style="font-size:11px;padding:1px 8px">当前主用</span>
            </div>
            <div class="text-xs text-muted mt-1 leading-relaxed">使用 Google Cloud 控制台 Cookie 直连私有 batchGraphql</div>
          </div>
        </label>
      </div>
    </div>

    <div class="card p-5">
      <div class="card-hd flex-wrap">
        <div>
          <div class="card-title"><span class="card-dot"></span>Express API Key 凭证池<span class="helpq" onclick="hlp(this,'h_keys')">?</span></div>
          <div class="text-xs text-muted mt-0.5">支持多账号轮询与自动故障隔离，输入移开焦点即自动保存</div>
        </div>
        <div class="flex items-center gap-2 flex-wrap">
          <span id="keys-count" class="pill pill-accent">0 个</span>
          <button class="btn-soft px-3 py-1.5 text-xs sm:text-sm" onclick="probeAllKeys()">🔎 探测全部项目</button>
          <button class="btn px-3.5 py-1.5 text-xs sm:text-sm" onclick="addKeyRow()">＋ 添加 Key</button>
        </div>
      </div>
      <div id="h_keys" class="helpbox">
        <b>自动保存 &amp; 即添即探</b>：点「＋ 添加 Key」粘贴新 Key 后，移开输入框（或按回车）即<b>自动保存并后台探测归属 Project ID</b>；勾选启用/禁用、修改 Project ID 或点「删除」也会立即落盘生效。<br>
        <b>多 Key 轮换与隔离</b>：每个 Key 绑定各自的 GCP Project ID；遇到 429/500 自动切换下一个健康 Key，遇到 401/欠费自动隔离并在运行日志标红提醒。
      </div>
      <div id="keys-list" class="space-y-2.5"></div>
      <div id="keys-empty" class="note-box hidden">当前没有任何 Express Key。点右上角「＋ 添加 Key」粘贴即可自动保存并探测项目。</div>
    </div>

    <div id="cookie-box" class="card p-5">
      <div class="card-hd">
        <div>
          <div class="card-title"><span class="card-dot"></span>Studio Cookie 直连凭证<span class="helpq" onclick="hlp(this,'h_ck')">?</span></div>
          <div class="text-xs text-muted mt-0.5">用于 Studio 直连模式，或作为 Express Key 池的跨通道备用凭证</div>
        </div>
      </div>
      <div id="h_ck" class="helpbox">
        <b>Cookie 维护说明</b>：Cookie 通常可维持数周甚至更久。当日志出现 <code>aiplatform.endpoints.predict denied</code> 权限错误时，90% 以上是因为浏览器会话已过期，重新从已登录的 <code>console.cloud.google.com</code> 复制完整 Request Cookie（或用 Cookie-Editor 导出 JSON）粘贴保存即可。
      </div>
      <div class="grid grid-cols-1 gap-3">
        <div>
          <div class="lbl mb-1.5">Google Cookie <span class="code-tag">含 SAPISID / SID</span></div>
          <textarea id="cookie-input" rows="3" class="inp log" placeholder="粘贴 console.cloud.google.com 的完整 Cookie（支持 Header 字符串或 Cookie-Editor JSON 自动解析）"></textarea>
        </div>
        <div>
          <div class="lbl mb-1.5">Google Cloud Project ID <span class="code-tag">支持粘贴整条控制台 URL 自动提取</span></div>
          <div class="flex flex-col sm:flex-row gap-2.5">
            <input id="project-input" class="inp flex-1" placeholder="例如：my-gcp-project-123456（或粘贴含 ?project=xxx 的链接）">
            <button class="btn px-5 py-2 text-sm whitespace-nowrap" onclick="saveCookie()">💾 保存并校验 Cookie</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Tab 3: Params -->
  <section id="view-params" class="tab-pane hidden">
    <div class="subnav-seg mb-4">
      <button id="tab-btn-params-model" type="button" class="active" onclick="switchParamsSubTab('model')">
        <span>🤖 按模型参数配置</span>
      </button>
      <button id="tab-btn-params-global" type="button" onclick="switchParamsSubTab('global')">
        <span>🌐 全局基础设施设置</span>
      </button>
    </div>

    <!-- 子面板 1：按模型参数配置 -->
    <div id="pane-params-model">
      <div class="card p-4 sm:p-5 mb-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div class="flex flex-col sm:flex-row sm:items-center gap-2.5 flex-1">
            <span class="lbl whitespace-nowrap font-semibold" style="color:var(--fg)">配置作用域</span>
            <select id="model-sel" class="inp sm:w-72" onchange="renderCaps()">
              <option value="__global__">＊ 全局默认（对所有模型）</option>
            </select>
            <span id="ov-badge" class="pill pill-accent hidden">★ 已设专属参数</span>
          </div>
          <div class="flex items-center gap-2 flex-wrap">
            <button id="btn-scope-clear" class="btn-soft px-3.5 py-2 text-xs sm:text-sm hidden" onclick="clearModelOverride()">↩️ 恢复全局默认</button>
            <button id="btn-scope-save" class="btn px-4 py-2 text-xs sm:text-sm" onclick="saveModelOverride()">💾 保存为全局默认</button>
          </div>
        </div>
        <div class="mt-3 pt-3 border-t divider-line flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <span id="ov-hint" class="text-xs text-muted"></span>
          <div id="caps-summary" class="flex flex-wrap gap-1.5"></div>
        </div>
      </div>

      <div class="grid md:grid-cols-2 gap-4 items-start">
        <!-- 左栏：思考 + 预填充与抗截断 -->
        <div class="space-y-4">
          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>思考强度与控制</div>
              <span class="code-tag">thinking_config</span>
            </div>
            <div class="space-y-3.5">
              <div id="wrap-g3level">
                <div class="lbl mb-1.5">思考档位 <span class="code-tag">thinking_level</span></div>
                <select id="thinking_g3_level" class="inp">
                  <option value="">自动（按模型默认档位）</option>
                  <option value="minimal">minimal（最低思考 · 3.6及更早）</option>
                  <option value="low">low（轻量思考）</option>
                  <option value="medium">medium（均衡思考）</option>
                  <option value="high">high（深度思考）</option>
                  <option value="xhigh">xhigh（超高思考 · Gemini 4）</option>
                </select>
              </div>
              <div>
                <div class="lbl mb-1.5">原生思考控制 <span class="code-tag">native_thinking_mode</span><span class="helpq" onclick="hlp(this,'h_ntm')">?</span></div>
                <select id="native_thinking_mode" class="inp">
                  <option value="request">跟随客户端请求（默认）</option>
                  <option value="off">关闭原生思考（酒馆角色扮演推荐）</option>
                  <option value="console">强制使用上方档位（忽略客户端）</option>
                </select>
                <div id="h_ntm" class="helpbox">
                  🎭 <b>酒馆（SillyTavern）角色扮演推荐选「关闭原生思考」</b>：自动忽略前端传来的 <code>reasoning_effort=xhigh</code>，将思考压到该模型最低合法档并剥离原生思考块，避免挤占正文额度，让预设自带的思维链接管。
                </div>
              </div>
            </div>
            <div id="think-note" class="note-box"></div>
          </div>

          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>预填充兼容与抗截断</div>
              <span class="code-tag">Prefill &amp; Sandbox</span>
            </div>
            <div>
              <div class="set-row">
                <span class="set-lbl">🛡️ Function Call 抗截断传输 <span class="code-tag">compose_reply</span><span class="helpq" onclick="hlp(this,'h_sbox')">?</span></span>
                <label class="switch"><input type="checkbox" id="synthetic_sandbox_enabled"><span class="slider"></span></label>
              </div>
              <div id="h_sbox" class="helpbox">
                <b>原生 Function Call 抗截断传输</b>：<br>
                • 使用固定工具 <code>compose_reply</code> 与 <code>mode: ANY</code> 受限解码包裹回复正文，保持 Vertex 隐式前缀缓存命中；<br>
                • 自动适配普通回复与预填充续写，网关层在真流式、假流式、非流式下透明拆包为标准文本（响应 ID 带 <code>bad_woman</code> 标识），生图与联网搜索请求自动旁路。
              </div>

              <div class="set-row">
                <span class="set-lbl">预填充兼容模式<span class="helpq" onclick="hlp(this,'h_pfm')">?</span></span>
                <select id="prefill_mode" class="inp" style="width:160px">
                  <option value="smart">智能并入（默认）</option>
                  <option value="keep_turn">保留模型轮次</option>
                  <option value="minimal">最小占位</option>
                  <option value="off">关闭兼容</option>
                </select>
              </div>
              <div id="h_pfm" class="helpbox">
                解决 Gemini 3.x / 4.x 拒绝以 assistant 轮次结尾的限制（内置未闭合标签思维链守卫与流式去重）：<br>
                • <b>智能并入（默认）</b>：将末尾预填充并入最后一条 user 末尾，并在输出侧自动拼回与去重，最稳健。<br>
                • <b>保留模型轮次</b>：保留 assistant 预填充轮次，在其后补简短 user 推动语。<br>
                • <b>最小占位</b>：仅补占位 user 保证不报错，不拼回预填充。<br>
                • <b>关闭兼容</b>：原样下发。
              </div>

              <div class="set-row">
                <span class="set-lbl">预填充时自动压制原生思考<span class="helpq" onclick="hlp(this,'h_pst')">?</span></span>
                <label class="switch"><input type="checkbox" id="prefill_suppress_thinking"><span class="slider"></span></label>
              </div>
              <div id="h_pst" class="helpbox">检测到请求末尾带预填充时，自动将模型的原生思考压到最低并不回传，让预设自带的思维链接管。</div>
              <input type="checkbox" id="prefill_cot_guard" class="hidden" checked>

              <details class="adv-box mt-3">
                <summary>
                  <span>高级外挂注入与自定义续写模板（轻量客户端可选）</span>
                  <span class="adv-arrow">▸</span>
                </summary>
                <div class="space-y-3 mt-3 pt-2.5 border-t divider-line">
                  <div>
                    <div class="lbl mb-1">附加 system 指令<span class="helpq" onclick="hlp(this,'h_injs')">?</span></div>
                    <textarea id="inject_system_instruction" rows="2" class="inp log" placeholder="留空 = 不注入（酒馆用户已有预设请留空）"></textarea>
                    <div id="h_injs" class="helpbox">追加到客户端 system 提示词之后，适合 NextChat / RikkaHub 等无预设管理器的轻量客户端。</div>
                  </div>
                  <div>
                    <div class="lbl mb-1">自动注入预填充<span class="helpq" onclick="hlp(this,'h_injp')">?</span></div>
                    <textarea id="inject_prefill" rows="2" class="inp log" placeholder="留空 = 不注入"></textarea>
                    <div id="h_injp" class="helpbox">仅当客户端请求未带预填充时自动补一条 assistant 预填充；若客户端已自带预填充则自动跳过。</div>
                  </div>
                  <div>
                    <div class="lbl mb-1">自定义续写引导模板<span class="helpq" onclick="hlp(this,'h_pfi')">?</span></div>
                    <textarea id="prefill_instruction" rows="2" class="inp log" placeholder="留空 = 使用内置默认续写引导语"></textarea>
                    <div id="h_pfi" class="helpbox">留空即使用内置默认续写引导语（生图模型留空时会自动换用要求输出图片的专用指令）。</div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </div>

        <!-- 右栏：采样参数与策略 + 生图 -->
        <div class="space-y-4">
          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>采样参数与策略</div>
              <span class="text-xs text-muted">留空即使用模型默认</span>
            </div>
            <div class="grid grid-cols-3 gap-3">
              <div><div class="lbl mb-1.5">temperature</div><input id="default_temperature" type="number" step="0.1" class="inp" placeholder="默认"></div>
              <div><div class="lbl mb-1.5">top_p</div><input id="default_top_p" type="number" step="0.05" class="inp" placeholder="默认"></div>
              <div><div class="lbl mb-1.5">max_tokens</div><input id="default_max_tokens" type="number" class="inp" placeholder="默认"></div>
            </div>
            <div class="set-row mt-3 pt-3 border-t divider-line">
              <span class="set-lbl">采样参数下发策略<span class="helpq" onclick="hlp(this,'h_sp')">?</span></span>
              <select id="sampling_policy" class="inp" style="max-width:165px">
                <option value="auto">自动判定（推荐）</option>
                <option value="deprecated">强制剥离</option>
                <option value="allowed">强制保留</option>
              </select>
            </div>
            <div id="h_sp" class="helpbox">决定是否将 <code>temperature / top_p / top_k</code> 下发给上游。自 Gemini 3.6 / 3.5-flash-lite / 4.x 起官方已废弃采样参数，「自动判定」会自动剥离以防 400 报错。</div>
            <div id="sampling-note" class="note-box"></div>
          </div>

          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>生图模型参数</div>
              <span class="code-tag">Image Config</span>
            </div>
            <div class="grid grid-cols-2 gap-3">
              <div>
                <div class="lbl mb-1.5">默认分辨率 <span class="code-tag">image_size</span></div>
                <select id="image_size" class="inp"><option value="512">512</option><option value="1K" selected>1K</option><option value="2K">2K</option><option value="4K">4K</option></select>
              </div>
              <div>
                <div class="lbl mb-1.5">默认宽高比 <span class="code-tag">aspect_ratio</span></div>
                <select id="image_aspect_ratio" class="inp"></select>
              </div>
            </div>
            <div class="mt-2 pt-2 border-t divider-line">
              <div class="set-row">
                <span class="set-lbl">生图请求下发 system 指令<span class="helpq" onclick="hlp(this,'h_isi')">?</span></span>
                <label class="switch"><input type="checkbox" id="image_system_instruction"><span class="slider"></span></label>
              </div>
              <div id="h_isi" class="helpbox">开启后，system 提示词中的画风、构图要求会随生图请求一并下发给生图模型。</div>
              <div class="set-row">
                <span class="set-lbl">生图请求允许注入预填充<span class="helpq" onclick="hlp(this,'h_ipf')">?</span></span>
                <label class="switch"><input type="checkbox" id="inject_prefill_for_image"><span class="slider"></span></label>
              </div>
              <div id="h_ipf" class="helpbox">默认关闭，避免角色扮演的思考标签误注入给生图模型。若需用预填充引导画风可针对生图模型单独开启。</div>
            </div>
            <div id="image-note" class="note-box"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 子面板 2：全局基础设施设置 -->
    <div id="pane-params-global" class="hidden">
      <div class="card p-4 sm:p-5 mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div class="text-sm font-semibold">全局基础设施与容灾策略</div>
          <div class="text-xs text-muted mt-0.5">对所有模型与通道统一生效（区域钉定、退避重试、输入图片压缩与流式缓冲）</div>
        </div>
        <button class="btn px-5 py-2 text-xs sm:text-sm whitespace-nowrap" onclick="saveSettings()">💾 保存全局设置</button>
      </div>

      <div class="grid md:grid-cols-2 gap-4 items-start">
        <!-- 左栏：区域路由 + 重试容灾 + 图片压缩 -->
        <div class="space-y-4">
          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>标准模式区域路由<span class="helpq" onclick="hlp(this,'h_loc')">?</span></div>
              <span class="code-tag">express_location</span>
            </div>
            <div class="set-row" style="padding:2px 0">
              <span class="set-lbl">目标部署区域</span>
              <select id="express_location" class="inp" style="max-width:210px">
                <option value="global">global（默认·推荐）</option>
                <option value="">默认（后端自选）</option>
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
              决定 Express 标准模式请求钉定到哪个区域，优先使用当前选中 Key 自有的 Project ID 拼装完整资源路径；若钉定路径失败会自动回退裸模型名重试一次。
            </div>
          </div>

          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>重试与跨通道容灾</div>
              <span class="code-tag">Retry &amp; Fallback</span>
            </div>
            <div class="grid grid-cols-2 gap-3 mb-2">
              <div><div class="lbl mb-1.5">最大重试次数</div><input id="retry_max" type="number" class="inp"></div>
              <div><div class="lbl mb-1.5">退避间隔（秒）</div><input id="retry_backoff_seconds" type="number" step="0.5" class="inp"></div>
            </div>
            <div class="set-row pt-2.5 border-t divider-line">
              <span class="set-lbl">允许跨通道兜底重试（Key ↔ Cookie）<span class="helpq" onclick="hlp(this,'h_ccf')">?</span></span>
              <label class="switch"><input type="checkbox" id="cross_channel_fallback"><span class="slider"></span></label>
            </div>
            <div id="h_ccf" class="helpbox">开启后，当主通道的所有凭证（如全部 Express Key 或 Cookie）报错/限流/失效时，自动无缝切换至另一通道的可用凭证接力完成回复。</div>
          </div>

          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>输入图片自动压缩</div>
              <label class="switch"><input type="checkbox" id="img_compress_enabled"><span class="slider"></span></label>
            </div>
            <div class="grid grid-cols-3 gap-3">
              <div><div class="lbl mb-1.5">最长边 (px)</div><input id="img_compress_max_dim" type="number" class="inp"></div>
              <div><div class="lbl mb-1.5">触发阈值 (MB)</div><input id="img_compress_max_mb" type="number" step="0.1" class="inp"></div>
              <div><div class="lbl mb-1.5">JPEG 质量</div><input id="img_compress_quality" type="number" class="inp"></div>
            </div>
          </div>
        </div>

        <!-- 右栏：流式保活、弱网续传与调试 -->
        <div class="space-y-4">
          <div class="card p-5">
            <div class="card-hd">
              <div class="card-title"><span class="card-dot"></span>流式保活、弱网续传 &amp; 调试</div>
              <span class="code-tag">Streaming &amp; Debug</span>
            </div>

            <div class="set-row">
              <span class="set-lbl">假流式保活模式 <span class="code-tag">fake_streaming</span><span class="helpq" onclick="hlp(this,'h_fks')">?</span></span>
              <label class="switch"><input type="checkbox" id="fake_streaming"><span class="slider"></span></label>
            </div>
            <div id="h_fks" class="helpbox">开启后在等待长思考模型（如 gemini-4-argon）生成期间，按下方间隔持续向客户端发送保活帧，防止前端 60s 超时断连。双通道均生效。</div>
            <div class="set-row">
              <span class="set-lbl text-muted" style="font-size:12.5px">└ 假流式心跳间隔（秒）</span>
              <input id="fake_streaming_interval" type="number" step="0.5" class="inp inp-sm">
            </div>

            <div class="set-row">
              <span class="set-lbl">弱网可靠续传缓冲 <span class="code-tag">resumable</span><span class="helpq" onclick="hlp(this,'h_rsm')">?</span></span>
              <label class="switch"><input type="checkbox" id="resumable_enabled"><span class="slider"></span></label>
            </div>
            <div id="h_rsm" class="helpbox">将上游回复写入带序号内存缓冲区，客户端断网重连可凭 <code>Last-Event-ID</code> 或 <code>?resume_from=</code> 无损续传；首包产出前若前端主动停止，后台 1.5s 内立即止损取消。</div>
            <div class="set-row">
              <span class="set-lbl text-muted" style="font-size:12.5px">└ 续传缓冲保留时长（秒）</span>
              <input id="resumable_ttl_seconds" type="number" step="60" class="inp inp-sm">
            </div>
            <div class="set-row">
              <span class="set-lbl text-muted" style="font-size:12.5px">└ 续传缓冲最大条数</span>
              <input id="resumable_max_streams" type="number" step="10" class="inp inp-sm">
            </div>

            <div class="set-row">
              <span class="set-lbl">多 Key 顺序轮询 <span class="code-tag">roundrobin</span><span class="helpq" onclick="hlp(this,'h_rr')">?</span></span>
              <label class="switch"><input type="checkbox" id="roundrobin" checked><span class="slider"></span></label>
            </div>
            <div id="h_rr" class="helpbox">开启后新请求按顺序轮流从下一个 Express Key 开始调用；关闭则随机挑选起始 Key。无论是否开启，单次请求触发重试时都会自动遍历其余健康 Key。</div>

            <div class="set-row">
              <span class="set-lbl">回复末尾附加安全评分<span class="helpq" onclick="hlp(this,'h_ss')">?</span></span>
              <label class="switch"><input type="checkbox" id="safety_score"><span class="slider"></span></label>
            </div>
            <div id="h_ss" class="helpbox">在模型回复末尾附加可折叠的 Google 安全分类评分详情块。</div>

            <div class="set-row">
              <span class="set-lbl">出站与原始响应诊断日志<span class="helpq" onclick="hlp(this,'h_dbg')">?</span></span>
              <label class="switch"><input type="checkbox" id="debug_outbound"><span class="slider"></span></label>
            </div>
            <div id="h_dbg" class="helpbox">在运行日志打印发给 Google 的完整出站参数、工具 Schema 与末尾消息断点，并在流结束或异常时输出首尾候选包原始结构。</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Tab 4: Logs -->
  <section id="view-logs" class="tab-pane hidden">
    <div class="card overflow-hidden">
      <div class="px-4 py-2.5 border-b log-header flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-2">
          <div class="flex gap-1.5"><div class="w-3 h-3 rounded-full" style="background:#f68697"></div><div class="w-3 h-3 rounded-full" style="background:#f8c27a"></div><div class="w-3 h-3 rounded-full" style="background:#5fd3bc"></div></div>
          <span class="ml-1.5 text-xs text-muted log">terminal · 实时运行日志（点击任意单行可复制）</span>
        </div>
        <div class="flex items-center gap-2">
          <button id="btn-autoscroll" class="pill pill-mint text-xs" style="cursor:pointer" onclick="toggleAutoscroll()">⬇️ 自动置底：开</button>
          <button class="btn-soft px-2.5 py-1 text-xs" onclick="copyAllLogs()">📋 复制全部</button>
          <button class="btn-soft px-2.5 py-1 text-xs" onclick="clearLogs()">🧹 清空</button>
        </div>
      </div>
      <div id="logwin" class="log p-4 space-y-1.5 overflow-y-auto" style="height:64vh"></div>
    </div>
  </section>
</div>
<div id="toast" class="toast"></div>

<script>
var $ = function(id){ return document.getElementById(id); };
var CAPS = {}, chart = null, curAR = "", lastStats = {s:0, e:0, r:0};
var GLOBAL_SETTINGS = {}, OVERRIDES = {};
var PER_MODEL_KEYS = [
  'native_thinking_mode','thinking_g3_level','image_size','image_aspect_ratio',
  'default_temperature','default_top_p','default_max_tokens',
  'inject_system_instruction','inject_prefill','prefill_instruction',
  'image_system_instruction','inject_prefill_for_image','sampling_policy',
  'prefill_cot_guard','prefill_mode','prefill_suppress_thinking',
  'synthetic_sandbox_enabled'
];
var COMMON_ARS = ["1:1","3:2","2:3","3:4","4:3","4:5","5:4","9:16","16:9","21:9","1:4","4:1","1:8","8:1","9:21"];

function toast(m){ var t=$('toast'); t.textContent=m; t.classList.add('show'); setTimeout(function(){t.classList.remove('show');},2200); }
function fmt(n){ return (n||0).toLocaleString('en-US'); }
async function logout(){ try{ await fetch('/api/logout',{method:'POST'}); }catch(e){} location.href='/'; }

async function copyEndpoint(){
  var url = window.location.origin + '/v1';
  try{
    await navigator.clipboard.writeText(url);
    toast('🔗 已复制接口地址：' + url);
  }catch(e){ toast('接口地址：' + url); }
}

function isDarkMode(){ return document.documentElement.classList.contains('dark'); }
function syncThemeSeg(){
  var dark = isDarkMode();
  var l=$('t-light'), d=$('t-dark');
  if(l) l.classList.toggle('on', !dark);
  if(d) d.classList.toggle('on', dark);
}
function setTheme(mode){
  var wantDark = (mode==='dark');
  document.documentElement.classList.toggle('dark', wantDark);
  try{ localStorage.setItem('ap_theme', wantDark ? 'dark' : 'light'); }catch(e){}
  syncThemeSeg();
  renderChart(lastStats.s, lastStats.e, lastStats.r);
  rerenderAllLogs();
}
syncThemeSeg();

function syncAutoscrollBtn(){
  var b = $('btn-autoscroll');
  if(!b) return;
  b.textContent = autoscroll ? '⬇️ 自动置底：开' : '⏸️ 自动置底：关';
  b.className = autoscroll ? 'pill pill-mint text-xs' : 'pill text-xs';
}
function toggleAutoscroll(){
  autoscroll = !autoscroll;
  syncAutoscrollBtn();
  if(autoscroll) scrollLogsToBottom();
}
function scrollLogsToBottom(){
  var lw=$('logwin');
  if(!lw) return;
  requestAnimationFrame(function(){ lw.scrollTop = lw.scrollHeight; });
}

function switchTab(t){
  document.querySelectorAll('.tab').forEach(function(e){ e.classList.toggle('active', e.dataset.tab===t); });
  ['overview','channel','params','logs'].forEach(function(v){ $('view-'+v).classList.toggle('hidden', v!==t); });
  if(t==='logs'){
    autoscroll = true;
    syncAutoscrollBtn();
    scrollLogsToBottom();
  }
}

function switchParamsSubTab(sub){
  var isModel = (sub === 'model');
  var bModel = $('tab-btn-params-model');
  var bGlobal = $('tab-btn-params-global');
  if(bModel) bModel.classList.toggle('active', isModel);
  if(bGlobal) bGlobal.classList.toggle('active', !isModel);
  var pModel = $('pane-params-model');
  var pGlobal = $('pane-params-global');
  if(pModel) pModel.classList.toggle('hidden', !isModel);
  if(pGlobal) pGlobal.classList.toggle('hidden', isModel);
}

function renderChart(s,e,r){
  lastStats = {s:s, e:e, r:r};
  var ctx=$('donut').getContext('2d');
  var dark = isDarkMode();
  var isEmpty = (s===0 && e===0 && r===0);
  var labels = isEmpty ? ['暂无请求'] : ['成功响应','拥堵重试','错误 / 拦截'];
  var data   = isEmpty ? [1] : [s, r, e];
  var emptyColor = dark ? '#2b263b' : '#ede7f8';
  var borderColor = dark ? '#1d1a28' : '#ffffff';
  var colors = isEmpty ? [emptyColor] : ['#5fd3bc', '#f8c27a', '#f68697'];

  var totalDone = s + e;
  var rateEl = $('health-rate');
  var pctEl  = $('donut-pct');
  var subEl  = $('donut-sub');
  if(totalDone === 0){
    if(rateEl){ rateEl.textContent = '待命'; rateEl.className = 'pill pill-accent'; }
    if(pctEl){ pctEl.textContent = '100%'; pctEl.style.color = 'var(--fg)'; }
    if(subEl){ subEl.textContent = '待命'; }
  } else {
    var pctNum = (s / totalDone) * 100;
    var pct = pctNum.toFixed(1).replace(/\.0$/, '');
    if(rateEl){
      rateEl.textContent = (pctNum >= 90 ? '健康 ' : '波动 ') + pct + '%';
      rateEl.className = pctNum >= 90 ? 'pill pill-mint' : 'pill pill-apricot';
    }
    if(pctEl){
      pctEl.textContent = pct + '%';
      pctEl.style.color = pctNum >= 90 ? 'var(--mint-text)' : 'var(--apricot-text)';
    }
    if(subEl){ subEl.textContent = '成功率'; }
  }

  if(chart){
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].backgroundColor = colors;
    chart.data.datasets[0].borderColor = borderColor;
    chart.update();
    return;
  }
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{ data: data, backgroundColor: colors, borderWidth: 3, borderColor: borderColor, hoverOffset: 4 }]
    },
    options: { cutout: '74%', plugins: { legend: { display: false }, tooltip: { backgroundColor: '#4a3f63', cornerRadius: 8, padding: 9 } } }
  });
}

async function fetchStats(){
  try{
    var d=await (await fetch('/api/stats')).json();
    $('s-total').textContent=fmt(d.total);
    $('s-success').textContent=fmt(d.success);
    $('s-error').textContent=fmt(d.error);
    $('s-retries').textContent=fmt(d.retries);
    var pTk = d.prompt_tokens || 0;
    var cTk = d.completion_tokens || 0;
    var totTk = pTk + cTk;
    $('t-prompt').textContent=fmt(pTk);
    $('t-comp').textContent=fmt(cTk);
    $('t-total').textContent=fmt(totTk);
    $('bar-prompt').style.width = totTk > 0 ? Math.max(4, Math.round((pTk / totTk) * 100)) + '%' : '0%';
    $('bar-comp').style.width   = totTk > 0 ? Math.max(4, Math.round((cTk / totTk) * 100)) + '%' : '0%';
    $('uptime').textContent='已运行 '+(d.uptime/3600).toFixed(1)+' h';
    renderChart(d.success||0, d.error||0, d.retries||0);
  }catch(e){}
}

function applyChannelHighlight(m){
  var isWeb = (m==='web_proxy');
  $('opt-api').classList.toggle('active', !isWeb);
  $('opt-web').classList.toggle('active', isWeb);
  $('badge-api').classList.toggle('hidden', isWeb);
  $('badge-web').classList.toggle('hidden', !isWeb);
  $('mode-pill').textContent = isWeb ? '主通道 Cookie 直连' : '主通道 Express API';
}
async function updateMode(m){
  applyChannelHighlight(m);
  await fetch('/api/settings/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  toast(m==='web_proxy' ? '已切换至 Cookie 直连反代通道' : '已切换至 Express API Key 标准通道');
}
function parseCookies(str){
  str=(str||'').trim();
  if(str.startsWith('[')&&str.endsWith(']')){
    try{
      var a=JSON.parse(str);
      if(Array.isArray(a)) return a.map(function(c){var n=c.name||c.key,v=c.value; return (n&&v)?(n+'='+v):'';}).filter(Boolean).join('; ');
    }catch(e){}
  }
  return str;
}
async function saveCookie(){
  var ck=parseCookies($('cookie-input').value); var pid=$('project-input').value.trim();
  var m=pid.match(/[?&]project=([^&]+)/)||pid.match(/\/projects\/([^\/]+)/); if(m) pid=m[1];
  $('cookie-input').value=ck; $('project-input').value=pid;
  if(!pid){ toast('请填写 Project ID'); return; }
  try{
    var r=await fetch('/api/cookie',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cookie:ck,project_id:pid})});
    var d=await r.json(); toast(r.ok?(d.message||'已保存并激活'):('❌ '+(d.error||'保存失败')));
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function loadRuntime(){
  try{
    var s=await (await fetch('/api/settings/runtime')).json();
    var m=s.use_web_proxy?'web_proxy':'api_key';
    var radio=document.querySelector('input[name=mode][value=' + m + ']');
    if(radio) radio.checked=true;
    applyChannelHighlight(m);
    var ci=$('cookie-input');
    if(ci){
      ci.value='';
      ci.placeholder = s.google_cookie_configured
        ? ('已配置：'+s.google_cookie+'　（留空则保持不变，需更新时粘贴新 Cookie）')
        : '粘贴完整 Cookie 头或 Cookie-Editor 导出内容';
    }
    if(s.google_project_id) $('project-input').value=s.google_project_id;
  }catch(e){}
}

var KEYS = [];
function keySrcPill(rec){
  if(rec.project_source==='auto') return '<span class="pill pill-mint" style="font-size:11px;padding:2px 8px">自动探测</span>';
  if(rec.project_source==='manual') return '<span class="pill pill-accent" style="font-size:11px;padding:2px 8px">手动指定</span>';
  return '<span class="pill pill-apricot" style="font-size:11px;padding:2px 8px">待探测</span>';
}
function renderKeys(){
  var box=$('keys-list');
  box.innerHTML = KEYS.map(function(rec,i){
    var kVal = (rec.key||'').replace(/"/g,'&quot;');
    var pVal = (rec.project_id||'').replace(/"/g,'&quot;');
    var chk = (rec.enabled===false) ? '' : 'checked';
    var detStr = rec.detected_at ? ('探测于 ' + new Date(rec.detected_at*1000).toLocaleString()) : '';
    var msgColor = (rec._msg_ok===false) ? 'var(--coral-text)' : 'var(--mint-text)';
    var msgHtml = rec._msg ? ('<div class="text-xs mt-2 pt-1.5 border-t divider-line" style="color:' + msgColor + '">' + rec._msg + '</div>') : '';
    return '<div class="key-row rounded-xl p-3.5" data-i="' + i + '">'
      + '<div class="flex flex-col sm:flex-row sm:items-center gap-2 mb-2.5">'
      +   '<div class="flex items-center gap-2 flex-1 min-w-0">'
      +     '<span class="pill pill-accent" style="min-width:34px;justify-content:center">#' + (i+1) + '</span>'
      +     '<input class="inp log k-key" style="flex:1;min-width:0" placeholder="粘贴 Express API Key（移开焦点自动保存）" value="' + kVal + '" onchange="onKeyFieldChange(' + i + ')" onkeydown="if(event.key===\'Enter\'){this.blur();}">'
      +   '</div>'
      +   '<div class="flex items-center justify-end gap-2 flex-shrink-0">'
      +     '<label class="flex items-center gap-1.5 text-xs text-muted whitespace-nowrap cursor-pointer select-none px-1"><input type="checkbox" class="k-enabled accent-[#8b7cf6]" ' + chk + ' onchange="onKeyFieldChange(' + i + ')">启用</label>'
      +     '<button class="btn-soft px-2.5 py-1 text-xs" onclick="probeOneKey(' + i + ')">🔎 探测</button>'
      +     '<button class="btn-danger-soft px-2.5 py-1 text-xs" onclick="removeKeyRow(' + i + ')">删除</button>'
      +   '</div>'
      + '</div>'
      + '<div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">'
      +   '<div class="flex items-center gap-2 flex-1 min-w-0">'
      +     '<span class="lbl whitespace-nowrap" style="min-width:62px">Project ID</span>'
      +     '<input class="inp log k-project" style="flex:1;min-width:140px;padding:5.5px 10px;font-size:12.5px" placeholder="留空 = 自动探测获取" value="' + pVal + '" onchange="onKeyFieldChange(' + i + ')" onkeydown="if(event.key===\'Enter\'){this.blur();}">'
      +   '</div>'
      +   '<div class="flex items-center justify-end gap-2 flex-wrap">'
      +     keySrcPill(rec)
      +     (detStr ? ('<span class="text-[11px] text-muted-light">' + detStr + '</span>') : '')
      +   '</div>'
      + '</div>'
      + msgHtml
      + '</div>';
  }).join('');
  $('keys-count').textContent = KEYS.filter(function(r){return r.key;}).length + ' 个';
  $('keys-empty').classList.toggle('hidden', KEYS.length>0);
}
function syncKeysFromDom(){
  Array.prototype.slice.call(document.querySelectorAll('#keys-list [data-i]')).forEach(function(row){
    var i=Number(row.dataset.i);
    if(!KEYS[i]) return;
    KEYS[i].key = row.querySelector('.k-key').value.trim();
    var pid = row.querySelector('.k-project').value.trim();
    if(pid !== (KEYS[i].project_id||'')){ KEYS[i].project_id = pid; KEYS[i].project_source = pid?'manual':''; }
    KEYS[i].enabled = row.querySelector('.k-enabled').checked;
  });
}
async function autoSaveExpressKeys(customMsg){
  syncKeysFromDom();
  var emptyRows = KEYS.filter(function(r){return !r.key;});
  var payload = KEYS.filter(function(r){return r.key;});
  try{
    var r=await fetch('/api/express-keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:payload})});
    var d=await r.json();
    if(r.ok){
      KEYS=(d.keys||[]).map(function(x){return Object.assign({},x);}).concat(emptyRows);
      renderKeys();
      toast(customMsg || ('✅ 已自动保存（共 ' + payload.length + ' 个有效 Key）'));
      if(payload.some(function(k){return !k.project_id;})){
        setTimeout(refreshKeysQuietly, 2200);
      }
    } else {
      toast('❌ '+(d.error||'自动保存失败'));
    }
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function refreshKeysQuietly(){
  try{
    syncKeysFromDom();
    var emptyRows = KEYS.filter(function(r){return !r.key;});
    var d=await (await fetch('/api/express-keys')).json();
    KEYS=(d.keys||[]).map(function(r){return Object.assign({},r);}).concat(emptyRows);
    renderKeys();
  }catch(e){}
}
async function onKeyFieldChange(i){
  syncKeysFromDom();
  var rec = KEYS[i];
  if(!rec || !rec.key) return;
  await autoSaveExpressKeys();
}
function addKeyRow(){
  syncKeysFromDom();
  KEYS.push({key:'',project_id:'',project_source:'',enabled:true});
  renderKeys();
  var inputs = document.querySelectorAll('#keys-list .k-key');
  if(inputs.length > 0) inputs[inputs.length - 1].focus();
}
async function removeKeyRow(i){
  syncKeysFromDom();
  var target = KEYS[i];
  var hadKey = !!(target && target.key);
  KEYS.splice(i,1);
  if(!hadKey){
    renderKeys();
    return;
  }
  await autoSaveExpressKeys('🗑️ 已删除 Key 并自动保存');
}
async function loadExpressKeys(){
  try{
    var d=await (await fetch('/api/express-keys')).json();
    KEYS=(d.keys||[]).map(function(r){return Object.assign({},r);});
    renderKeys();
  }catch(e){}
}
async function probeOneKey(i){
  syncKeysFromDom();
  var rec=KEYS[i];
  if(!rec||!rec.key){ toast('请先在该行填入 Express API Key'); return; }
  var targetKey = rec.key;
  var emptyRows = KEYS.filter(function(r){return !r.key;});
  var payload = KEYS.filter(function(r){return r.key;});
  toast('正在自动保存并探测归属项目…');
  try{
    var r=await fetch('/api/express-keys/probe',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({key:targetKey, force:true, keys:payload})
    });
    var d=await r.json();
    if(!r.ok){ toast('❌ '+(d.error||'探测失败')); return; }
    var res=(d.results||[])[0]||{};
    KEYS=(d.keys||[]).map(function(x){return Object.assign({},x);}).concat(emptyRows);
    var hit=KEYS.find(function(x){return x.key===targetKey;});
    if(hit){ hit._msg=res.message||''; hit._msg_ok=!!res.ok; }
    renderKeys();
    toast(res.ok?('✅ 已保存并探测成功：'+(res.project_id||'—')):('⚠️ Key 已保存，但探测未果：'+(res.message||'请手动填写项目')));
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function probeAllKeys(){
  syncKeysFromDom();
  var emptyRows = KEYS.filter(function(r){return !r.key;});
  var payload = KEYS.filter(function(r){return r.key;});
  if(payload.length===0){ toast('请先至少添加一个 Express API Key'); return; }
  toast('正在自动保存并探测全部 Key…');
  try{
    var r=await fetch('/api/express-keys/probe',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({all:true, force:true, keys:payload})
    });
    var d=await r.json();
    if(!r.ok){ toast('❌ '+(d.error||'探测失败')); return; }
    var results=d.results||[];
    KEYS=(d.keys||[]).map(function(x){return Object.assign({},x);}).concat(emptyRows);
    results.forEach(function(res){ var hit=KEYS.find(function(x){return x.key===res.key;}); if(hit){ hit._msg=res.message||''; hit._msg_ok=!!res.ok; } });
    renderKeys();
    toast('已保存并完成探测：成功 '+results.filter(function(x){return x.ok;}).length+' / '+results.length);
  }catch(e){ toast('❌ 网络请求失败'); }
}

function setV(id,v){ var el=$(id); if(!el) return; if(el.type==='checkbox') el.checked=!!v; else el.value=(v===null||v===undefined)?'':v; }
async function loadParams(){
  try{
    var s=await (await fetch('/api/settings')).json();
    GLOBAL_SETTINGS=s;
    curAR = s.image_aspect_ratio || "";
    [
      'native_thinking_mode','thinking_g3_level','image_size',
      'default_temperature','default_top_p','default_max_tokens',
      'img_compress_max_dim','img_compress_max_mb','img_compress_quality',
      'retry_max','retry_backoff_seconds','fake_streaming_interval',
      'prefill_mode','prefill_instruction','inject_system_instruction','inject_prefill',
      'sampling_policy','express_location','resumable_ttl_seconds','resumable_max_streams'
    ].forEach(function(k){setV(k,s[k]);});
    [
      'img_compress_enabled','fake_streaming','roundrobin','safety_score',
      'debug_outbound','prefill_suppress_thinking',
      'image_system_instruction','inject_prefill_for_image','prefill_cot_guard','resumable_enabled',
      'cross_channel_fallback','synthetic_sandbox_enabled'
    ].forEach(function(k){setV(k,s[k]);});
  }catch(e){}
  try{
    var c=await (await fetch('/api/capabilities')).json();
    CAPS=c.capabilities||{};
    OVERRIDES=c.overrides||{};
    $('model-sel').innerHTML='<option value="__global__">＊ 全局默认（对所有模型）</option>'
      + (c.models||[]).map(function(m){var star=OVERRIDES[m]?' ★':''; return '<option value="' + m + '">' + m + star + '</option>';}).join('');
    renderCaps();
  }catch(e){ renderCaps(); }
}

function applyModelParamFields(model){
  var isGlobal = (model==='__global__');
  var ov = isGlobal ? {} : (OVERRIDES[model] || {});
  PER_MODEL_KEYS.forEach(function(k){
    var v = (k in ov) ? ov[k] : GLOBAL_SETTINGS[k];
    if(k==='image_aspect_ratio'){ curAR = v || ""; }
    else setV(k, v);
  });
  var has = !isGlobal && !!OVERRIDES[model] && Object.keys(OVERRIDES[model]).length>0;
  $('ov-badge').classList.toggle('hidden', !has);
  $('btn-scope-save').textContent = isGlobal ? '💾 保存为全局默认' : '💾 保存为该模型专属';
  $('btn-scope-clear').classList.toggle('hidden', isGlobal);
  $('ov-hint').textContent = isGlobal
    ? '当前编辑：所有未单独定制（无 ★）模型的全局默认参数'
    : (has ? ('当前编辑：' + model + ' 的专属覆盖参数（★ 已激活）')
           : ('当前编辑：' + model + '（当前继承全局默认，点右上保存即可创建专属覆盖）'));
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
    synthetic_sandbox_enabled:$('synthetic_sandbox_enabled').checked,
  };
}

async function saveModelOverride(){
  var m=$('model-sel').value; if(!m) return;
  var patch = collectSectionOnePatch();
  if(m==='__global__'){
    try{
      var r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
      if(r.ok){ GLOBAL_SETTINGS=await r.json(); toast('✅ 已保存为所有模型的全局默认配置'); }
      else toast('❌ 保存失败');
    }catch(e){ toast('❌ 保存失败'); }
    return;
  }
  try{
    var r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    var d=await r.json();
    if(r.ok){ OVERRIDES[m]=d.override||patch; toast('✅ 已保存 '+m+' 的专属参数'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('❌ 保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

async function clearModelOverride(){
  var m=$('model-sel').value; if(!m || m==='__global__') return;
  if(!OVERRIDES[m]){ toast('该模型当前没有专属参数'); return; }
  try{
    var r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'DELETE'});
    if(r.ok){ delete OVERRIDES[m]; toast('已清除 '+m+' 专属参数，恢复继承全局默认'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('❌ 清除失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

function refreshModelSelStars(){
  var sel=$('model-sel'); var cur=sel.value;
  Array.prototype.slice.call(sel.options).forEach(function(o){
    var base=o.value;
    if(base==='__global__'){
      o.textContent = '＊ 全局默认（对所有模型）';
    } else {
      o.textContent = base + (OVERRIDES[base] ? ' ★' : '');
    }
  });
  sel.value=cur;
}

function chip(t, accent){ return '<span class="pill' + (accent?' pill-accent':'') + '" style="font-size:11px;padding:2px 8px">' + t + '</span>'; }
function fillARFor(cap){
  var sel=$('image_aspect_ratio');
  var list = (cap && cap.is_image && cap.image_aspect_ratios.length) ? cap.image_aspect_ratios : COMMON_ARS;
  sel.innerHTML='<option value="">自动（不强制指定）</option>' + list.map(function(a){return '<option value="' + a + '">' + a + '</option>';}).join('');
  sel.value = (curAR && list.includes(curAR)) ? curAR : "";
}
function renderCaps(){
  var m=$('model-sel').value;
  if(m==='__global__'){
    applyModelParamFields(m);
    fillARFor(null);
    $('caps-summary').innerHTML = chip('🌐 作用域：全局默认', true) + chip('切换具体模型可查看其能力白名单');
    ['thinking_g3_level','image_size','image_aspect_ratio'].forEach(function(id){var el=$(id); if(el) el.disabled=false;});
    $('wrap-g3level').style.opacity='1';
    $('think-note').textContent='全局默认：选「自动」时采用模型官方原生默认档位；压制思考或选 minimal 时，3.7+ 及 4.x 全系模型最低自动发送 low。';
    $('image-note').textContent='全局默认：分辨率与宽高比仅在调用生图模型（-image / nano-banana）时生效。';
    $('sampling-note').textContent='全局默认：遇到已废弃采样参数的模型（如 Gemini 3.6+ / 4.x）会自动剥离以防报错。';
    return;
  }
  var cap=CAPS[m]; if(!cap) return;
  applyModelParamFields(m);
  var th=cap.thinking||{};
  var sum=[chip('家族 '+cap.family, true)];
  if(th.kind==='level') sum.push(chip('思考档位 '+(th.levels||[]).join(' / ')));
  else sum.push(chip('无思考调节'));
  if(cap.is_image){
    sum.push(chip('分辨率 '+cap.image_sizes.join('/')));
    sum.push(chip('比例 '+cap.image_aspect_ratios.length+' 种'));
    sum.push(chip('生图专用'));
  } else {
    sum.push(chip('采样 '+(cap.sampling_advice==='deprecated'?'已废弃·自动剥离':cap.sampling_advice==='recommend_default'?'可调·建议默认':'可调')));
    sum.push(chip('支持函数调用'));
  }
  if(cap.supports_search) sum.push(chip('支持联网搜索'));
  $('caps-summary').innerHTML=sum.join('');

  var isLevel=th.kind==='level';
  $('thinking_g3_level').disabled=!isLevel;
  $('wrap-g3level').style.opacity=isLevel?'1':'.4';
  var tn='';
  if(cap.is_image) tn='生图模型思考档位：选「自动」即采用官方原生默认，亦可在上方手动指定档位。';
  else if(isLevel) tn='支持思考档位：' + (th.levels||[]).join(' / ') + ((th.levels && !th.levels.includes('minimal')) ? '（该模型最低合法档为 low，不支持 minimal）' : '（最低合法档为 minimal）') + '。';
  else tn='该模型不支持思考档位调节。';
  $('think-note').textContent=tn;

  var imgOn=cap.is_image;
  $('image_size').disabled=!imgOn; $('image_aspect_ratio').disabled=!imgOn;
  fillARFor(cap);
  $('image-note').textContent = imgOn
    ? ('支持宽高比：' + cap.image_aspect_ratios.join('、') + '；支持分辨率：' + cap.image_sizes.join(' / ') + '。')
    : '当前所选为文本模型，无需配置生图分辨率与宽高比。';

  $('sampling-note').textContent = cap.is_image
    ? '生图模型调用时会自动剥离文本采样参数。'
    : (cap.sampling_advice==='deprecated'
        ? '⚠️ 该模型官方已废弃 temperature / top_p / top_k，网关会自动剥离以防 400 报错。'
        : cap.sampling_advice==='recommend_default'
          ? '官方建议保持默认采样值；上方留空即使用模型内置默认。'
          : '该模型支持自定义采样参数；留空即使用模型内置默认。');
}
function numOrNull(id){ var v=$(id).value.trim(); return v===''?null:Number(v); }
function numOr(id,d){ var v=$(id).value.trim(); return v===''?d:Number(v); }
function hlp(el,id){var b=document.getElementById(id);var on=b.classList.toggle('show');el.classList.toggle('on',on);}

async function saveSettings(){
  var patch={
    express_location:$('express_location').value,
    img_compress_enabled:$('img_compress_enabled').checked,
    img_compress_max_dim:numOr('img_compress_max_dim',1536),
    img_compress_max_mb:numOr('img_compress_max_mb',1.5),
    img_compress_quality:numOr('img_compress_quality',85),
    retry_max:numOr('retry_max',10),
    retry_backoff_seconds:numOr('retry_backoff_seconds',5),
    cross_channel_fallback:$('cross_channel_fallback').checked,
    fake_streaming:$('fake_streaming').checked,
    fake_streaming_interval:numOr('fake_streaming_interval',1),
    resumable_enabled:$('resumable_enabled').checked,
    resumable_ttl_seconds:numOr('resumable_ttl_seconds',1800),
    resumable_max_streams:numOr('resumable_max_streams',200),
    roundrobin:$('roundrobin').checked,
    safety_score:$('safety_score').checked,
    debug_outbound:$('debug_outbound').checked,
  };
  try{
    var r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    if(r.ok){
      GLOBAL_SETTINGS=await r.json();
      toast('✅ 全局基础设施设置已保存');
    } else toast('❌ 保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

var logwin=$('logwin'); var autoscroll=true;
var LOG_RAW_LINES = [];
logwin.addEventListener('scroll',function(){
  if($('view-logs').classList.contains('hidden')) return;
  var nearBottom = (logwin.scrollHeight - logwin.scrollTop - logwin.clientHeight) < 60;
  if(nearBottom !== autoscroll){
    autoscroll = nearBottom;
    syncAutoscrollBtn();
  }
});
function logLine(t, idx){
  var dark = isDarkMode();
  var c = dark ? '#ccc4de' : '#4d4460', bg = 'transparent', bl = '3px solid transparent';
  if(t.includes('✅')||t.includes('🎉')){
    c = dark ? '#6de0c9' : '#1f8270';
    bg = dark ? 'rgba(95,211,188,.10)' : '#f1fcf9';
    bl = '3px solid #5fd3bc';
  }
  else if(t.includes('⚠️')||t.includes('WARN')||t.includes('🔄')||t.includes('重试')||t.includes('🛑')){
    c = dark ? '#fad096' : '#b56c14';
    bg = dark ? 'rgba(248,194,122,.11)' : '#fffaf2';
    bl = '3px solid #f8c27a';
  }
  else if(t.includes('❌')||t.includes('ERROR')||t.includes('🔴')){
    c = dark ? '#ff9bab' : '#c43c54';
    bg = dark ? 'rgba(246,134,151,.14)' : '#fff4f6';
    bl = '3px solid #f68697';
  }
  else if(t.includes('💰')){
    c = dark ? '#c2b6ff' : '#5e4bc4';
    bg = dark ? 'rgba(139,124,246,.13)' : '#f7f4ff';
    bl = '3px solid #8b7cf6';
  }
  var hlColor = dark ? '#65dbc4' : '#239480';
  var s = t.replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s = s.replace(/^(\[\d{2}:\d{2}:\d{2}\])\s*/, '<span class="log-ts">$1</span>');
  s = s.replace(/(gemini-[a-zA-Z0-9.\-]+)/g,'<span style="color:' + hlColor + ';font-weight:600">$1</span>');
  return '<div class="log-item" data-idx="' + idx + '" onclick="copySingleLog(this)" title="点击复制本条日志" style="color:' + c + ';background:' + bg + ';border-left:' + bl + ';padding:6px 10px;border-radius:6px;white-space:pre-wrap;word-break:break-word">' + s + '</div>';
}
async function copySingleLog(el){
  var sel = window.getSelection ? window.getSelection().toString() : '';
  if(sel && sel.length > 0) return;
  var idx = Number(el.getAttribute('data-idx'));
  var raw = LOG_RAW_LINES[idx] || el.innerText || '';
  try{
    await navigator.clipboard.writeText(raw);
    toast('📋 已复制该条日志');
  }catch(e){ toast('复制失败，请手动选中复制'); }
}
async function copyAllLogs(){
  if(LOG_RAW_LINES.length === 0){ toast('当前没有可复制的日志'); return; }
  try{
    await navigator.clipboard.writeText(LOG_RAW_LINES.join('\n'));
    toast('📋 已复制全部 ' + LOG_RAW_LINES.length + ' 条日志');
  }catch(e){ toast('复制失败，请手动选中复制'); }
}
function rerenderAllLogs(){
  logwin.innerHTML = LOG_RAW_LINES.map(function(line, i){ return logLine(line, i); }).join('');
  if(autoscroll) scrollLogsToBottom();
}
function clearLogs(){
  LOG_RAW_LINES = [];
  logwin.innerHTML = '';
  toast('已清空当前面板日志');
}
try{
  var es=new EventSource('/stream-logs');
  es.onmessage=function(e){
    if(e.data.includes('keep-alive')) return;
    if(LOG_RAW_LINES.length >= 100){
      LOG_RAW_LINES.shift();
      LOG_RAW_LINES.push(e.data);
      rerenderAllLogs();
      return;
    }
    var idx = LOG_RAW_LINES.length;
    LOG_RAW_LINES.push(e.data);
    logwin.insertAdjacentHTML('beforeend', logLine(e.data, idx));
    if(autoscroll) scrollLogsToBottom();
  };
}catch(e){}

$('project-input').addEventListener('input',function(e){ var v=e.target.value.trim(); var m=v.match(/[?&]project=([^&]+)/)||v.match(/\/projects\/([^\/]+)/); if(m) e.target.value=m[1]; });
$('image_aspect_ratio').addEventListener('change', function(e){ curAR=e.target.value; });

fetchStats(); setInterval(fetchStats,3000); loadRuntime(); loadParams(); loadExpressKeys();
</script>
</body>
</html>
"""

LOGIN_HTML = LOGIN_HTML.replace("__AVATAR_URI__", AVATAR_DATA_URI)
DASHBOARD_HTML = DASHBOARD_HTML.replace("__AVATAR_URI__", AVATAR_DATA_URI)




@app.get("/favicon.ico")
async def favicon_endpoint():
    return Response(content=AVATAR_BYTES, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})

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
