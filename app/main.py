import asyncio
import os
import secrets
import threading
import time
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

express_key_manager = ExpressKeyManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_loader import refresh_models_config_cache
    print("🚀 [服务启动] agentplatform2api 适配器已启动（Express API Key / Cookie 直连 双通道）。")

    # S-1：默认口令 + 公开托管 + 明文 Cookie 是很危险的组合，必须让人看见。
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
    if express_key_manager.get_total_keys() > 0:
        print(f"✅ [密钥配置] 已加载 {express_key_manager.get_total_keys()} 个 Express API Key。")
    else:
        print("⚠️ [密钥配置] 未检测到 VERTEX_EXPRESS_API_KEY。若不启用 Cookie 直连模式，聊天请求将会报错。")
    await refresh_models_config_cache()
    yield

app = FastAPI(title="agentplatform2api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # 本服务使用 Bearer / Basic 鉴权，不依赖浏览器 Cookie；
    # 关闭 allow_credentials 以符合 CORS 规范（通配符 + 凭证不合法）。
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
            if response.status_code >= 400:
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

# P2-2：会话 token 改为随机值存内存，不再用 sha256(常量 + API_KEY) 这种确定值。
# 确定值意味着同一个 API_KEY 永远对应同一个 cookie，无法单独失效某个会话。
_sessions: dict = {}          # token -> 过期时间戳
_sessions_lock = threading.Lock()

# 登录失败计数：{ip: [失败次数, 最近失败时间]}，指数退避
_login_failures: dict = {}
_login_lock = threading.Lock()
LOGIN_LOCK_BASE_SECONDS = 2
LOGIN_LOCK_MAX_SECONDS = 300


def _issue_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _sessions_lock:
        for t, exp in list(_sessions.items()):     # 顺手清理过期会话
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
    """该 IP 还需等待多少秒才能再次尝试登录（0 = 可以尝试）。"""
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
    """S-1：控制台只回显掩码，不再把完整 Google 会话 Cookie 明文吐回前端。"""
    if not cookie_str:
        return ""
    names = []
    for seg in cookie_str.split(";"):
        name = seg.strip().split("=", 1)[0].strip()
        if name:
            names.append(name)
    head = cookie_str.strip()[:6]
    tail = cookie_str.strip()[-4:]
    return f"{head}…{tail}（共 {len(names)} 个 cookie 字段，{len(cookie_str)} 字符）"


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
<style>body{background:#fbfbfd;font-family:'Inter',system-ui,sans-serif;color:#18181b}
.inp{border:1px solid #e8e8ec;border-radius:10px;padding:11px 13px;width:100%;outline:none;font-size:15px}
.inp:focus{border-color:#4f46e5;box-shadow:0 0 0 3px rgba(79,70,229,.12)}
.btn{background:#4f46e5;color:#fff;border-radius:10px;font-weight:600;padding:11px;width:100%;transition:.15s}
.btn:hover{background:#4338ca}</style>
</head><body class="min-h-screen flex items-center justify-center px-4">
  <div class="w-full max-w-sm">
    <div class="flex items-center gap-3 mb-6 justify-center">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none"><rect width="40" height="40" rx="11" fill="url(#lg)"/>
      <path d="M11 16 H27" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/><path d="M23 12 L28.5 16 L23 20" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M29 24 H13" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-opacity=".92"/><path d="M17 20 L11.5 24 L17 28" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".92"/>
      <defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs></svg>
      <span class="text-lg font-bold tracking-tight">agentplatform2api</span>
    </div>
    <div class="bg-white border border-neutral-200 rounded-2xl p-6 shadow-sm">
      <p class="text-sm text-neutral-500 mb-4">请输入访问密码（即 API_KEY）</p>
      <form id="f" onsubmit="return doLogin(event)">
        <input id="pw" type="password" class="inp mb-3" placeholder="密码" autofocus autocomplete="current-password">
        <button class="btn" type="submit">进入控制台</button>
      </form>
      <p id="err" class="text-xs text-rose-600 mt-3 h-4"></p>
    </div>
  </div>
<script>
async function doLogin(e){
  e.preventDefault();
  const pw=document.getElementById('pw').value;
  const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
  if(r.ok){ location.href='/'; } else { document.getElementById('err').textContent='密码错误，请重试'; }
  return false;
}
</script>
</body></html>
"""


# ==========================================
# 控制台（浅色 Vercel 风格，单文件，免构建）
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
  :root { --border:#e8e8ec; --muted:#6b7280; --fg:#18181b; --bg:#fbfbfd; --accent:#4f46e5; --accent2:#7c3aed; --accent-hover:#4338ca; }
  * { -webkit-font-smoothing:antialiased; }
  body { background:var(--bg); color:var(--fg); font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }
  .card { background:#fff; border:1px solid var(--border); border-radius:14px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
  .lbl { font-size:12px; color:var(--muted); font-weight:500; }
  .val { font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .btn { background:var(--accent); color:#fff; border-radius:9px; font-weight:600; transition:.15s; box-shadow:0 1px 2px rgba(79,70,229,.25); }
  .btn:hover { background:var(--accent-hover); }
  .btn:disabled { opacity:.5; cursor:not-allowed; }
  .btn-ghost { background:#fff; color:var(--fg); border:1px solid var(--border); border-radius:9px; font-weight:500; }
  .btn-ghost:hover { background:#f7f7f9; }
  .inp { border:1px solid var(--border); border-radius:9px; background:#fff; font-size:14px; padding:8px 10px; width:100%; outline:none; transition:.15s; color:var(--fg); }
  .inp:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(79,70,229,.12); }
  .inp:disabled { background:#f4f4f6; color:#a1a1aa; }
  .tab { padding:14px 4px; font-size:14px; font-weight:500; color:var(--muted); border-bottom:2px solid transparent; cursor:pointer; white-space:nowrap; }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab:hover:not(.active){ color:var(--fg); }
  .pill { display:inline-flex; align-items:center; gap:6px; font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--border); background:#fff; color:#52525b; }
  .pill-accent { border-color:transparent; background:rgba(79,70,229,.08); color:var(--accent); font-weight:600; }
  .log { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12.5px; word-break:break-all; }
  ::-webkit-scrollbar{width:8px;height:8px}
  ::-webkit-scrollbar-thumb{background:#dcdce3;border-radius:8px}
  ::-webkit-scrollbar-thumb:hover{background:#c0c0ca}
  .toast { position:fixed; bottom:22px; left:50%; transform:translateX(-50%); background:var(--fg); color:#fff; padding:10px 18px; border-radius:10px; font-size:14px; opacity:0; transition:.25s; pointer-events:none; z-index:50; }
  .toast.show { opacity:1; }
  .switch { position:relative; width:40px; height:22px; }
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#e4e4e7;border-radius:999px;transition:.2s;cursor:pointer}
  .slider:before{content:"";position:absolute;height:16px;width:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.2s;box-shadow:0 1px 2px rgba(0,0,0,.2)}
  input:checked + .slider{background:var(--accent)}
  input:checked + .slider:before{transform:translateX(18px)}
  .hero { background:linear-gradient(180deg,#fff, #fbfbfd); border:1px solid var(--border); border-radius:16px; }
</style>
</head>
<body class="min-h-screen">
<div class="max-w-5xl mx-auto px-5 md:px-8 py-6">

  <!-- Header -->
  <header class="hero px-5 py-4 mb-6 flex items-center justify-between flex-wrap gap-3">
    <div class="flex items-center gap-3.5">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
        <rect width="40" height="40" rx="11" fill="url(#lg)"/>
        <path d="M11 16 H27" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M23 12 L28.5 16 L23 20" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M29 24 H13" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-opacity=".92"/>
        <path d="M17 20 L11.5 24 L17 28" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" stroke-opacity=".92"/>
        <defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>
      </svg>
      <div>
        <h1 class="text-xl font-bold tracking-tight leading-none">agentplatform<span style="color:var(--accent)">2api</span></h1>
        <p class="text-xs text-neutral-500 mt-1.5">OpenAI 兼容代理 · Gemini Agent Platform 双通道</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <span class="pill pill-accent" id="mode-pill">通道 —</span>
      <span class="pill"><span class="relative flex h-2 w-2"><span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span><span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span></span><span id="uptime">运行中</span></span>
      <button onclick="logout()" class="pill" style="cursor:pointer" title="退出登录">退出</button>
    </div>
  </header>

  <!-- Tabs -->
  <nav class="flex gap-6 border-b border-neutral-200 mb-6 overflow-x-auto">
    <div class="tab active" data-tab="overview" onclick="switchTab('overview')">数据概览</div>
    <div class="tab" data-tab="channel" onclick="switchTab('channel')">通道与凭证</div>
    <div class="tab" data-tab="params" onclick="switchTab('params')">模型参数</div>
    <div class="tab" data-tab="logs" onclick="switchTab('logs')">运行日志</div>
  </nav>

  <!-- Overview -->
  <section id="view-overview">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
      <div class="card p-4"><div class="lbl">总请求</div><div id="s-total" class="val text-2xl font-bold mt-1">0</div></div>
      <div class="card p-4"><div class="lbl">成功响应</div><div id="s-success" class="val text-2xl font-bold mt-1 text-emerald-600">0</div></div>
      <div class="card p-4"><div class="lbl">拥堵重试</div><div id="s-retries" class="val text-2xl font-bold mt-1 text-amber-500">0</div></div>
      <div class="card p-4"><div class="lbl">错误 / 拦截</div><div id="s-error" class="val text-2xl font-bold mt-1 text-rose-600">0</div></div>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
      <div class="card p-5 flex flex-col items-center justify-center">
        <div class="lbl w-full mb-3">服务健康度</div>
        <div class="w-36 h-36"><canvas id="donut"></canvas></div>
      </div>
      <div class="card p-5 md:col-span-2">
        <div class="lbl mb-4">Token 算力消耗 <span class="text-neutral-400" style="text-transform:none;font-weight:400">· 仅标准 Express 通道计入（Cookie 直连接口不回传用量）</span></div>
        <div class="space-y-4">
          <div><div class="flex justify-between text-sm mb-1"><span class="text-neutral-600">Prompt（输入）</span><span id="t-prompt" class="val font-semibold">0</span></div><div class="w-full bg-neutral-100 rounded-full h-1.5"><div class="bg-black h-1.5 rounded-full" style="width:70%"></div></div></div>
          <div><div class="flex justify-between text-sm mb-1"><span class="text-neutral-600">Completion（输出）</span><span id="t-comp" class="val font-semibold">0</span></div><div class="w-full bg-neutral-100 rounded-full h-1.5"><div class="bg-neutral-400 h-1.5 rounded-full" style="width:45%"></div></div></div>
          <div class="pt-4 border-t border-neutral-100 flex justify-between items-center"><span class="lbl">总计</span><span id="t-total" class="val text-xl font-bold">0</span></div>
        </div>
      </div>
    </div>
  </section>

  <!-- Channel -->
  <section id="view-channel" class="hidden">
    <div class="card p-5 mb-4">
      <div class="lbl mb-3">上游调用通道</div>
      <div class="flex flex-col sm:flex-row gap-3">
        <label class="flex-1 border border-neutral-200 rounded-lg p-3 cursor-pointer flex items-start gap-3 hover:bg-neutral-50" id="opt-api">
          <input type="radio" name="mode" value="api_key" class="mt-1" onchange="updateMode('api_key')">
          <div><div class="font-medium text-sm">Express API Key（标准）</div><div class="text-xs text-neutral-500 mt-0.5">用 VERTEX_EXPRESS_API_KEY 调官方 SDK</div></div>
        </label>
        <label class="flex-1 border border-neutral-200 rounded-lg p-3 cursor-pointer flex items-start gap-3 hover:bg-neutral-50" id="opt-web">
          <input type="radio" name="mode" value="web_proxy" class="mt-1" onchange="updateMode('web_proxy')">
          <div><div class="font-medium text-sm">Cookie 直连反代</div><div class="text-xs text-neutral-500 mt-0.5">用控制台 Cookie 调 batchGraphql</div></div>
        </label>
      </div>
    </div>
    <div id="cookie-box" class="card p-5 hidden">
      <div class="lbl mb-2">Google Cookie（含 HttpOnly 字段）</div>
      <textarea id="cookie-input" rows="3" class="inp log mb-3" placeholder="粘贴 console.cloud.google.com 的完整 Cookie（支持 Cookie-Editor 导出的 JSON / Header String，自动解析）"></textarea>
      <div class="lbl mb-2">Google Cloud Project ID</div>
      <input id="project-input" class="inp mb-3" placeholder="可直接粘贴含 ?project=xxx 的整条 URL，自动提取">
      <div class="flex items-center justify-between">
        <span class="text-xs text-neutral-500">保存后自动校验是否包含 SAPISID</span>
        <button class="btn px-4 py-2 text-sm" onclick="saveCookie()">保存并激活</button>
      </div>
      <p class="text-xs text-neutral-500 mt-3 leading-relaxed">💡 Cookie 通常较为持久（可维持数周甚至更久，取决于账号会话是否有效）；仅当出现 Permission Denied 等权限错误时再重新获取粘贴即可，并非只有 1–2 小时。</p>
    </div>
  </section>

  <!-- Params -->
  <section id="view-params" class="hidden">
    <div class="card p-5 mb-4">
      <div class="flex items-center justify-between flex-wrap gap-3">
        <div>
          <div class="lbl mb-1">选择模型 <span id="ov-badge" class="pill pill-accent hidden" style="text-transform:none">已有专属参数</span></div>
          <select id="model-sel" class="inp" style="min-width:240px" onchange="renderCaps()"></select>
        </div>
        <div id="caps-summary" class="text-xs text-neutral-600 flex flex-wrap gap-2 max-w-xl"></div>
      </div>
      <p class="text-xs text-neutral-500 mt-3">下方参数默认为<b>全局默认值</b>（对所有模型生效）。可为<b>当前所选模型</b>单独保存专属值：<b>思考、生图、采样默认</b>这三类支持按模型覆盖。优先级：<b>单次请求 &gt; 模型专属 &gt; 全局默认 &gt; 内置</b>，最后仍按模型能力自动裁剪。</p>
      <div class="flex items-center gap-2 mt-3 flex-wrap">
        <button class="btn px-4 py-2 text-sm" onclick="saveModelOverride()">💾 保存为该模型专属</button>
        <button class="px-4 py-2 text-sm rounded-lg border border-neutral-300 hover:bg-neutral-50" onclick="clearModelOverride()">清除该模型专属</button>
        <span id="ov-hint" class="text-xs text-neutral-500"></span>
      </div>
    </div>

    <div class="grid md:grid-cols-2 gap-4">
      <!-- 思考 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">思考强度</div>
        <div class="mb-3" id="wrap-g3level">
          <div class="lbl mb-1">Gemini 3.x 思考档位（thinking_level）</div>
          <select id="thinking_g3_level" class="inp"><option value="">自动（按模型默认）</option><option value="minimal">minimal</option><option value="low">low</option><option value="medium">medium</option><option value="high">high</option></select>
        </div>
        <div id="wrap-g25budget">
          <div class="lbl mb-1">Gemini 2.5 思考预算（thinking_budget，-1=动态，0=关闭*）</div>
          <input id="thinking_g25_budget" type="number" class="inp" placeholder="-1">
        </div>
        <p id="think-note" class="text-xs text-neutral-500 mt-2"></p>
        <div class="mt-3 pt-3 border-t border-neutral-100">
          <div class="lbl mb-1">原生思考控制（native_thinking_mode）</div>
          <select id="native_thinking_mode" class="inp">
            <option value="request">跟随请求（默认）</option>
            <option value="off">关闭原生思考（角色扮演推荐）</option>
            <option value="console">强制用上方档位（忽略前端）</option>
          </select>
          <p class="text-xs text-neutral-500 mt-2 leading-relaxed">🎭 <b>酒馆预设"卡原生思维链"选“关闭原生思考”即可（可用“保存为该模型专属”只对 3.6-flash 生效）。</b><br>• <b>跟随请求</b>：用前端发来的 <code>reasoning_effort</code>；SillyTavern 常发 <code>xhigh</code> → 高强度原生思考。<br>• <b>关闭原生思考</b>：忽略前端参数，把档位压到该模型最低（3.x=minimal、2.5-flash 预算 0），并<b>不返回思考</b>。⚠️ 实测 Studio(batchGraphql) 会忽略 includeThoughts，故本项会在<b>压到 minimal</b> 的同时于响应侧<b>剥离思考块</b>——这也是重预设下 3.6-flash 能出正文（不被思考阶段截断）的关键。<br>• <b>强制用上方档位</b>：忽略前端，用你在上面选的档位（返回思考）。<br>（预填充预设另见"开关 &amp; 预填充"卡片的压制开关。）</p>
        </div>
      </div>

      <!-- 生图 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">生图</div>
        <div class="mb-3">
          <div class="lbl mb-1">默认分辨率（image_size）</div>
          <select id="image_size" class="inp"><option value="512">512</option><option value="1K">1K</option><option value="2K">2K</option><option value="4K">4K</option></select>
        </div>
        <div>
          <div class="lbl mb-1">默认宽高比（aspect_ratio）</div>
          <select id="image_aspect_ratio" class="inp"></select>
        </div>
        <p id="image-note" class="text-xs text-neutral-500 mt-2"></p>
      </div>

      <!-- 采样默认 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">采样默认值 <span class="text-xs font-normal text-neutral-400">（留空=不注入，交给模型默认）</span></div>
        <div class="grid grid-cols-3 gap-3">
          <div><div class="lbl mb-1">temperature</div><input id="default_temperature" type="number" step="0.1" class="inp" placeholder="—"></div>
          <div><div class="lbl mb-1">top_p</div><input id="default_top_p" type="number" step="0.05" class="inp" placeholder="—"></div>
          <div><div class="lbl mb-1">max_tokens</div><input id="default_max_tokens" type="number" class="inp" placeholder="—"></div>
        </div>
        <p id="sampling-note" class="text-xs text-neutral-500 mt-2"></p>
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

      <!-- 开关 -->
      <div class="card p-5">
        <div class="text-sm font-semibold mb-3">开关 & 预填充</div>
        <div class="space-y-3">
          <div class="flex items-center justify-between"><span class="text-sm">假流式（fake streaming）</span><label class="switch"><input type="checkbox" id="fake_streaming"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between"><span class="text-sm">假流式心跳间隔(秒)</span><input id="fake_streaming_interval" type="number" step="0.5" class="inp" style="width:90px"></div>
          <div class="flex items-center justify-between"><span class="text-sm">多 Key 轮询（round-robin）</span><label class="switch"><input type="checkbox" id="roundrobin"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between"><span class="text-sm">输出附加安全分</span><label class="switch"><input type="checkbox" id="safety_score"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between"><span class="text-sm">出站参数调试日志 <span class="text-xs text-neutral-400">（两条通道都打印实际发出的思考/采样参数，实机验证必开）</span></span><label class="switch"><input type="checkbox" id="debug_outbound"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between"><span class="text-sm">Cookie 通道额外诊断 <span class="text-xs text-neutral-400">（无正文时的原始响应样本总是自动记录，无需开启）</span></span><label class="switch"><input type="checkbox" id="cookie_debug"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between"><span class="text-sm">生图下发 system 指令 <span class="text-xs text-neutral-400">（默认关＝沿用旧行为；开启前请先真机验证目标模型）</span></span><label class="switch"><input type="checkbox" id="image_system_instruction"><span class="slider"></span></label></div>
          <div class="flex items-center justify-between gap-3"><span class="text-sm">预填充兼容模式 <span class="text-xs text-neutral-400">（3.x 起官方取消预填充。「保留模型轮次」把预填充留在模型自己的轮次里，预设思维链才不会被当成参考文本）</span></span>
            <select id="prefill_mode" class="inp" style="width:190px"><option value="keep_turn">保留模型轮次（默认）</option><option value="smart">智能（旧行为）</option><option value="minimal">最小</option><option value="off">关闭</option></select>
          </div>
          <div class="flex items-center justify-between"><span class="text-sm">预填充时压制原生思考 <span class="text-xs text-neutral-400">（卡思维链）</span></span><label class="switch"><input type="checkbox" id="prefill_suppress_thinking"><span class="slider"></span></label></div>
          <div>
            <div class="lbl mb-1">续写指令模板 <span class="text-xs text-neutral-400">（留空=内置默认。「智能」模式下预填充文本会附在其后；「保留模型轮次」模式下它只是末尾那句推动语）</span></div>
            <textarea id="prefill_instruction" rows="2" class="inp log" placeholder="[继续输出] 下面是你这条回复已经写好的开头，请从断点处无缝继续，不要重复开头内容，也不要添加任何前言、解释或标注："></textarea>
          </div>
          <p class="text-xs text-neutral-500 leading-relaxed">智能模式：2.5 及更早模型<b>原生透传</b>预填充（模型直接续写）；3.x 拒绝 model 结尾，自动转为末尾 user 续写指令。两者都会把预填充拼回输出开头并自动去重。压制思考时：3.x 压至最低档并不回传思考（无法全关），2.5-flash 预算 0 全关、2.5-pro 降至 128。<b>注意：预填充压制仅在请求真的带预填充时触发；若你的酒馆预设把思维链写在 system 提示里（无预填充），请改用上方“思考强度”卡片的“强制用此设置”。</b></p>
        </div>
      </div>
    </div>

    <div class="flex items-center justify-end gap-3 mt-4">
      <span class="text-xs text-neutral-500">此按钮保存<b>全局默认 + 基础设施项</b>，不影响已设专属参数的模型</span>
      <button class="btn px-5 py-2.5 text-sm" onclick="saveSettings()">保存全局设置</button>
    </div>
  </section>

  <!-- Logs -->
  <section id="view-logs" class="hidden">
    <div class="card overflow-hidden">
      <div class="px-4 py-2.5 border-b border-neutral-200 flex items-center gap-2">
        <div class="flex gap-1.5"><div class="w-3 h-3 rounded-full bg-rose-400"></div><div class="w-3 h-3 rounded-full bg-amber-400"></div><div class="w-3 h-3 rounded-full bg-emerald-400"></div></div>
        <span class="ml-2 text-xs text-neutral-400 log">terminal · 实时监控</span>
      </div>
      <div id="logwin" class="log p-4 space-y-1.5 overflow-y-auto bg-[#fbfbfb]" style="height:60vh"></div>
    </div>
  </section>
</div>
<div id="toast" class="toast"></div>

<script>
const $ = id => document.getElementById(id);
let CAPS = {}, chart = null, curAR = "";
let GLOBAL_SETTINGS = {}, OVERRIDES = {};
const PER_MODEL_KEYS = ['native_thinking_mode','thinking_g3_level','thinking_g25_budget','image_size','image_aspect_ratio','default_temperature','default_top_p','default_max_tokens'];
const COMMON_ARS = ["1:1","3:2","2:3","3:4","4:3","4:5","5:4","9:16","16:9","21:9","1:4","4:1","1:8","8:1","9:21"];

function toast(m){ const t=$('toast'); t.textContent=m; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),1800); }
function fmt(n){ return (n||0).toLocaleString('en-US'); }
async function logout(){ try{ await fetch('/api/logout',{method:'POST'}); }catch(e){} location.href='/'; }

function switchTab(t){
  document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active', e.dataset.tab===t));
  ['overview','channel','params','logs'].forEach(v=>$('view-'+v).classList.toggle('hidden', v!==t));
}

/* ---------- Overview ---------- */
function renderChart(s,e,r){
  const ctx=$('donut').getContext('2d');
  let data=[s,e,r], colors=['#171717','#e11d48','#f59e0b'];
  if(s===0&&e===0&&r===0){ data=[1]; colors=['#ededed']; }
  if(chart){ chart.data.datasets[0].data=data; chart.data.datasets[0].backgroundColor=colors; chart.update(); return; }
  chart=new Chart(ctx,{type:'doughnut',data:{labels:['成功','错误','重试'],datasets:[{data,backgroundColor:colors,borderWidth:2,borderColor:'#fff'}]},options:{cutout:'72%',plugins:{legend:{display:false}}}});
}
async function fetchStats(){
  try{
    const d=await (await fetch('/api/stats')).json();
    $('s-total').textContent=fmt(d.total); $('s-success').textContent=fmt(d.success);
    $('s-error').textContent=fmt(d.error); $('s-retries').textContent=fmt(d.retries);
    $('t-prompt').textContent=fmt(d.prompt_tokens); $('t-comp').textContent=fmt(d.completion_tokens);
    $('t-total').textContent=fmt((d.prompt_tokens||0)+(d.completion_tokens||0));
    $('uptime').textContent='已运行 '+(d.uptime/3600).toFixed(1)+' h';
    renderChart(d.success,d.error,d.retries);
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
    /* S-1：后端只返回掩码，绝不回填到输入框（否则保存时会把真实 Cookie 覆盖成掩码）。
       输入框留空 = 保持现有 Cookie；填了才更新。 */
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

/* ---------- Params ---------- */
function setV(id,v){ const el=$(id); if(!el) return; if(el.type==='checkbox') el.checked=!!v; else el.value=(v===null||v===undefined)?'':v; }
async function loadParams(){
  try{
    const s=await (await fetch('/api/settings')).json();
    GLOBAL_SETTINGS=s;
    curAR = s.image_aspect_ratio || "";
    ['native_thinking_mode','thinking_g3_level','thinking_g25_budget','image_size','default_temperature','default_top_p','default_max_tokens','img_compress_max_dim','img_compress_max_mb','img_compress_quality','retry_max','retry_backoff_seconds','fake_streaming_interval','prefill_mode','prefill_instruction'].forEach(k=>setV(k,s[k]));
    ['img_compress_enabled','fake_streaming','roundrobin','safety_score','cookie_debug','debug_outbound','prefill_suppress_thinking','image_system_instruction'].forEach(k=>setV(k,s[k]));
    // 向后兼容：旧版布尔开关映射到新的 native_thinking_mode 下拉
    if((!s.native_thinking_mode || s.native_thinking_mode==='request')){
      if(s.hide_thoughts) setV('native_thinking_mode','off');
      else if(s.thinking_force_console) setV('native_thinking_mode','console');
    }
  }catch(e){}
  try{
    const c=await (await fetch('/api/capabilities')).json();
    CAPS=c.capabilities||{};
    OVERRIDES=c.overrides||{};
    $('model-sel').innerHTML=(c.models||[]).map(m=>{const star=OVERRIDES[m]?' ★':''; return `<option value="${m}">${m}${star}</option>`;}).join('');
    renderCaps();
  }catch(e){}
}
// 按所选模型把“可覆盖的 7 个参数字段”填成：有专属值用专属，否则回退全局
function applyModelParamFields(model){
  const ov = OVERRIDES[model] || {};
  PER_MODEL_KEYS.forEach(k=>{
    const v = (k in ov) ? ov[k] : GLOBAL_SETTINGS[k];
    if(k==='image_aspect_ratio'){ curAR = v || ""; }
    else setV(k, v);
  });
  const has = !!OVERRIDES[model] && Object.keys(OVERRIDES[model]).length>0;
  $('ov-badge').classList.toggle('hidden', !has);
  $('ov-hint').textContent = has ? '当前显示该模型的专属值（带 ★）；改动后点“保存为该模型专属”更新。' : '当前显示全局默认值；改动后点“保存为该模型专属”即可只对此模型生效。';
}
async function saveModelOverride(){
  const m=$('model-sel').value; if(!m) return;
  const patch={
    native_thinking_mode:$('native_thinking_mode').value,
    thinking_g3_level:$('thinking_g3_level').value,
    thinking_g25_budget:numOr('thinking_g25_budget',-1),
    image_size:$('image_size').value,
    image_aspect_ratio:$('image_aspect_ratio').value,
    default_temperature:numOrNull('default_temperature'),
    default_top_p:numOrNull('default_top_p'),
    default_max_tokens:numOrNull('default_max_tokens'),
  };
  try{
    const r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    const d=await r.json();
    if(r.ok){ OVERRIDES[m]=d.override||patch; toast('已保存 '+m+' 专属参数'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}
async function clearModelOverride(){
  const m=$('model-sel').value; if(!m) return;
  if(!OVERRIDES[m]){ toast('该模型没有专属参数'); return; }
  try{
    const r=await fetch('/api/model-overrides/'+encodeURIComponent(m),{method:'DELETE'});
    if(r.ok){ delete OVERRIDES[m]; toast('已清除 '+m+' 专属参数，回退全局'); refreshModelSelStars(); applyModelParamFields(m); }
    else toast('清除失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}
function refreshModelSelStars(){
  const sel=$('model-sel'); const cur=sel.value;
  [...sel.options].forEach(o=>{ const base=o.value; o.textContent = base + (OVERRIDES[base]?' ★':''); });
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
  const m=$('model-sel').value; const cap=CAPS[m]; if(!cap) return;
  applyModelParamFields(m);   // 按模型填入专属/全局参数值（含 curAR）
  const th=cap.thinking||{};
  let sum=[chip('家族 '+cap.family, true)];
  if(th.kind==='level') sum.push(chip('思考档位 '+(th.levels||[]).join(' / ')));
  else if(th.kind==='budget') sum.push(chip('思考预算 '+th.budget_min+'~'+th.budget_max));
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

  const isLevel=th.kind==='level', isBudget=th.kind==='budget';
  $('thinking_g3_level').disabled=!isLevel;
  $('thinking_g25_budget').disabled=!isBudget;
  $('wrap-g3level').style.opacity=isLevel?'1':'.4';
  $('wrap-g25budget').style.opacity=isBudget?'1':'.4';
  let tn='';
  if(cap.is_image) tn='生图模型不接受思考参数（模型内部自行思考）。';
  else if(isLevel) tn='该模型用思考档位' + ((th.levels && !th.levels.includes('minimal')) ? '（此模型最低 low，无 minimal，且无法关闭）' : '（无法完全关闭，最省为 minimal）') + '。';
  else if(isBudget) tn=(cap.family==='g25' && m.includes('pro')) ? '2.5 Pro 最低预算 128，无法设 0 关闭；-1 为动态。' : '2.5 Flash 可设 0 关闭思考，-1 为动态。';
  else tn='该模型不支持思考调节。';
  $('think-note').textContent=tn;

  const imgOn=cap.is_image;
  $('image_size').disabled=!imgOn; $('image_aspect_ratio').disabled=!imgOn;
  fillARFor(cap);
  $('image-note').textContent = imgOn
    ? ('该模型支持比例：' + cap.image_aspect_ratios.join('、') + '；分辨率 ' + cap.image_sizes.join('/') + '。选到不支持的比例不会报错，会自动回退为“由模型决定”。')
    : '仅生图模型（名称含 image）使用此项。';

  $('sampling-note').textContent = cap.is_image
    ? '生图模型会自动剥离采样参数。'
    : (cap.sampling_advice==='deprecated'
        ? '⚠️ 该模型已废弃 temperature/top_p/top_k（官方要求移除，现忽略、未来 400），代理会自动移除；如需更确定的输出请改用系统指令。candidate_count 在 3.x 也不支持。'
        : cap.sampling_advice==='recommend_default'
          ? '官方建议 Gemini 3.x 保持采样默认值（可调，但改动可能致循环/降智）；留空即用模型默认。'
          : '该模型支持采样参数；留空即用模型默认。');
}
function numOrNull(id){ const v=$(id).value.trim(); return v===''?null:Number(v); }
function numOr(id,d){ const v=$(id).value.trim(); return v===''?d:Number(v); }
async function saveSettings(){
  // 基础设施级设置：始终作为全局保存
  const patch={
    img_compress_enabled:$('img_compress_enabled').checked,
    img_compress_max_dim:numOr('img_compress_max_dim',1536),
    img_compress_max_mb:numOr('img_compress_max_mb',1.5),
    img_compress_quality:numOr('img_compress_quality',85),
    retry_max:numOr('retry_max',10),
    retry_backoff_seconds:numOr('retry_backoff_seconds',5),
    fake_streaming:$('fake_streaming').checked,
    fake_streaming_interval:numOr('fake_streaming_interval',1),
    roundrobin:$('roundrobin').checked,
    safety_score:$('safety_score').checked,
    cookie_debug:$('cookie_debug').checked,
    debug_outbound:$('debug_outbound').checked,
    image_system_instruction:$('image_system_instruction').checked,
    prefill_mode:$('prefill_mode').value,
    prefill_suppress_thinking:$('prefill_suppress_thinking').checked,
    prefill_instruction:$('prefill_instruction').value,
  };
  // 7 个可覆盖参数：仅当所选模型“没有专属配置”时，才作为全局默认保存，
  // 避免把某模型的专属值误存成全局（专属值请用“保存为该模型专属”）。
  const m=$('model-sel').value;
  const overriding = !!(OVERRIDES[m] && Object.keys(OVERRIDES[m]).length>0);
  if(!overriding){
    Object.assign(patch,{
      native_thinking_mode:$('native_thinking_mode').value,
      thinking_g3_level:$('thinking_g3_level').value,
      thinking_g25_budget:numOr('thinking_g25_budget',-1),
      image_size:$('image_size').value,
      image_aspect_ratio:$('image_aspect_ratio').value,
      default_temperature:numOrNull('default_temperature'),
      default_top_p:numOrNull('default_top_p'),
      default_max_tokens:numOrNull('default_max_tokens'),
    });
  }
  const scope = overriding
    ? '将保存全局的基础设施项（图压缩/重试/假流式/预填充/思考控制开关等）。\n当前所选模型有专属思考/生图/采样参数，不会被改动。'
    : '将保存为全局默认，影响所有【未设置专属参数】的模型。\n已设专属参数的模型不受影响。';
  if(!confirm(scope + '\n\n确定保存全局设置吗？')) return;
  try{
    const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
    if(r.ok){ GLOBAL_SETTINGS=await r.json(); toast(overriding?'已保存全局(基础设施)设置；该模型的思考/生图/采样为专属值，未改全局':'全局设置已保存'); }
    else toast('保存失败');
  }catch(e){ toast('❌ 网络请求失败'); }
}

/* ---------- Logs ---------- */
const logwin=$('logwin'); let autoscroll=true;
logwin.addEventListener('scroll',()=>{ autoscroll = logwin.scrollHeight-logwin.scrollTop-logwin.clientHeight<50; });
function logLine(t){
  let c='#525252',bg='transparent',bl='2px solid transparent';
  if(t.includes('✅')||t.includes('🎉')){c='#0369a1';bl='2px solid #38bdf8';}
  else if(t.includes('⚠️')||t.includes('WARN')||t.includes('🔄')||t.includes('重试')){c='#b45309';bg='#fffbeb';bl='2px solid #f59e0b';}
  else if(t.includes('❌')||t.includes('ERROR')){c='#be123c';bg='#fef2f2';bl='2px solid #f43f5e';}
  else if(t.includes('💰')){c='#6d28d9';bg='#faf5ff';bl='2px solid #a855f7';}
  let s=t.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/(gemini-[a-zA-Z0-9.\-]+)/g,'<span style="color:#059669;font-weight:600">$1</span>');
  return `<div style="color:${c};background:${bg};border-left:${bl};padding:5px 9px;border-radius:4px">${s}</div>`;
}
try{
  const es=new EventSource('/stream-logs');
  es.onmessage=e=>{ if(e.data.includes('keep-alive')) return; logwin.insertAdjacentHTML('beforeend',logLine(e.data)); if(autoscroll) logwin.scrollTop=logwin.scrollHeight; };
}catch(e){}

/* ---------- Project ID auto-extract ---------- */
$('project-input').addEventListener('input',e=>{ const v=e.target.value.trim(); const m=v.match(/[?&]project=([^&]+)/)||v.match(/\/projects\/([^\/]+)/); if(m) e.target.value=m[1]; });
$('image_aspect_ratio').addEventListener('change', e=>{ curAR=e.target.value; });

/* init */
fetchStats(); setInterval(fetchStats,3000); loadRuntime(); loadParams();
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

    # P2-2：失败三次后指数退避，避免对口令（同时也是 API Key）无限爆破
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
            # 反代后 request.url.scheme 可能是 http，这里同时看 x-forwarded-proto
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
        # S-1：只回显掩码。完整 Cookie 等价于该 Google 账号的完整访问权，
        # 没有任何理由让它出现在前端 JS / 浏览器缓存 / 截图里。
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
    caps = {m: mc.capabilities_summary(m) for m in models}
    # 附带各模型是否已有专属参数覆盖，供前端标示
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


class CookieSetting(BaseModel):
    cookie: str = ""          # 留空 = 保持现有 Cookie
    project_id: str = ""


@app.post("/api/cookie")
async def set_google_cookie(setting: CookieSetting, _auth: bool = Depends(require_auth)):
    """保存 Cookie 与 Project ID。

    S-1：cookie 传空字符串表示「保持现有 Cookie 不变，只更新 Project ID」，
    这样前端就不需要为了改 Project ID 而把完整 Cookie 再取回来一次。
    """
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
    async def log_generator():
        q = rt_logger.subscribe()
        try:
            for msg in rt_logger.snapshot_history():
                yield f"data: {msg}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive heartbeat\n\n"
        finally:
            rt_logger.unsubscribe(q)
    return StreamingResponse(log_generator(), media_type="text/event-stream")


app.include_router(models_api.router)
app.include_router(chat_api.router)
