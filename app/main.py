import os
import time
import httpx
import asyncio
import secrets
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from auth import get_api_key
from credentials_manager import CredentialManager
from express_key_manager import ExpressKeyManager
from vertex_ai_init import init_vertex_ai
from routes import models_api, chat_api

# 引入重写后的日志模块
from logger import rt_logger, stats
import config

credential_manager = CredentialManager()
express_key_manager = ExpressKeyManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_vertex_ai(credential_manager)
    yield 

app = FastAPI(title="OpenAI to Gemini Adapter", lifespan=lifespan)

# CORS 配置，修复第三方调用拦截
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.credential_manager = credential_manager
app.state.express_key_manager = express_key_manager

# 【跨域 Bug 修复】：通过后台任务或者安全包装拦截状态，不要用 try...except 吞噬跨域头
@app.middleware("http")
async def stats_tracker_middleware(request: Request, call_next):
    if "chat/completions" in request.url.path:
        response = await call_next(request)
        # 如果返回的是成功代码，则统计成功；如果不成功，统计错误
        stats.add_request(success=(response.status_code == 200))
        return response
    return await call_next(request)

security = HTTPBasic()
def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not secrets.compare_digest(credentials.password, config.API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ==========================================
# 💎 全新现代化 API 仪表盘 HTML (OneAPI 风格)
# ==========================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vertex2OpenAI | 管理控制台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0B0F19; color: #E2E8F0; font-family: 'Inter', system-ui, sans-serif; }
        .glass-panel { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.08); }
        .log-container { font-family: 'Fira Code', monospace; font-size: 0.85rem; }
        .nav-item { cursor: pointer; transition: all 0.2s; border-left: 3px solid transparent; }
        .nav-item.active { background: rgba(59, 130, 246, 0.1); border-left-color: #3B82F6; color: #60A5FA; }
        .nav-item:hover:not(.active) { background: rgba(255, 255, 255, 0.05); }
        /* 隐藏滚动条但保留功能 */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="h-screen flex overflow-hidden">
    <!-- 左侧导航栏 -->
    <aside class="w-64 glass-panel border-r border-slate-800 flex flex-col z-20">
        <div class="h-16 flex items-center px-6 border-b border-slate-800">
            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-white shadow-lg mr-3">V</div>
            <span class="font-bold text-lg tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">Vertex2OpenAI</span>
        </div>
        <nav class="flex-1 py-4 flex flex-col gap-1">
            <div onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-item active px-6 py-3 flex items-center gap-3">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                数据大盘
            </div>
            <div onclick="switchTab('logs')" id="nav-logs" class="nav-item px-6 py-3 flex items-center gap-3">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                实时日志
            </div>
            <div class="mt-auto px-6 py-4">
                <div class="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                    <div class="text-xs text-slate-400 mb-1">系统状态</div>
                    <div class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span class="text-sm text-emerald-400 font-medium">运行正常 (Running)</span>
                    </div>
                </div>
            </div>
        </nav>
    </aside>

    <!-- 右侧内容区 -->
    <main class="flex-1 flex flex-col relative z-10 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-opacity-5">
        <!-- 顶栏 -->
        <header class="h-16 glass-panel border-b border-slate-800 flex items-center justify-between px-8">
            <h1 id="page-title" class="text-lg font-semibold">数据大盘</h1>
            <div class="flex gap-3">
                <button onclick="fetchStats()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-1.5 rounded-lg text-sm transition-colors border border-slate-700 flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                    手动刷新
                </button>
            </div>
        </header>

        <!-- 内容渲染区 -->
        <div class="flex-1 overflow-y-auto p-8 relative">
            
            <!-- 视图 1：数据看板 -->
            <div id="view-dashboard" class="max-w-6xl mx-auto space-y-6">
                <!-- 顶部四个数据卡片 -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="glass-panel p-6 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-24 h-24 bg-blue-500/10 rounded-full blur-xl group-hover:bg-blue-500/20 transition-all"></div>
                        <h3 class="text-slate-400 text-sm font-medium mb-2">总请求次数</h3>
                        <p id="stat-total" class="text-3xl font-bold text-white">0</p>
                    </div>
                    <div class="glass-panel p-6 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-24 h-24 bg-emerald-500/10 rounded-full blur-xl group-hover:bg-emerald-500/20 transition-all"></div>
                        <h3 class="text-slate-400 text-sm font-medium mb-2">成功响应</h3>
                        <p id="stat-success" class="text-3xl font-bold text-emerald-400">0</p>
                    </div>
                    <div class="glass-panel p-6 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-24 h-24 bg-rose-500/10 rounded-full blur-xl group-hover:bg-rose-500/20 transition-all"></div>
                        <h3 class="text-slate-400 text-sm font-medium mb-2">拦截/异常</h3>
                        <p id="stat-error" class="text-3xl font-bold text-rose-400">0</p>
                    </div>
                    <div class="glass-panel p-6 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-24 h-24 bg-purple-500/10 rounded-full blur-xl group-hover:bg-purple-500/20 transition-all"></div>
                        <h3 class="text-slate-400 text-sm font-medium mb-2">运行时长</h3>
                        <p id="stat-uptime" class="text-3xl font-bold text-purple-400">0 h</p>
                    </div>
                </div>

                <!-- 图表与 Token 区 -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- 左侧：成功率环形图 -->
                    <div class="glass-panel p-6 rounded-2xl lg:col-span-1 flex flex-col items-center justify-center">
                        <h3 class="text-slate-400 text-sm font-medium w-full text-left mb-4">请求成功率监控</h3>
                        <div class="w-48 h-48 relative">
                            <canvas id="successChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- 右侧：Token 消耗进度条 -->
                    <div class="glass-panel p-6 rounded-2xl lg:col-span-2 flex flex-col justify-center">
                        <h3 class="text-slate-400 text-sm font-medium mb-6">Token 算力消耗量</h3>
                        <div class="space-y-6">
                            <div>
                                <div class="flex justify-between text-sm mb-2">
                                    <span class="text-slate-300 flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-blue-500"></span> Prompt Tokens (输入)</span>
                                    <span id="stat-prompt" class="font-mono text-blue-400 font-bold tracking-wider">0</span>
                                </div>
                                <div class="w-full bg-slate-800 rounded-full h-2.5"><div class="bg-gradient-to-r from-blue-600 to-blue-400 h-2.5 rounded-full" style="width: 80%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-sm mb-2">
                                    <span class="text-slate-300 flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-purple-500"></span> Completion Tokens (输出)</span>
                                    <span id="stat-comp" class="font-mono text-purple-400 font-bold tracking-wider">0</span>
                                </div>
                                <div class="w-full bg-slate-800 rounded-full h-2.5"><div class="bg-gradient-to-r from-purple-600 to-purple-400 h-2.5 rounded-full" style="width: 60%"></div></div>
                            </div>
                            <div class="pt-4 border-t border-slate-700/50 mt-4 flex justify-between items-center">
                                <span class="text-sm text-slate-400">总计消耗 (Total)</span>
                                <span id="stat-total-tokens" class="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 font-mono">0</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 视图 2：极客日志终端 -->
            <div id="view-logs" class="hidden h-full max-w-6xl mx-auto flex flex-col glass-panel rounded-2xl overflow-hidden shadow-2xl">
                <div class="bg-slate-900/80 px-4 py-2 border-b border-slate-700/50 flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full bg-rose-500"></div>
                    <div class="w-3 h-3 rounded-full bg-amber-500"></div>
                    <div class="w-3 h-3 rounded-full bg-emerald-500"></div>
                    <span class="ml-4 text-xs text-slate-500 font-mono">root@vertex-proxy:~# tail -f /var/log/api.log</span>
                </div>
                <div id="log-window" class="log-container p-4 flex-1 overflow-y-auto space-y-1 tracking-wide leading-relaxed">
                    <!-- Logs injection here -->
                </div>
            </div>

        </div>
    </main>

    <script>
        // --- 核心逻辑 ---
        let chartInstance = null;

        function formatNumber(num) { return num.toLocaleString('en-US'); }

        // 切换 Tab
        function switchTab(tabId) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            document.getElementById('nav-' + tabId).classList.add('active');
            
            document.getElementById('view-dashboard').classList.add('hidden');
            document.getElementById('view-logs').classList.add('hidden');
            document.getElementById('view-' + tabId).classList.remove('hidden');
            
            document.getElementById('page-title').innerText = tabId === 'dashboard' ? '数据大盘' : '实时日志';
        }

        // 初始化/更新图表
        function renderChart(success, error) {
            const ctx = document.getElementById('successChart').getContext('2d');
            if (chartInstance) {
                chartInstance.data.datasets[0].data = [success, error || (success === 0 ? 1 : 0)];
                chartInstance.update();
                return;
            }
            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['成功', '异常'],
                    datasets: [{
                        data: [success, error || (success === 0 ? 1 : 0)],
                        backgroundColor: ['#10B981', '#F43F5E'],
                        borderWidth: 0, hoverOffset: 4
                    }]
                },
                options: { cutout: '75%', plugins: { legend: { display: false } } }
            });
        }

        // 获取后端监控数据
        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('stat-total').innerText = formatNumber(data.total);
                document.getElementById('stat-success').innerText = formatNumber(data.success);
                document.getElementById('stat-error').innerText = formatNumber(data.error);
                
                let hours = (data.uptime / 3600).toFixed(1);
                document.getElementById('stat-uptime').innerText = hours + ' h';
                
                document.getElementById('stat-prompt').innerText = formatNumber(data.prompt_tokens);
                document.getElementById('stat-comp').innerText = formatNumber(data.completion_tokens);
                document.getElementById('stat-total-tokens').innerText = formatNumber(data.prompt_tokens + data.completion_tokens);
                
                renderChart(data.success, data.error);
            } catch (e) {
                console.error("Fetch stats failed", e);
            }
        }

        // --- 日志终端渲染逻辑 ---
        const logWindow = document.getElementById('log-window');
        let isAutoScroll = true;
        logWindow.addEventListener('scroll', () => {
            isAutoScroll = logWindow.scrollHeight - logWindow.scrollTop - logWindow.clientHeight < 50;
        });

        function formatLogText(text) {
            let color = "#94A3B8"; // Default slate
            if(text.includes("INFO") || text.includes("✅")) color = "#38BDF8";
            else if(text.includes("WARN") || text.includes("⚠️")) color = "#FBBF24";
            else if(text.includes("ERROR") || text.includes("❌")) color = "#F87171";
            else if(text.includes("💰")) color = "#D946EF";
            
            // Highlight model names
            let safeText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            safeText = safeText.replace(/(gemini-[a-zA-Z0-9\-\.]+)/g, '<span class="text-emerald-400 font-bold">$1</span>');
            return `<div style="color: ${color};">${safeText}</div>`;
        }

        const evtSource = new EventSource('/stream-logs');
        evtSource.onmessage = (e) => {
            if(e.data.includes("keep-alive heartbeat")) return;
            logWindow.insertAdjacentHTML('beforeend', formatLogText(e.data));
            if (isAutoScroll) logWindow.scrollTop = logWindow.scrollHeight;
        };

        // 启动轮询 (每 3 秒刷新一次面板数据)
        fetchStats();
        setInterval(fetchStats, 3000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(username: str = Depends(verify_auth)):
    return DASHBOARD_HTML

# 【新增】专供前端拉取统计数据的 API 接口
@app.get("/api/stats")
async def get_stats_api(username: str = Depends(verify_auth)):
    return JSONResponse(content=stats.get_json_stats())

@app.get("/stream-logs")
async def stream_logs_endpoint(request: Request, username: str = Depends(verify_auth)):
    async def log_generator():
        q = asyncio.Queue()
        rt_logger.queues.append(q)
        try:
            for msg in rt_logger.history:
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
            if q in rt_logger.queues:
                rt_logger.queues.remove(q)
    return StreamingResponse(log_generator(), media_type="text/event-stream")

app.include_router(models_api.router) 
app.include_router(chat_api.router)