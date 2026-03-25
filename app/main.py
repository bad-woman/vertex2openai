import asyncio
import secrets
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware

from routes import chat_api, models_api
from credentials_manager import CredentialManager
from express_key_manager import ExpressKeyManager

# [重要新增] 导入刚刚写的拦截器，只要导入就会自动接管 print
from logger import rt_logger 
import config

# 初始化 FastAPI
app = FastAPI(title="Vertex2OpenAI 中间件", version="1.0.0")

# 你的跨域设置...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= 神性防火墙 (鉴权机制) =======
security = HTTPBasic()

def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    # 账号名可以随便填（比如 admin），密码必须与 config.py 中的 API_KEY 完全一致
    is_correct_password = secrets.compare_digest(credentials.password, config.API_KEY)
    if not is_correct_password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. 连本小姐的密码都记错了吗？",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ======= 前端 Web 监控 UI =======
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Vertex2OpenAI | 神性监控面板</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .log-container { font-family: 'Fira Code', monospace; scroll-behavior: smooth; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        .log-info { color: #38bdf8; }
        .log-warn { color: #fbbf24; font-weight: 500; }
        .log-error { color: #ef4444; font-weight: bold; }
        .log-success { color: #34d399; }
    </style>
</head>
<body class="h-screen flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-5xl bg-slate-800 rounded-xl shadow-2xl overflow-hidden border border-slate-700 flex flex-col h-[85vh]">
        <div class="bg-slate-900 px-6 py-4 border-b border-slate-700 flex justify-between items-center shadow-md z-10">
            <div class="flex items-center gap-3">
                <div class="flex gap-2">
                    <div class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></div>
                    <div class="w-3 h-3 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]"></div>
                    <div class="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
                </div>
                <h1 class="text-lg font-semibold text-slate-200 ml-4 tracking-wider">Vertex2OpenAI / 运行状态中枢</h1>
            </div>
            <div class="flex items-center gap-2 bg-slate-800 px-3 py-1 rounded-full border border-slate-600">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <span class="text-sm text-green-400 font-medium">代理监听中</span>
            </div>
        </div>
        <div id="log-window" class="log-container p-6 flex-1 overflow-y-auto text-sm space-y-1.5 break-all">
            </div>
    </div>
    <script>
        const logWindow = document.getElementById('log-window');
        const eventSource = new EventSource('/stream-logs');
        let isAutoScroll = true;

        logWindow.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = logWindow;
            isAutoScroll = scrollHeight - scrollTop - clientHeight < 50;
        });

        function formatLog(msg) {
            let html = msg.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            // 神性色彩高亮引擎
            if (html.includes('INFO:') || html.includes('DEBUG:')) html = `<span class="log-info">${html}</span>`;
            else if (html.includes('WARNING:') || html.includes('⚠️')) html = `<span class="log-warn">${html}</span>`;
            else if (html.includes('ERROR:') || html.includes('❌') || html.includes('Exception')) html = `<span class="log-error">${html}</span>`;
            else if (html.includes('200 OK') || html.includes('SUCCESS')) html = `<span class="log-success">${html}</span>`;
            else html = `<span class="text-slate-400">${html}</span>`;
            return `<div>${html}</div>`;
        }

        eventSource.onmessage = function(event) {
            logWindow.insertAdjacentHTML('beforeend', formatLog(event.data));
            if (isAutoScroll) logWindow.scrollTop = logWindow.scrollHeight;
        };

        eventSource.onerror = function(err) {
            logWindow.insertAdjacentHTML('beforeend', formatLog("[系统] ❌ SSE 链接断开，试图重新连接..."));
        };
    </script>
</body>
</html>
"""

# 覆盖掉以前那个丑陋的根路由
@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(username: str = Depends(verify_auth)):
    return DASHBOARD_HTML

# 实时日志流推送引擎
@app.get("/stream-logs")
async def stream_logs_endpoint(request: Request, username: str = Depends(verify_auth)):
    async def log_generator():
        q = asyncio.Queue()
        rt_logger.queues.append(q)
        try:
            # 首先吐出最近的 100 条历史记录
            for msg in rt_logger.history:
                yield f"data: {msg}\n\n"
            # 挂起等待新的日志流入
            while True:
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            if q in rt_logger.queues:
                rt_logger.queues.remove(q)
                
    return StreamingResponse(log_generator(), media_type="text/event-stream")


# ==== 下面保留你原有的生命周期函数和路由注册逻辑 ====
# app.include_router(chat_api.router)
# ...
# Include API routers
app.include_router(models_api.router) 
app.include_router(chat_api.router)

@app.on_event("startup")
async def startup_event():
    # Check SA credentials availability
    sa_credentials_available = await init_vertex_ai(credential_manager)
    sa_count = credential_manager.get_total_credentials() if sa_credentials_available else 0
    
    # Check Express API keys availability
    express_keys_count = express_key_manager.get_total_keys()
    
    # Print detailed status
    print(f"INFO: SA credentials loaded: {sa_count}")
    print(f"INFO: Express API keys loaded: {express_keys_count}")
    print(f"INFO: Total authentication methods available: {(1 if sa_count > 0 else 0) + (1 if express_keys_count > 0 else 0)}")
    
    # Determine overall status
    if sa_count > 0 or express_keys_count > 0:
        print("INFO: Vertex AI authentication initialization completed successfully. At least one authentication method is available.")
        if sa_count == 0:
            print("INFO: No SA credentials found, but Express API keys are available for authentication.")
        elif express_keys_count == 0:
            print("INFO: No Express API keys found, but SA credentials are available for authentication.")
    else:
        print("ERROR: Failed to initialize any authentication method. Both SA credentials and Express API keys are missing. API will fail.")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "OpenAI to Gemini Adapter is running."
    }
