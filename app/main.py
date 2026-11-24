"""
agentplatform2api 主入口（应用生命周期、中间件、静态资源挂载与三套路由注册）。

控制台静态页面位于 app/static/，后台管理 API 位于 app/routes/admin_api.py。
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
import express_key_probe
from express_key_manager import ExpressKeyManager
from logger import stats
from resumable import resumable_store
from routes import admin_api, chat_api, models_api
from routes.admin_api import DEFAULT_API_KEY
from runtime_state import app_state

express_key_manager = ExpressKeyManager()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


async def _probe_express_projects_on_startup() -> None:
    """启动后异步探测尚无 Project ID 的 Express Key（失败不影响服务）。"""
    try:
        pending = [r["key"] for r in app_state.get_express_keys() if not r.get("project_id")]
        if not pending:
            return
        print(f"🔎 [密钥探测] 启动探测：{len(pending)} 个 Key 尚无 Project ID，正在探测归属项目。")
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
        print(
            "🔴 [安全警告] API_KEY 仍是默认值 123456！它既是本代理的 Key，也是控制台登录口令，"
            "请立刻改成强口令。"
        )

    app_state.import_env_express_keys_once(config.VERTEX_EXPRESS_API_KEY_VAL)
    if express_key_manager.get_total_keys() > 0:
        print(f"✅ [密钥配置] 已加载 {express_key_manager.get_total_keys()} 个 Express API Key。")
    else:
        print(
            "⚠️ [密钥配置] 未配置任何 Express API Key（控制台「通道与凭证」页可随时添加）。"
            "若不启用 Cookie 直连模式，聊天请求将会报错。"
        )
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


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(admin_api.router)
app.include_router(models_api.router)
app.include_router(chat_api.router)
