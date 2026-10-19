from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    API_KEY: str = "123456"
    VERTEX_EXPRESS_API_KEY: Optional[str] = None
    FAKE_STREAMING: bool = False
    FAKE_STREAMING_INTERVAL: float = 1.0
    MODELS_CONFIG_URL: str = ""
    ROUNDROBIN: bool = False
    SAFETY_SCORE: bool = False
    PROXY_URL: Optional[str] = None
    SSL_CERT_FILE: Optional[str] = None
    VERTEX_BASE_URL: Optional[str] = None

    # Cookie direct mode settings
    GOOGLE_COOKIE: Optional[str] = None
    GOOGLE_PROJECT_ID: Optional[str] = None
    EXPERIMENT_FLAGS: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_settings = AppSettings()

API_KEY = _settings.API_KEY

raw_vertex_keys = _settings.VERTEX_EXPRESS_API_KEY
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(",") if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

FAKE_STREAMING_ENABLED = _settings.FAKE_STREAMING
FAKE_STREAMING_INTERVAL_SECONDS = _settings.FAKE_STREAMING_INTERVAL
MODELS_CONFIG_URL = _settings.MODELS_CONFIG_URL
ROUNDROBIN = _settings.ROUNDROBIN
SAFETY_SCORE = _settings.SAFETY_SCORE
PROXY_URL = _settings.PROXY_URL
SSL_CERT_FILE = _settings.SSL_CERT_FILE
VERTEX_BASE_URL = _settings.VERTEX_BASE_URL

GOOGLE_COOKIE = _settings.GOOGLE_COOKIE
GOOGLE_PROJECT_ID = _settings.GOOGLE_PROJECT_ID
EXPERIMENT_FLAGS = _settings.EXPERIMENT_FLAGS

REASONING_TAG = "agent_platform_think_tag"
VERTEX_REASONING_TAG = REASONING_TAG


# ============================================================
# 控制台可调的运行时默认值（可在大盘热更新，持久化到 web_state.json）
# 优先级：单次请求 > 模型专属 > 控制台全局默认 > 代码内置兜底
# ============================================================
DEFAULT_SETTINGS = {
    # 思考
    "thinking_g3_level": "",
    "native_thinking_mode": "request",
    "thinking_force_console": False,
    "hide_thoughts": False,
    # 生图
    "image_size": "4K",
    "image_aspect_ratio": "",
    "image_system_instruction": False,
    "inject_prefill_for_image": False,
    # 采样默认（客户端未显式传时使用；None=不注入）
    "default_temperature": None,
    "default_top_p": None,
    "default_max_tokens": None,
    "sampling_policy": "auto",
    # 预填充、注入与工具抗截断（支持按模型单独保存）
    "inject_system_instruction": "",
    "inject_prefill": "",
    "prefill_instruction": "",
    "prefill_mode": "smart",
    "prefill_suppress_thinking": True,
    "prefill_cot_guard": True,
    "anti_truncation_mode": "off",
    # 输入图片压缩（全局）
    "img_compress_enabled": True,
    "img_compress_max_dim": 1536,
    "img_compress_max_mb": 1.5,
    "img_compress_quality": 85,
    # 重试（全局）
    "retry_max": 10,
    "retry_backoff_seconds": 5,
    # 开关与基础设施（全局）
    "fake_streaming": FAKE_STREAMING_ENABLED,
    "fake_streaming_interval": FAKE_STREAMING_INTERVAL_SECONDS,
    "roundrobin": ROUNDROBIN,
    "safety_score": SAFETY_SCORE,
    "debug_outbound": False,
    "cookie_debug": False,
    "express_location": "global",
    "resumable_enabled": True,
    "resumable_ttl_seconds": 1800,
    "resumable_max_streams": 200,
    # 按模型单独保存的参数覆盖：{ "模型ID": { 键: 值, ... } }
    "model_overrides": {},
}

# 允许按模型单独保存（覆盖全局默认）的参数键（严格对应控制台「① 按模型参数」区的所有控件）。
PER_MODEL_KEYS = [
    "native_thinking_mode",
    "thinking_g3_level",
    "image_size",
    "image_aspect_ratio",
    "default_temperature",
    "default_top_p",
    "default_max_tokens",
    "inject_system_instruction",
    "inject_prefill",
    "prefill_instruction",
    "image_system_instruction",
    "inject_prefill_for_image",
    "sampling_policy",
    "prefill_cot_guard",
    "prefill_mode",
    "prefill_suppress_thinking",
    "anti_truncation_mode",
]