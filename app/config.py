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

    # Cookie 直连模式配置（推荐 Render 等云端部署使用）
    GOOGLE_COOKIE: Optional[str] = None         # Google Cookie 字符串（从浏览器 DevTools 复制）
    GOOGLE_PROJECT_ID: Optional[str] = None     # Google Cloud 项目 ID（从 Console URL 中获取）
    GOOGLE_REGION: str = "us-central1"          # Vertex AI 区域，默认 us-central1
    EXPERIMENT_FLAGS: Optional[str] = None       # experimentFlagsBinary（从 F12 Network 中获取，Express Mode 权限标识）

    # 无头浏览器模式配置（仅本地部署可用）
    HEADLESS_MODE: bool = True
    CREDENTIAL_REFRESH_INTERVAL: int = 180

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

GOOGLE_COOKIE = _settings.GOOGLE_COOKIE
GOOGLE_PROJECT_ID = _settings.GOOGLE_PROJECT_ID
GOOGLE_REGION = _settings.GOOGLE_REGION
EXPERIMENT_FLAGS = _settings.EXPERIMENT_FLAGS
HEADLESS_MODE = _settings.HEADLESS_MODE
CREDENTIAL_REFRESH_INTERVAL = _settings.CREDENTIAL_REFRESH_INTERVAL

VERTEX_REASONING_TAG = "vertex_think_tag"
