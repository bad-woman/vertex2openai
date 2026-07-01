from typing import Optional
from google.genai import types
import config as app_config

def get_http_options(base_url: Optional[str] = None, headers: Optional[dict] = None) -> Optional[types.HttpOptions]:
    """构造 google-genai HTTP 选项，支持代理、自定义证书及自定义 Headers。"""
    client_args = {}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE

    kwargs = {}
    if headers:
        kwargs["headers"] = headers
    if base_url:
        kwargs["base_url"] = base_url
        
    if client_args:
        kwargs["client_args"] = client_args
        kwargs["async_client_args"] = client_args

    if kwargs:
        return types.HttpOptions(**kwargs)
    return None