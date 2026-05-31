from typing import Optional
from google.genai import types
import config as app_config


def get_http_options(base_url: Optional[str] = None) -> Optional[types.HttpOptions]:
    """构造 google-genai HTTP 选项，仅处理代理和自定义证书。"""
    client_args = {}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE

    if base_url:
        return types.HttpOptions(
            base_url=base_url,
            client_args=client_args if client_args else None,
            async_client_args=client_args if client_args else None,
        )
    if client_args:
        return types.HttpOptions(
            client_args=client_args,
            async_client_args=client_args,
        )
    return None
