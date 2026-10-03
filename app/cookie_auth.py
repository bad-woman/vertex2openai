"""
Cookie 认证模块 (batchGraphql 专用)

从 Google Cookie 字符串中提取 SAPISID 族 cookie，
计算三段式 SAPISIDHASH 认证头，直接对接 batchGraphql 端点。

Authorization 格式：
  SAPISIDHASH <ts>_<hash> SAPISID1PHASH <ts>_<hash> SAPISID3PHASH <ts>_<hash>

其中 hash = SHA1(timestamp + " " + SAPISID_VALUE + " " + origin)
"""

import hashlib
import time
import re
from typing import Optional, Dict


# batchGraphql 固定端点与参数
BATCH_GRAPHQL_URL = (
    "https://cloudconsole-pa.clients6.google.com"
    "/v3/entityServices/AiplatformEntityService"
    "/schemas/AIPLATFORM_GRAPHQL:batchGraphql"
    "?key=AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g&prettyPrint=false"
)

# StreamGenerateContent 的固定签名（登录模式）
STREAM_GENERATE_QUERY_SIGNATURE = "2/Hc4FpJfYmM+gO5TB0hcjYY0Iwj0rqLIVDZOhurBbu/I="
STREAM_GENERATE_OPERATION_NAME = "StreamGenerateContent"

# SAPISIDHASH 计算使用的 origin
SAPISIDHASH_ORIGIN = "https://console.cloud.google.com"


def parse_cookie_value(cookie_str: str, name: str) -> str:
    """从 cookie 字符串中提取指定 cookie 的值"""
    pattern = rf'(?:^|;\s*){re.escape(name)}=([^;]*)'
    match = re.search(pattern, cookie_str)
    return match.group(1).strip() if match else ""


def _compute_hash(sapisid_value: str, origin: str) -> str:
    """计算单个 SAPISIDHASH"""
    timestamp = int(time.time())
    hash_input = f"{timestamp} {sapisid_value} {origin}"
    hash_value = hashlib.sha1(hash_input.encode()).hexdigest()
    return f"{timestamp}_{hash_value}"


def build_authorization_header(cookie_str: str) -> Optional[str]:
    """
    构建三段式 Authorization 头
    
    格式: SAPISIDHASH <h> SAPISID1PHASH <h> SAPISID3PHASH <h>
    """
    sapisid = parse_cookie_value(cookie_str, "SAPISID")
    sapisid_1p = parse_cookie_value(cookie_str, "__Secure-1PAPISID")
    sapisid_3p = parse_cookie_value(cookie_str, "__Secure-3PAPISID")
    
    # 至少需要一个 SAPISID
    primary = sapisid or sapisid_3p or sapisid_1p
    if not primary:
        return None
    
    # 计算 hash（它们通常值相同，hash 也相同）
    h_main = _compute_hash(sapisid or primary, SAPISIDHASH_ORIGIN)
    h_1p = _compute_hash(sapisid_1p or primary, SAPISIDHASH_ORIGIN)
    h_3p = _compute_hash(sapisid_3p or primary, SAPISIDHASH_ORIGIN)
    
    return f"SAPISIDHASH {h_main} SAPISID1PHASH {h_1p} SAPISID3PHASH {h_3p}"


def build_headers(cookie_str: str) -> Optional[Dict[str, str]]:
    """
    构建 batchGraphql 请求所需的完整 HTTP 头
    
    每次调用都重新计算 SAPISIDHASH（因为包含实时时间戳）
    """
    auth = build_authorization_header(cookie_str)
    if not auth:
        print("⚠️ [Cookie 认证] Cookie 中未找到 SAPISID 族字段，无法计算认证头")
        return None
    
    headers = {
        "authorization": auth,
        "cookie": cookie_str,
        "content-type": "application/json",
        "x-goog-authuser": "0",
        "x-same-domain": "1",
        "origin": "https://console.cloud.google.com",
        "referer": "https://console.cloud.google.com/",
    }
    
    return headers


def validate_cookie(cookie_str: str) -> dict:
    """验证 Cookie 是否包含必要字段"""
    sapisid = parse_cookie_value(cookie_str, "SAPISID")
    sapisid_3p = parse_cookie_value(cookie_str, "__Secure-3PAPISID")
    sid = parse_cookie_value(cookie_str, "SID") or parse_cookie_value(cookie_str, "__Secure-1PSID")
    
    has_sapisid = bool(sapisid or sapisid_3p)
    has_sid = bool(sid)
    
    if has_sapisid and has_sid:
        return {"valid": True, "has_sapisid": True, "has_sid": True,
                "message": "✅ Cookie 有效：包含 SAPISID 和 SID，可以直连 batchGraphql"}
    elif has_sapisid:
        return {"valid": True, "has_sapisid": True, "has_sid": False,
                "message": "⚠️ Cookie 部分有效：有 SAPISID 但缺少 SID，可尝试使用"}
    elif has_sid:
        return {"valid": False, "has_sapisid": False, "has_sid": True,
                "message": "❌ Cookie 无效：有 SID 但缺少 SAPISID，无法计算 SAPISIDHASH 认证头"}
    else:
        return {"valid": False, "has_sapisid": False, "has_sid": False,
                "message": "❌ Cookie 无效：未找到必要的认证字段 (SAPISID / SID)"}
