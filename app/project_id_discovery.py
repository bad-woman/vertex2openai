import httpx
import re
import json
from typing import Dict, Optional
import config

PROJECT_ID_CACHE: Dict[str, str] = {}

async def discover_project_id(api_key: str) -> str:
    """
    通过 httpx 发现项目 ID，采用全版本兼容的 proxy 参数
    """
    if api_key in PROJECT_ID_CACHE:
        print(f"INFO: 使用缓存的项目 ID: {PROJECT_ID_CACHE[api_key]}")
        return PROJECT_ID_CACHE[api_key]
    
    error_url = "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.7-pro-preview-05-06:streamGenerateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}]
    }
    
    client_args = {'timeout': 20.0}
    if config.PROXY_URL:
        client_args['proxy'] = config.PROXY_URL
    if getattr(config, "SSL_CERT_FILE", None):
        client_args['verify'] = config.SSL_CERT_FILE
        
    async with httpx.AsyncClient(**client_args) as client:
        try:
            response = await client.post(error_url, params={"key": api_key}, json=payload)
            response_text = response.text
            
            try:
                error_data = response.json()
                if isinstance(error_data, list) and len(error_data) > 0:
                    error_data = error_data[0]
                
                if "error" in error_data:
                    error_message = error_data["error"].get("message", "")
                    match = re.search(r'projects/(\d+)/locations/', error_message)
                    if match:
                        project_id = match.group(1)
                        PROJECT_ID_CACHE[api_key] = project_id
                        print(f"INFO: 成功发现项目 ID: {project_id}")
                        return project_id
            except json.JSONDecodeError:
                match = re.search(r'projects/(\d+)/locations/', response_text)
                if match:
                    project_id = match.group(1)
                    PROJECT_ID_CACHE[api_key] = project_id
                    print(f"INFO: 从原始响应文本中发现项目 ID: {project_id}")
                    return project_id
            
            raise Exception(f"未能发现项目 ID。状态码: {response.status_code}, 响应: {response_text[:500]}")
            
        except Exception as e:
            print(f"ERROR: 发现项目 ID 失败 (本地网络/证书/代理可能存在限制): {e}")
            raise