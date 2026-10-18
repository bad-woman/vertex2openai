"""Express API Key 的 GCP Project ID 自动探测。"""

import asyncio
import re
import time
from typing import Optional, Tuple

import httpx

import config as app_config
from runtime_state import app_state

GENERATE_CONTENT_URL = (
    "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
)
PROBE_FAKE_MODEL = "vertex2openai-probe-nonexistent-model"
PROBE_TIMEOUT_SECONDS = 15.0


def _client_args() -> dict:
    args = {"timeout": PROBE_TIMEOUT_SECONDS}
    if app_config.PROXY_URL:
        args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        args["verify"] = app_config.SSL_CERT_FILE
    return args


def mask_key(key: str) -> str:
    """日志里的 Key 掩码（控制台按用户要求明文显示，日志仍打掩码）。"""
    k = str(key or "")
    if len(k) <= 10:
        return "****"
    return f"{k[:6]}…{k[-4:]}"


async def probe_project_id(key: str) -> Tuple[Optional[str], str]:
    """探测单个 Key 所属的 GCP Project ID。返回 (project_id, 说明)。"""
    key = (key or "").strip()
    if not key:
        return None, "Key 为空。"
    url = GENERATE_CONTENT_URL.format(model=PROBE_FAKE_MODEL)
    body = {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}
    try:
        async with httpx.AsyncClient(**_client_args()) as client:
            resp = await client.post(url, params={"key": key}, json=body)
    except Exception as e:
        return None, f"探测请求失败：{type(e).__name__} - {e}"

    text = resp.text or ""
    if resp.status_code == 404:
        m = re.search(r"projects/([^/\"'`\s]+)", text)
        if m:
            project = m.group(1)
            tag = "（项目编号，可直接用于资源路径）" if project[:1].isdigit() else ""
            return project, f"探测成功{tag}：从上游 404 错误信息中解析出项目。"
        return None, f"上游返回 404，但错误信息里没有项目路径：{text[:200]}"
    if resp.status_code in (400, 401, 403):
        return None, f"Key 可能无效或无权访问（HTTP {resp.status_code}）：{text[:200]}"
    return None, f"探测接口返回 HTTP {resp.status_code}：{text[:200]}"


async def probe_and_store(key: str, *, force: bool = False) -> dict:
    """探测并写回状态。若该 Key 尚未保存入库，自动将其追加到列表中再探测。"""
    clean_key = (key or "").strip()
    if not clean_key:
        return {"key": key, "ok": False, "project_id": "", "message": "Key 为空。"}

    current_list = app_state.get_express_keys()
    records = {r["key"]: r for r in current_list}
    rec = records.get(clean_key)
    if rec is None:
        # 用户在控制台刚添加该 Key 尚未点保存就直接点探测：自动入库保存，避免丢失
        current_list.append({"key": clean_key, "project_id": "", "project_source": "", "enabled": True})
        app_state.set_express_keys(current_list)
        rec = {"key": clean_key, "project_id": "", "project_source": "", "enabled": True}

    if rec.get("project_id") and not force:
        return {"key": clean_key, "ok": True, "project_id": rec["project_id"], "message": "已有 Project ID，跳过探测。"}
    if rec.get("project_source") == "manual" and not force:
        return {"key": clean_key, "ok": True, "project_id": rec.get("project_id", ""), "message": "人工填写值，跳过探测。"}

    project_id, message = await probe_project_id(clean_key)
    if project_id:
        app_state.update_express_key_project(clean_key, project_id, source="auto")
        print(f"🔎 [密钥探测] Key {mask_key(clean_key)} 归属项目：{project_id}")
        return {"key": clean_key, "ok": True, "project_id": project_id, "message": message}
    print(f"⚠️ [密钥探测] Key {mask_key(clean_key)} 未能自动探测到 Project ID：{message}")
    return {"key": clean_key, "ok": False, "project_id": "", "message": message}


async def probe_all(*, force: bool = False, keys: Optional[list] = None) -> list:
    """批量探测（启动时与控制台「探测全部」共用）。"""
    targets = keys if keys is not None else [r["key"] for r in app_state.get_express_keys()]
    if not targets:
        return []
    started = time.time()
    results = await asyncio.gather(*[probe_and_store(k, force=force) for k in targets],
                                   return_exceptions=True)
    out = []
    for key, res in zip(targets, results):
        if isinstance(res, Exception):
            out.append({"key": key, "ok": False, "project_id": "", "message": f"探测异常：{res}"})
        else:
            out.append(res)
    ok = sum(1 for r in out if r.get("ok"))
    print(f"🔎 [密钥探测] 共探测 {len(out)} 个 Key，成功 {ok} 个，耗时 {time.time() - started:.1f}s。")
    return out