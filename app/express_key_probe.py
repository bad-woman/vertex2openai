"""Express API Key 的 GCP Project ID 自动探测。

背景：Express Key 是绑定到某个 Google 账号/项目上的，而「标准模式 location 钉定」
需要拼出 `projects/{project}/locations/{location}/...` 的完整资源路径。
多账号轮换时每个 Key 的项目都不一样，不能再共用 Cookie 通道那份全局 Project ID。

探测方式：用「调模型的正确 URL」，把模型名换成一个不存在的模型名发一次请求：
  POST https://aiplatform.googleapis.com/v1/publishers/google/models/{fake}:generateContent?key={key}
（这正是 Express 标准模式实际在用的全局端点格式。）
模型不存在时 Google 返回 404，错误信息里会带上按该 Key 解析出的完整资源路径，
形如 …projects/{project}/locations/…/publishers/google/models/{fake}… was not found，
从中提取 project。请求在模型解析阶段即失败，不产生任何 token、不消耗配额；
Key 本身非法时则返回 400/401/403，顺带校验 Key 有效性。
探测失败不影响调用：拿不到项目就退回全局 Project ID，再退回裸模型名（旧行为）。
"""

import asyncio
import re
import time
from typing import Optional, Tuple

import httpx

import config as app_config
from runtime_state import app_state

# 调模型的正确 URL（Express 全局端点格式），{model} 处填一个不存在的模型名来探测
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
    """探测单个 Key 所属的 GCP Project ID。返回 (project_id, 说明)。

    用不存在的模型名调一次 generateContent：Key 有效时上游返回 404，
    错误信息里带 `projects/{project}/...` 路径，从中提取项目；
    Key 无效时返回 400/401/403。全程不消耗配额。
    """
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
    """探测并写回状态。force=False 时已有 Project ID 的 Key 直接跳过。"""
    records = {r["key"]: r for r in app_state.get_express_keys()}
    rec = records.get((key or "").strip())
    if rec is None:
        return {"key": key, "ok": False, "message": "该 Key 不在列表中（可能刚被删除）。"}
    if rec.get("project_id") and not force:
        return {"key": key, "ok": True, "project_id": rec["project_id"], "message": "已有 Project ID，跳过探测。"}
    if rec.get("project_source") == "manual" and not force:
        return {"key": key, "ok": True, "project_id": rec.get("project_id", ""), "message": "人工填写值，跳过探测。"}

    project_id, message = await probe_project_id(key)
    if project_id:
        app_state.update_express_key_project(key, project_id, source="auto")
        print(f"🔎 [密钥探测] Key {mask_key(key)} 归属项目：{project_id}")
        return {"key": key, "ok": True, "project_id": project_id, "message": message}
    print(f"⚠️ [密钥探测] Key {mask_key(key)} 未能自动探测到 Project ID：{message}")
    return {"key": key, "ok": False, "project_id": "", "message": message}


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
