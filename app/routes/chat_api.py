from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from models import OpenAIRequest
from auth import get_api_key

import asyncio
import json

from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.cookie_proxy import CookieProxyUpstream
from api_helpers import extract_upstream_error, create_openai_error_response

from resumable import (
    resumable_store,
    compute_stream_id,
    is_enabled as resumable_enabled,
    DetachedRequest,
    pump_stream,
    sse_from_buffer,
    frame_with_id,
    DONE_FRAME,
)

router = APIRouter()

express_upstream = ExpressSDKUpstream()
cookie_upstream = CookieProxyUpstream()


async def _call_upstream(request_obj: OpenAIRequest, fastapi_request: Request):
    if app_state.is_web_proxy_enabled():
        return await cookie_upstream.chat_completions(request_obj, fastapi_request)
    else:
        return await express_upstream.chat_completions(request_obj, fastapi_request)


def _request_fingerprint(request_obj: OpenAIRequest) -> str:
    try:
        return request_obj.model_dump_json()
    except Exception:
        return json.dumps(request_obj.model_dump(), sort_keys=True, default=str)


def _resume_start_seq(fastapi_request: Request) -> int:
    try:
        leid = (fastapi_request.headers.get("last-event-id") or "").strip()
        if leid:
            return max(0, int(leid) + 1)
    except (TypeError, ValueError):
        pass
    try:
        q = fastapi_request.query_params.get("resume_from")
        if q is not None:
            return max(0, int(str(q).strip()))
    except (TypeError, ValueError):
        pass
    return 0


def _resume_headers(stream_id: str, replay: bool = False) -> dict:
    headers = {"X-Resume-Id": stream_id, "Cache-Control": "no-store"}
    if replay:
        headers["X-Resumable-Replay"] = "1"
    return headers


def _extract_json_response_error(resp: JSONResponse, status: int) -> str:
    try:
        raw = getattr(resp, "body", b"") or b""
        if isinstance(raw, bytes):
            data = json.loads(raw.decode("utf-8", errors="replace"))
        else:
            data = json.loads(str(raw))
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return f"HTTP {status}：{err['message']}"
            if isinstance(err, str) and err:
                return f"HTTP {status}：{err}"
    except Exception:
        pass
    return f"上游返回非流式错误响应（HTTP {status}）。"


async def _abort_placeholder(buf, stream_id: str, message: str, code: int = 500) -> None:
    err = {"error": {"message": message[:1024], "type": "upstream_error", "code": code}}
    await buf.append(frame_with_id(buf.next_seq, f"data: {json.dumps(err, ensure_ascii=False)}\n\n"))
    await buf.append(frame_with_id(buf.next_seq, DONE_FRAME))
    await buf.finish("error", error=message)
    await resumable_store.drop(stream_id)


async def _resumable_stream(fastapi_request: Request, request_obj: OpenAIRequest, api_key: str):
    stream_id, id_src = compute_stream_id(
        fastapi_request.headers, api_key or "", _request_fingerprint(request_obj))

    buf, is_new = await resumable_store.create_if_absent(
        stream_id, model=request_obj.model, kind="stream")
    if not is_new and (
        buf.kind != "stream"
        or buf.fully_delivered
        or buf.status == "error"
        or (buf.done and not buf.has_model_output)
    ):
        await resumable_store.drop(stream_id)
        buf, is_new = await resumable_store.create_if_absent(
            stream_id, model=request_obj.model, kind="stream")
    if not is_new:
        start = _resume_start_seq(fastapi_request)
        print(f"↩️ [续传命中] {stream_id}（{id_src}）：直接重放/续传，不再请求上游。")
        return StreamingResponse(
            sse_from_buffer(buf, start, fastapi_request=fastapi_request),
            media_type="text/event-stream",
            headers=_resume_headers(stream_id, replay=True),
        )

    detached = DetachedRequest(fastapi_request, buf)
    try:
        resp = await _call_upstream(request_obj, detached)
    except asyncio.CancelledError:
        await _abort_placeholder(buf, stream_id, "客户端断开，上游调用被取消。", 499)
        raise
    except Exception as e:
        err_code, err_msg = extract_upstream_error(e)
        await _abort_placeholder(buf, stream_id, err_msg, err_code)
        raise
    if not isinstance(resp, StreamingResponse):
        status = getattr(resp, "status_code", 500)
        detail_msg = _extract_json_response_error(resp, status)
        await _abort_placeholder(buf, stream_id, detail_msg, status)
        return resp

    pump_task = asyncio.create_task(pump_stream(buf, resp.body_iterator))
    resumable_store.register_pump(stream_id, pump_task)
    print(f"📦 [续传缓冲] {stream_id}（{id_src}）：后台泵已启动，支持弱网断线续传与首包前取消止损。")

    start = _resume_start_seq(fastapi_request)
    return StreamingResponse(
        sse_from_buffer(buf, start, fastapi_request=fastapi_request),
        media_type="text/event-stream",
        headers=_resume_headers(stream_id),
    )


async def _resumable_json(fastapi_request: Request, request_obj: OpenAIRequest, api_key: str):
    stream_id, id_src = compute_stream_id(
        fastapi_request.headers, api_key or "", _request_fingerprint(request_obj))

    buf, is_new = await resumable_store.create_if_absent(
        stream_id, model=request_obj.model, kind="json")
    if not is_new and (
        buf.kind != "json"
        or buf.fully_delivered
        or buf.status == "error"
        or buf.status_code >= 400
    ):
        await resumable_store.drop(stream_id)
        buf, is_new = await resumable_store.create_if_absent(
            stream_id, model=request_obj.model, kind="json")

    async def _wait_buffer_with_disconnect(target_buf) -> bool:
        deadline = asyncio.get_running_loop().time() + 600.0
        target_buf.readers += 1
        try:
            while not target_buf.done:
                remain = deadline - asyncio.get_running_loop().time()
                if remain <= 0:
                    return False
                await target_buf.wait_for(2 ** 62, min(0.5, remain))
                if target_buf.done:
                    break
                if await fastapi_request.is_disconnected():
                    return False
            return target_buf.payload is not None
        finally:
            target_buf.readers = max(0, target_buf.readers - 1)

    if not is_new:
        print(f"↩️ [重放命中] {stream_id}（{id_src}）：等待或返回已缓冲的完整回复，不再重复请求上游。")
        ok = await _wait_buffer_with_disconnect(buf)
        if not ok or buf.payload is None:
            if await fastapi_request.is_disconnected():
                if buf.readers <= 0 and not buf.done:
                    await resumable_store.drop(stream_id)
                return JSONResponse(
                    status_code=499,
                    content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
                )
            await resumable_store.drop(stream_id)
            return JSONResponse(
                status_code=504,
                content=create_openai_error_response(504, "上游响应超时，缓冲未完成。", "timeout"),
                headers=_resume_headers(stream_id, replay=True),
            )
        if buf.status == "error" or buf.status_code >= 400:
            await resumable_store.drop(stream_id)
        else:
            buf.mark_delivered()
        return JSONResponse(status_code=buf.status_code, content=buf.payload,
                            headers=_resume_headers(stream_id, replay=True))

    detached = DetachedRequest(fastapi_request, buf)

    async def _pump_json():
        try:
            resp = await _call_upstream(request_obj, detached)
        except asyncio.CancelledError:
            await buf.finish("error", error="客户端已取消请求", status_code=499)
            raise
        except Exception as e:
            err_code, err_msg = extract_upstream_error(e)
            print(f"❌ [非流式缓冲报错] {stream_id} | 模型={request_obj.model} | HTTP {err_code} | {err_msg[:450]}")
            await buf.finish("error", error=err_msg,
                             payload=create_openai_error_response(err_code, err_msg[:1024], "upstream_error"),
                             status_code=err_code)
            return
        body = getattr(resp, "body", b"") or b""
        status = getattr(resp, "status_code", 200)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {"_raw": body.decode("utf-8", "replace")[:4000]}
        is_ok = 200 <= status < 300
        if is_ok:
            buf.has_model_output = True
        await buf.finish("done" if is_ok else "error",
                         payload=payload, status_code=status)

    pump_task = asyncio.create_task(_pump_json())
    resumable_store.register_pump(stream_id, pump_task)

    ok = await _wait_buffer_with_disconnect(buf)
    if not ok or buf.payload is None:
        if await fastapi_request.is_disconnected():
            print(f"🛑 [客户端取消] {stream_id} 非流式请求期间客户端主动断开，正在终止上游任务。")
            if buf.readers <= 0 and not buf.done:
                await resumable_store.drop(stream_id)
            return JSONResponse(
                status_code=499,
                content=create_openai_error_response(499, "客户端已断开连接，请求已取消。", "client_closed_request"),
            )
        await resumable_store.drop(stream_id)
        return JSONResponse(
            status_code=504,
            content=create_openai_error_response(504, "上游响应超时，缓冲未完成。", "timeout"),
            headers=_resume_headers(stream_id),
        )

    if buf.status == "error" or buf.status_code >= 400:
        await resumable_store.drop(stream_id)
    else:
        buf.mark_delivered()
    print(f"📦 [响应缓冲] {stream_id}（{id_src}）：非流式完整回复已落缓冲（HTTP {buf.status_code}）。")
    return JSONResponse(status_code=buf.status_code, content=buf.payload,
                        headers=_resume_headers(stream_id))


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        if resumable_enabled():
            if request.stream:
                return await _resumable_stream(fastapi_request, request, api_key)
            return await _resumable_json(fastapi_request, request, api_key)
        return await _call_upstream(request, fastapi_request)
    except Exception as e:
        code, msg = extract_upstream_error(e)
        channel = "Studio(Cookie)" if app_state.is_web_proxy_enabled() else "Express(API Key)"
        print(f"❌ [路由兜底报错] 通道={channel} | 模型={request.model} | HTTP {code} | {msg[:450]}")
        return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))


@router.get("/v1/resumable/{stream_id}")
async def resumable_poll(stream_id: str, fastapi_request: Request,
                         api_key: str = Depends(get_api_key),
                         from_: int = Query(0, alias="from"),
                         limit: int = Query(200),
                         wait: float = Query(0.0)):
    buf = await resumable_store.get(stream_id)
    if buf is None:
        return JSONResponse(
            status_code=404,
            content=create_openai_error_response(
                404, f"未找到响应缓冲 {stream_id}（可能已过期被清理）。", "not_found"))
    limit = max(1, min(limit, 500))
    if wait > 0 and not buf.done:
        await buf.wait_for(max(0, from_), min(wait, 25.0))
    start = max(0, from_)
    events = buf.snapshot(start, limit)
    end = start + len(events)
    done = buf.done and end >= buf.next_seq
    if done and start <= buf.next_seq and buf.status != "error":
        buf.mark_delivered()
    return JSONResponse(content={
        "id": stream_id,
        "model": buf.model,
        "kind": buf.kind,
        "status": buf.status,
        "error": buf.error,
        "events": [{"seq": start + i, "data": frame} for i, frame in enumerate(events)],
        "next": end,
        "done": done,
    })