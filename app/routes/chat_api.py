from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from models import OpenAIRequest
from auth import get_api_key

import asyncio
import json

# 引入运行状态管理器与多通道分发策略
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.cookie_proxy import CookieProxyUpstream
from api_helpers import extract_upstream_error, create_openai_error_response

# 第二轮：弱网可靠传输 —— 服务端响应缓冲 + 序列号事件 + 断线重放 / 分片轮询。
# 上游的读取由后台泵任务驱动，与前端连接解绑；Google 的回复总能完整落缓冲，
# 前端凭 Last-Event-ID / ?resume_from= 续传，或轮询 /v1/resumable/{id} 拉分片。
# 不用 SSE 心跳保活，不维持单条长连接。
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

# 实例化多通道策略
express_upstream = ExpressSDKUpstream()
cookie_upstream = CookieProxyUpstream()


async def _call_upstream(request_obj: OpenAIRequest, fastapi_request: Request):
    """按控制台开关分流到两条上游通道（与原来完全一致）。"""
    if app_state.is_web_proxy_enabled():
        return await cookie_upstream.chat_completions(request_obj, fastapi_request)
    else:
        return await express_upstream.chat_completions(request_obj, fastapi_request)


def _request_fingerprint(request_obj: OpenAIRequest) -> str:
    """稳定请求指纹：同一请求原样重发即命中同一缓冲区（不再烧上游配额）。"""
    try:
        return request_obj.model_dump_json()
    except Exception:
        return json.dumps(request_obj.model_dump(), sort_keys=True, default=str)


def _resume_start_seq(fastapi_request: Request) -> int:
    """续传起点：优先 Last-Event-ID 请求头，其次 ?resume_from= 查询参数，缺省 0。"""
    try:
        leid = (fastapi_request.headers.get("last-event-id") or "").strip()
        if leid:
            return max(0, int(leid))
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


async def _abort_placeholder(buf, stream_id: str, message: str, code: int = 500) -> None:
    """占位缓冲没能接上上游：给已经在跟读它的并发请求补一个错误帧 + 终止帧，再撤掉占位。

    撤掉占位是为了不让一个「空壳」缓冲挡住后续重试（跟读者手里还拿着缓冲对象，读得完）。
    """
    err = {"error": {"message": message[:1024], "type": "upstream_error", "code": code}}
    await buf.append(frame_with_id(buf.next_seq, f"data: {json.dumps(err, ensure_ascii=False)}\n\n"))
    await buf.append(frame_with_id(buf.next_seq, DONE_FRAME))
    await buf.finish("error", error=message)
    await resumable_store.drop(stream_id)


async def _resumable_stream(fastapi_request: Request, request_obj: OpenAIRequest, api_key: str):
    """流式：缓冲命中则重放/续传；否则建缓冲、后台泵抽帧，当前连接只做缓冲的读者。

    建缓冲走 `create_if_absent` 原子占位：并发的同一请求只有第一个去打上游，
    其余直接跟读同一份缓冲（不重复烧配额，也不会把先建的缓冲/泵任务覆盖掉）。
    """
    stream_id, id_src = compute_stream_id(
        fastapi_request.headers, api_key or "", _request_fingerprint(request_obj))

    buf, is_new = await resumable_store.create_if_absent(
        stream_id, model=request_obj.model, kind="stream")
    if not is_new and (buf.kind != "stream" or buf.fully_delivered):
        # 已经完整交付过（或类型对不上）：这次算新一轮生成，撤掉旧缓冲重新占位。
        await resumable_store.drop(stream_id)
        buf, is_new = await resumable_store.create_if_absent(
            stream_id, model=request_obj.model, kind="stream")
    if not is_new:
        # 命中在用的缓冲：重放已缓冲的部分并继续跟随上游直到本轮结束，不再请求上游。
        start = _resume_start_seq(fastapi_request)
        print(f"↩️ [续传命中] {stream_id}（{id_src}）：直接重放/续传，不再请求上游。")
        return StreamingResponse(
            sse_from_buffer(buf, start),
            media_type="text/event-stream",
            headers=_resume_headers(stream_id, replay=True),
        )

    # 新一轮：上游调用与前端解绑（DetachedRequest 让 is_disconnected() 恒为 False，
    # 前端断开不会中断上游读取），生成器由后台泵任务消费。
    detached = DetachedRequest(fastapi_request)
    try:
        resp = await _call_upstream(request_obj, detached)
    except asyncio.CancelledError:
        # 客户端在上游返回前断开：同样撤掉占位，避免留下一个永远等不到泵的
        # 僵尸缓冲（同指纹重试会跟读它直到被清扫，默认配置下最长 2 小时）。
        await _abort_placeholder(buf, stream_id, "客户端断开，上游调用被取消。", 499)
        raise
    except Exception as e:
        # 占位已经建了，跟读者可能在等：先把错误落进缓冲再撤占位，然后交给路由兜底。
        await _abort_placeholder(buf, stream_id, f"{type(e).__name__}: {e}")
        raise
    if not isinstance(resp, StreamingResponse):
        # 上游直接给了非流式响应（通常是错误 JSON）：不进缓冲，原样返回；
        # 撤掉占位，重试会走一次新的上游调用（429 等本就应该重打）。
        status = getattr(resp, "status_code", 500)
        await _abort_placeholder(buf, stream_id, f"上游返回非流式响应（HTTP {status}）。", status)
        return resp

    pump_task = asyncio.create_task(pump_stream(buf, resp.body_iterator))
    resumable_store.register_pump(stream_id, pump_task)
    print(f"📦 [续传缓冲] {stream_id}（{id_src}）：后台泵已启动，上游读取与前端连接解绑。")

    start = _resume_start_seq(fastapi_request)
    return StreamingResponse(
        sse_from_buffer(buf, start),
        media_type="text/event-stream",
        headers=_resume_headers(stream_id),
    )


async def _resumable_json(fastapi_request: Request, request_obj: OpenAIRequest, api_key: str):
    """非流式：完整回复落缓冲；原样重发命中未交付缓冲则直接重放。

    同样走 `create_if_absent` 原子占位：并发的同一请求只打一次上游，
    后来的请求等同一个后台泵的结果。
    """
    stream_id, id_src = compute_stream_id(
        fastapi_request.headers, api_key or "", _request_fingerprint(request_obj))

    buf, is_new = await resumable_store.create_if_absent(
        stream_id, model=request_obj.model, kind="json")
    if not is_new and (buf.kind != "json" or buf.fully_delivered):
        # 已完整交付过（或类型对不上）：这次算新一轮生成，撤掉旧缓冲重新占位。
        await resumable_store.drop(stream_id)
        buf, is_new = await resumable_store.create_if_absent(
            stream_id, model=request_obj.model, kind="json")
    if not is_new:
        # 命中在用的缓冲：已出结果的直接重放；上游还在算的，等同一个后台泵的结果。
        print(f"↩️ [重放命中] {stream_id}（{id_src}）：直接返回已缓冲的完整回复，不再请求上游。")
        if buf.payload is None:
            await buf.wait_for(2 ** 62, 600.0)
        if buf.payload is None:
            return JSONResponse(
                status_code=504,
                content=create_openai_error_response(504, "上游响应超时，缓冲未完成。", "timeout"),
                headers=_resume_headers(stream_id, replay=True))
        buf.mark_delivered()
        return JSONResponse(status_code=buf.status_code, content=buf.payload,
                            headers=_resume_headers(stream_id, replay=True))

    detached = DetachedRequest(fastapi_request)

    async def _pump_json():
        # 后台跑上游调用：这条前端连接断了，Google 的回复也要收完进缓冲。
        try:
            resp = await _call_upstream(request_obj, detached)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"❌ [响应缓冲] {stream_id} 上游调用失败：{msg[:200]}")
            await buf.finish("error", error=msg,
                             payload={"error": {"message": str(e)[:1024],
                                                 "type": "upstream_error", "code": 500}},
                             status_code=500)
            return
        body = getattr(resp, "body", b"") or b""
        status = getattr(resp, "status_code", 200)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {"_raw": body.decode("utf-8", "replace")[:4000]}
        await buf.finish("done" if 200 <= status < 300 else "error",
                         payload=payload, status_code=status)

    pump_task = asyncio.create_task(_pump_json())
    resumable_store.register_pump(stream_id, pump_task)

    # 当前连接等待缓冲完成（等同于原来的直通等待；断线不影响后台泵）。
    # wait_for(seq=极大值) 等价于“等到 done 或超时”。
    await buf.wait_for(2 ** 62, 600.0)
    if buf.payload is None:
        return JSONResponse(
            status_code=504,
            content=create_openai_error_response(504, "上游响应超时，缓冲未完成。", "timeout"),
            headers=_resume_headers(stream_id))
    # 交给 ASGI 层即视为已交付：半路丢包的重试会走“新一轮生成”（正确，仅多一次上游调用）；
    # 上游还在算时断线的重试会命中上面的重放分支（不烧配额）。
    buf.mark_delivered()
    print(f"📦 [响应缓冲] {stream_id}（{id_src}）：非流式完整回复已落缓冲（HTTP {buf.status_code}）。")
    return JSONResponse(status_code=buf.status_code, content=buf.payload,
                        headers=_resume_headers(stream_id))


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    """
    /v1/chat/completions 动态分流路由器
    根据大盘设置的全局开关，决定将 OpenAI 请求路由至：
    - True  -> CookieProxyUpstream (Cookie 直连反代通道，规避 429 限流)
    - False -> ExpressSDKUpstream (官方 API Key 标准通道)

    弱网可靠传输（resumable_enabled 开启时，默认开）：
    响应先落服务端缓冲（带序列号），上游读取与前端连接解绑；
    前端断线可凭 Last-Event-ID / ?resume_from= 续传，或轮询 /v1/resumable/{id} 拉分片。

    统一异常兜底：把上游抛出的 404/403/400 等如实转成 OpenAI 错误格式，
    避免笼统的 500 Internal Server Error（非流式路径此前直接 raise）。
    统计由 main.py 的中间件按响应状态码统一计入，这里不重复计数。
    """
    try:
        if resumable_enabled():
            if request.stream:
                return await _resumable_stream(fastapi_request, request, api_key)
            return await _resumable_json(fastapi_request, request, api_key)
        return await _call_upstream(request, fastapi_request)
    except Exception as e:
        code, msg = extract_upstream_error(e)
        print(f"❌ [路由兜底] 模型 {request.model} 调用失败 | HTTP {code} | {msg[:200]}")
        return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))


@router.get("/v1/resumable/{stream_id}")
async def resumable_poll(stream_id: str, fastapi_request: Request,
                         api_key: str = Depends(get_api_key),
                         from_: int = Query(0, alias="from"),
                         limit: int = Query(200),
                         wait: float = Query(0.0)):
    """
    分片轮询拉取：GET /v1/resumable/{id}?from=<seq>&limit=200&wait=10

    短连接天生抗抖动；返回带序号的事件分片、next 游标与 done 标志，
    客户端按 seq 去重、按 done 判定本轮完整结束。wait>0 时为长轮询（最多等 25 秒）。
    响应头 X-Resume-Id 即为这里的 {id}。
    """
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
    if done and start <= buf.next_seq:
        # 轮询客户端已经把本轮读到末尾（start 在缓冲范围内），与 SSE 读者交付完终止帧等价：
        # 标记为已交付，下一次同指纹请求才会被当成新一轮生成，而不是重放旧回复。
        # start 越过末尾属于游标异常，不标记——避免一次坏游标挡住后续的正常重放。
        buf.mark_delivered()
    return JSONResponse(content={
        "id": stream_id,
        "model": buf.model,
        "kind": buf.kind,
        "status": buf.status,
        "error": buf.error,
        "events": [{"seq": start + i, "data": frame} for i, frame in enumerate(events)],
        "next": end,
        "done": buf.done and end >= buf.next_seq,
    })
