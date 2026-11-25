"""
用量守恒校核（含隐式前缀缓存命中）、实况脉冲联动与持久化调用追踪器（CallTraceTracker）。

从原 api_helpers.py 精准拆离，同时接入：
  - usage_metadata.cached_content_token_count 隐式前缀缓存命中提取；
  - metrics_store SQLite 长久持久化（分模型聚合 + Key 指纹命中次数 + 调用记录）；
  - live_tracker 在途请求实况脉冲与服务端中止信号。
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Dict, List, Optional

import config as app_config
import model_capabilities as mc
from live_tracker import live_tracker
from logger import call_recorder, stats
from metrics_store import metrics_store
from models import OpenAIRequest
from runtime_state import app_state


def _um_get_int(um: Any, *keys: str) -> Optional[int]:
    """跨 SDK 版本安全读取 usage_metadata 字段（兼容属性、dict 与 Pydantic model_extra）。"""
    if um is None:
        return None
    if isinstance(um, dict):
        for k in keys:
            v = um.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    extra = getattr(um, "model_extra", None)
    for k in keys:
        v = getattr(um, k, None)
        if v is None and isinstance(extra, dict):
            v = extra.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None


def _extract_raw_usage_fields(resp: Any) -> tuple[int, int, int, int, int]:
    """提取未经守恒折算的原始 (prompt_tk, explicit_th_tk, explicit_out_tk, tool_prompt_tk, total_tk)。"""
    um = resp.get("usageMetadata") if isinstance(resp, dict) else getattr(resp, "usage_metadata", None)
    if not um:
        return 0, 0, 0, 0, 0
    p_tk = _um_get_int(um, "prompt_token_count", "promptTokenCount") or 0
    th_tk = _um_get_int(
        um, "thoughts_token_count", "thoughtsTokenCount", "thought_token_count", "thoughtTokenCount"
    ) or 0
    out_tk = _um_get_int(um, "candidates_token_count", "candidatesTokenCount") or 0
    tool_p_tk = _um_get_int(um, "tool_use_prompt_token_count", "toolUsePromptTokenCount") or 0
    tot_tk = _um_get_int(um, "total_token_count", "totalTokenCount") or 0
    return p_tk, th_tk, out_tk, tool_p_tk, tot_tk


def _extract_cached_tokens(resp: Any) -> int:
    """提取 Vertex AI 隐式前缀缓存命中 Token 数（cached_content_token_count / cachedContentTokenCount）。"""
    um = resp.get("usageMetadata") if isinstance(resp, dict) else getattr(resp, "usage_metadata", None)
    if not um:
        return 0
    cached = _um_get_int(um, "cached_content_token_count", "cachedContentTokenCount") or 0
    return max(0, int(cached))


def _reconcile_usage_counts(
    p_tk: int,
    explicit_th: int,
    explicit_out: int,
    tool_p_tk: int,
    raw_tot: int,
    visible_chars: int = 0,
) -> tuple[int, int, int, int]:
    """基于 total_token_count 总量守恒校核并补齐隐藏思考或流式分片漏计的正文 Token。"""
    th_tk = max(0, int(explicit_th or 0))
    out_tk = max(0, int(explicit_out or 0))
    p_tk = max(0, int(p_tk or 0))
    tool_p_tk = max(0, int(tool_p_tk or 0))
    raw_tot = max(0, int(raw_tot or 0))

    if raw_tot > 0 and p_tk > 0:
        residual_gen = max(0, raw_tot - p_tk - tool_p_tk)
        if residual_gen > (th_tk + out_tk):
            if th_tk > 16 and out_tk <= 32 and visible_chars > 120:
                out_tk = max(out_tk, residual_gen - th_tk)
            else:
                th_tk = max(th_tk, residual_gen - out_tk)

    comp_tk = th_tk + out_tk
    t_tk = max(raw_tot, p_tk + tool_p_tk + comp_tk) if (p_tk or comp_tk or raw_tot) else 0
    return p_tk, th_tk, out_tk, t_tk


def _extract_usage_detail(resp: Any, visible_chars: int = 0) -> tuple[int, int, int, int]:
    """从 Gemini 响应对象提取经总量守恒校核后的精确 (输入Tokens, 原生思考Tokens, 正文输出Tokens, 总Tokens)。"""
    p_tk, raw_th, raw_out, tool_p_tk, raw_tot = _extract_raw_usage_fields(resp)
    return _reconcile_usage_counts(p_tk, raw_th, raw_out, tool_p_tk, raw_tot, visible_chars=visible_chars)


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    p_tk, th_tk, out_tk, t_tk = _extract_usage_detail(resp)
    return p_tk, (th_tk + out_tk), t_tk


def _record_usage(resp: Any, visible_chars: int = 0) -> dict:
    usage_metadata = getattr(resp, "usage_metadata", None)
    traffic_type = getattr(usage_metadata, "traffic_type", None) if usage_metadata else None
    if traffic_type:
        value = getattr(traffic_type, "value", None) or str(traffic_type)
        print(f"🚦 [流量等级] 上游实际 traffic_type={value}")

    p_tk, th_tk, out_tk, t_tk = _extract_usage_detail(resp, visible_chars=visible_chars)
    cached_tk = min(p_tk, _extract_cached_tokens(resp)) if p_tk > 0 else _extract_cached_tokens(resp)
    comp_tk = th_tk + out_tk
    if p_tk or comp_tk:
        stats.add_tokens(p_tk, comp_tk, cached_tokens=cached_tk)
        cache_note = f" (缓存命中: {cached_tk})" if cached_tk > 0 else ""
        print(f"💰 [算力消耗统计] 输入: {p_tk}{cache_note} | 思考: {th_tk} | 正文: {out_tk} | 总计: {t_tk} Tokens")
    else:
        stats.add_success()
    res: Dict[str, Any] = {
        "prompt_tokens": p_tk,
        "cached_tokens": cached_tk,
        "completion_tokens": comp_tk,
        "total_tokens": t_tk,
        "thoughts_tokens": th_tk,
        "output_tokens": out_tk,
    }
    return res


def _short_cred_name(target: Optional[dict]) -> str:
    if not target:
        return "—"
    if target.get("channel") == "express":
        rec = target.get("key_record") or {}
        return f"Key #{rec.get('index', 0) + 1}"
    return "Studio Cookie"


def _format_thinking_for_record(tk_dict: Optional[dict]) -> tuple[str, str, bool]:
    """返回 (展示文案, 徽章样式 muted|accent|apricot, 是否流式回传原生思考)。"""
    if not tk_dict or tk_dict.get("mode") is None:
        return "无", "muted", False
    inc = bool(tk_dict.get("include_thoughts", True))
    lv = str(tk_dict.get("level") or "").strip().upper()
    if not inc:
        return (f"压制 ({lv})" if lv else "压制"), "muted", False
    if lv == "XHIGH":
        return "XHIGH", "apricot", True
    if lv:
        return lv, "accent", True
    return "自动", "accent", True


def _chunk_has_model_token(chunk: Any, include_thoughts: bool = True) -> tuple[bool, bool]:
    """检测上游流式 chunk 是否包含实际生成的模型 token（精确测量真实首字响应 TTFT）。"""
    cands = getattr(chunk, "candidates", None) or []
    has_tok = False
    has_thought = False
    for cand in cands:
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        for p in parts:
            is_th = getattr(p, "thought", None) is True
            txt = getattr(p, "text", None)
            if is_th:
                if txt:
                    has_thought = True
                    if include_thoughts:
                        has_tok = True
            elif txt:
                has_tok = True
            else:
                fc = getattr(p, "function_call", None)
                if fc is not None and (
                    getattr(fc, "name", None)
                    or getattr(fc, "args", None)
                    or getattr(fc, "partial_args", None)
                ):
                    has_tok = True
                elif getattr(p, "inline_data", None) is not None or getattr(p, "file_data", None) is not None:
                    has_tok = True
    return has_tok, has_thought


class CallTraceTracker:
    """单次聊天请求的生命周期指标追踪器（联动 live_tracker + CallRecorder + SQLite metrics_store，幂等零重复记账）。"""

    def __init__(self, request_obj: OpenAIRequest, primary_channel: str):
        self.created_wall = time.time()
        self.start_mono = time.monotonic()
        self.primary_channel = primary_channel
        self.finalized = False

        raw_model = str(getattr(request_obj, "model", "") or "unknown").strip()
        base_m = raw_model
        while True:
            low_m = base_m.lower()
            if low_m.endswith("-search"):
                base_m = base_m[: -len("-search")]
                continue
            m_th = re.search(r"-think-(?:minimal|low|medium|high|xhigh|off|none)$", low_m)
            if m_th:
                base_m = base_m[: m_th.start()]
                continue
            break
        self.model_display = base_m or raw_model
        prof = mc.get_profile(self.model_display)
        self.is_image = bool(prof.get("is_image", False))
        self.is_stream_req = bool(getattr(request_obj, "stream", False))
        fake_on = bool(app_state.get_setting("fake_streaming", app_config.FAKE_STREAMING_ENABLED))

        eff_settings = app_state.get_effective_settings(self.model_display)
        msgs = getattr(request_obj, "messages", None) or []
        rough_prefill = bool(
            msgs
            and str(getattr(msgs[-1], "role", "") or "") == "assistant"
            and eff_settings.get("prefill_mode", "smart") != "off"
        )
        tk_init = mc.resolve_thinking(raw_model, request_obj, eff_settings, prefill_active=rough_prefill)
        self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(tk_init)

        self.sandbox = bool(eff_settings.get("synthetic_sandbox_enabled", False) and not self.is_image)
        self.prefill = rough_prefill
        self.search = bool(raw_model.lower().endswith("-search"))
        self.tool_calls = False

        if self.is_image:
            self.mode_label = "生图假流" if self.is_stream_req else "生图非流"
            self.mode_class = "pill-accent" if self.is_stream_req else "pill-sky"
            self.is_unary = True
        elif not self.is_stream_req:
            self.mode_label = "非流式"
            self.mode_class = "pill-sky"
            self.is_unary = True
        elif fake_on:
            self.mode_label = "假流式"
            self.mode_class = "pill-accent"
            self.is_unary = True
        else:
            self.mode_label = "真流式"
            self.mode_class = "pill-mint"
            self.is_unary = False

        self.retries = 0
        self.retry_wait_total = 0.0
        self.unique_creds: List[str] = []
        self.failed_steps: List[str] = []
        self.last_cred_short: str = ""
        self.last_channel: str = primary_channel

        # 注册至实况脉冲追踪器
        self.req_id: str = live_tracker.start(
            model=self.model_display,
            channel=primary_channel,
            is_stream=self.is_stream_req,
            sandbox_active=self.sandbox,
        )

    def is_live_aborted(self) -> bool:
        return live_tracker.is_aborted(self.req_id)

    def finish_live(
        self,
        *,
        status: str = "ok",
        finish_reason: str = "",
        total_sec: Optional[float] = None,
        ttft: Optional[float] = None,
        tps: Optional[float] = None,
        cache_pct: Optional[float] = None,
        out_chars: Optional[int] = None,
        note: str = "",
    ) -> None:
        live_tracker.finish(
            self.req_id,
            status=status,
            finish_reason=finish_reason,
            total_sec=total_sec,
            ttft=ttft,
            tps=tps,
            cache_pct=cache_pct,
            out_chars=out_chars,
            note=note,
        )

    def note_live_phase(
        self,
        phase: Optional[str] = None,
        *,
        add_chars: int = 0,
        add_thought_chars: int = 0,
        note: Optional[str] = None,
    ) -> None:
        live_tracker.update(
            self.req_id,
            phase=phase,
            add_chars=add_chars,
            add_thought_chars=add_thought_chars,
            retries=self.retries,
            note=note,
        )

    def sync_from_express_ctx(self, ctx: Optional[dict]) -> None:
        if not ctx:
            return
        self.model_display = ctx.get("base_model_name") or self.model_display
        self.sandbox = bool(ctx.get("synthetic_tool_name"))
        self.prefill = bool(ctx.get("prefill_active", False))
        self.search = bool(ctx.get("has_search", False))
        tk_res = ctx.get("thinking_resolved")
        if tk_res is not None:
            self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(tk_res)
        live_tracker.update(
            self.req_id,
            model=self.model_display,
            sandbox_active=self.sandbox,
        )

    def sync_from_cookie_telemetry(self, tel: Optional[dict]) -> None:
        if not tel:
            return
        if tel.get("base_model"):
            self.model_display = tel["base_model"]
        if "sandbox" in tel:
            self.sandbox = bool(tel["sandbox"])
        if "prefill" in tel:
            self.prefill = bool(tel["prefill"])
        if "search" in tel:
            self.search = bool(tel["search"])
        if tel.get("thinking_resolved") is not None:
            self.thinking_label, self.think_style, self.include_thoughts = _format_thinking_for_record(
                tel["thinking_resolved"]
            )
        if "is_unary" in tel:
            self.is_unary = bool(tel["is_unary"])
        if tel.get("mode_label"):
            raw_ml = str(tel["mode_label"])
            for prefix in ("🖼️ ", "📦 ", "💓 ", "⚡ "):
                if raw_ml.startswith(prefix):
                    raw_ml = raw_ml[len(prefix):]
            self.mode_label = raw_ml
            if "真流式" in self.mode_label:
                self.mode_class = "pill-mint"
            elif "假流" in self.mode_label:
                self.mode_class = "pill-accent"
            else:
                self.mode_class = "pill-sky"
        live_tracker.update(
            self.req_id,
            model=self.model_display,
            sandbox_active=self.sandbox,
        )

    def note_attempt_start(self, target: dict) -> None:
        short = _short_cred_name(target)
        ch = "cookie" if target.get("channel") == "cookie" else "express"
        self.last_cred_short = short
        self.last_channel = ch
        if short not in self.unique_creds:
            self.unique_creds.append(short)

        # 持久化记录每把 Key / Cookie 的累计命中次数
        raw_key = ""
        pid = ""
        if ch == "express":
            rec = target.get("key_record") or {}
            raw_key = str(rec.get("key") or "")
            pid = str(rec.get("project_id") or "")
        metrics_store.record_key_hit(ch, raw_key)

        live_tracker.update(
            self.req_id,
            channel=ch,
            cred_label=short,
            project_id=pid,
            phase="connecting",
            retries=self.retries,
        )

    def note_attempt_failure(self, target: dict, err_code: int, wait_sec: Optional[float] = None) -> None:
        short = _short_cred_name(target)
        if wait_sec is not None:
            self.retries += 1
            self.retry_wait_total += wait_sec
            self.failed_steps.append(f"{html.escape(short)} (HTTP {err_code}，等 {wait_sec:.2f}s)")
            live_tracker.update(
                self.req_id,
                phase="retrying",
                retries=self.retries,
                note=f"HTTP {err_code}，{wait_sec:.1f}s 后切换凭证",
            )
        else:
            self.failed_steps.append(f"{html.escape(short)} (HTTP {err_code})")

    def _build_time_strs(self) -> tuple[str, str]:
        lt = time.localtime(self.created_wall)
        return time.strftime("%H:%M:%S", lt), time.strftime("%Y-%m-%d %H:%M:%S", lt)

    def finalize_success(
        self,
        target: dict,
        *,
        is_cookie: bool = False,
        ttft: Optional[float] = None,
        gen_time: float = 0.0,
        saw_thought_stream: bool = False,
        finish_reason: str = "stop",
        raw_finish_reason: str = "",
        has_tool_calls: bool = False,
        has_usage: bool = False,
        p_tk: int = 0,
        cached_tk: int = 0,
        th_tk: int = 0,
        out_tk: int = 0,
        out_chars: int = 0,
    ) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        short_cred = _short_cred_name(target)
        ch = "cookie" if (is_cookie or target.get("channel") == "cookie") else "express"

        if target.get("channel") != self.primary_channel:
            cred_note = "跨通道兜底"
        elif is_cookie:
            cred_note = "Cookie 模式"
        else:
            note = str((target.get("key_record") or {}).get("note") or "").strip()
            cred_note = note[:16] if note else "标准模式"

        is_trunc = (str(finish_reason or "").lower() == "length")
        if is_trunc:
            badge_type = "trunc"
        elif self.retries > 0:
            badge_type = "retry_ok"
        else:
            badge_type = "ok"

        clean_ttft = round(float(ttft), 2) if (not self.is_unary and ttft is not None and ttft >= 0) else None
        clean_gen = round(max(0.0, float(gen_time)), 2)
        clean_cached = max(0, min(int(p_tk or 0), int(cached_tk or 0))) if p_tk > 0 else max(0, int(cached_tk or 0))
        cache_pct = round((clean_cached / p_tk) * 100, 1) if (p_tk > 0 and clean_cached > 0) else 0.0

        tps: Optional[float] = None
        tps_approx = False
        if has_usage:
            if not self.is_unary:
                streamed_tk = (th_tk + out_tk) if (self.include_thoughts and saw_thought_stream) else out_tk
                if clean_gen >= 0.12 and streamed_tk > 0:
                    tps = round(streamed_tk / clean_gen, 1)
                    tps_approx = False
                elif (th_tk + out_tk) > 0 and ((clean_ttft or 0.0) + clean_gen) >= 0.1:
                    tps = round((th_tk + out_tk) / ((clean_ttft or 0.0) + clean_gen), 1)
                    tps_approx = True
            else:
                tot_gen_tk = th_tk + out_tk
                if clean_gen >= 0.1 and tot_gen_tk > 0:
                    tps = round(tot_gen_tk / clean_gen, 1)
                    tps_approx = True

        # 估算三段耗时占比（用于详情彩色分段条）
        conn_t = round(min(clean_ttft if clean_ttft is not None else total_wall * 0.15, total_wall), 2)
        think_t = 0.0
        if th_tk > 0 and (th_tk + out_tk) > 0:
            rem_t = max(0.0, total_wall - conn_t)
            think_t = round(rem_t * (th_tk / (th_tk + out_tk)), 2)
        body_t = round(max(0.01, total_wall - conn_t - think_t), 2)

        parts_det: List[str] = []
        if self.failed_steps:
            chain_str = " → ".join(self.failed_steps + [f"<b>{html.escape(short_cred)} 成功</b>"])
            parts_det.append(
                f"重试轨迹（重试 {self.retries} 次，退避共耗 {self.retry_wait_total:.1f}s）：{chain_str}。"
            )
        else:
            parts_det.append(f"一次直通成功：使用凭证 <b>{html.escape(short_cred)}</b>（{html.escape(cred_note)}）。")

        if not self.is_unary and clean_ttft is not None:
            parts_det.append(f"首字响应 <b>{clean_ttft:.2f}s</b>，首字后纯流式输出耗时 <b>{clean_gen:.2f}s</b>（请求总历时 {total_wall:.2f}s）。")
        else:
            parts_det.append(f"上游原生整包生成耗时 <b>{clean_gen:.2f}s</b>（请求总历时 {total_wall:.2f}s）。")

        if has_usage:
            spd_txt = f"，速度 <b>{'≈' if tps_approx else ''}{tps:.1f} tok/s</b>" if tps else ""
            cache_txt = f"（隐式缓存命中 <b>{clean_cached:,}</b> / {cache_pct:.1f}%）" if clean_cached > 0 else ""
            parts_det.append(
                f"Token 明细：输入 <b>{p_tk:,}</b>{cache_txt} · 思考 <b>{th_tk:,}</b> · 正文 <b>{out_tk:,}</b>（共 {p_tk + th_tk + out_tk:,} Tokens{spd_txt}）。"
            )
        elif is_cookie:
            ch_txt = f"，正文共 <b>{out_chars:,}</b> 字符" if out_chars > 0 else ""
            parts_det.append(f"Cookie 私有接口不回传 Token 统计{ch_txt}。")

        raw_fr_up = str(raw_finish_reason or "").strip().upper()
        if is_trunc:
            if raw_fr_up in ("MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", "UNEXPECTED_TOOL_CALL"):
                parts_det.append(
                    f"<b>上游格式中断</b>：上游返回 <code>{html.escape(raw_fr_up)}</code>（并非超过 max_tokens，而是模型工具参数外层 JSON 编码中途出错被上游中止）。"
                )
            elif raw_fr_up == "MAX_TOKENS":
                parts_det.append(
                    "<b>达上限截断</b>：上游返回 <code>MAX_TOKENS</code>（正文与隐性思考合计触及 max_tokens 上限提前停止）。"
                )
            else:
                raw_note = f"（上游={html.escape(raw_fr_up)}）" if raw_fr_up else ""
                parts_det.append(f"<b>提前截断</b>：finish_reason=length{raw_note}。")

        t_short, t_full = self._build_time_strs()
        final_fr_display = raw_fr_up or ("STOP" if str(finish_reason or "stop") == "stop" else str(finish_reason or "").upper())
        self.finish_live(
            status="trunc" if is_trunc else "ok",
            finish_reason=final_fr_display,
            total_sec=total_wall,
            ttft=clean_ttft,
            tps=tps,
            cache_pct=cache_pct if clean_cached > 0 else None,
            out_chars=out_chars,
        )
        call_recorder.record(
            {
                "time": t_short,
                "full_time": t_full,
                "group": "ok",
                "badge_type": badge_type,
                "retries": self.retries,
                "code": 200,
                "finish_reason": str(finish_reason or "stop"),
                "raw_finish_reason": raw_fr_up or ("STOP" if str(finish_reason or "stop") == "stop" else str(finish_reason or "").upper()),
                "model": self.model_display,
                "channel": ch,
                "sandbox": self.sandbox,
                "prefill": self.prefill,
                "search": self.search,
                "tool_calls": bool(has_tool_calls or finish_reason == "tool_calls"),
                "thinking": self.thinking_label,
                "think_style": self.think_style,
                "mode": self.mode_label,
                "mode_class": self.mode_class,
                "cred": short_cred,
                "cred_note": cred_note,
                "is_unary": self.is_unary,
                "ttft": clean_ttft,
                "gen_time": clean_gen,
                "conn_time": conn_t,
                "think_time": think_t,
                "body_time": body_t,
                "total_time": total_wall,
                "is_cookie": is_cookie,
                "has_usage": has_usage,
                "p_tk": p_tk,
                "cached_tk": clean_cached,
                "cache_pct": cache_pct,
                "th_tk": th_tk,
                "out_tk": out_tk,
                "out_chars": out_chars,
                "tps": tps,
                "tps_approx": tps_approx,
                "detail": " ".join(parts_det),
            },
            channel=ch,
            created_at=self.created_wall,
        )

    def finalize_error(self, err_code: int, err_msg: str, preflight: bool = False) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        if preflight or not self.unique_creds:
            cred = "未发往上游"
            cred_note = "前置校验拦截"
        elif len(self.unique_creds) == 1:
            cred = self.unique_creds[0]
            cred_note = f"单凭证重试×{self.retries}" if self.retries > 0 else "未触发重试"
        else:
            cred = f"试 {len(self.unique_creds)} 凭证均败"
            cred_note = f"止于 {self.last_cred_short}"

        parts_det: List[str] = []
        if self.failed_steps:
            parts_det.append(f"尝试轨迹：{' → '.join(self.failed_steps)}。")
        clean_reason = html.escape(str(err_msg or "未知错误").strip()[:600])
        parts_det.append(f"<b>失败原因（HTTP {err_code} · 总耗时 {total_wall:.2f}s）</b>：{clean_reason}")

        t_short, t_full = self._build_time_strs()
        ch = "cookie" if (self.last_channel == "cookie" or cred == "Studio Cookie") else "express"
        self.finish_live(
            status="error",
            finish_reason=f"HTTP {int(err_code or 500)}",
            total_sec=total_wall,
            note=str(err_msg or "上游异常").strip()[:72],
        )
        call_recorder.record(
            {
                "time": t_short,
                "full_time": t_full,
                "group": "err",
                "badge_type": "err",
                "retries": self.retries,
                "code": int(err_code or 500),
                "finish_reason": "error",
                "raw_finish_reason": f"HTTP_{int(err_code or 500)}",
                "model": self.model_display,
                "channel": ch,
                "sandbox": self.sandbox,
                "prefill": self.prefill,
                "search": self.search,
                "tool_calls": False,
                "thinking": self.thinking_label,
                "think_style": self.think_style,
                "mode": self.mode_label,
                "mode_class": self.mode_class,
                "cred": cred,
                "cred_note": cred_note,
                "is_unary": self.is_unary,
                "ttft": None,
                "gen_time": total_wall,
                "conn_time": total_wall,
                "think_time": 0.0,
                "body_time": 0.0,
                "total_time": total_wall,
                "is_cookie": (ch == "cookie"),
                "has_usage": False,
                "p_tk": 0,
                "cached_tk": 0,
                "cache_pct": 0.0,
                "th_tk": 0,
                "out_tk": 0,
                "out_chars": 0,
                "tps": None,
                "tps_approx": False,
                "detail": " ".join(parts_det),
            },
            channel=ch,
            created_at=self.created_wall,
        )

    def finalize_cancelled(self, partial_output: bool = False, by_console: bool = False) -> None:
        if self.finalized:
            return
        self.finalized = True
        total_wall = round(max(0.01, time.monotonic() - self.start_mono), 2)
        cred = self.last_cred_short or "—"
        if by_console:
            cred_note = "控制台手动中止"
            msg = f"<b>控制台手动中止（HTTP 499 · 已耗时 {total_wall:.2f}s）</b>：已从实况脉冲舱触发中止信号并安全回收上游连接。"
        elif partial_output:
            cred_note = "输出中途断开"
            msg = f"<b>客户端中途断开（HTTP 499 · 已耗时 {total_wall:.2f}s）</b>：流式输出期间连接关闭，后台已安全回收上游流。"
        else:
            cred_note = "前端主动停止"
            msg = f"<b>前端手动停止（HTTP 499 · 已耗时 {total_wall:.2f}s）</b>：首包产出前检测到客户端断开，后台已立即取消上游任务与重试。"

        t_short, t_full = self._build_time_strs()
        ch = "cookie" if (self.last_channel == "cookie" or cred == "Studio Cookie") else "express"
        self.finish_live(
            status="cancelled",
            finish_reason="CANCELLED",
            total_sec=total_wall,
            note=cred_note,
        )
        call_recorder.record(
            {
                "time": t_short,
                "full_time": t_full,
                "group": "cancel",
                "badge_type": "cancel",
                "retries": self.retries,
                "code": 499,
                "finish_reason": "cancelled",
                "raw_finish_reason": "CANCELLED",
                "model": self.model_display,
                "channel": ch,
                "sandbox": self.sandbox,
                "prefill": self.prefill,
                "search": self.search,
                "tool_calls": False,
                "thinking": self.thinking_label,
                "think_style": self.think_style,
                "mode": self.mode_label,
                "mode_class": self.mode_class,
                "cred": cred,
                "cred_note": cred_note,
                "is_unary": self.is_unary,
                "ttft": None,
                "gen_time": total_wall,
                "conn_time": total_wall,
                "think_time": 0.0,
                "body_time": 0.0,
                "total_time": total_wall,
                "is_cookie": (ch == "cookie"),
                "has_usage": False,
                "p_tk": 0,
                "cached_tk": 0,
                "cache_pct": 0.0,
                "th_tk": 0,
                "out_tk": 0,
                "out_chars": 0,
                "tps": None,
                "tps_approx": False,
                "detail": msg,
            },
            channel=ch,
            created_at=self.created_wall,
        )
