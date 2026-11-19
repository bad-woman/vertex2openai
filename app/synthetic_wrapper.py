"""
Function Call 通用抗截断传输模块 (Compose Reply Anti-Truncation Wrapper)

核心机制：
1. 固定合成工具名 `compose_reply`（仅在客户端工具同名冲突时生成随机后缀），保证 Vertex 隐式前缀缓存命中；
2. 零侵入消息历史，与所有客户端及预填充模式正交解耦；
3. 零前瞻确定性流式词法状态机（代码块围栏/行内代码作用域隔离 + 弹性括号配对栈 + HTML 标签开闭计数）；
4. 流式动态隔离缓冲（In-Flight 只防真死循环与显式工具泄漏）+ 流末严格尾锚定悬空闭合修剪（At-EOF），彻底杜绝误伤 Python 代码与酒馆前端美化；
5. 支持 Express SDK 与 Studio Cookie 双通道的真流式 (partialArgs)、假流式与非流式透明拆包。
"""
import json
import random
import re
import time
import unicodedata
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types
import model_capabilities as mc

DEFAULT_SYNTHETIC_TOOL_NAME = "compose_reply"
DOWNSTREAM_ID_PREFIX = "chatcmpl-bad_woman"

CONTENT_KEYS = (
    "content",
    "text",
    "continuation",
    "narrative",
    "response",
    "output",
    "answer",
    "result",
    "message",
)

_EXPLICIT_GOOGLE_SEARCH_NAMES = {
    "google_search",
    "googlesearch",
    "google_search_retrieval",
}

_BUILTIN_SEARCH_NAMES = {
    "google_search",
    "googlesearch",
    "google_search_retrieval",
    "web_search",
    "websearch",
    "search_web",
    "browse",
    "search",
}


def is_builtin_search_tool(tool: Any) -> bool:
    """准确识别 OpenAI tools 列表里的内置搜索占位工具，避免误劫持带参数 Schema 的客户端自定义函数（如 search / browse）。"""
    if not isinstance(tool, dict):
        return False
    t_type = str(tool.get("type") or "").strip().lower().replace("-", "_")
    if t_type and t_type != "function" and t_type in _BUILTIN_SEARCH_NAMES:
        return True
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
    raw_name = (fn.get("name") if fn is not None else None) or tool.get("name") or ""
    norm_name = str(raw_name).strip().lower().replace("-", "_")
    if not norm_name or norm_name not in _BUILTIN_SEARCH_NAMES:
        return False
    if norm_name in _EXPLICIT_GOOGLE_SEARCH_NAMES:
        return True
    # 对于 search / browse / web_search 等通用名：若带有效 parameters.properties，说明是客户端自定义函数，不可劫持
    params = (fn.get("parameters") if fn is not None else None) or tool.get("parameters")
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict) and len(props) > 0:
            return False
    return True

_VOID_HTML_TAGS = {
    "br", "hr", "img", "input", "meta", "link", "area", "base",
    "col", "embed", "param", "source", "track", "wbr",
}

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CLOSE_TO_OPEN = {")": "(", "]": "[", "}": "{"}
_CLOSER_CHARS = set("})]")
_PURE_LAYOUT_CHARS = set(" \t\r-_=*~.─━═—…#*+/|\\0")
_KNOWN_LEAK_TOKENS = ("endasi", "end_assistant", "исповедь", "<|endoftext|>", "<|im_end|>")

_OPEN_TAG_HEAD_RE = re.compile(
    r"^<([A-Za-z\u4e00-\u9fff\u3040-\u30ff][A-Za-z0-9_\-~.:\u4e00-\u9fff\u3040-\u30ff]{0,48})"
    r"(?:[\s/>]|$)"
)
_CLOSE_TAG_FULL_RE = re.compile(
    r"^<\s*/\s*([A-Za-z\u4e00-\u9fff\u3040-\u30ff][A-Za-z0-9_\-~.:\u4e00-\u9fff\u3040-\u30ff]{0,48})\s*>$"
)


def _check_sdk_supports_stream_fc_args() -> bool:
    try:
        fields = getattr(types.FunctionCallingConfig, "model_fields", None) or {}
        if "stream_function_call_arguments" in fields:
            return True
        ann = getattr(types.FunctionCallingConfig, "__annotations__", None) or {}
        return "stream_function_call_arguments" in ann
    except Exception:
        return False


def _check_sdk_supports_params_json_schema() -> bool:
    try:
        fields = getattr(types.FunctionDeclaration, "model_fields", None) or {}
        if "parameters_json_schema" in fields:
            return True
        ann = getattr(types.FunctionDeclaration, "__annotations__", None) or {}
        return "parameters_json_schema" in ann
    except Exception:
        return False


def _resolve_schema_prop_ordering_key() -> Optional[str]:
    try:
        schema_cls = getattr(types, "Schema", None)
        if schema_cls is None:
            return "property_ordering"
        fields = getattr(schema_cls, "model_fields", None) or {}
        if "property_ordering" in fields:
            return "property_ordering"
        if "propertyOrdering" in fields:
            return "propertyOrdering"
        ann = getattr(schema_cls, "__annotations__", None) or {}
        if "property_ordering" in ann:
            return "property_ordering"
        if "propertyOrdering" in ann:
            return "propertyOrdering"
    except Exception:
        pass
    return None


_SDK_SUPPORTS_STREAM_FC_ARGS = _check_sdk_supports_stream_fc_args()
_SDK_SUPPORTS_PARAMS_JSON_SCHEMA = _check_sdk_supports_params_json_schema()
_SCHEMA_PROP_ORDER_KEY = _resolve_schema_prop_ordering_key()

_MALFORMED_TOOL_FINISH_REASONS = frozenset({
    "MALFORMED_FUNCTION_CALL",
    "MALFORMED_RESPONSE",
    "UNEXPECTED_TOOL_CALL",
})


def is_malformed_tool_finish(resp_or_chunk: Any) -> bool:
    """检测上游候选是否因工具调用/解码格式畸形中断（MALFORMED_FUNCTION_CALL / MALFORMED_RESPONSE / UNEXPECTED_TOOL_CALL）。"""
    cands = getattr(resp_or_chunk, "candidates", None) or []
    for cand in cands:
        fr = getattr(cand, "finish_reason", None)
        if not fr:
            continue
        fr_str = (getattr(fr, "name", None) or str(fr)).strip().upper()
        if any(tag in fr_str for tag in _MALFORMED_TOOL_FINISH_REASONS):
            return True
    return False


@dataclass
class SyntheticSandboxSpec:
    enabled: bool = False
    emit_tool_name: str = ""
    strip_client_tools: bool = False
    real_tool_names: List[str] = field(default_factory=list)
    prefill_text: str = field(default="", repr=False)
    has_prefill: bool = False
    is_streaming: bool = False
    bypass_reason: str = ""


def make_bad_woman_response_id(default_prefix: str = "chatcmpl", sandbox_enabled: bool = False) -> str:
    """生成下游 OpenAI 响应 ID：开启抗截断时使用 chatcmpl-bad_woman 前缀。"""
    ts = int(time.time())
    suffix = uuid.uuid4().hex[:8]
    if sandbox_enabled:
        return f"{DOWNSTREAM_ID_PREFIX}-{ts}-{suffix}"
    return f"{default_prefix}-{ts}-{random.randint(1000, 9999)}"


def should_enable_sandbox(model_name: str, request_obj: Any, settings: Optional[dict]) -> Tuple[bool, str]:
    """判定当前请求是否启用抗截断传输工具。"""
    settings = settings or {}
    if not bool(settings.get("synthetic_sandbox_enabled", False)):
        return False, "未开启"

    raw_model = str(getattr(request_obj, "model", None) or model_name or "").strip()
    profile = mc.get_profile(raw_model)
    if profile.get("is_image"):
        return False, "生图模型自动旁路"

    if raw_model.lower().endswith("-search"):
        return False, "联网搜索后缀自动旁路"

    for tool in (getattr(request_obj, "tools", None) or []):
        if is_builtin_search_tool(tool):
            return False, "内置搜索工具自动旁路"

    resp_fmt = getattr(request_obj, "response_format", None)
    if resp_fmt is not None:
        fmt_type = resp_fmt.get("type", "") if isinstance(resp_fmt, dict) else getattr(resp_fmt, "type", "")
        if str(fmt_type).lower() in ("json_object", "json_schema"):
            return False, "强制 JSON 格式输出自动旁路"

    tc = getattr(request_obj, "tool_choice", None)
    if isinstance(tc, dict) or (isinstance(tc, str) and tc.strip().lower() == "required"):
        return False, "客户端强制指定工具调用自动旁路"

    if (getattr(request_obj, "n", None) or 1) != 1:
        return False, "多候选请求自动旁路"

    return True, ""


def create_sandbox_spec(model_name: str, request_obj: Any, settings: Optional[dict]) -> SyntheticSandboxSpec:
    """为单次请求构建抗截断工具规格（默认使用固定名 compose_reply 以命中 Vertex 隐式缓存）。"""
    settings = settings or {}
    ok, reason = should_enable_sandbox(model_name, request_obj, settings)
    if not ok:
        return SyntheticSandboxSpec(enabled=False, bypass_reason=reason)

    tc = getattr(request_obj, "tool_choice", None)
    tc_is_none = isinstance(tc, str) and tc.strip().lower() == "none"
    has_tool_history = any(
        getattr(m, "role", None) == "tool" or bool(getattr(m, "tool_calls", None))
        for m in (getattr(request_obj, "messages", None) or [])
    )
    # 历史含工具轮次时，即使 tool_choice=none 也要保留工具声明（由 allowed_function_names=[compose_reply] 禁止本轮调用客户端工具）
    strip_client = bool(tc_is_none and not has_tool_history)

    all_client_names: List[str] = []
    for tool in (getattr(request_obj, "tools", None) or []):
        if not isinstance(tool, dict) or is_builtin_search_tool(tool):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("name") or "").strip()
        if name and name not in all_client_names:
            all_client_names.append(name)

    real_names = [] if tc_is_none else list(all_client_names)

    if any(n.startswith(DEFAULT_SYNTHETIC_TOOL_NAME) for n in all_client_names):
        emit_name = f"compose_{uuid.uuid4().hex[:24]}"
        while emit_name in all_client_names:
            emit_name = f"compose_{uuid.uuid4().hex[:24]}"
    else:
        emit_name = DEFAULT_SYNTHETIC_TOOL_NAME

    return SyntheticSandboxSpec(
        enabled=True,
        emit_tool_name=emit_name,
        strip_client_tools=strip_client,
        real_tool_names=real_names,
        is_streaming=bool(getattr(request_obj, "stream", False)),
    )


def build_emit_tool_declaration(
    emit_tool_name: str,
    for_cookie: bool = False,
) -> Dict[str, Any]:
    """构建单字段 (content) 抗截断传输工具声明：
    1. 全局静态固定工具名与极简中性描述，100% 保证命中 Vertex 隐式前缀缓存；
    2. 去掉冗余的 done 布尔字段，消除长文末尾从 content 切换到 ,done:true} 时的闭合悬崖与恐慌循环；
    3. Express 通道在 SDK 支持时采用 parameters_json_schema + additionalProperties: False（只允许 content 单键），
       Cookie 通道或旧版 SDK 自动回退为标准 parameters 单字段 Schema。
    """
    desc = (
        "Deliver the complete user-visible reply as text; "
        "end the call when the requested content is complete."
    )
    content_desc = (
        "The requested reply itself, preserving its language, sections, formatting and markup."
    )
    if not for_cookie and _SDK_SUPPORTS_PARAMS_JSON_SCHEMA:
        return {
            "name": emit_tool_name,
            "description": desc,
            "parameters_json_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": content_desc,
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        }
    return {
        "name": emit_tool_name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": content_desc,
                },
            },
            "required": ["content"],
        },
    }


def build_sandbox_control_instruction(
    emit_tool_name: str,
    has_prefill: bool = False,
    unclosed_tag: Optional[str] = None,
    is_keep_turn: bool = False,
    has_callable_client_tools: bool = False,
) -> str:
    """构建极简中性传输格式指令（单字段 content，去除易诱发模型自检复读的闭合焦虑措辞）。"""
    target_text = "only the new continuation text" if has_prefill else "the entire user-visible answer"
    if has_callable_client_tools:
        transport_head = (
            f"Transport format for this reply: put all user-visible text, if any, in exactly one call of {emit_tool_name}; "
            f"its content string holds {target_text}, preserving all requested formatting, markup, language and sections. "
            f"{emit_tool_name} only transports text and performs no external action. "
            f"Call the other tools whenever the task needs them, with or without {emit_tool_name}."
        )
    else:
        transport_head = (
            f"Transport format for this reply: call {emit_tool_name} exactly once. "
            f"Put {target_text} in its content string, preserving all requested formatting, markup, language and sections. "
            f"The function only transports text and performs no external action."
        )

    transport_tail = (
        " This transport does not change the task. Do not add unsolicited commentary about this transport, "
        "tool selection or serialization. Once the requested content is complete, end the call without "
        "repeating finished content or adding filler. Preserve repetition explicitly requested by the task."
    )
    transport_line = transport_head + transport_tail

    if has_prefill and unclosed_tag:
        pos_ref = "上一条回复已写好的开头" if is_keep_turn else "下方已写好的开头"
        cot_line = (
            f"（格式硬性要求：{pos_ref}正停在未闭合的 <{unclosed_tag}> 标签内部；"
            f"请直接从断点处继续写出 <{unclosed_tag}> 内的完整思考内容，"
            f"写完后用 </{unclosed_tag}> 闭合，然后才开始写正文；"
            f"不要再次重复输出 <{unclosed_tag}> 开标签，不要跳过思考直接写正文，也不允许只写一个空标签。）"
        )
        return f"{cot_line}\n{transport_line}"

    return transport_line


def supports_stream_fc_args(model_name: str, check_sdk: bool = True) -> bool:
    """按 Google Vertex AI 官方规范判断模型与 SDK 是否支持 streamFunctionCallArguments。
    官方文档明确：Gemini 3+ / 4+ 文本模型（含 3.5-flash-lite）均支持，唯独 gemini-3.1-flash-lite 与生图模型不支持；
    且该参数仅可用于流式接口 streamGenerateContent，不可传入一元 generateContent。
    """
    if check_sdk and not _SDK_SUPPORTS_STREAM_FC_ARGS:
        return False
    prof = mc.get_profile(model_name or "")
    return bool(prof.get("supports_stream_fc_args", False) and not prof.get("is_image", False))


def strip_stream_fc_args_from_config(gen_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """返回剥离了 stream_function_call_arguments 的配置字典副本（供一元 generate_content 或不支持该特性的模型降级使用）。"""
    if not isinstance(gen_config_dict, dict):
        return gen_config_dict
    tc = gen_config_dict.get("tool_config")
    if not isinstance(tc, dict):
        return gen_config_dict
    fcc = tc.get("function_calling_config")
    if not isinstance(fcc, dict) or (
        "stream_function_call_arguments" not in fcc and "streamFunctionCallArguments" not in fcc
    ):
        return gen_config_dict
    new_fcc = {
        k: v
        for k, v in fcc.items()
        if k not in ("stream_function_call_arguments", "streamFunctionCallArguments")
    }
    new_tc = dict(tc)
    new_tc["function_calling_config"] = new_fcc
    new_cfg = dict(gen_config_dict)
    new_cfg["tool_config"] = new_tc
    return new_cfg


def is_stream_fc_args_unsupported_error(err: Any) -> bool:
    """识别上游返回的“不支持 stream_function_call_arguments”协议错误（用于无损剥离该标志重试，同时保留抗截断工具）。"""
    s = f"{getattr(err, 'status', '')} {getattr(err, 'message', '')} {err}".lower()
    has_field_hint = any(
        k in s
        for k in (
            "stream_function_call_arguments",
            "streamfunctioncallarguments",
            "streaming function call",
            "partial_args",
            "partialargs",
        )
    )
    if not has_field_hint:
        return False
    return any(
        r in s
        for r in (
            "not supported",
            "unsupported",
            "unknown field",
            "unknown name",
            "unrecognized",
            "invalid argument",
            "invalid_argument",
            "cannot",
            "only supported",
        )
    )


def inject_express_gen_config(
    gen_config_dict: Dict[str, Any],
    spec: Optional[SyntheticSandboxSpec],
    enable_stream_fc_args: bool = False,
) -> None:
    """向 Express SDK 的 generation_config 字典注入 compose_reply 声明与 tool_config（不注入 stop_sequences）。"""
    if not spec or not spec.enabled:
        return

    emit_decl = build_emit_tool_declaration(
        spec.emit_tool_name,
        for_cookie=False,
    )

    existing_decls: List[Dict[str, Any]] = []
    if not spec.strip_client_tools:
        for tool_group in (gen_config_dict.get("tools") or []):
            if isinstance(tool_group, dict) and isinstance(tool_group.get("function_declarations"), list):
                existing_decls.extend(tool_group["function_declarations"])

    all_decls = existing_decls + [emit_decl]
    gen_config_dict["tools"] = [{"function_declarations": all_decls}]

    fc_cfg: Dict[str, Any] = {"mode": "ANY"}
    if not spec.real_tool_names:
        fc_cfg["allowed_function_names"] = [spec.emit_tool_name]
    if enable_stream_fc_args and _SDK_SUPPORTS_STREAM_FC_ARGS:
        fc_cfg["stream_function_call_arguments"] = True

    gen_config_dict["tool_config"] = {"function_calling_config": fc_cfg}


# ============================================================================
# 确定性流式词法状态追踪器与分期防溢出引擎（保护 Python 代码与酒馆前端美化）
# ============================================================================


class CharMeta:
    """单字符词法注解，用于精准区分“合法配对闭合”与“悬空溢出残片”。"""
    __slots__ = (
        "in_code",
        "is_orphan_closer",
        "orphan_dquote",
        "orphan_close_tag_start",
        "paired_close_tag",
    )

    def __init__(self) -> None:
        self.in_code: bool = False
        self.is_orphan_closer: bool = False
        self.orphan_dquote: bool = False
        self.orphan_close_tag_start: int = -1
        self.paired_close_tag: bool = False


class LexicalState:
    """零前瞻确定性有限状态自动机（DFA），满足任意流式切分恒等性：feed(A + B) === feed(A); feed(B)。
    实时追踪：
      1. CommonMark 围栏代码块（``` / ~~~，含嵌套反引号长度匹配与单行 ```inline``` 识别）；
      2. 行内反引号代码作用域（含颜文字孤立反引号跨空行防锁死恢复）；
      3. 代码块外弹性括号配对栈 ()[]{}，精确识别未配对悬空右括号；
      4. 同段半角直双引号奇偶配对与 HTML/XML 标签开闭深度计数（含属性引号感知）。
    """

    def __init__(self) -> None:
        self.in_fence: bool = False
        self.fence_char: str = ""
        self.fence_len: int = 0

        self.at_bol: bool = True
        self.bol_spaces: int = 0
        self.run_char: str = ""
        self.run_len: int = 0
        self.run_at_bol: bool = False

        self.pending_open_char: str = ""
        self.pending_open_len: int = 0
        self.closing_fence_ok: bool = False

        self.inline_tick_len: int = 0
        self.inline_newlines: int = 0
        self.inline_chars: int = 0
        self.consec_newlines: int = 0

        self.bracket_stack: List[str] = []
        self.dquote_open: bool = False
        self.prev_backslash: bool = False

        self.tag_buf: str = ""
        self.tag_overflow: bool = False
        self.tag_rel_offset: int = 0
        self.tag_quote: str = ""
        self.tag_prev_ch: str = ""
        self.html_open_counts: Dict[str, int] = {}

    def clone(self) -> "LexicalState":
        c = LexicalState()
        c.in_fence = self.in_fence
        c.fence_char = self.fence_char
        c.fence_len = self.fence_len
        c.at_bol = self.at_bol
        c.bol_spaces = self.bol_spaces
        c.run_char = self.run_char
        c.run_len = self.run_len
        c.run_at_bol = self.run_at_bol
        c.pending_open_char = self.pending_open_char
        c.pending_open_len = self.pending_open_len
        c.closing_fence_ok = self.closing_fence_ok
        c.inline_tick_len = self.inline_tick_len
        c.inline_newlines = self.inline_newlines
        c.inline_chars = self.inline_chars
        c.consec_newlines = self.consec_newlines
        c.bracket_stack = list(self.bracket_stack)
        c.dquote_open = self.dquote_open
        c.prev_backslash = self.prev_backslash
        c.tag_buf = self.tag_buf
        c.tag_overflow = self.tag_overflow
        c.tag_rel_offset = self.tag_rel_offset
        c.tag_quote = self.tag_quote
        c.tag_prev_ch = self.tag_prev_ch
        c.html_open_counts = dict(self.html_open_counts)
        return c

    def _flush_run_before(self) -> None:
        if self.run_len == 0:
            return
        rc = self.run_char
        rl = self.run_len
        was_bol = self.run_at_bol
        self.run_char = ""
        self.run_len = 0
        self.run_at_bol = False

        if self.in_fence:
            if was_bol and rc == self.fence_char and rl >= self.fence_len:
                self.closing_fence_ok = True
            return

        if self.pending_open_len > 0:
            if self.pending_open_char == "`" and rc == "`":
                if rl == self.pending_open_len:
                    self.pending_open_char = ""
                    self.pending_open_len = 0
                else:
                    self.inline_tick_len = self.pending_open_len
                    self.inline_newlines = 0
                    self.inline_chars = 0
                    self.pending_open_char = ""
                    self.pending_open_len = 0
            return

        if self.inline_tick_len > 0:
            if rc == "`" and rl == self.inline_tick_len:
                self.inline_tick_len = 0
                self.inline_newlines = 0
                self.inline_chars = 0
            return

        if was_bol and rl >= 3 and rc in ("`", "~"):
            self.pending_open_char = rc
            self.pending_open_len = rl
        elif rc == "`":
            self.inline_tick_len = rl
            self.inline_newlines = 0
            self.inline_chars = 0

    def is_currently_in_code(self) -> bool:
        if self.in_fence:
            # 若正好停在无尾换行的合法闭合围栏末尾，视为围栏已完成闭合
            if self.closing_fence_ok:
                return False
            if self.run_at_bol and self.run_char == self.fence_char and self.run_len >= self.fence_len:
                return False
            return True
        if self.pending_open_len > 0 or self.inline_tick_len > 0:
            return True
        if self.run_len > 0 and (self.run_char == "`" or (self.run_at_bol and self.run_len >= 3)):
            return True
        return False

    def feed(self, text: str) -> None:
        if not text:
            return
        for ch in text:
            self._step(ch, None, 0)

    def scan(self, text: str) -> Tuple[List[CharMeta], "LexicalState"]:
        nxt = self.clone()
        metas: List[CharMeta] = []
        for idx, ch in enumerate(text):
            m = CharMeta()
            nxt._step(ch, m, idx)
            metas.append(m)
        return metas, nxt

    def _step(self, ch: str, meta: Optional[CharMeta], idx: int) -> None:
        can_start_tilde = (
            self.at_bol and self.bol_spaces <= 7
            and self.inline_tick_len == 0 and self.pending_open_len == 0
        )
        if ch == "`" or (ch == "~" and (self.run_char == "~" or can_start_tilde)):
            if self.run_len == 0:
                self.run_char = ch
                self.run_len = 1
                self.run_at_bol = (self.at_bol and self.bol_spaces <= 7)
                self.at_bol = False
            elif self.run_char == ch:
                self.run_len += 1
            else:
                self._flush_run_before()
                self.run_char = ch
                self.run_len = 1
                self.run_at_bol = False
            self.consec_newlines = 0
            self.prev_backslash = False
            if self.tag_buf:
                self.tag_rel_offset += 1
                if not self.tag_overflow and len(self.tag_buf) < 96:
                    self.tag_buf += ch
                else:
                    self.tag_overflow = True
            if meta is not None:
                meta.in_code = True
            return

        if self.run_len > 0:
            self._flush_run_before()

        if ch == "\n":
            self.consec_newlines += 1
            was_in_code = (
                self.in_fence or self.pending_open_len > 0 or self.inline_tick_len > 0
            )
            if self.in_fence:
                if self.closing_fence_ok:
                    self.in_fence = False
                    self.fence_char = ""
                    self.fence_len = 0
                    self.closing_fence_ok = False
            else:
                if self.pending_open_len >= 3:
                    self.in_fence = True
                    self.fence_char = self.pending_open_char
                    self.fence_len = self.pending_open_len
                    self.pending_open_char = ""
                    self.pending_open_len = 0
                    was_in_code = True
                elif self.inline_tick_len > 0:
                    self.inline_newlines += 1
                    if self.consec_newlines >= 2 or self.inline_newlines >= 2 or self.inline_chars >= 80:
                        self.inline_tick_len = 0
                        self.inline_newlines = 0
                        self.inline_chars = 0

            if not self.in_fence:
                self.dquote_open = False
                if self.consec_newlines >= 2:
                    self.tag_buf = ""
                    self.tag_overflow = False
                    self.tag_rel_offset = 0
                    self.tag_quote = ""
                    self.tag_prev_ch = ""
                elif self.tag_buf:
                    self.tag_rel_offset += 1
                    if not self.tag_overflow and len(self.tag_buf) < 96:
                        self.tag_buf += ch
                    else:
                        self.tag_overflow = True

            self.at_bol = True
            self.bol_spaces = 0
            self.closing_fence_ok = False
            self.prev_backslash = False
            if meta is not None:
                meta.in_code = was_in_code
            return

        if ch != "\r":
            self.consec_newlines = 0

        if self.at_bol:
            if ch == " ":
                self.bol_spaces += 1
            elif ch == "\t":
                self.bol_spaces += 4
            elif ch == "\r":
                pass
            else:
                self.at_bol = False

        if self.in_fence:
            if self.closing_fence_ok and ch not in (" ", "\t", "\r"):
                self.closing_fence_ok = False
            self.prev_backslash = False
            if meta is not None:
                meta.in_code = True
            return

        if self.pending_open_len > 0:
            self.prev_backslash = False
            if meta is not None:
                meta.in_code = True
            return

        if self.inline_tick_len > 0:
            self.inline_chars += 1
            if self.inline_chars <= 200:
                self.prev_backslash = False
                if meta is not None:
                    meta.in_code = True
                return
            # 孤立未闭合的单行反引号超过 200 字符自动失效，防止致盲后续整段文本的防溢出检测
            self.inline_tick_len = 0
            self.inline_newlines = 0
            self.inline_chars = 0

        if meta is not None:
            meta.in_code = False

        if ch == '"' and not self.prev_backslash:
            if self.dquote_open:
                self.dquote_open = False
                if meta is not None:
                    meta.orphan_dquote = False
            else:
                self.dquote_open = True
                if meta is not None:
                    meta.orphan_dquote = True

        self.prev_backslash = (ch == "\\" and not self.prev_backslash)

        if ch in _OPEN_TO_CLOSE:
            if len(self.bracket_stack) < 512:
                self.bracket_stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            want = _CLOSE_TO_OPEN[ch]
            if self.bracket_stack and self.bracket_stack[-1] == want:
                self.bracket_stack.pop()
            else:
                found_idx = -1
                low_bound = max(0, len(self.bracket_stack) - 3)
                for k in range(len(self.bracket_stack) - 1, low_bound - 1, -1):
                    if self.bracket_stack[k] == want:
                        found_idx = k
                        break
                if found_idx != -1:
                    del self.bracket_stack[found_idx:]
                elif meta is not None:
                    meta.is_orphan_closer = True

        if ch == "<" and not self.tag_quote:
            self.tag_buf = "<"
            self.tag_overflow = False
            self.tag_rel_offset = 0
            self.tag_quote = ""
            self.tag_prev_ch = ""
        elif self.tag_buf:
            self.tag_rel_offset += 1
            if self.tag_rel_offset == 1 and not (
                ch.isalpha() or ch in ("/", "!", "?") or ("\u4e00" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff")
            ):
                # 排除数学小于号等非标签 `< 5` / `< b`，防止后续撇号被误认为属性引号
                self.tag_buf = ""
                self.tag_overflow = False
                self.tag_rel_offset = 0
                self.tag_quote = ""
                self.tag_prev_ch = ""
                return
            if not self.tag_overflow and len(self.tag_buf) < 96:
                self.tag_buf += ch
            else:
                self.tag_overflow = True

            if self.tag_quote:
                if ch == self.tag_quote:
                    self.tag_quote = ""
                if ch not in (" ", "\t", "\r", "\n"):
                    self.tag_prev_ch = ch
            elif ch in ('"', "'") and len(self.tag_buf) > 2 and not self.tag_buf.startswith("<!--"):
                self.tag_quote = ch
                self.tag_prev_ch = ch
            elif ch == ">":
                if self.tag_buf.startswith("<!--") and not (
                    self.tag_buf.endswith("-->") or (self.tag_overflow and self.tag_prev_ch == "-")
                ):
                    self.tag_prev_ch = ch
                else:
                    full_t = self.tag_buf
                    is_ovf = self.tag_overflow
                    rel_off = self.tag_rel_offset
                    prev_non_ws = self.tag_prev_ch
                    self.tag_buf = ""
                    self.tag_overflow = False
                    self.tag_rel_offset = 0
                    self.tag_quote = ""
                    self.tag_prev_ch = ""

                    if not full_t.startswith(("<!--", "<!", "<?")):
                        m_close = _CLOSE_TAG_FULL_RE.match(full_t) if not is_ovf else None
                        if m_close:
                            tname = m_close.group(1).lower()
                            if tname not in _VOID_HTML_TAGS:
                                cur_cnt = self.html_open_counts.get(tname, 0)
                                if cur_cnt > 0:
                                    self.html_open_counts[tname] = cur_cnt - 1
                                    if meta is not None:
                                        meta.paired_close_tag = True
                                elif meta is not None:
                                    meta.orphan_close_tag_start = idx - rel_off
                        elif not (prev_non_ws == "/" or full_t.rstrip().endswith("/>")):
                            m_open = _OPEN_TAG_HEAD_RE.match(full_t)
                            if m_open:
                                tname = m_open.group(1).lower()
                                if tname not in _VOID_HTML_TAGS and not tname.startswith(
                                    ("http:", "https:", "mailto:", "ftp:")
                                ):
                                    self.html_open_counts[tname] = min(
                                        self.html_open_counts.get(tname, 0) + 1, 512
                                    )
            elif self.tag_rel_offset > 2048:
                self.tag_buf = ""
                self.tag_overflow = False
                self.tag_rel_offset = 0
                self.tag_quote = ""
                self.tag_prev_ch = ""
            elif ch not in (" ", "\t", "\r", "\n"):
                self.tag_prev_ch = ch


def prefill_appears_repeated(prefill_text: str, output_head: str) -> bool:
    """判断模型输出开头是否原样复述了 prefill（用于决定是否用 prefill 预热词法状态机）。"""
    if not prefill_text or not output_head:
        return False
    p = prefill_text.lstrip(" \t\r\n")
    o = output_head.lstrip(" \t\r\n")
    if not p or not o:
        return False
    probe_len = min(len(p), len(o), 24)
    if probe_len >= 4 and p[:probe_len] == o[:probe_len]:
        return True
    return False


# ============================================================================
# 融合尾部保护引擎 (SyntheticTailGuard)：
# 结合 vertex2openai-access v8.2.5 TailGuard 低误杀架构与 Gemini 4 实战跑飞补丁：
#   1. 流中零激进规则：不因闭合标签(</div>)或闭合括号(]})截断，周期重复阈值收紧至 >=12次 且 >=600字；
#   2. 未闭合行内反引号超过 100 字或跨行时回溯取消保护，防止乱码里的孤立 ` 致盲整段检测；
#   3. 四类高置信实锤狙击：
#      - 显式工具名/工具调用自检独白与控制伪词；
#      - 48 字高密度货币/符号乱码窗 (find_symbol_salad)；
#      - 英文自检句尾粘连孤立闭合括号 (_META_TAIL，含 »/》/?> 变体) 与同义词恐慌循环；
#      - 句末孤立 } + 400 字多语种收尾念叨观察窗 (_stray_close + _is_chatter)；
#   4. 仅在流末 (flush) 才清理孤立未配对的尾部闭合残片 (]}) / Done!}}) 与围栏粘连尾。
# ============================================================================

TAIL_HOLD = 128            # 流式暂扣尾窗长度（覆盖单句自检 + 符号乱码窗）
TAIL_HISTORY = 1600        # 重复检测与句末前瞻回看长度
TAIL_MAX_UNIT = 200        # 重复检测最大单元长度
TAIL_MIN_REPEATS = 12      # 周期死循环最小重复次数
TAIL_MIN_SPAN = 600        # 周期死循环最小总跨度字符数
TAIL_SUSPECT = 400         # 句末孤立 } 后的多语种收尾念叨观察窗长度
TAIL_INLINE_WAIT = 100     # 行内反引号同一行内等待闭合的最大字符数

_DECO_LAYOUT_EXTRA = set("·•‣⁃～〜―⁃•◦▪▫■□●○◆◇★☆✦✧❖✿❀")


def _is_pure_layout_unit(non_ws: List[str]) -> bool:
    """仅豁免真正的排版分割线/进度条/装饰符号（如 ─━═█░✦◆·），绝不豁免 ! ? > < 》 » , : ) } ] 等乱码/闭合符号。"""
    if not non_ws:
        return True
    for ch in non_ws:
        if ch in _PURE_LAYOUT_CHARS or ch in _DECO_LAYOUT_EXTRA:
            continue
        cp = ord(ch)
        if 0x2500 <= cp <= 0x27BF:
            continue
        if unicodedata.category(ch) == "Zs":
            continue
        return False
    return True


def _short_reason_preview(snippet: str, limit: int = 36) -> str:
    clean = (snippet or "").replace("\r", "\\r").replace("\n", "\\n")
    return clean[:limit] + ("…" if len(clean) > limit else "")


def _build_leak_pattern(tool_name: str = "") -> re.Pattern:
    """构造显式工具名/工具调用自检独白与控制伪词正则（不含易误杀正常编程讨论的宽泛词组）。"""
    names = ["compose_reply", "router_emit"]
    tn = (tool_name or "").strip()
    if tn and tn not in names:
        names.append(tn)
    name_alt = "|".join(re.escape(n) for n in names)
    return re.compile(
        r"(?:"
        r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?"
        r"\bLet['’]s\s+(?:check|finalize|make\s+sure|verify|review|ensure|confirm"
        r"|finish|close|end|stop|complete|fix"
        r"|just\s+(?:close|finish|end|stop))\b"
        r"[^\n]{0,140}?"
        r"(?:" + name_alt + r"|\btool\s*call\b|\bfunction\s*call\b"
        r"|\bjson\s*string\b|\bcontent\s*string\b|\bproper\s*closure\b"
        r"|\bgenerating\s*tokens?\b|\bdone\b[\"']?\s*[:=]\s*true\b)"
        r"|"
        r"(?:[\"'»》!?,][ \t]*)?[\"']?\bdone\b[\"']?\s*[:=]\s*true[ \t]*[}\])]"
        r"|"
        r"\b(?:I\s+am\s+still\s+inside\s+the\s+(?:tool|function)\s*call"
        r"|the\s+[`'\"]?content[`'\"]?\s+string\s+(?:is\s+)?already\s+closed)\b"
        r"|"
        r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?(?:endasi|end_assistant|исповедь|<\|endoftext\|>|<\|im_end\|>)(?=[\s}\])\"'!.?]*\Z|\s)"
        r"|"
        r"[!?\s]*\}\s*(?:endasi|end_assistant|" + name_alt + r")\b"
        r")",
        re.IGNORECASE,
    )


# 英文自检起手词 + 句末标点后紧跟孤立闭合括号残渣（兼容 .!}?) / .!»}] / .!»?>}] / .!》?>}]）
_META_TAIL_RE = re.compile(
    r"\b(?:Let['’]s|Let\s+me|I\s+(?:must|should|need\s+to|have\s+to|will|am\s+still)|Wait)\b"
    r"[^\n}]{0,90}?[.!?]+(?:[ \t]*[!?\"'»》>]){0,5}\s*\}\s*[\])?]",
    re.IGNORECASE,
)

# 同义词轮换型收尾恐慌循环（即使不含 }，如 Let's stop.!》?> Let's end.!》?> Let's finish.!》?>）
_PANIC_CLOSURE_CLAUSE = (
    r"\bLet['’]s\s+(?:stop|end|finish|close|complete|fix|check|do\s+it|review|ensure|make\s+sure"
    r"|just\s+(?:close|stop|finish|end|output)|not\s+(?:output|put))\b"
    r"[^.!?\n]{0,48}[.!?]+(?:[ \t]*[»》>?}\])\"'!]){2,}"
)
_PANIC_CLOSURE_SINGLE_RE = re.compile(_PANIC_CLOSURE_CLAUSE, re.IGNORECASE)
_PANIC_CLOSURE_LOOP_RE = re.compile(
    r"(?:" + _PANIC_CLOSURE_CLAUSE + r"[ \t]*){3,}",
    re.IGNORECASE,
)

# 句末标点（后面可带收尾引号/书名号及至多两个空白）紧接孤立 }
_AFTER_SENTENCE_RE = re.compile(r'[.!?。！？…~～]["\'”’」』»》>?]{0,3}[ \t\r\n]{0,2}$')
_STRAY_BRACE_REASON = "模型写完正文后多写了一个 } 想结束工具调用并开始收尾念叨"

# 孤立 } 后面的多语种「已结束/无需再输出」收尾念叨特征
_CHATTER_RE = re.compile(
    r"end\s+of\s+(?:turn|response|output|message)|no\s+(?:further|more)\s+(?:output|text|words|content)"
    r"|nothing\s+(?:follows|more|else)|(?:tool|function)\s*call|empty\s+after"
    r"|工具调用|函数调用|通过(?:函数|工具)|无需再|不再(?:输出|生成)|(?:完整|全部)交付"
    r"|เสร็จสิ้น|keine\s+weiteren|ツール|関数呼び出し|도구\s*호출",
    re.IGNORECASE,
)
_PAREN_END_LINE_RE = re.compile(r"^[ \t]*\([^()\n]{1,40}\)[ \t]*$", re.MULTILINE)

# 流末专属：只在整段正文最末尾且全部括号均未配对时才清理的残渣
_DONE_TAIL_RE = re.compile(r"Done!\s*[}\])\s]{2,}\Z")
_CLOSURE_TAIL_RE = re.compile(
    r"(?:[\"']{1,3}[ \t]*[}\])][\s}\])]*|(?:\]\}\)\}|\]\}\)\)|\}\]\)\}|\]\}\)|[}\])][\s}\])]{2,24}))\Z"
)

# 流末围栏粘连工具溢出残片（例如: ```")} 或 ~~~]})) ）
_FENCE_GLUED_OVERFLOW_RE = re.compile(
    r"^([ \t]{0,7})(`{3,}|~{3,})([ \t]*)"
    r"("
    r"[\"']\s*[}\])][\s}\])\"']*"
    r"|[}\])]{3,}[\s}\])\"']*"
    r"|(?:[\"']?\s*[}\])]*\s*)?(?:endasi|end_assistant|исповедь)\b[\s\S]*"
    r"|(?:[\"']?\s*[}\])]*\s*)?\bLet['’]s\s+(?:check|finalize|make\s+sure|verify|finish|close|end|stop)\b[\s\S]*"
    r")\s*\Z",
    re.IGNORECASE,
)

# 48 字高密度货币/符号乱码检测常量（v8.2.0 实锤规则）
_SALAD_WINDOW = 48
_SALAD_MIN_DISTINCT = 18
_SALAD_MIN_CURRENCY = 6
_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_SENTENCE_END_CHARS = "。！？.!?”’」』）)…~～"


def _is_chatter(after: str) -> bool:
    """判断孤立 } 后面是否是多语种收尾念叨或连续三行 (End)/(Done)/(Stop)。"""
    probe = after[:TAIL_SUSPECT]
    return bool(_CHATTER_RE.search(probe)) or len(_PAREN_END_LINE_RE.findall(probe)) >= 3


def _is_symbol_ch(ch: str) -> bool:
    return not (ch.isalnum() or ch.isspace())


def _is_currency_ch(ch: str) -> bool:
    return ch in "$¢£¤¥֏؋৲৳૱௹฿៛" or "₠" <= ch <= "⃏"


def _is_rare_symbol_ch(ch: str) -> bool:
    return "←" <= ch <= "⏿" or "─" <= ch <= "▟"


def find_symbol_salad(text: str, start: int = 0) -> Optional[int]:
    """检测正文后突然冒出的 48 字高密度货币/符号乱码，返回乱码起始下标；未命中返回 None。"""
    n = len(text)
    if n - start < _SALAD_WINDOW:
        return None
    for i in range(start, n - _SALAD_WINDOW + 1):
        if not _is_symbol_ch(text[i]) or text[i] in _SENTENCE_END_CHARS:
            continue
        window = text[i : i + _SALAD_WINDOW]
        symbols = {c for c in window if _is_symbol_ch(c)}
        currency = sum(1 for c in symbols if _is_currency_ch(c))
        hit = currency >= _SALAD_MIN_CURRENCY
        if not hit and sum(1 for c in symbols if c in _ASCII_PUNCT) >= _SALAD_MIN_DISTINCT:
            wordish = sum(1 for c in window if c.isalnum())
            rare = sum(1 for c in window if _is_rare_symbol_ch(c))
            hit = wordish <= _SALAD_WINDOW // 4 and rare >= 3
        if not hit:
            continue
        line_start = text.rfind("\n", 0, i) + 1
        j = i
        while j > line_start:
            ch = text[j - 1]
            if (_is_symbol_ch(ch) and ch not in _SENTENCE_END_CHARS) or ch in " \t":
                j -= 1
                continue
            if ch.isalpha() and j - 2 >= line_start and not text[j - 2].isalnum():
                j -= 1
                continue
            break
        return j
    return None


def find_runaway_repetition(text: str) -> Optional[int]:
    """检测尾部是否同一小段连续重复 >=12 次且总跨度 >=600 字（豁免纯空白与排版分隔符），
    命中时返回保留第一遍后的切点下标，否则返回 None。"""
    n = len(text)
    if n < TAIL_MIN_SPAN:
        return None
    best: Optional[int] = None
    max_sz = min(TAIL_MAX_UNIT, n // TAIL_MIN_REPEATS)
    for size in range(1, max_sz + 1):
        unit = text[n - size :]
        non_ws = [c for c in unit if not c.isspace()]
        if not non_ws or _is_pure_layout_unit(non_ws):
            continue
        repeats = 1
        pos = n - size
        while pos >= size and text[pos - size : pos] == unit:
            repeats += 1
            pos -= size
        if repeats >= TAIL_MIN_REPEATS and repeats * size >= TAIL_MIN_SPAN:
            cut = pos + size
            if best is None or cut < best:
                best = cut
    return best


class _TailTextState:
    """轻量流式词法状态机：跨尾窗保留围栏代码块、单行三反引号、行内代码与括号配对状态。"""

    def __init__(self) -> None:
        self.fence: str = ""
        self.fence_len: int = 0
        self.fence_open_line: bool = False
        self.inline: int = 0
        self.run: str = ""
        self.run_len: int = 0
        self.run_bol: bool = False
        self.indent: int = 0
        self.escaped: bool = False
        self.stack: List[str] = []
        self.inline_pos: Optional[int] = None

    def scan(self, text: str) -> Tuple[List[bool], List[bool]]:
        protected: List[bool] = []
        orphan: List[bool] = []
        self.inline_pos = None
        for i, ch in enumerate(text):
            if self.run and ch != self.run:
                if self.fence:
                    inline_close = (
                        self.fence == "`"
                        and self.fence_open_line
                        and self.run_len == self.fence_len
                    )
                    if self.run == self.fence and (
                        inline_close
                        or (self.run_bol and self.run_len >= self.fence_len and ch.isspace())
                    ):
                        self.fence = ""
                elif self.inline:
                    if self.run == "`" and self.run_len == self.inline:
                        self.inline = 0
                elif self.run_bol and self.run_len >= 3:
                    self.fence, self.fence_len = self.run, self.run_len
                    self.fence_open_line = True
                elif self.run == "`":
                    self.inline = self.run_len
                    self.inline_pos = max(0, i - self.run_len)
                self.run = ""
                self.run_len = 0

            code = bool(self.fence or self.inline)
            delimiter = ch in "`~" and (not self.escaped or code)
            if delimiter:
                if not self.run:
                    self.run, self.run_bol = ch, 0 <= self.indent <= 7
                self.run_len += 1
            is_code = code or delimiter
            unmatched = False
            if not is_code:
                if ch in _OPEN_TO_CLOSE:
                    if len(self.stack) < 512:
                        self.stack.append(ch)
                elif ch in _CLOSE_TO_OPEN:
                    opening = _CLOSE_TO_OPEN[ch]
                    if self.stack and self.stack[-1] == opening:
                        self.stack.pop()
                    else:
                        # 弹性容错 3 层内嵌未闭合括号，防止正文颜文字/半括号污染后续正常闭合
                        found_idx = -1
                        low_bound = max(0, len(self.stack) - 3)
                        for k in range(len(self.stack) - 1, low_bound - 1, -1):
                            if self.stack[k] == opening:
                                found_idx = k
                                break
                        if found_idx != -1:
                            del self.stack[found_idx:]
                        else:
                            unmatched = True
            protected.append(is_code)
            orphan.append(unmatched)
            if ch == "\n":
                self.fence_open_line = False
                if self.inline and not self.fence:
                    # 行内代码不跨行：到行尾未闭合则回溯取消从反引号起的所有保护
                    for idx_p in range(self.inline_pos or 0, len(protected)):
                        protected[idx_p] = False
                    self.inline = 0
                    self.inline_pos = None
            self.indent = 0 if ch == "\n" else (self.indent + 1 if ch == " " and self.indent >= 0 else -1)
            self.escaped = (ch == "\\" and not self.escaped)
        return protected, orphan


def _scan_tail_window(state: _TailTextState, text: str) -> Tuple[List[bool], List[bool], _TailTextState]:
    """扫描暂扣窗口并返回 (代码保护标记, 孤立括号标记, 末态副本)。
    核心保障（v8.2.2）：孤立未闭合的行内反引号同一行超过 100 字或跨行后，
    立即回溯取消从该反引号起的代码保护，防止乱码里的反引号致盲整段检测。
    """
    probe = deepcopy(state)
    carried = bool(probe.inline) and not probe.fence
    protected, orphan = probe.scan(text)
    closing_now = (probe.run == "`" and probe.run_len == probe.inline)
    if probe.inline and not probe.fence and not closing_now:
        if probe.inline_pos is not None:
            start = probe.inline_pos if len(text) - probe.inline_pos > TAIL_INLINE_WAIT else len(text)
        elif carried:
            nl = text.find("\n")
            start = len(text) if nl < 0 else nl
        else:
            start = len(text)
        for idx_p in range(start, len(text)):
            protected[idx_p] = False
    return protected, orphan, probe


def _first_unprotected_match(pattern: re.Pattern, text: str, protected: List[bool]) -> Optional[re.Match]:
    for m in pattern.finditer(text):
        if not any(protected[m.start() : m.end()]):
            return m
    return None


class SyntheticTailGuard:
    """流式与非流式通用的正文尾部保护器（零激进误杀 + 高精准跑飞熔断）。"""

    def __init__(self, tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME, prefill_text: str = "") -> None:
        self.tool_name: str = (tool_name or DEFAULT_SYNTHETIC_TOOL_NAME).strip()
        self._leak_re: re.Pattern = _build_leak_pattern(self.tool_name)
        self._held: str = ""
        self._sent_tail: str = ""
        self._sent_code: List[bool] = []
        self._state: _TailTextState = _TailTextState()
        self._prefill_seeded: bool = False
        self.halted: bool = False
        self.trimmed_at_eof: bool = False
        self.chatter_trimmed: bool = False
        self.reason: str = ""
        self._released: int = 0
        self._ok_abs: int = -1
        if prefill_text:
            self.seed_prefill(prefill_text)

    def seed_prefill(self, prefill_text: str) -> None:
        """用未复述的 prefill 文本预热括号与代码围栏状态（仅在尚未放出任何正文前生效）。"""
        if prefill_text and not self._prefill_seeded and self._released == 0:
            self._state.scan(prefill_text)
            self._prefill_seeded = True

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.reason = reason
        self._held = ""

    def _release(self, out: str) -> str:
        if out:
            self._released += len(out)
            protected, _ = self._state.scan(out)
            self._sent_code = (self._sent_code + protected)[-TAIL_HISTORY:]
            self._sent_tail = (self._sent_tail + out)[-TAIL_HISTORY:]
        return out

    def _stray_close(self, text: str, protected: List[bool], orphan: List[bool]) -> Optional[int]:
        for i, ch in enumerate(text):
            if ch != "}" or not orphan[i] or protected[i] or self._released + i <= self._ok_abs:
                continue
            if _AFTER_SENTENCE_RE.search((self._sent_tail + text[:i])[-12:]):
                return i
        return None

    def feed(self, delta: str) -> str:
        """喂入新解码出的正文增量，返回已确认安全、可立即下发给客户端的正文。"""
        if self.halted or not delta:
            return ""
        self._held += delta
        protected, orphan, end_probe = _scan_tail_window(self._state, self._held)

        cuts: List[Tuple[int, str]] = []

        # 1. 显式工具名/工具调用自检独白与控制伪词
        m_leak = _first_unprotected_match(self._leak_re, self._held, protected)
        if m_leak is not None:
            cut_pos = m_leak.start()
            while cut_pos < m_leak.end() and self._held[cut_pos] in _CLOSER_CHARS and not orphan[cut_pos]:
                cut_pos += 1
            cuts.append((cut_pos, f"工具自检/控制词泄漏({_short_reason_preview(self._held[cut_pos:m_leak.end()])})"))

        # 2. 48 字高密度货币/符号乱码（代码保护位等长替换为 'a' 以保持下标一致）
        masked_for_salad = "".join("a" if p else c for c, p in zip(self._held, protected))
        salad_pos = find_symbol_salad(masked_for_salad)
        if salad_pos is not None:
            cuts.append((salad_pos, "正文后冒出高密度符号/货币乱码"))

        # 3. 英文自检句尾粘连孤立闭合括号残渣（要求匹配到的 } 为孤立未配对括号，彻底防误伤合法 JSON/代码）
        for m_meta in _META_TAIL_RE.finditer(self._held):
            if any(protected[m_meta.start() : m_meta.end()]):
                continue
            brace_idx = self._held.rfind("}", m_meta.start(), m_meta.end())
            if brace_idx != -1 and orphan[brace_idx]:
                cuts.append((m_meta.start(), f"英文收尾自检残渣({_short_reason_preview(m_meta.group(0))})"))
                break

        # 4. 同义词轮换型收尾恐慌循环（即使没有 }，连续 3 句即熔断；单句出现先暂扣观察）
        pending: Optional[int] = None
        m_panic = _PANIC_CLOSURE_LOOP_RE.search(self._held)
        if m_panic is not None:
            if not end_probe.fence or len(re.findall(_PANIC_CLOSURE_CLAUSE, m_panic.group(0), re.IGNORECASE)) >= 5:
                cuts.append((m_panic.start(), f"工具收尾恐慌循环({_short_reason_preview(m_panic.group(0))})"))
        elif not end_probe.fence:
            m_p1 = _PANIC_CLOSURE_SINGLE_RE.search(self._held)
            if m_p1 is not None and len(self._held) - m_p1.start() < TAIL_SUSPECT:
                pending = m_p1.start()

        # 5. 句末孤立 } + 400 字多语种收尾念叨观察窗
        while True:
            brace = self._stray_close(self._held, protected, orphan)
            if brace is None:
                break
            after = self._held[brace + 1 :]
            if _is_chatter(after):
                cuts.append((brace, _STRAY_BRACE_REASON))
            elif len(after) < TAIL_SUSPECT:
                pending = brace if pending is None else min(pending, brace)
            else:
                self._ok_abs = self._released + brace
                continue
            break

        if cuts:
            cut, why = min(cuts, key=lambda x: x[0])
            out = self._held[:cut].rstrip(" \t")
            self._halt(why)
            return self._release(out)

        # 6. 严格周期死循环检测（连续 >=12 次 且 >=600 字，回看 1600 字已放行历史）
        window = self._sent_tail + self._held
        rep_cut = find_runaway_repetition(window)
        if rep_cut is not None and not any((self._sent_code + protected)[rep_cut:]):
            keep = max(0, rep_cut - len(self._sent_tail))
            out = self._held[:keep]
            self._halt("同一小段文字无限重复（≥12次/600字）")
            return self._release(out)

        hold_from = len(self._held) - TAIL_HOLD
        if pending is not None:
            hold_from = min(hold_from, pending - TAIL_HOLD)
        if hold_from <= 0:
            return ""
        out, self._held = self._held[:hold_from], self._held[hold_from:]
        return self._release(out)

    def flush(self, full_text_so_far: str = "") -> str:
        """流/调用结束时放行暂扣尾窗，并仅在全文最末尾清理孤立未配对闭合残渣与念叨。"""
        if self.halted:
            return ""
        out, self._held = self._held, ""
        if not out:
            return ""

        protected, orphan, end_probe = _scan_tail_window(self._state, out)

        # 0. 流末围栏未换行直接粘连工具溢出尾巴（如 ```")} ）
        if end_probe.fence:
            last_nl = out.rfind("\n")
            line_start = last_nl + 1
            last_line = out[line_start:]
            m_glue = _FENCE_GLUED_OVERFLOW_RE.match(last_line)
            if m_glue:
                f_chars = m_glue.group(2)
                if f_chars[0] == end_probe.fence and len(f_chars) >= end_probe.fence_len:
                    keep_end = line_start + len(m_glue.group(1)) + len(f_chars)
                    self.trimmed_at_eof = True
                    self.reason = f"清理围栏末尾粘连工具尾({_short_reason_preview(out[keep_end:])})"
                    return self._release(out[:keep_end])

        # 1. 句末孤立 } 后跟着念叨，或流就停在一个孤立多余的 } 上
        while out:
            brace = self._stray_close(out, protected, orphan)
            if brace is None:
                break
            after = out[brace + 1 :]
            if (_is_chatter(after) or not after.strip()) and (full_text_so_far + out[:brace]).strip():
                self.trimmed_at_eof = True
                if _is_chatter(after):
                    self.chatter_trimmed = True
                    self.reason = _STRAY_BRACE_REASON
                else:
                    self.reason = "去掉了正文末尾孤立多余的 }"
                out = out[:brace].rstrip(" \t")
                return self._release(out)
            self._ok_abs = self._released + brace

        # 2. 流末 Done!}}] 或全未配对孤立右括号残渣（如 "]}) / ]})}"）
        for pattern in (_DONE_TAIL_RE, _CLOSURE_TAIL_RE):
            m = pattern.search(out)
            if m is None:
                continue
            if any(protected[m.start() : m.end()]):
                continue
            closer_positions = [k for k in range(m.start(), m.end()) if out[k] in _CLOSER_CHARS]
            if not closer_positions or not all(orphan[k] for k in closer_positions):
                continue
            if not (full_text_so_far + out[: m.start()]).strip():
                continue
            cluster = out[m.start() : m.end()]
            closer_types = {out[k] for k in closer_positions}
            # 若并非 Done! 且未以引号开头，要求至少 3 个混合括号且含 } 或 ]，避免误切正文正常单/双括号
            if pattern is _CLOSURE_TAIL_RE and cluster[0] not in ('"', "'"):
                if not (len(closer_positions) >= 3 and len(closer_types) >= 2 and ("}" in closer_types or "]" in closer_types)):
                    continue
            self.trimmed_at_eof = True
            self.reason = f"去掉了正文末尾多余的闭合残片({_short_reason_preview(cluster)})"
            out = out[: m.start()]
            break

        return self._release(out)


def clean_synthetic_complete_text(
    text: str,
    tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME,
    prefill_text: str = "",
) -> Tuple[str, bool, str]:
    """一次性清洗完整文本（非流式/假流式）。返回 (干净正文, 是否命中跑飞熔断, 原因)。"""
    if not text:
        return "", False, ""
    use_pf = prefill_text if (prefill_text and not prefill_appears_repeated(prefill_text, text)) else ""
    guard = SyntheticTailGuard(tool_name=tool_name, prefill_text=use_pf)
    out = guard.feed(text)
    out += guard.flush(out)
    runaway_cut = bool(guard.halted or guard.chatter_trimmed)
    return out, runaway_cut, guard.reason


def analyze_synthetic_tail(
    text: str,
    emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME,
    base_lex: Optional[LexicalState] = None,
    at_eof: bool = True,
    reason_out: Optional[List[str]] = None,
) -> Tuple[str, bool, bool]:
    """兼容旧签名的尾部检测入口（底层统一委托 SyntheticTailGuard）。
    返回: (干净文本, 是否触发了溢出切除, 流中是否需暂扣观察)
    """
    if not text:
        return "", False, False
    guard = SyntheticTailGuard(tool_name=emit_tool_name)
    if base_lex is not None:
        guard._state.fence = base_lex.fence_char if base_lex.in_fence else ""
        guard._state.fence_len = base_lex.fence_len if base_lex.in_fence else 0
        guard._state.inline = base_lex.inline_tick_len
        guard._state.stack = list(base_lex.bracket_stack)
    emitted = guard.feed(text)
    if guard.halted:
        if reason_out is not None and guard.reason:
            reason_out.append(guard.reason)
        return emitted, True, False
    if at_eof:
        flushed = guard.flush(emitted)
        full = emitted + flushed
        trimmed = bool(guard.trimmed_at_eof or guard.chatter_trimmed)
        if trimmed and reason_out is not None and guard.reason:
            reason_out.append(guard.reason)
        return full, trimmed, False
    return text, False, len(guard._held) > TAIL_HOLD


def sanitize_synthetic_tail(
    text: str,
    emit_tool_name: str = "",
    base_lex: Optional[LexicalState] = None,
    at_eof: bool = True,
) -> Tuple[str, bool]:
    """检测并平滑切除尾部闭合溢出（兼容原函数签名）。"""
    clean, halted, _ = analyze_synthetic_tail(
        text,
        emit_tool_name=emit_tool_name or DEFAULT_SYNTHETIC_TOOL_NAME,
        base_lex=base_lex,
        at_eof=at_eof,
    )
    return clean, halted


def _scan_truncated_json_content(raw: str) -> Optional[str]:
    """当字符串形式的 args 被截断导致 json.loads 失败时，有界提取 "content" 字符串已生成部分。"""
    s = (raw or "").strip()
    if not s.startswith("{") or '"content"' not in s:
        return None
    key_idx = s.find('"content"')
    rest = s[key_idx + len('"content"') :]
    colon = rest.find(":")
    if colon < 0:
        return None
    rest = rest[colon + 1 :].lstrip(" \t\r\n")
    if not rest.startswith('"'):
        return None
    rest = rest[1:]
    out: List[str] = []
    esc = False
    uni = 0
    uni_val = 0
    surr = 0
    for c in rest:
        if uni > 0:
            if "0" <= c <= "9":
                v = ord(c) - ord("0")
            elif "a" <= c <= "f":
                v = ord(c) - ord("a") + 10
            elif "A" <= c <= "F":
                v = ord(c) - ord("A") + 10
            else:
                v = -1
            if v >= 0:
                uni_val = (uni_val << 4) | v
                uni -= 1
                if uni == 0:
                    if surr != 0:
                        if 0xDC00 <= uni_val <= 0xDFFF:
                            out.append(chr(((surr - 0xD800) << 10) + (uni_val - 0xDC00) + 0x10000))
                        else:
                            out.append("\ufffd")
                            out.append(chr(uni_val))
                        surr = 0
                    elif 0xD800 <= uni_val <= 0xDBFF:
                        surr = uni_val
                    else:
                        out.append(chr(uni_val))
                    uni_val = 0
            else:
                out.append("\ufffd")
                uni = 0
                uni_val = 0
                surr = 0
        elif esc:
            esc = False
            if c in ('"', "\\", "/"):
                out.append(c)
            elif c == "b":
                out.append("\b")
            elif c == "f":
                out.append("\f")
            elif c == "n":
                out.append("\n")
            elif c == "r":
                out.append("\r")
            elif c == "t":
                out.append("\t")
            elif c == "u":
                uni = 4
                uni_val = 0
            else:
                out.append("\ufffd")
        elif c == "\\":
            esc = True
        elif c == '"':
            break
        else:
            out.append(c)
    res = "".join(out)
    return res if res else None


def _unwrap_nested_json_if_needed(text: str, emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME) -> str:
    """仅当模型误将合成工具参数双重序列化为裸单键 JSON（如 '{"content": "..."}'）时解包；
    严禁剥离 ```json 等 Markdown 代码块包裹的用户内容或普通多字段业务 JSON。
    """
    s = (text or "").strip()
    if not s or s.startswith(("```", "~~~")):
        return text
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict) and parsed:
                keys = set(parsed.keys())
                if (
                    keys in ({"content"}, {"content", "done"})
                    and isinstance(parsed.get("content"), str)
                    and parsed["content"]
                ):
                    return parsed["content"]
                p_name = str(parsed.get("name") or parsed.get("tool") or "").strip()
                if p_name and p_name in (emit_tool_name, DEFAULT_SYNTHETIC_TOOL_NAME):
                    inner = parsed.get("args") or parsed.get("arguments") or parsed
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            pass
                    if isinstance(inner, dict):
                        for k in CONTENT_KEYS:
                            val = inner.get(k)
                            if isinstance(val, str) and val:
                                return val
        except Exception:
            pass
    elif s.startswith("{"):
        recovered = _scan_truncated_json_content(s)
        if recovered:
            return recovered
    return text


def extract_raw_synthetic_text(args: Any, emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME) -> str:
    """从合成工具 args 中提取未经尾部裁剪的原始文本（供流式组装器统一经尾部保护器处理）。"""
    if args is None:
        return ""
    if isinstance(args, str):
        return _unwrap_nested_json_if_needed(args, emit_tool_name=emit_tool_name)
    if isinstance(args, dict):
        for k in CONTENT_KEYS:
            val = args.get(k)
            if isinstance(val, str) and val:
                return _unwrap_nested_json_if_needed(val, emit_tool_name=emit_tool_name)
        for val in args.values():
            if isinstance(val, str) and val:
                return _unwrap_nested_json_if_needed(val, emit_tool_name=emit_tool_name)
    return ""


def extract_synthetic_content(
    args: Any,
    emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME,
    prefill_text: str = "",
    halted_out: Optional[List[str]] = None,
) -> str:
    """从合成工具的 args 字典中提取正文 content，并经融合尾部保护器精确清理跑飞内容与残片。"""
    raw_val = extract_raw_synthetic_text(args, emit_tool_name=emit_tool_name)
    if not raw_val:
        return ""
    clean_val, runaway_cut, reason = clean_synthetic_complete_text(
        raw_val,
        tool_name=emit_tool_name or DEFAULT_SYNTHETIC_TOOL_NAME,
        prefill_text=prefill_text,
    )
    if runaway_cut and halted_out is not None:
        halted_out.append(reason or "跑飞截断")
    return clean_val


def is_content_json_path(path_tokens: List[Any], raw_path: str = "") -> bool:
    if path_tokens:
        return str(path_tokens[0]) in CONTENT_KEYS
    low = (raw_path or "").lower()
    return any(k in low for k in CONTENT_KEYS)


def is_done_json_path(path_tokens: List[Any], raw_path: str = "") -> bool:
    if path_tokens:
        return str(path_tokens[0]).lower() in ("done", "is_finished")
    low = (raw_path or "").strip().lower()
    return low in ("$.done", ".done", "done", "$.is_finished", ".is_finished", "is_finished")
