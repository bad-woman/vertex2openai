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
import uuid
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


_SDK_SUPPORTS_STREAM_FC_ARGS = _check_sdk_supports_stream_fc_args()
_SDK_SUPPORTS_PARAMS_JSON_SCHEMA = _check_sdk_supports_params_json_schema()


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
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        t_name = str(fn.get("name") or tool.get("name") or tool.get("type") or "").strip().lower().replace("-", "_")
        if t_name in _BUILTIN_SEARCH_NAMES:
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
    strip_client = isinstance(tc, str) and tc.strip().lower() == "none"

    all_client_names: List[str] = []
    for tool in (getattr(request_obj, "tools", None) or []):
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("name") or "").strip()
        if name and name.lower().replace("-", "_") not in _BUILTIN_SEARCH_NAMES:
            if name not in all_client_names:
                all_client_names.append(name)

    real_names = [] if strip_client else list(all_client_names)

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
    """构建有序双必填字段 (content + done) 抗截断传输工具声明：
    1. 全局静态固定描述与 Schema，100% 保证命中 Vertex 隐式前缀缓存；
    2. 协议级带外收敛：受限解码器（FSM）要求模型在写完 content 后必须闭合双引号去填写 `, "done": true}`，
       为模型注意力提供确定性的结构出口台阶，从根源消除在一元 generateContent / 假流式 / 真流式下把外层容器括号幻觉进 content 或卡死未闭合的问题；
    3. Express 模式使用 parameters_json_schema + propertyOrdering + additionalProperties: False；
    4. Cookie 模式使用标准 parameters + propertyOrdering: ["content", "done"]（不含布尔型 additionalProperties，兼容 batchGraphql）。
    """
    desc = "Deliver the complete user-visible reply as text, then set done to true to end the call."
    content_desc = (
        "The complete user-visible reply itself, preserving its language, code blocks, sections, formatting and markup."
    )
    done_desc = "Set to true immediately after content ends to finalize and close the call."
    base_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": content_desc,
            },
            "done": {
                "type": "boolean",
                "description": done_desc,
            },
        },
        "required": ["content", "done"],
        "propertyOrdering": ["content", "done"],
    }
    if not for_cookie and _SDK_SUPPORTS_PARAMS_JSON_SCHEMA:
        json_schema = {k: v for k, v in base_schema.items() if k != "propertyOrdering"}
        json_schema["propertyOrdering"] = ["content", "done"]
        json_schema["additionalProperties"] = False
        return {
            "name": emit_tool_name,
            "description": desc,
            "parameters_json_schema": json_schema,
        }
    return {
        "name": emit_tool_name,
        "description": desc,
        "parameters": base_schema,
    }


def build_sandbox_control_instruction(
    emit_tool_name: str,
    has_prefill: bool = False,
    unclosed_tag: Optional[str] = None,
    is_keep_turn: bool = False,
    has_callable_client_tools: bool = False,
) -> str:
    """构建通用传输格式指令（配合 content + done 双字段收敛约束，防工具自检与防尾部重复填充）。"""
    target_text = "only the new continuation text" if has_prefill else "the entire user-visible answer"
    if has_callable_client_tools:
        transport_head = (
            f"Transport format for this reply: put all user-visible text, if any, in exactly one call of {emit_tool_name}; "
            f"its content string holds {target_text} (preserving all requested formatting, code blocks, markup, language and sections), "
            f"then set done to true. {emit_tool_name} only transports text and performs no external action. "
            f"Call the other tools whenever the task needs them, with or without {emit_tool_name}."
        )
    else:
        transport_head = (
            f"Transport format for this reply: call {emit_tool_name} exactly once. "
            f"Put {target_text} in its content string (preserving all requested formatting, code blocks, markup, language and sections), "
            f"then set done to true. The function only transports text and performs no external action."
        )

    transport_tail = (
        " This transport does not change the task. Do not add unsolicited commentary about this transport, "
        "tool selection or serialization. Once the requested content is complete, close the content string and set done to true "
        "immediately without repeating finished content or adding filler. Preserve repetition explicitly requested by the task."
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
        self.consec_newlines: int = 0

        self.bracket_stack: List[str] = []
        self.dquote_open: bool = False
        self.prev_backslash: bool = False

        self.tag_buf: str = ""
        self.tag_overflow: bool = False
        self.tag_rel_offset: int = 0
        self.tag_quote: str = ""
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
        c.consec_newlines = self.consec_newlines
        c.bracket_stack = list(self.bracket_stack)
        c.dquote_open = self.dquote_open
        c.prev_backslash = self.prev_backslash
        c.tag_buf = self.tag_buf
        c.tag_overflow = self.tag_overflow
        c.tag_rel_offset = self.tag_rel_offset
        c.tag_quote = self.tag_quote
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
                    self.pending_open_char = ""
                    self.pending_open_len = 0
            return

        if self.inline_tick_len > 0:
            if rc == "`" and rl == self.inline_tick_len:
                self.inline_tick_len = 0
                self.inline_newlines = 0
            return

        if was_bol and rl >= 3 and rc in ("`", "~"):
            self.pending_open_char = rc
            self.pending_open_len = rl
        elif rc == "`":
            self.inline_tick_len = rl
            self.inline_newlines = 0

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
                    if self.consec_newlines >= 2 or self.inline_newlines >= 2:
                        self.inline_tick_len = 0
                        self.inline_newlines = 0

            if not self.in_fence:
                self.dquote_open = False
                if self.consec_newlines >= 2:
                    self.tag_buf = ""
                    self.tag_overflow = False
                    self.tag_rel_offset = 0
                    self.tag_quote = ""
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

        if self.pending_open_len > 0 or self.inline_tick_len > 0:
            self.prev_backslash = False
            if meta is not None:
                meta.in_code = True
            return

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
        elif self.tag_buf:
            self.tag_rel_offset += 1
            if not self.tag_overflow and len(self.tag_buf) < 96:
                self.tag_buf += ch
            else:
                self.tag_overflow = True

            if self.tag_quote:
                if ch == self.tag_quote:
                    self.tag_quote = ""
            elif ch in ('"', "'") and len(self.tag_buf) > 2 and not self.tag_buf.startswith("<!--"):
                self.tag_quote = ch
            elif ch == ">":
                if self.tag_buf.startswith("<!--") and not self.tag_buf.endswith("-->"):
                    pass
                else:
                    full_t = self.tag_buf
                    is_ovf = self.tag_overflow
                    rel_off = self.tag_rel_offset
                    self.tag_buf = ""
                    self.tag_overflow = False
                    self.tag_rel_offset = 0
                    self.tag_quote = ""

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
                        elif not full_t.rstrip().endswith("/>"):
                            m_open = _OPEN_TAG_HEAD_RE.match(full_t)
                            if m_open:
                                tname = m_open.group(1).lower()
                                if tname not in _VOID_HTML_TAGS and not tname.startswith(
                                    ("http:", "https:", "mailto:", "ftp:")
                                ):
                                    self.html_open_counts[tname] = min(
                                        self.html_open_counts.get(tname, 0) + 1, 512
                                    )
            elif self.tag_rel_offset > 256:
                self.tag_buf = ""
                self.tag_overflow = False
                self.tag_rel_offset = 0
                self.tag_quote = ""


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


# 显式工具自检独白与控制伪词正则（无嵌套歧义量词，保证线性时间 O(N) 无回溯风险）
_SELF_CHECK_RE = re.compile(
    r"(?:"
    r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?"
    r"\bLet['’]s\s+(?:check|finalize|make\s+sure|verify|review|ensure|confirm)\b"
    r"[\s\S]{0,180}?"
    r"(?:compose_reply|router_emit|\btool\s*call\b|\bfunction\s*call\b|\bjson\s*string\b"
    r"|Done![ \t\r\n\"']*[}\])][ \t\r\n\"']*[}\])])"
    r"|"
    r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?Done![ \t\r\n\"']*[}\])][ \t\r\n\"']*[}\])][\s\"'}\])]*\Z"
    r"|"
    r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?(?:endasi|end_assistant|исповедь|<\|endoftext\|>|<\|im_end\|>)(?=[\s}\])\"'!.?]*\Z|\s)"
    r"|"
    r"[!?\s]*\}\s*(?:endasi|end_assistant|compose_reply|router_emit)\b"
    r")",
    re.IGNORECASE,
)

# 流中疑似正在形成工具自检前缀（尚未吐完后续关键词，触发动态隔离窗口暂扣观察）
_SUSPECT_SELF_CHECK_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?\bLet['’]s\s*(?:(?:check|finalize|make|verify|review|ensure|confirm)\b[^\n]{0,120})?\Z"
    r"|"
    r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?Done![\s\"'}\])]{0,12}\Z"
    r"|"
    r"[}\])\"']+\s*(?:end|comp|rout|исп)[a-z_а-я]{0,14}\Z"
    r")",
    re.IGNORECASE,
)

# EOF 末尾未换行闭合围栏直接粘连工具溢出残片（例如: ```")} 或 ~~~]})) ）
_FENCE_GLUED_OVERFLOW_RE = re.compile(
    r"^([ \t]{0,7})(`{3,}|~{3,})([ \t]*)"
    r"("
    r"[\"']\s*[}\])][\s}\])\"']*"
    r"|[}\])]{3,}[\s}\])\"']*"
    r"|(?:[\"']?\s*[}\])]*\s*)?(?:endasi|end_assistant|исповедь)\b[\s\S]*"
    r"|(?:[\"']?\s*[}\])]*\s*)?\bLet['’]s\s+(?:check|finalize|make\s+sure|verify)\b[\s\S]*"
    r")\s*\Z",
    re.IGNORECASE,
)

# EOF 末尾悬空闭合括号簇候选（必须通过词法栈 100% 未配对悬空校验方可裁剪）
_EOF_CLOSER_CLUSTER_RE = re.compile(
    r"([\"']{1,3}[ \t]*[}\])][\s}\])]*|[}\])][\s}\])]{1,24})\Z"
)


def _primitive_period(s: str) -> int:
    """求字符串 s 的最小本原周期长度（例如 '------' -> 1, '</div></div>' -> 6）。"""
    n = len(s)
    if n <= 1:
        return n
    for p in range(1, n // 2 + 1):
        if n % p == 0 and s == s[:p] * (n // p):
            return p
    return n


def analyze_synthetic_tail(
    text: str,
    emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME,
    base_lex: Optional[LexicalState] = None,
    at_eof: bool = True,
) -> Tuple[str, bool, bool]:
    """词法状态感知 + 分期尾窗检测与清洗核心引擎。
    返回: (干净文本, 是否触发了溢出切除, 流中是否需扩容隔离窗口暂缓吐出)
    """
    if not text:
        return "", False, False

    lex = base_lex.clone() if base_lex is not None else LexicalState()
    metas, end_lex = lex.scan(text)
    n = len(text)

    # 消解同段内已配对闭合直双引号的 orphan_dquote 标记
    open_dq_idx = -1
    prev_bs = base_lex.prev_backslash if base_lex is not None else False
    dq_open = base_lex.dquote_open if base_lex is not None else False
    for i, ch in enumerate(text):
        if metas[i].in_code:
            prev_bs = False
            continue
        if ch == "\n":
            dq_open = False
            open_dq_idx = -1
            prev_bs = False
            continue
        if ch == '"' and not prev_bs:
            if dq_open:
                dq_open = False
                if open_dq_idx != -1:
                    metas[open_dq_idx].orphan_dquote = False
                    open_dq_idx = -1
                metas[i].orphan_dquote = False
            else:
                dq_open = True
                open_dq_idx = i
                metas[i].orphan_dquote = True
        prev_bs = (ch == "\\" and not prev_bs)

    # 0. EOF 特判：最后一行闭合围栏未换行直接粘连工具溢出尾巴（如 ```")} ）
    if at_eof and end_lex.in_fence:
        last_nl = text.rfind("\n")
        line_start = last_nl + 1
        last_line = text[line_start:]
        m_glue = _FENCE_GLUED_OVERFLOW_RE.match(last_line)
        if m_glue:
            f_chars = m_glue.group(2)
            if f_chars[0] == end_lex.fence_char and len(f_chars) >= end_lex.fence_len:
                keep_end = line_start + len(m_glue.group(1)) + len(f_chars)
                return text[:keep_end], True, False

    # 1. 显式工具自检独白与控制伪词拦截（仅在代码块与行内代码作用域之外生效）
    for m in _SELF_CHECK_RE.finditer(text):
        start_pos = m.start()
        if start_pos < n and not metas[start_pos].in_code:
            matched_str = m.group(0)
            if matched_str.lstrip("\"'}])\t !?").lower().startswith("done!"):
                orphan_cnt = sum(
                    1 for k in range(m.start(), m.end())
                    if text[k] in _CLOSER_CHARS and metas[k].is_orphan_closer
                )
                if orphan_cnt < 2:
                    continue
            cut_pos = start_pos
            if cut_pos < m.end() and text[cut_pos] == '"' and not metas[cut_pos].orphan_dquote:
                cut_pos += 1
            while (
                cut_pos < m.end()
                and text[cut_pos] in _CLOSER_CHARS
                and not metas[cut_pos].is_orphan_closer
            ):
                cut_pos += 1
            return text[:cut_pos], True, False

    if emit_tool_name and emit_tool_name not in ("compose_reply", "router_emit") and emit_tool_name in text:
        dyn_pat = re.compile(
            r"(?:[\"']?[ \t]*[}\])]+[!?\s]*)?(?:\bLet['’]s\s+[^\n]{0,100}?\b"
            + re.escape(emit_tool_name)
            + r"\b|\}\s*"
            + re.escape(emit_tool_name)
            + r"\b)",
            re.IGNORECASE,
        )
        for m_dyn in dyn_pat.finditer(text):
            if not metas[m_dyn.start()].in_code:
                cut_pos = m_dyn.start()
                if cut_pos < m_dyn.end() and text[cut_pos] == '"' and not metas[cut_pos].orphan_dquote:
                    cut_pos += 1
                while (
                    cut_pos < m_dyn.end()
                    and text[cut_pos] in _CLOSER_CHARS
                    and not metas[cut_pos].is_orphan_closer
                ):
                    cut_pos += 1
                return text[:cut_pos], True, False

    # 2. 悬空多余闭合标签刷屏检测（完整保留合法配对的 </div>，仅切掉超出开启深度的多余闭合标签）
    idx_scan = n - 1
    while idx_scan >= 0 and text[idx_scan] in " \t\r\n":
        idx_scan -= 1
    if idx_scan >= 0 and text[idx_scan] == ">" and metas[idx_scan].orphan_close_tag_start != -1:
        orphan_tag_starts: List[int] = []
        pos = idx_scan
        first_tag_norm = None
        while pos >= 0:
            while pos >= 0 and text[pos] in " \t\r\n":
                pos -= 1
            if pos < 0 or text[pos] != ">" or metas[pos].orphan_close_tag_start == -1:
                break
            t_start = metas[pos].orphan_close_tag_start
            raw_tag = re.sub(r"\s+", "", text[t_start : pos + 1]).lower()
            if first_tag_norm is None:
                first_tag_norm = raw_tag
            elif raw_tag != first_tag_norm:
                break
            orphan_tag_starts.append(t_start)
            pos = t_start - 1

        orphan_tag_cnt = len(orphan_tag_starts)
        halt_tag_thresh = 5 if at_eof else 7
        if orphan_tag_cnt >= halt_tag_thresh:
            cut_at = orphan_tag_starts[-1]
            while cut_at > 0 and text[cut_at - 1] in " \t":
                cut_at -= 1
            if cut_at > 0 and text[cut_at - 1] == "\n":
                cut_at -= 1
            return text[:cut_at], True, False
        if not at_eof and orphan_tag_cnt >= 2:
            return text, False, True

    # 3. 单字符悬空右括号刷屏死循环（保留前面合法配对的括号，仅从首个悬空右括号切起）
    p = n - 1
    while p >= 0 and text[p] in " \t\r\n":
        p -= 1
    if p >= 0 and text[p] in _CLOSER_CHARS and not metas[p].in_code and metas[p].is_orphan_closer:
        orphan_closer_indices: List[int] = []
        q = p
        while q >= 0 and (text[q] in _CLOSER_CHARS or text[q] in " \t"):
            if text[q] in _CLOSER_CHARS:
                if metas[q].in_code or not metas[q].is_orphan_closer:
                    break
                orphan_closer_indices.append(q)
            q -= 1
        if len(orphan_closer_indices) >= 16:
            first_orphan = orphan_closer_indices[-1]
            if first_orphan > 0 and text[first_orphan - 1] == '"' and metas[first_orphan - 1].orphan_dquote:
                first_orphan -= 1
            return text[:first_orphan], True, False
        if not at_eof and len(orphan_closer_indices) >= 3:
            return text, False, True

    # 4. 本原周期退化死循环检测
    suspect_hold = False
    max_u = min(48, n // 2)
    best_cut: Optional[int] = None

    for ulen in range(2, max_u + 1):
        unit = text[n - ulen :]
        if _primitive_period(unit) != ulen:
            continue
        non_ws = [c for c in unit if c not in " \t\r\n"]
        if not non_ws:
            continue
        if all(c in _PURE_LAYOUT_CHARS for c in non_ws):
            continue

        repeats = 1
        curr = n - ulen
        while curr >= ulen and text[curr - ulen : curr] == unit:
            repeats += 1
            curr -= ulen

        if repeats < 2:
            continue

        total_rep_chars = repeats * ulen
        unit_low = unit.lower()

        if any(tok in unit_low for tok in _KNOWN_LEAK_TOKENS) or (
            emit_tool_name and emit_tool_name.lower() in unit_low
        ):
            if not metas[curr].in_code:
                best_cut = curr if best_cut is None else min(best_cut, curr)
                continue

        if all(c in " \t\r\n}])\"'`" for c in unit) and any(c in _CLOSER_CHARS for c in unit):
            first_all_orphan_rep_start: Optional[int] = None
            orphan_reps = 0
            for r_idx in range(repeats):
                seg_start = curr + r_idx * ulen
                seg_end = seg_start + ulen
                closers_in_seg = [
                    k for k in range(seg_start, seg_end) if text[k] in _CLOSER_CHARS
                ]
                if closers_in_seg and all(
                    (not metas[k].in_code and metas[k].is_orphan_closer) for k in closers_in_seg
                ):
                    if first_all_orphan_rep_start is None:
                        first_all_orphan_rep_start = seg_start
                    orphan_reps += 1
                else:
                    first_all_orphan_rep_start = None
                    orphan_reps = 0

            if orphan_reps >= 2 and first_all_orphan_rep_start is not None:
                if at_eof or (orphan_reps >= 4 and orphan_reps * ulen >= 10):
                    best_cut = (
                        first_all_orphan_rep_start
                        if best_cut is None
                        else min(best_cut, first_all_orphan_rep_start)
                    )
                    continue
                else:
                    suspect_hold = True

        stripped_u = unit.strip()
        if stripped_u.startswith("</") and stripped_u.endswith(">"):
            continue

        is_table_row_seg = ("|" in unit and "\n" not in unit)
        min_total_len = 360 if is_table_row_seg else 120
        min_reps = 16 if is_table_row_seg else 10

        if repeats >= min_reps and total_rep_chars >= min_total_len:
            best_cut = curr if best_cut is None else min(best_cut, curr)
        elif not at_eof and repeats >= 4 and total_rep_chars >= 36:
            suspect_hold = True

    if best_cut is not None:
        return text[:best_cut], True, False

    # 5. 流末 (at_eof=True) 专属：切除全文最末尾单发悬空工具闭合残片（如 ")} 、 "]})} 、 ]})) 、 """) ）
    if at_eof and not end_lex.is_currently_in_code():
        m_eof = _EOF_CLOSER_CLUSTER_RE.search(text)
        if m_eof:
            c_start, c_end = m_eof.span()
            if not metas[c_start].in_code:
                closer_positions = [
                    k for k in range(c_start, c_end) if text[k] in _CLOSER_CHARS
                ]
                if closer_positions and all(metas[k].is_orphan_closer for k in closer_positions):
                    cluster = text[c_start:c_end]
                    closer_types = {text[k] for k in closer_positions}
                    starts_with_quote = cluster[0] in ('"', "'")

                    should_trim = False
                    if starts_with_quote:
                        if cluster.startswith(('"""', "'''")) and len(closer_positions) >= 1:
                            should_trim = True
                        elif cluster[0] == '"' and metas[c_start].orphan_dquote:
                            if ("}" in closer_types or ")" in closer_types) and (
                                len(closer_positions) >= 2 or "}" in closer_types
                            ):
                                should_trim = True
                    else:
                        if len(closer_positions) >= 3 and len(closer_types) >= 2 and (
                            "}" in closer_types or "]" in closer_types
                        ):
                            should_trim = True

                    if should_trim:
                        return text[:c_start], True, False

    # 6. 流中 (at_eof=False) 检查末尾是否有半截自检前缀，需暂扣在隔离窗口观察
    if not at_eof and not end_lex.is_currently_in_code():
        tail_probe = text[-140:] if n > 140 else text
        m_susp = _SUSPECT_SELF_CHECK_PREFIX_RE.search(tail_probe)
        if m_susp:
            abs_pos = (n - len(tail_probe)) + m_susp.start()
            if not metas[abs_pos].in_code:
                suspect_hold = True

    return text, False, suspect_hold


def sanitize_synthetic_tail(
    text: str,
    emit_tool_name: str = "",
    base_lex: Optional[LexicalState] = None,
    at_eof: bool = True,
) -> Tuple[str, bool]:
    """检测并平滑切除尾部闭合溢出（兼容原函数签名，内部委托词法感知引擎）。
    返回: (干净文本, 是否触发了闭合熔断)
    """
    clean, halted, _ = analyze_synthetic_tail(
        text,
        emit_tool_name=emit_tool_name or DEFAULT_SYNTHETIC_TOOL_NAME,
        base_lex=base_lex,
        at_eof=at_eof,
    )
    return clean, halted


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
                # 情况 1: 纯单键 {"content": "..."} 或双键 {"content": "...", "done": true} 双重序列化
                if (
                    keys in ({"content"}, {"content", "done"})
                    and isinstance(parsed.get("content"), str)
                    and parsed["content"]
                ):
                    return parsed["content"]
                # 情况 2: 显式携带合成工具名的自包装对象 {"name": "compose_reply", "arguments": ...}
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
    return text


def extract_raw_synthetic_text(args: Any, emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME) -> str:
    """从合成工具 args 中提取未经尾部裁剪的原始文本（供流式组装器统一经词法追踪器处理）。"""
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
) -> str:
    """从合成工具的 args 字典中提取正文 content，并基于词法状态精确清理尾部闭合溢出残片。"""
    raw_val = extract_raw_synthetic_text(args, emit_tool_name=emit_tool_name)
    if not raw_val:
        return ""
    base_lex: Optional[LexicalState] = None
    if prefill_text and not prefill_appears_repeated(prefill_text, raw_val):
        base_lex = LexicalState()
        base_lex.feed(prefill_text)
    clean_val, _ = sanitize_synthetic_tail(
        raw_val, emit_tool_name=emit_tool_name, base_lex=base_lex, at_eof=True
    )
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
