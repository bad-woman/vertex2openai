"""
Function Call 通用抗截断传输模块 (Router Emit Anti-Truncation Wrapper)

核心机制：
1. 固定合成工具名 `router_emit`（仅在客户端工具同名冲突时生成随机后缀），保证 Vertex 隐式前缀缓存命中；
2. 零侵入消息历史，与所有客户端及预填充模式正交解耦；
3. 服务端规范约束 (; end the call...) + 服务端原生 stop_sequences (["}]})", "]})}"]) + 网关滑动尾窗熔断三重防线；
4. 支持 Express SDK 与 Studio Cookie 双通道的真流式 (partialArgs)、假流式与非流式透明拆包。
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

DEFAULT_SYNTHETIC_TOOL_NAME = "router_emit"
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
    prefill_text: str = ""
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
    """为单次请求构建抗截断工具规格（默认使用固定名 router_emit 以命中 Vertex 隐式缓存）。"""
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
        emit_name = f"router_{uuid.uuid4().hex[:24]}"
        while emit_name in all_client_names:
            emit_name = f"router_{uuid.uuid4().hex[:24]}"
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
    prefill_text: str = "",
    has_prefill: bool = False,
    for_cookie: bool = False,
) -> Dict[str, Any]:
    """构建通用单字段 content 传输工具声明：
    1. 明确绑定生命周期终止条件 (; end the call when the requested content is complete.)；
    2. Express 模式使用 parameters_json_schema + additionalProperties: False；
    3. Cookie 模式使用标准 parameters（不含 additionalProperties，防止 batchGraphql 校验拒绝）。
    """
    is_cont = bool(has_prefill or (prefill_text and prefill_text.strip()))
    desc = (
        "Deliver only the new continuation text to append to the existing message; end the call when the requested content is complete."
        if is_cont
        else "Deliver the complete user-visible reply as text; end the call when the requested content is complete."
    )
    content_desc = (
        "The new continuation text itself, preserving its language, sections, formatting and markup."
        if is_cont
        else "The requested reply itself, preserving its language, sections, formatting and markup."
    )
    base_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": content_desc,
            },
        },
        "required": ["content"],
    }
    if not for_cookie and _SDK_SUPPORTS_PARAMS_JSON_SCHEMA:
        json_schema = dict(base_schema)
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
    """构建通用传输格式指令（含 0.13.1 防工具自检与防尾部重复填充约束）。"""
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
        "tool selection or serialization. Once the requested content is complete, end the call without repeating "
        "finished content or adding filler. Preserve repetition explicitly requested by the task."
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


def build_anti_truncation_stop_sequences(
    existing_stops: Optional[List[str]],
) -> List[str]:
    """构建服务端原生 stop_sequences（最多 5 个），包含纯结构闭合特征符作为硬件安全气囊。"""
    stops: List[str] = []
    for s in (existing_stops or []):
        if isinstance(s, str) and s and s not in stops:
            stops.append(s)
    for guard in ("}]})", "]})}", "}]})}"):
        if len(stops) >= 5:
            break
        if guard not in stops:
            stops.append(guard)
    return stops[:5]


def inject_express_gen_config(
    gen_config_dict: Dict[str, Any],
    spec: Optional[SyntheticSandboxSpec],
    enable_stream_fc_args: bool = False,
) -> None:
    """向 Express SDK 的 generation_config 字典注入 router_emit 声明、tool_config 与原生 stop_sequences。"""
    if not spec or not spec.enabled:
        return

    emit_decl = build_emit_tool_declaration(
        spec.emit_tool_name,
        prefill_text=spec.prefill_text,
        has_prefill=spec.has_prefill,
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
    gen_config_dict["stop_sequences"] = build_anti_truncation_stop_sequences(
        gen_config_dict.get("stop_sequences")
    )


_TAIL_OVERFLOW_RE = re.compile(
    r"(?:[!?\s]*\}(?![^\n]*\{)[^\n]{0,60}?)?"
    r"(?:"
    r"Let's\s+(?:check|finalize|make\s+sure|verify)\b[^\n]{0,100}?(?:router_emit|tool\s*call|Done!|[}\])]{2,})"
    r"|Done!\s*[}\])\s]{2,}"
    r"|\]\}\)\}"
    r"|\]\}\)\)"
    r"|\]\}\)"
    r")",
    re.IGNORECASE,
)


def sanitize_synthetic_tail(text: str, emit_tool_name: str = "") -> Tuple[str, bool]:
    """检测并平滑切除尾部闭合溢出（同时用于流式 96 字符滑动尾窗与非流式/假流式完整文本）。
    返回: (干净文本, 是否触发了闭合熔断)
    """
    if not text:
        return "", False
    m = _TAIL_OVERFLOW_RE.search(text)
    if m:
        return text[: m.start()], True
    if emit_tool_name and emit_tool_name in text:
        pattern = (
            r"(?:[!?\s]*\}(?![^\n]*\{)[^\n]{0,40}?)?Let\'s\s+[^\n]{0,60}?"
            + re.escape(emit_tool_name)
            + r"\b"
        )
        m_tool = re.search(pattern, text, re.IGNORECASE)
        if m_tool:
            return text[: m_tool.start()], True
    return text, False


def _unwrap_nested_json_if_needed(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return text
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3].strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                for k in CONTENT_KEYS:
                    val = parsed.get(k)
                    if isinstance(val, str) and val:
                        return val
        except Exception:
            pass
    return text


def extract_synthetic_content(args: Any, emit_tool_name: str = DEFAULT_SYNTHETIC_TOOL_NAME) -> str:
    """从合成工具的 args 字典中提取正文 content，并清理 stop_sequences 命中前可能残留的闭合自检尾巴。"""
    raw_val = ""
    if args is None:
        return ""
    elif isinstance(args, str):
        raw_val = _unwrap_nested_json_if_needed(args)
    elif isinstance(args, dict):
        for k in CONTENT_KEYS:
            val = args.get(k)
            if isinstance(val, str) and val:
                raw_val = _unwrap_nested_json_if_needed(val)
                break
        if not raw_val:
            for val in args.values():
                if isinstance(val, str) and val:
                    raw_val = _unwrap_nested_json_if_needed(val)
                    break
    if not raw_val:
        return ""
    clean_val, _ = sanitize_synthetic_tail(raw_val, emit_tool_name=emit_tool_name)
    return clean_val


def is_content_json_path(path_tokens: List[Any], raw_path: str = "") -> bool:
    if path_tokens:
        return str(path_tokens[0]) in CONTENT_KEYS
    low = (raw_path or "").lower()
    return any(k in low for k in CONTENT_KEYS)
