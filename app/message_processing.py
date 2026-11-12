import base64
import re
import json
import time
import uuid
import random
import httpx
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import config as app_config
import model_capabilities as mc
from runtime_state import app_state
from signature_store import (
    signature_store,
    image_signature_store,
    SKIP_VALIDATOR_SENTINEL,
    SignatureRecord,
    SignatureState,
)
from synthetic_wrapper import (
    LexicalState,
    SyntheticSandboxSpec,
    analyze_synthetic_tail,
    build_sandbox_control_instruction,
    extract_raw_synthetic_text,
    extract_synthetic_content,
    is_content_json_path,
    make_bad_woman_response_id,
    prefill_appears_repeated,
    sanitize_synthetic_tail,
)

from google.genai import types
from models import OpenAIMessage, ContentPartText, ContentPartImage, normalize_content_part

import io
try:
    from PIL import Image
except ImportError:
    Image = None


def optimize_image_bytes(image_data: bytes, original_mime: str, max_size_bytes: int = None) -> Tuple[bytes, str]:
    """输入图片压缩引擎：可在控制台配置开关/边长/质量/体积阈值。
    超过阈值的图会限制最长边并重采样，避免多轮修图卡死。"""
    if Image is None:
        return image_data, original_mime

    settings = app_state.get_settings()
    if not settings.get("img_compress_enabled", True):
        return image_data, original_mime

    if max_size_bytes is None:
        try:
            max_size_bytes = int(float(settings.get("img_compress_max_mb", 1.5)) * 1024 * 1024)
        except (TypeError, ValueError):
            max_size_bytes = int(1.5 * 1024 * 1024)
    try:
        max_dim = int(settings.get("img_compress_max_dim", 1536) or 1536)
    except (TypeError, ValueError):
        max_dim = 1536
    try:
        quality = int(settings.get("img_compress_quality", 85) or 85)
    except (TypeError, ValueError):
        quality = 85

    if len(image_data) <= max_size_bytes:
        return image_data, original_mime

    try:
        with Image.open(io.BytesIO(image_data)) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA' or img.mode == 'LA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            opt_data = output.getvalue()

            if len(opt_data) > max_size_bytes:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=max(40, quality - 15), optimize=True)
                opt_data = output.getvalue()

            return opt_data, "image/jpeg"
    except Exception as e:
        print(f"⚠️ [图片处理] 输入图片压缩失败，已回退为原图传输：{e}")
        return image_data, original_mime


SUPPORTED_ROLES = ["user", "model", "function"]

LEGACY_THOUGHT_SEP = "__thought__"


def _signature_bytes(part: Any, fc: Any = None) -> Optional[bytes]:
    """从 part / function_call 上取出思考签名，统一成 bytes。"""
    sig = None
    if isinstance(part, dict):
        sig = part.get("thoughtSignature") or part.get("thought_signature")
    elif part is not None:
        sig = getattr(part, "thought_signature", None)

    if sig is None:
        if isinstance(fc, dict):
            sig = fc.get("thoughtSignature") or fc.get("thought_signature")
        elif fc is not None:
            sig = getattr(fc, "thought_signature", None)

    if isinstance(sig, bytes):
        return sig or None
    if isinstance(sig, str) and sig:
        try:
            return base64.b64decode(sig)
        except Exception:
            return sig.encode("utf-8")
    return None


def encode_thought_signature(signature: Optional[bytes]) -> Optional[str]:
    """Encode SDK signature bytes for the documented OpenAI extension."""
    if not signature:
        return None
    return base64.b64encode(signature).decode("ascii")


def thought_signature_extra(signature: Optional[bytes], part_kind: Optional[str] = None) -> Optional[dict]:
    encoded = encode_thought_signature(signature)
    if not encoded:
        return None
    google = {"thought_signature": encoded}
    if part_kind:
        google["thought_signature_part"] = part_kind
    return {"google": google}


def _extra_google(value: Any) -> dict:
    extra = value.get("extra_content") if isinstance(value, dict) else getattr(value, "extra_content", None)
    return (extra.get("google") or {}) if isinstance(extra, dict) and isinstance(extra.get("google"), dict) else {}


def signature_from_extra(value: Any) -> Tuple[Optional[bytes], Optional[str]]:
    google = _extra_google(value)
    encoded = google.get("thought_signature")
    if not encoded:
        return None, google.get("thought_signature_part")
    if isinstance(encoded, bytes):
        return encoded, google.get("thought_signature_part")
    try:
        return base64.b64decode(str(encoded), validate=True), google.get("thought_signature_part")
    except Exception:
        return str(encoded).encode("utf-8"), google.get("thought_signature_part")


def ordinary_part_metadata(kind: str, text: str, signature: Optional[bytes]) -> dict:
    item = {"kind": kind, "text": text}
    encoded = encode_thought_signature(signature)
    if encoded:
        item["thought_signature"] = encoded
    return item


def ordinary_parts_from_extra(value: Any) -> Optional[List[types.Part]]:
    """Restore exact flattened ordinary Part boundaries from bridge metadata."""
    raw = _extra_google(value).get("ordinary_parts")
    if not isinstance(raw, list):
        return None
    parts: List[types.Part] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        kind = item.get("kind")
        text = item.get("text")
        if kind not in {"text", "thought", "signature_only"} or not isinstance(text, str):
            return None
        if kind == "text" and "data:image/" in text:
            return None
        encoded = item.get("thought_signature")
        signature = None
        if encoded:
            try:
                signature = base64.b64decode(str(encoded), validate=True)
            except Exception:
                signature = str(encoded).encode("utf-8")
        parts.append(types.Part(
            text=text,
            thought=(True if kind == "thought" else None),
            thought_signature=signature,
        ))
    return parts


def build_tool_call_id(fc: Any, part: Any,
                       missing_state: SignatureState = SignatureState.UNKNOWN) -> str:
    """Return a short OpenAI call id and cache signed/unsigned topology fallback."""
    real_id = getattr(fc, "id", None) or ""
    if not isinstance(real_id, str):
        real_id = str(real_id)
    if not real_id:
        real_id = "call_" + uuid.uuid4().hex[:16]

    sig = _signature_bytes(part, fc)
    if sig:
        signature_store.put_record(real_id, SignatureRecord(SignatureState.SIGNED, sig))
    elif missing_state is SignatureState.UNSIGNED_FOLLOWER:
        signature_store.put_unsigned_follower(real_id)
    else:
        signature_store.put_unknown(real_id)
    return real_id


def resolve_tool_call_signature(tool_call_id: str,
                                require_signature: bool = False,
                                explicit_signature: Optional[bytes] = None,
                                explicit_unsigned: bool = False) -> Tuple[str, Optional[bytes]]:
    """Restore (real id, signature) with explicit OpenAI metadata first."""
    real_id = tool_call_id or ""
    sig: Optional[bytes] = explicit_signature

    if LEGACY_THOUGHT_SEP in real_id:
        real_id, _, encoded = real_id.partition(LEGACY_THOUGHT_SEP)
        if sig is None:
            try:
                sig = base64.b64decode(encoded) or None
            except Exception:
                sig = None

    record = signature_store.get_record(real_id) if sig is None and not explicit_unsigned else None
    if sig is None and record and record.state is SignatureState.SIGNED:
        sig = record.signature
    is_unsigned_follower = explicit_unsigned or bool(
        record and record.state is SignatureState.UNSIGNED_FOLLOWER)

    if sig is None and require_signature and not is_unsigned_follower:
        sig = SKIP_VALIDATOR_SENTINEL
        print("⚠️ [工具调用] 必需的思考签名无法恢复，已使用官方跳过校验哨兵。"
              "请求不会失败，但模型表现可能下降。")

    return real_id, sig


def _requires_signature(model_name: str) -> bool:
    """该模型是否强校验思考签名（Gemini 3.x / 4.x 家族及生图模型）。"""
    if not model_name:
        return False
    try:
        prof = mc.get_profile(model_name)
        if prof.get("is_image"):
            return True
        return prof.get("family") in ("g3", "g4")
    except Exception:
        return False




def _extract_markdown_images_to_parts(
    text: str, for_model_role: bool = False
) -> Tuple[List[types.Part], str]:
    parts = []
    remaining_text = text
    pattern = r"!\[[^\]]*\]\(data:(image/[a-zA-Z0-9+.-]+);base64,([a-zA-Z0-9+/=]+)\)"
    matches = list(re.finditer(pattern, text))

    if matches:
        for match in reversed(matches):
            mime_type = match.group(1)
            b64_data = match.group(2)
            if not mime_type.startswith("image/"):
                continue
            try:
                raw_bytes = base64.b64decode(b64_data)
                if for_model_role:
                    # 模型历史轮次中的图片：若命中原始思考签名，严禁有损压缩篡改字节，保证加密签名与原图 100% 匹配
                    sig = image_signature_store.get(raw_bytes)
                    if sig:
                        part = types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
                        try:
                            part = part.model_copy(update={"thought_signature": sig})
                        except Exception:
                            setattr(part, "thought_signature", sig)
                    else:
                        opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                        sig = image_signature_store.get(opt_bytes)
                        part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                        if sig:
                            try:
                                part = part.model_copy(update={"thought_signature": sig})
                            except Exception:
                                setattr(part, "thought_signature", sig)
                else:
                    # user 角色上传的图片：正常执行大图压缩，且绝不附加模型专属的 thought_signature
                    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                    part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                parts.append(part)
                start, end = match.span()
                remaining_text = remaining_text[:start] + remaining_text[end:]
            except Exception as e:
                print(f"⚠️ [图片处理] 提取 Markdown 图片失败，已跳过该图片：{e}")
        parts.reverse()
        remaining_text = re.sub(r"[ \t]+", " ", remaining_text).strip()

    return parts, remaining_text


def _coerce_tool_response(content: Any) -> Dict[str, Any]:
    """把 OpenAI 工具结果安全地转成 function_response 需要的对象。"""
    if content is None:
        return {"result": ""}
    if not isinstance(content, str):
        try:
            return {"result": json.dumps(content, ensure_ascii=False)}
        except Exception:
            return {"result": str(content)}
    s = content.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"result": content}
    return {"result": content}


def _message_text(content: Any) -> str:
    """从 OpenAIMessage.content（str 或分段 list）里提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif hasattr(p, "text") and isinstance(getattr(p, "text", None), str):
                parts.append(p.text)
        return "".join(parts)
    return ""


def _is_empty_message(msg: OpenAIMessage) -> bool:
    if getattr(msg, "tool_calls", None):
        return False
    return not _message_text(msg.content).strip()


DEFAULT_PREFILL_INSTRUCTION = (
    "[继续输出] 下面是你这条回复已经写好的开头，请紧接在最后一个字符之后无缝往下续写，"
    "严禁重复已经写出的开头内容或标签，也不要添加任何前言、解释或标注："
)


DEFAULT_KEEP_TURN_NUDGE = (
    "[继续] 请紧接在你上一条回复的断点处无缝往下写，严禁重复已写出的内容或开标签，不要任何前言或解释。"
)


DEFAULT_IMAGE_PREFILL_NUDGE = (
    "[继续] 按上面说的风格与要求，直接输出图片本身，不要输出任何文字描述或字符画。"
)


def apply_console_injection(
    messages: List[OpenAIMessage],
    system_text: str = "",
    prefill_text: str = "",
    has_tools: bool = False,
    is_image_model: bool = False,
    allow_image_prefill: bool = False,
) -> Tuple[List[OpenAIMessage], List[str]]:
    """把控制台配置的 system 指令 / 预填充注入到消息里。"""
    notes: List[str] = []
    system_text = (system_text or "").strip()
    prefill_text = (prefill_text or "").strip()
    if not system_text and not prefill_text:
        return messages, notes

    new_msgs = list(messages or [])

    if system_text:
        insert_at = 0
        for i, m in enumerate(new_msgs):
            if m.role == "system":
                insert_at = i + 1
        new_msgs.insert(insert_at, OpenAIMessage(role="system", content=system_text))
        notes.append(f"💉 [控制台注入] 已追加 system 指令（{len(system_text)} 字）。")

    if prefill_text:
        if has_tools:
            notes.append("💉 [控制台注入] 请求带函数调用，已跳过预填充注入（避免打乱工具往返）。")
        elif is_image_model and not allow_image_prefill:
            notes.append("💉 [控制台注入] 生图模型，已跳过预填充注入"
                         "（如需用预填充引导画风，请在控制台打开「生图也注入预填充」）。")
        else:
            idx = len(new_msgs) - 1
            while idx >= 0 and _is_empty_message(new_msgs[idx]):
                idx -= 1
            client_has_prefill = (idx >= 0 and new_msgs[idx].role == "assistant"
                                  and not getattr(new_msgs[idx], "tool_calls", None))
            if client_has_prefill:
                notes.append("💉 [控制台注入] 客户端已自带预填充，跳过注入（不覆盖前端预设）。")
            else:
                new_msgs = new_msgs[:idx + 1]
                new_msgs.append(OpenAIMessage(role="assistant", content=prefill_text))
                notes.append(f"💉 [控制台注入] 已注入预填充（{len(prefill_text)} 字），"
                             "下面按预填充兼容模式处理。")

    return new_msgs, notes


_VOID_HTML_TAGS = {
    "br", "hr", "img", "input", "meta", "link", "area", "base",
    "col", "embed", "param", "source", "track", "wbr",
}

_ANY_TAG_TOKEN_RE = re.compile(
    r"<\s*(/)?\s*([A-Za-z\u4e00-\u9fff\u3040-\u30ff][A-Za-z0-9_\-~.:\u4e00-\u9fff\u3040-\u30ff]{0,48})"
    r"(?:\s+[^<>]*?)?\s*(/)?\s*>"
)

_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?(?:-->|\Z)")

_COT_EXACT_NAMES = {
    "think", "thinking", "thought", "thoughts", "cot",
    "reasoning", "reason", "reflection", "reflect",
    "analysis", "analyze", "analyse", "plan", "planning",
    "scratchpad", "monologue", "brainstorm", "brainstorming",
    "draft", "ponder", "deliberate", "meditate", "introspection", "logic",
}

_COT_SUBSTR_HINTS = (
    "think", "thought", "cot", "reason", "reflect", "scratchpad", "monologue",
    "思考", "思维", "推理", "推演", "构思", "分析", "独白", "心声", "心理", "草稿", "审视", "规划",
)

_NON_COT_BODY_OR_LAYOUT_HINTS = (
    "content", "output", "response", "reply", "answer", "story", "narrative",
    "body", "card", "box", "panel", "status", "style", "script", "widget", "container",
    "正文", "回复", "美化", "状态", "面板", "界面",
)

_STANDARD_HTML_LAYOUT_TAGS = {
    "html", "head", "body", "div", "span", "p", "a", "b", "i", "u", "s",
    "strong", "em", "small", "mark", "del", "ins", "sub", "sup",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "dl", "dt", "dd",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup",
    "details", "summary", "dialog", "section", "article", "aside", "nav", "header", "footer", "main",
    "figure", "figcaption", "blockquote", "pre", "code", "kbd", "samp", "var",
    "style", "script", "svg", "path", "g", "circle", "rect", "line", "polygon", "canvas",
    "form", "label", "button", "select", "option", "textarea", "fieldset", "legend",
    "font", "center", "marquee", "ruby", "rt", "rp", "iframe", "audio", "video",
}


def _mask_code_and_comments(text: str) -> str:
    """将围栏代码块、行内反引号代码与 HTML 注释等长替换为空格，防止代码中的 `<` `>` 干扰标签栈识别。"""
    if not text:
        return ""
    lex = LexicalState()
    metas, _ = lex.scan(text)
    chars = [" " if metas[i].in_code and ch != "\n" else ch for i, ch in enumerate(text)]
    masked = "".join(chars)
    if "<!--" in masked:
        masked = _HTML_COMMENT_RE.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)), masked)
    return masked


def _scan_unclosed_tag_stack(text: str) -> List[str]:
    """扫描文本（跳过代码块与注释）并返回所有开启但尚未闭合的标签栈（保留原始大小写）。"""
    if not text or "<" not in text:
        return []
    masked = _mask_code_and_comments(text)
    stack: List[str] = []
    for m in _ANY_TAG_TOKEN_RE.finditer(masked):
        is_close = bool(m.group(1))
        name = m.group(2)
        is_self_close = bool(m.group(3)) or m.group(0).rstrip().endswith("/>")
        low_name = name.lower()
        if low_name in _VOID_HTML_TAGS or low_name.startswith(("http:", "https:", "mailto:", "ftp:")):
            continue
        if is_close:
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx].lower() == low_name:
                    stack.pop(idx)
                    break
        elif not is_self_close:
            stack.append(name)
    return stack


def _is_cot_semantic_tag(tag_name: str) -> bool:
    """判定标签名是否属于思维链/推理类语义标签（排除 HTML 前端美化标签与正文容器标签）。"""
    if not tag_name:
        return False
    low = tag_name.strip().lower()
    # 兼容命名空间前缀（如 antml:thinking -> thinking）
    if ":" in low:
        low = low.rsplit(":", 1)[-1]
    norm = low.strip("_-~.")
    if not norm or norm in _STANDARD_HTML_LAYOUT_TAGS:
        return False
    if norm in _COT_EXACT_NAMES:
        return True
    if any(bad in norm for bad in _NON_COT_BODY_OR_LAYOUT_HINTS):
        return False
    return any(hint in norm for hint in _COT_SUBSTR_HINTS)


def detect_unclosed_tag(text: str) -> Optional[str]:
    """找出文本里（忽略代码块与注释后）"开了但没闭合"的最后一个标签名，用于输出侧防双开标签去重。"""
    stack = _scan_unclosed_tag_stack(text)
    return stack[-1] if stack else None


def detect_unclosed_cot_tag(text: str) -> Optional[str]:
    """找出预填充末尾真正处于未闭合状态的思维链标签名；若停在前端美化 HTML 或正文标签内则返回 None。"""
    stack = _scan_unclosed_tag_stack(text)
    if not stack:
        return None
    for idx in range(len(stack) - 1, -1, -1):
        cand = stack[idx]
        low = cand.lower()
        if any(b in low for b in ("content", "output", "response", "reply", "answer", "正文", "回复")):
            return None
        if _is_cot_semantic_tag(cand):
            return cand
    return None


def build_cot_guard(tag: str, is_keep_turn: bool = False) -> str:
    """生成"必须先写完思维链再写正文"的硬性格式要求（思维链守卫）。"""
    pos_ref = "上一条回复已写好的开头" if is_keep_turn else "下方已写好的开头"
    return (
        f"（格式硬性要求：{pos_ref}正停在未闭合的 <{tag}> 标签内部；"
        f"请直接从断点处继续写出 <{tag}> 内的完整思考内容，写完后用 </{tag}> 闭合，然后才开始写正文；"
        f"不要再次重复输出 <{tag}> 开标签，不要跳过思考直接写正文，也不允许只写一个空标签。）"
    )


def _append_control_to_last_user(messages: List[OpenAIMessage], control_text: str) -> List[OpenAIMessage]:
    """当请求无 prefill 时，将合成工具传输指令追加到最后一条 user 消息末尾（若仅含 system 消息则补一条 user）。"""
    if not control_text or not messages:
        return messages
    new_msgs = list(messages)
    sep = "\n\n"
    for i in range(len(new_msgs) - 1, -1, -1):
        if new_msgs[i].role == "user":
            prev_u = new_msgs[i]
            if isinstance(prev_u.content, str):
                new_msgs[i] = prev_u.model_copy(update={"content": prev_u.content + sep + control_text.strip()})
            elif isinstance(prev_u.content, list):
                merged_list = list(prev_u.content) + [{"type": "text", "text": sep + control_text.strip()}]
                new_msgs[i] = prev_u.model_copy(update={"content": merged_list})
            return new_msgs
    new_msgs.append(OpenAIMessage(role="user", content=control_text.strip()))
    return new_msgs


def apply_prefill_compat(
    messages: List[OpenAIMessage],
    mode: str = "smart",
    allow_model_last: bool = False,
    instruction_template: str = "",
    cot_guard: bool = True,
    synthetic_tool_name: str = "",
    has_callable_client_tools: bool = False,
) -> Tuple[List[OpenAIMessage], str, bool]:
    """
    预填充(prefill)兼容：Gemini 3.x / 4.x 拒绝以 assistant/model 结尾的请求（400）。
    自动穿透末尾或中间夹杂的 system 消息识别真实的最后一条对话轮次，并保留全部 system 消息。
    返回 (处理后的消息列表, 需拼回输出开头的预填充文本, 是否检测到预填充)。
    """
    if not messages:
        return messages, "", False

    if mode == "off":
        if synthetic_tool_name:
            ctrl = build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=False,
                has_callable_client_tools=has_callable_client_tools,
            )
            return _append_control_to_last_user(messages, ctrl), "", False
        return messages, "", False

    # 倒序寻找最后一条非空且非 system 的实际对话消息（因为 system 会被抽入 system_instruction，不进入 contents）
    idx = len(messages) - 1
    while idx >= 0 and (_is_empty_message(messages[idx]) or messages[idx].role == "system"):
        idx -= 1
    if idx < 0:
        if synthetic_tool_name:
            ctrl = build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=False,
                has_callable_client_tools=has_callable_client_tools,
            )
            return _append_control_to_last_user(messages, ctrl), "", False
        return messages, "", False

    last = messages[idx]
    if last.role != "assistant" or getattr(last, "tool_calls", None):
        if synthetic_tool_name:
            ctrl = build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=False,
                has_callable_client_tools=has_callable_client_tools,
            )
            return _append_control_to_last_user(messages, ctrl), "", False
        return messages, "", False

    # 向上收集末尾所有连续的 assistant 预填充消息（穿透中间夹杂的空消息或 system 消息）
    start_idx = idx
    while start_idx > 0:
        prev_m = messages[start_idx - 1]
        if _is_empty_message(prev_m) or prev_m.role == "system":
            start_idx -= 1
            continue
        if prev_m.role == "assistant" and not getattr(prev_m, "tool_calls", None):
            start_idx -= 1
            continue
        break
    # 将 start_idx 向前收紧到第一条实际的 assistant 预填充处（其前面的 system 留在 prefix_msgs 中）
    while start_idx < idx and (messages[start_idx].role == "system" or _is_empty_message(messages[start_idx])):
        start_idx += 1

    prefill_segments: List[str] = []
    trailing_system_msgs: List[OpenAIMessage] = []
    for k in range(start_idx, len(messages)):
        m_item = messages[k]
        if m_item.role == "system":
            if not _is_empty_message(m_item):
                trailing_system_msgs.append(m_item)
        elif not _is_empty_message(m_item) and m_item.role == "assistant":
            seg_text = _message_text(m_item.content)
            if seg_text and seg_text.strip():
                prefill_segments.append(seg_text.lstrip("\r\n").rstrip(" \t\r"))

    raw_prefill = "\n\n".join(prefill_segments)
    if not raw_prefill or not raw_prefill.strip():
        if synthetic_tool_name:
            ctrl = build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=False,
                has_callable_client_tools=has_callable_client_tools,
            )
            return _append_control_to_last_user(messages, ctrl), "", False
        return messages, "", False

    prefill = raw_prefill.lstrip("\r\n").rstrip(" \t\r")
    if prefill.endswith("\n\n"):
        prefill = prefill.rstrip("\n") + "\n"

    prefix_msgs = list(messages[:start_idx])
    merged_assistant_msg = OpenAIMessage(role="assistant", content=prefill)

    if mode == "minimal":
        new_msgs = prefix_msgs + trailing_system_msgs + [merged_assistant_msg]
        tail_text = "(请继续)"
        if synthetic_tool_name:
            unclosed_min = detect_unclosed_cot_tag(prefill) if cot_guard else None
            tail_text += "\n" + build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=True,
                unclosed_tag=unclosed_min,
                has_callable_client_tools=has_callable_client_tools,
            )
        new_msgs.append(OpenAIMessage(role="user", content=tail_text))
        return new_msgs, "", True

    if allow_model_last and not synthetic_tool_name:
        return prefix_msgs + trailing_system_msgs + [merged_assistant_msg], prefill, True

    unclosed_tag = detect_unclosed_cot_tag(prefill) if cot_guard else None

    if mode == "keep_turn":
        new_msgs = prefix_msgs + trailing_system_msgs + [merged_assistant_msg]
        nudge = (instruction_template or "").strip() or DEFAULT_KEEP_TURN_NUDGE
        if synthetic_tool_name:
            nudge = nudge + "\n" + build_sandbox_control_instruction(
                synthetic_tool_name,
                has_prefill=True,
                unclosed_tag=unclosed_tag,
                is_keep_turn=True,
                has_callable_client_tools=has_callable_client_tools,
            )
        elif unclosed_tag:
            nudge = nudge + "\n\n" + build_cot_guard(unclosed_tag, is_keep_turn=True)
        new_msgs.append(OpenAIMessage(role="user", content=nudge))
        return new_msgs, prefill, True

    # smart 模式：将合成工具传输指令与标签守卫置于 intro 内（位于 prefill 原文之前），
    # 并穿透末尾 system 找到最后一条真实 user 消息进行无损合并，确保最终对话严格以 prefill 原文收尾！
    intro = (instruction_template or "").strip() or DEFAULT_PREFILL_INSTRUCTION
    if synthetic_tool_name:
        intro = intro + "\n" + build_sandbox_control_instruction(
            synthetic_tool_name,
            has_prefill=True,
            unclosed_tag=unclosed_tag,
            is_keep_turn=False,
            has_callable_client_tools=has_callable_client_tools,
        )
    elif unclosed_tag:
        intro = intro + "\n" + build_cot_guard(unclosed_tag, is_keep_turn=False)
    instruction = intro + "\n\n" + prefill
    join_sep = "\n\n"

    # 在 prefix_msgs 中倒序查找最后一条非 system 的对话消息，若为 user 则直接合并
    last_non_sys_idx = len(prefix_msgs) - 1
    while last_non_sys_idx >= 0 and prefix_msgs[last_non_sys_idx].role == "system":
        last_non_sys_idx -= 1

    new_msgs = list(prefix_msgs) + trailing_system_msgs
    if last_non_sys_idx >= 0 and new_msgs[last_non_sys_idx].role == "user":
        prev_user = new_msgs.pop(last_non_sys_idx)
        if isinstance(prev_user.content, str):
            merged_u = prev_user.model_copy(update={"content": prev_user.content + join_sep + instruction})
        elif isinstance(prev_user.content, list):
            merged_list = list(prev_user.content) + [{"type": "text", "text": join_sep + instruction}]
            merged_u = prev_user.model_copy(update={"content": merged_list})
        else:
            merged_u = OpenAIMessage(role="user", content=instruction)
        new_msgs.append(merged_u)
    else:
        new_msgs.append(OpenAIMessage(role="user", content=instruction))

    return new_msgs, prefill, True


def strip_prefill_overlap(prefill: str, output: str, min_overlap: int = 4) -> str:
    """预填充智能去重：裁掉模型在输出开头复述的预填充文本或重复开启的未闭合标签。"""
    if not prefill or not output:
        return output

    pre_clean = prefill.rstrip(" \t\r\n")
    if not pre_clean:
        return output

    out_lstripped = output.lstrip(" \t\r\n")
    leading_ws_len = len(output) - len(out_lstripped)

    if output.startswith(prefill):
        return output[len(prefill):]
    if out_lstripped.startswith(pre_clean):
        rest = out_lstripped[len(pre_clean):]
        if prefill.endswith("\n") and rest.startswith("\n"):
            rest = rest[1:]
        return rest

    kmax = min(len(pre_clean), len(out_lstripped))
    effective_min = 2 if (pre_clean.endswith(">") or out_lstripped.startswith("<")) else min(min_overlap, len(pre_clean))
    for k in range(kmax, max(1, effective_min) - 1, -1):
        if pre_clean[-k:] == out_lstripped[:k]:
            rest = out_lstripped[k:]
            if prefill.endswith("\n") and rest.startswith("\n"):
                rest = rest[1:]
            return rest

    unclosed_tag = detect_unclosed_tag(prefill)
    if unclosed_tag:
        tag_pattern = re.compile(
            rf"<\s*{re.escape(unclosed_tag)}(?:\s+[^<>]*?)?\s*>",
            re.IGNORECASE,
        )
        last_tag_match = None
        for m in tag_pattern.finditer(pre_clean):
            last_tag_match = m
        if last_tag_match is not None:
            tail_from_tag = pre_clean[last_tag_match.start():]
            if out_lstripped.startswith(tail_from_tag):
                rest = out_lstripped[len(tail_from_tag):]
                if prefill.endswith("\n") and rest.startswith("\n"):
                    rest = rest[1:]
                return rest
            out_tag_match = tag_pattern.match(out_lstripped)
            if out_tag_match:
                after_out_tag = out_lstripped[out_tag_match.end():]
                after_pre_tag = pre_clean[last_tag_match.end():].lstrip(" \t\r\n")
                after_out_lstripped = after_out_tag.lstrip(" \t\r\n")
                if after_pre_tag and after_out_lstripped.startswith(after_pre_tag):
                    rest = after_out_lstripped[len(after_pre_tag):]
                else:
                    rest = after_out_tag
                    if (prefill.endswith("\n") or pre_clean.endswith(">")) and rest.startswith("\n"):
                        rest = rest[1:]
                return rest

    if leading_ws_len > 0 and prefill.endswith("\n") and output.startswith("\n"):
        return output[1:]

    return output


class PrefillDeduper:
    """流式版预填充去重器。"""

    def __init__(self, prefill: str, window_cap: int = 65536):
        self.prefill = prefill or ""
        self.window = min(len(self.prefill) + 48, window_cap) if self.prefill else 0
        self.buffer = ""
        self.done = self.window == 0
        self._unclosed_tag = detect_unclosed_tag(self.prefill) if self.prefill else None

    def feed(self, text: str) -> str:
        if self.done:
            return text
        if not text:
            return ""
        self.buffer += text
        if len(self.buffer) >= self.window or not self._still_ambiguous():
            return self._resolve()
        return ""

    def _still_ambiguous(self) -> bool:
        """判断当前已攒下的前缀是否仍可能与 prefill 构成更长的重叠。"""
        buf = self.buffer
        pre = self.prefill.rstrip(" \t\r\n")
        if not buf or not pre:
            return False

        s = buf.lstrip(" \t\r\n")
        if not s:
            return len(buf) < 32

        if len(s) >= len(pre) + 16:
            return False

        if s.startswith("<") and ">" not in s and len(s) < 48:
            return True

        if self._unclosed_tag:
            open_tok = f"<{self._unclosed_tag}>"
            if open_tok.lower().startswith(s.lower()) or s.lower().startswith(open_tok.lower()):
                if len(s) < len(pre) + len(open_tok):
                    return True

        if s in pre or pre.startswith(s):
            return True

        max_k = min(len(pre), len(s))
        for k in range(max_k, 0, -1):
            if pre.endswith(s[:k]):
                if k == len(s):
                    return True
                break
        return False

    def flush(self) -> str:
        if self.done:
            return ""
        return self._resolve()

    def _resolve(self) -> str:
        out, self.buffer, self.done = self.buffer, "", True
        return strip_prefill_overlap(self.prefill, out)


# ============================================================
# 流式工具调用参数分片聚合器（支持 streamFunctionCallArguments / partialArgs / willContinue）
# ============================================================

_JSON_PATH_TOKEN_RE = re.compile(r"\.([A-Za-z0-9_$-]+)|\[(\d+)\]|\[['\"]([^'\"]+)['\"]\]")


def _parse_json_path_tokens(json_path: str) -> List[Any]:
    """将 RFC 9535 JSONPath（如 $.tables[0].name 或 $['a.b'][1]）解析为路径键列表。"""
    s = (json_path or "").strip()
    if s.startswith("$"):
        s = s[1:]
    if not s:
        return []
    tokens: List[Any] = []
    for m in _JSON_PATH_TOKEN_RE.finditer(s):
        if m.group(1) is not None:
            tokens.append(m.group(1))
        elif m.group(2) is not None:
            tokens.append(int(m.group(2)))
        elif m.group(3) is not None:
            tokens.append(m.group(3))
    if not tokens and s.startswith("."):
        tokens = [seg for seg in s.split(".") if seg]
    return tokens


def _extract_partial_arg_value(parg: Any) -> Tuple[bool, Any, bool]:
    """从单个 PartialArg（SDK 对象或字典）提取 (是否有值, 值, will_continue)。"""
    def _get(k_snake: str, k_camel: str):
        if isinstance(parg, dict):
            return parg.get(k_camel, parg.get(k_snake))
        return getattr(parg, k_snake, getattr(parg, k_camel, None))

    wc = bool(_get("will_continue", "willContinue"))

    sv = _get("string_value", "stringValue")
    if sv is not None:
        return True, str(sv), wc

    nv = _get("number_value", "numberValue")
    if nv is not None:
        if isinstance(nv, float) and nv.is_integer():
            return True, int(nv), wc
        return True, nv, wc

    bv = _get("bool_value", "boolValue")
    if bv is not None:
        return True, bool(bv), wc

    if isinstance(parg, dict) and ("nullValue" in parg or "null_value" in parg):
        return True, None, wc
    if not isinstance(parg, dict) and getattr(parg, "null_value", None) is not None:
        return True, None, wc

    return False, None, wc


def _set_json_path_value(root: Dict[str, Any], tokens: List[Any], value: Any, append_string: bool = False) -> None:
    """在嵌套字典/列表中按路径写入或追加（用于流式字符串分片）值。"""
    if not tokens:
        return
    curr: Any = root
    for idx in range(len(tokens) - 1):
        key = tokens[idx]
        next_key = tokens[idx + 1]
        want_list = isinstance(next_key, int)
        if isinstance(key, int):
            if not isinstance(curr, list):
                return
            while len(curr) <= key:
                curr.append([] if want_list else {})
            if curr[key] is None or not isinstance(curr[key], (list if want_list else dict)):
                curr[key] = [] if want_list else {}
            curr = curr[key]
        else:
            if not isinstance(curr, dict):
                return
            if key not in curr or not isinstance(curr[key], (list if want_list else dict)):
                curr[key] = [] if want_list else {}
            curr = curr[key]

    last_key = tokens[-1]
    if isinstance(last_key, int):
        if not isinstance(curr, list):
            return
        while len(curr) <= last_key:
            curr.append(None)
        if append_string and isinstance(curr[last_key], str) and isinstance(value, str):
            curr[last_key] = curr[last_key] + value
        else:
            curr[last_key] = value
    else:
        if not isinstance(curr, dict):
            return
        if append_string and isinstance(curr.get(last_key), str) and isinstance(value, str):
            curr[last_key] = curr[last_key] + value
        else:
            curr[last_key] = value


class FunctionCallStreamAssembler:
    """将 Gemini streamFunctionCallArguments 的分片 FunctionCall (partialArgs / willContinue)
    安全聚合成完整的函数调用事件，同时支持合成传输工具 (compose_reply) 的实时流式直通拆包、
    词法状态追踪（保护 Python 代码与酒馆前端美化）及动态隔离尾窗防闭合溢出熔断。"""

    _TAIL_HOLD_CHARS = 96
    _MAX_QUARANTINE_CHARS = 640

    def __init__(self, synthetic_tool_name: str = "", prefill_text: str = ""):
        self._pending: Optional[Dict[str, Any]] = None
        self.synthetic_tool_name: str = (synthetic_tool_name or "").strip()
        self.prefill_text: str = prefill_text or ""
        self.synthetic_hit: bool = False
        self.synthetic_halted: bool = False
        self._synth_eof_sealed: bool = False
        self._pending_is_synthetic: bool = False
        self._last_synth_path: str = ""
        self._synth_signature: Optional[bytes] = None
        self._captured_synth_sig: Optional[bytes] = None
        self._real_calls_emitted: int = 0
        self._synth_tail_buf: str = ""
        self._lex_state: LexicalState = LexicalState()
        self._lex_seeded: bool = not bool(self.prefill_text)

    def _attach_fallback_synth_sig(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """当同轮合成工具占用了唯一的真思考签名，而首个真实工具无签名时，自动将真签名继承给首个真实工具。"""
        for item in calls:
            if self._real_calls_emitted == 0 and not item.get("thought_signature") and self._captured_synth_sig:
                item["thought_signature"] = self._captured_synth_sig
            self._real_calls_emitted += 1
        return calls

    def _ensure_lex_seeded(self, force: bool = False) -> None:
        if self._lex_seeded:
            return
        probe_target = min(len(self.prefill_text.lstrip(" \t\r\n")), 24)
        cur_non_ws = len(self._synth_tail_buf.lstrip(" \t\r\n"))
        if not force and cur_non_ws < max(4, probe_target):
            return
        self._lex_seeded = True
        if not prefill_appears_repeated(self.prefill_text, self._synth_tail_buf):
            self._lex_state.feed(self.prefill_text)

    def has_pending(self) -> bool:
        return self._pending is not None

    def flush_synthetic_tail(self) -> str:
        """冲刷合成工具滑动尾窗中剩余的文本（流结束时调用，执行 EOF 级词法感知修剪）。"""
        if not self._synth_tail_buf:
            return ""
        raw = self._synth_tail_buf
        self._synth_tail_buf = ""
        if self.synthetic_halted or self._synth_eof_sealed:
            return ""
        self._ensure_lex_seeded(force=True)
        clean, _halted, _ = analyze_synthetic_tail(
            raw,
            emit_tool_name=self.synthetic_tool_name,
            base_lex=self._lex_state,
            at_eof=True,
        )
        self._lex_state.feed(clean)
        return clean

    def feed_with_synthetic(
        self, fc: Any, thought_signature: Optional[bytes] = None
    ) -> Tuple[List[Dict[str, Any]], str, Optional[bytes]]:
        """喂入一个 function_call 对象或字典。
        返回: (已聚合完成的真实工具调用列表, 合成工具实时提取出的正文增量文本, 合成工具携带的思考签名)
        """
        if fc is None:
            return [], "", None

        if isinstance(fc, dict):
            fc_id = str(fc.get("id") or "").strip() or None
            fc_name = str(fc.get("name") or "").strip()
            fc_args = fc.get("args")
            fc_partial = fc.get("partialArgs") if "partialArgs" in fc else fc.get("partial_args")
            fc_will_continue = bool(fc.get("willContinue") if "willContinue" in fc else fc.get("will_continue"))
        else:
            fc_id = str(getattr(fc, "id", "") or "").strip() or None
            fc_name = str(getattr(fc, "name", "") or "").strip()
            fc_args = getattr(fc, "args", None)
            fc_partial = getattr(fc, "partial_args", None)
            fc_will_continue = bool(getattr(fc, "will_continue", False))

        is_synth = False
        if self.synthetic_tool_name:
            if fc_name == self.synthetic_tool_name:
                is_synth = True
            elif not fc_name and self._pending_is_synthetic:
                is_synth = True

        if is_synth:
            if self.synthetic_halted or self._synth_eof_sealed:
                return [], "", None

            ready_real: List[Dict[str, Any]] = []
            if self._pending is not None:
                ready_real.append(self._finalize_pending())

            self.synthetic_hit = True
            self._pending_is_synthetic = bool(fc_will_continue)
            if thought_signature:
                self._synth_signature = thought_signature
                self._captured_synth_sig = thought_signature
            if ready_real:
                self._attach_fallback_synth_sig(ready_real)

            has_args = isinstance(fc_args, dict) and len(fc_args) > 0
            has_partial = isinstance(fc_partial, list) and len(fc_partial) > 0
            synth_deltas: List[str] = []

            if has_partial:
                for parg in fc_partial:
                    raw_path = (
                        parg.get("jsonPath", parg.get("json_path"))
                        if isinstance(parg, dict)
                        else getattr(parg, "json_path", getattr(parg, "jsonPath", None))
                    )
                    path_str = str(raw_path or "").strip() or self._last_synth_path
                    if path_str:
                        self._last_synth_path = path_str
                    has_val, val, _ = _extract_partial_arg_value(parg)
                    if has_val and isinstance(val, str) and val:
                        tokens = _parse_json_path_tokens(path_str)
                        if is_content_json_path(tokens, path_str) or not tokens:
                            synth_deltas.append(val)
            elif has_args:
                raw_extracted = extract_raw_synthetic_text(fc_args, emit_tool_name=self.synthetic_tool_name)
                if raw_extracted:
                    synth_deltas.append(raw_extracted)

            incoming_text = "".join(synth_deltas)
            if incoming_text:
                self._synth_tail_buf += incoming_text

            self._ensure_lex_seeded(force=(not fc_will_continue))

            if not fc_will_continue:
                # 合成工具本轮调用正常收敛结束：执行 EOF 级尾锚定词法修剪（不断流，保留后续 usage/tool_calls）
                self._last_synth_path = ""
                clean_buf, trimmed, _ = analyze_synthetic_tail(
                    self._synth_tail_buf,
                    emit_tool_name=self.synthetic_tool_name,
                    base_lex=self._lex_state,
                    at_eof=True,
                )
                self._synth_tail_buf = ""
                self._lex_state.feed(clean_buf)
                if trimmed:
                    self._synth_eof_sealed = True
                sig_to_emit = self._synth_signature
                self._synth_signature = None
                return ready_real, clean_buf, sig_to_emit

            # 流式进行期 (fc_will_continue=True)：仅拦截高置信工具泄漏与真退化死循环，对疑似片段启用动态隔离窗口
            if not self._lex_seeded:
                base_for_check = LexicalState()
                if self.prefill_text and not prefill_appears_repeated(self.prefill_text, self._synth_tail_buf):
                    base_for_check.feed(self.prefill_text)
            else:
                base_for_check = self._lex_state

            clean_buf, halted, suspect_hold = analyze_synthetic_tail(
                self._synth_tail_buf,
                emit_tool_name=self.synthetic_tool_name,
                base_lex=base_for_check,
                at_eof=False,
            )
            if halted:
                self.synthetic_halted = True
                self._pending_is_synthetic = False
                self._last_synth_path = ""
                self._synth_tail_buf = ""
                self._ensure_lex_seeded(force=True)
                self._lex_state.feed(clean_buf)
                sig_to_emit = self._synth_signature
                self._synth_signature = None
                return ready_real, clean_buf, sig_to_emit

            hold_limit = self._MAX_QUARANTINE_CHARS if suspect_hold else self._TAIL_HOLD_CHARS
            if len(self._synth_tail_buf) > hold_limit:
                self._ensure_lex_seeded(force=True)
                cut = len(self._synth_tail_buf) - hold_limit
                out_text = self._synth_tail_buf[:cut]
                self._synth_tail_buf = self._synth_tail_buf[cut:]
                self._lex_state.feed(out_text)
            else:
                out_text = ""

            sig_to_emit = None
            if out_text:
                sig_to_emit = self._synth_signature
                self._synth_signature = None
            return ready_real, out_text, sig_to_emit

        # 若此前处于合成工具分片但本块切换到了真实工具名，复位合成挂起状态
        if fc_name and self.synthetic_tool_name and fc_name != self.synthetic_tool_name:
            self._pending_is_synthetic = False
            self._last_synth_path = ""

        return self._attach_fallback_synth_sig(self.feed(fc, thought_signature=thought_signature)), "", None

    def feed(self, fc: Any, thought_signature: Optional[bytes] = None) -> List[Dict[str, Any]]:
        """喂入一个真实 function_call 对象或字典，返回目前已聚合完成的调用列表。"""
        if fc is None:
            return []

        if isinstance(fc, dict):
            fc_id = str(fc.get("id") or "").strip() or None
            fc_name = str(fc.get("name") or "").strip()
            fc_args = fc.get("args")
            fc_partial = fc.get("partialArgs") if "partialArgs" in fc else fc.get("partial_args")
            fc_will_continue = bool(fc.get("willContinue") if "willContinue" in fc else fc.get("will_continue"))
        else:
            fc_id = str(getattr(fc, "id", "") or "").strip() or None
            fc_name = str(getattr(fc, "name", "") or "").strip()
            fc_args = getattr(fc, "args", None)
            fc_partial = getattr(fc, "partial_args", None)
            fc_will_continue = bool(getattr(fc, "will_continue", False))

        has_args = isinstance(fc_args, dict) and len(fc_args) > 0
        has_partial = isinstance(fc_partial, list) and len(fc_partial) > 0

        if not fc_name and not has_args and not has_partial and not fc_will_continue and self._pending is None:
            return []

        ready: List[Dict[str, Any]] = []

        if self._pending is not None and fc_name and self._pending["name"] and fc_name != self._pending["name"]:
            ready.append(self._finalize_pending())

        if self._pending is None and fc_name and not has_partial and not fc_will_continue:
            ready.append({
                "id": fc_id,
                "name": fc_name,
                "args": dict(fc_args) if isinstance(fc_args, dict) else {},
                "thought_signature": thought_signature,
            })
            return ready

        if self._pending is None:
            self._pending = {
                "id": fc_id,
                "name": fc_name or "unknown",
                "args": dict(fc_args) if isinstance(fc_args, dict) else {},
                "thought_signature": thought_signature,
                "last_path": "",
                "string_continuing": False,
            }
        else:
            if fc_id and not self._pending.get("id"):
                self._pending["id"] = fc_id
            if fc_name and (not self._pending.get("name") or self._pending["name"] == "unknown"):
                self._pending["name"] = fc_name
            if thought_signature and not self._pending.get("thought_signature"):
                self._pending["thought_signature"] = thought_signature
            if has_args and not has_partial:
                self._pending["args"].update(fc_args)

        if has_partial:
            for parg in fc_partial:
                raw_path = (
                    parg.get("jsonPath", parg.get("json_path"))
                    if isinstance(parg, dict)
                    else getattr(parg, "json_path", getattr(parg, "jsonPath", None))
                )
                path_str = str(raw_path or "").strip() or self._pending["last_path"]
                has_val, val, arg_wc = _extract_partial_arg_value(parg)
                if has_val and path_str:
                    tokens = _parse_json_path_tokens(path_str)
                    append_str = bool(
                        isinstance(val, str)
                        and self._pending["string_continuing"]
                        and path_str == self._pending["last_path"]
                    )
                    _set_json_path_value(self._pending["args"], tokens, val, append_string=append_str)
                    self._pending["last_path"] = path_str
                    self._pending["string_continuing"] = bool(arg_wc and isinstance(val, str))
                elif not arg_wc:
                    self._pending["string_continuing"] = False

        if not fc_will_continue:
            ready.append(self._finalize_pending())

        return ready

    def flush(self) -> List[Dict[str, Any]]:
        """流结束时冲刷尚未收到 willContinue=False 尾块的残留函数调用。"""
        if self._pending is None:
            return []
        return self._attach_fallback_synth_sig([self._finalize_pending()])

    def _finalize_pending(self) -> Dict[str, Any]:
        item = self._pending
        self._pending = None
        return {
            "id": item.get("id"),
            "name": item.get("name") or "unknown",
            "args": item.get("args") or {},
            "thought_signature": item.get("thought_signature"),
        }


def sanitize_orphan_tool_turns(messages: List[OpenAIMessage]) -> List[OpenAIMessage]:
    """自愈历史消息中因删楼层、手动中断或上下文滑窗截断产生的不配对 tool_calls / tool 消息，
    防止上游 Gemini 抛出 400 (Function call turn must be followed by function response turn)。"""
    if not messages:
        return messages

    has_any_tool = any(
        getattr(m, "role", None) == "tool" or bool(getattr(m, "tool_calls", None))
        for m in messages
    )
    if not has_any_tool:
        return messages

    out: List[OpenAIMessage] = []
    n = len(messages)
    i = 0

    while i < n:
        msg = messages[i]
        role = getattr(msg, "role", "")
        tcs = getattr(msg, "tool_calls", None)

        if role in ("assistant", "model") and isinstance(tcs, list) and len(tcs) > 0:
            # 向后穿透夹杂的 system 消息，收集紧邻的连续 tool 响应消息
            j = i + 1
            interleaved_sys: List[OpenAIMessage] = []
            matched_tools: List[OpenAIMessage] = []
            while j < n:
                nxt = messages[j]
                nxt_role = getattr(nxt, "role", "")
                if nxt_role == "system":
                    interleaved_sys.append(nxt)
                    j += 1
                    continue
                if nxt_role == "tool":
                    matched_tools.append(nxt)
                    j += 1
                    continue
                break

            if not matched_tools:
                # 后面没有任何 tool 响应（例如中途手动停止后直接继续聊天）：平滑降级为普通文本历史
                summary_lines = []
                orig_text = _message_text(msg.content)
                if orig_text:
                    summary_lines.append(orig_text)
                for tc in tcs:
                    fn = (tc or {}).get("function") or {}
                    summary_lines.append(
                        f"[已中断的工具调用 · {fn.get('name') or 'tool'}({fn.get('arguments') or '{}'})]"
                    )
                downgraded = msg.model_copy(
                    update={"content": "\n".join(summary_lines), "tool_calls": None}
                )
                out.append(downgraded)
                out.extend(interleaved_sys)
                i = j
                continue

            # 存在 tool 响应：对齐每一个 tool_call，若有并行调用缺失响应则自动补齐占位结果
            expected_calls: List[Tuple[str, str]] = []
            for idx_tc, tc in enumerate(tcs):
                raw_id = str((tc or {}).get("id") or f"call_auto_{i}_{idx_tc}")
                norm_id = raw_id.partition(LEGACY_THOUGHT_SEP)[0]
                fn_name = str(((tc or {}).get("function") or {}).get("name") or "unknown")
                expected_calls.append((norm_id, fn_name))

            by_id: Dict[str, List[OpenAIMessage]] = {}
            unmatched_queue: List[OpenAIMessage] = []
            expected_id_set = {cid for cid, _ in expected_calls}
            for tm in matched_tools:
                tid = str(getattr(tm, "tool_call_id", "") or "").partition(LEGACY_THOUGHT_SEP)[0]
                if tid and tid in expected_id_set:
                    by_id.setdefault(tid, []).append(tm)
                else:
                    unmatched_queue.append(tm)

            aligned_tools: List[OpenAIMessage] = []
            for exp_id, exp_name in expected_calls:
                bucket = by_id.get(exp_id)
                if bucket:
                    tm_pick = bucket.pop(0)
                    if not getattr(tm_pick, "name", None) and exp_name:
                        tm_pick = tm_pick.model_copy(update={"name": exp_name})
                    aligned_tools.append(tm_pick)
                elif unmatched_queue:
                    tm_pick = unmatched_queue.pop(0)
                    aligned_tools.append(
                        tm_pick.model_copy(update={"tool_call_id": exp_id, "name": exp_name})
                    )
                else:
                    aligned_tools.append(
                        OpenAIMessage(
                            role="tool",
                            tool_call_id=exp_id,
                            name=exp_name,
                            content='{"status": "skipped", "note": "No tool output recorded in history."}',
                        )
                    )

            out.append(msg)
            out.extend(interleaved_sys)
            out.extend(aligned_tools)
            # 多余未配对的孤立 tool 消息降级为普通文本观测
            for extra_tm in unmatched_queue:
                t_body = _message_text(extra_tm.content)
                out.append(
                    OpenAIMessage(
                        role="user",
                        content=f"[工具执行结果 · {getattr(extra_tm, 'name', None) or 'tool'}]\n{t_body}",
                    )
                )
            i = j
            continue

        if role == "tool":
            # 前面没有对应 assistant.tool_calls 的孤立 tool 消息（常见于长对话上下文滑窗截断开头）
            t_body = _message_text(msg.content)
            out.append(
                OpenAIMessage(
                    role="user",
                    content=f"[工具执行结果 · {getattr(msg, 'name', None) or 'tool'}]\n{t_body}",
                )
            )
            i += 1
            continue

        out.append(msg)
        i += 1

    return out


def create_gemini_prompt(
    messages: List[OpenAIMessage],
    model_name: str = "",
    sandbox_spec: Optional[SyntheticSandboxSpec] = None,
) -> List[types.Content]:
    """OpenAI 消息 → Gemini contents。"""
    print("🔄 [消息转换] 正在将 OpenAI 格式消息转换为 Gemini contents。")
    messages = sanitize_orphan_tool_turns(messages)
    require_sig = _requires_signature(model_name)
    raw_gemini_messages = []
    tool_name_by_id = {}
    for prior in messages:
        for tool_call in (getattr(prior, "tool_calls", None) or []):
            call_id = str(tool_call.get("id") or "")
            call_name = str((tool_call.get("function") or {}).get("name") or "")
            if call_id and call_name:
                tool_name_by_id[call_id] = call_name
                tool_name_by_id[call_id.partition(LEGACY_THOUGHT_SEP)[0]] = call_name

    for idx, message in enumerate(messages):
        role = message.role
        if role == "system":
            continue

        parts = []
        current_gemini_role = ""

        if role == "tool":
            tool_call_id_str = message.tool_call_id or ""
            real_tool_id = tool_call_id_str.partition(LEGACY_THOUGHT_SEP)[0]
            function_name = message.name or tool_name_by_id.get(tool_call_id_str) or tool_name_by_id.get(real_tool_id)

            if not function_name:
                mock_text = f"[System Observation - Tool Result]:\n{message.content}"
                parts.append(types.Part.from_text(text=mock_text))
                current_gemini_role = "user"
            else:
                tool_output_data = _coerce_tool_response(message.content)

                func_resp_kwargs = {"name": function_name, "response": tool_output_data}
                if real_tool_id:
                    func_resp_kwargs["id"] = real_tool_id

                try:
                    resp_part = types.Part(
                        function_response=types.FunctionResponse(**func_resp_kwargs))
                except Exception as e:
                    print(f"⚠️ [工具调用] 构造 FunctionResponse 失败，将回退为基础形式：{e}")
                    resp_part = types.Part.from_function_response(name=function_name, response=tool_output_data)

                parts.append(resp_part)
                current_gemini_role = "user"

        elif role in ("assistant", "model") and message.tool_calls:
            current_gemini_role = "model"
            tool_parts = []
            for tool_index, tool_call in enumerate(message.tool_calls):
                function_call_data = tool_call.get("function", {})
                function_name = function_call_data.get("name", "unknown")
                arguments_str = function_call_data.get("arguments", "{}")
                tool_call_id_str = tool_call.get("id", "") or ""

                try:
                    parsed_arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else (arguments_str or {})
                except json.JSONDecodeError:
                    parsed_arguments = {}

                explicit_sig, _ = signature_from_extra(tool_call)
                real_tool_id, thought_sig_bytes = resolve_tool_call_signature(
                    tool_call_id_str,
                    require_signature=require_sig and tool_index == 0,
                    explicit_signature=explicit_sig,
                    explicit_unsigned=tool_index > 0,
                )

                fc_kwargs = {"name": function_name, "args": parsed_arguments}
                if real_tool_id:
                    fc_kwargs["id"] = real_tool_id

                try:
                    part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                    if thought_sig_bytes:
                        part_kwargs["thought_signature"] = thought_sig_bytes
                    fc_part = types.Part(**part_kwargs)
                except Exception as e:
                    print(f"⚠️ [工具调用] 构造 FunctionCall 失败，将回退为基础形式：{e}")
                    fc_part = types.Part.from_function_call(name=function_name, args=parsed_arguments)
                tool_parts.append(fc_part)

            ordinary_parts = ordinary_parts_from_extra(message)
            if ordinary_parts is None:
                ordinary_parts = []
                message_sig, signature_kind = signature_from_extra(message)
                reasoning = getattr(message, "reasoning_content", None)
                # 仅当客户端完整保留了本条消息的加密思考签名时才回放 thought=True 块，防止无签名思考块触发 400
                if reasoning and message_sig:
                    ordinary_parts.append(types.Part(text=reasoning, thought=True))
                if isinstance(message.content, str) and message.content:
                    image_parts, clean_text = _extract_markdown_images_to_parts(
                        message.content, for_model_role=True
                    )
                    if clean_text:
                        ordinary_parts.append(types.Part.from_text(text=clean_text))
                    ordinary_parts.extend(image_parts)

                if message_sig:
                    target = None
                    if signature_kind == "thought":
                        target = next((p for p in reversed(ordinary_parts) if getattr(p, "thought", None)), None)
                    elif signature_kind == "text":
                        target = next((p for p in reversed(ordinary_parts)
                                       if getattr(p, "text", None) is not None and not getattr(p, "thought", None)), None)
                    if target is None and signature_kind != "signature_only":
                        target = ordinary_parts[-1] if ordinary_parts else None
                    if target is not None:
                        ordinary_parts[ordinary_parts.index(target)] = target.model_copy(
                            update={"thought_signature": message_sig})
                    else:
                        ordinary_parts.append(types.Part(text="", thought_signature=message_sig))

            google_extra = _extra_google(message)
            part_order = google_extra.get("part_order")
            if isinstance(part_order, list):
                ordered, used_tools, used_ordinary = [], set(), set()
                for item in part_order:
                    if isinstance(item, dict) and item.get("type") == "tool_call":
                        i = item.get("index")
                        if isinstance(i, int) and 0 <= i < len(tool_parts):
                            ordered.append(tool_parts[i])
                            used_tools.add(i)
                    elif isinstance(item, dict) and item.get("type") == "ordinary":
                        i = item.get("index")
                        if isinstance(i, int) and 0 <= i < len(ordinary_parts):
                            ordered.append(ordinary_parts[i])
                            used_ordinary.add(i)
                ordered.extend(p for i, p in enumerate(ordinary_parts) if i not in used_ordinary)
                ordered.extend(p for i, p in enumerate(tool_parts) if i not in used_tools)
                # 若排在前面的 ordinary Part 已携带真实签名，避免同轮后续 function_call 重复挂同一签名或跳过哨兵
                seen_prior_sig = False
                for idx_ord, p_ord in enumerate(ordered):
                    p_sig = getattr(p_ord, "thought_signature", None)
                    if getattr(p_ord, "function_call", None) is not None:
                        if seen_prior_sig and p_sig:
                            try:
                                ordered[idx_ord] = p_ord.model_copy(update={"thought_signature": None})
                            except Exception:
                                pass
                    elif p_sig:
                        seen_prior_sig = True
                parts.extend(ordered)
            else:
                parts.extend(tool_parts)
                parts.extend(ordinary_parts)
        else:
            message_sig, signature_kind = signature_from_extra(message)
            reasoning = getattr(message, "reasoning_content", None)
            if message.content is None and not (reasoning and message_sig) and not message_sig:
                continue

            current_gemini_role = role
            if current_gemini_role in ("assistant", "model"):
                current_gemini_role = "model"
            if current_gemini_role not in SUPPORTED_ROLES:
                current_gemini_role = "user"
            is_model_turn = (current_gemini_role == "model")

            exact_ordinary_parts = ordinary_parts_from_extra(message) if is_model_turn else None
            if exact_ordinary_parts is not None:
                has_any_part_sig = any(getattr(ep, "thought_signature", None) for ep in exact_ordinary_parts) or bool(message_sig)
                if not has_any_part_sig:
                    exact_ordinary_parts = [ep for ep in exact_ordinary_parts if not getattr(ep, "thought", None)]
                parts.extend(exact_ordinary_parts)

            if reasoning and exact_ordinary_parts is None and is_model_turn and message_sig:
                parts.append(types.Part(text=reasoning, thought=True))

            if exact_ordinary_parts is None and isinstance(message.content, str):
                image_parts, clean_text = _extract_markdown_images_to_parts(
                    message.content, for_model_role=is_model_turn
                )
                if clean_text:
                    parts.append(types.Part.from_text(text=clean_text))
                parts.extend(image_parts)

            elif exact_ordinary_parts is None and isinstance(message.content, list):
                for part_item in message.content:
                    part_item = normalize_content_part(part_item)
                    if not isinstance(part_item, dict):
                        text_attr = getattr(part_item, "text", None)
                        if isinstance(text_attr, str):
                            parts.append(types.Part.from_text(text=text_attr))
                        continue

                    if part_item.get("type") == "text":
                        text_content = part_item.get("text", "\n")
                        image_parts, clean_text = _extract_markdown_images_to_parts(
                            text_content, for_model_role=is_model_turn
                        )
                        if clean_text:
                            parts.append(types.Part.from_text(text=clean_text))
                        parts.extend(image_parts)

                    elif part_item.get("type") == "image_url":
                        img_url_data = part_item.get("image_url") or {}
                        if isinstance(img_url_data, dict):
                            image_url = img_url_data.get("url", "")
                        else:
                            image_url = getattr(img_url_data, "url", "") or ""

                        if image_url.startswith("data:"):
                            mime_match = re.match(r"data:([^;]+);base64,(.+)", image_url, re.DOTALL)
                            if mime_match:
                                mime_type, b64_data = mime_match.groups()
                                raw_bytes = base64.b64decode(b64_data)
                                if is_model_turn:
                                    sig = image_signature_store.get(raw_bytes)
                                    if sig:
                                        img_part = types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
                                    else:
                                        opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                        sig = image_signature_store.get(opt_bytes)
                                        img_part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                                    if sig:
                                        try:
                                            img_part = img_part.model_copy(update={"thought_signature": sig})
                                        except Exception:
                                            setattr(img_part, "thought_signature", sig)
                                else:
                                    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                    img_part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                                parts.append(img_part)
                        elif image_url.startswith(("http://", "https://")):
                            fetched = fetch_remote_image(image_url)
                            if fetched:
                                img_bytes, mime_type = fetched
                                if is_model_turn:
                                    sig = image_signature_store.get(img_bytes)
                                    if sig:
                                        img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                                    else:
                                        opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                        sig = image_signature_store.get(opt_bytes)
                                        img_part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                                    if sig:
                                        try:
                                            img_part = img_part.model_copy(update={"thought_signature": sig})
                                        except Exception:
                                            setattr(img_part, "thought_signature", sig)
                                else:
                                    opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                    img_part = types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime)
                                parts.append(img_part)

            if message_sig and exact_ordinary_parts is None:
                target = None
                if signature_kind == "thought":
                    target = next((p for p in reversed(parts) if getattr(p, "thought", None)), None)
                elif signature_kind == "text":
                    target = next((p for p in reversed(parts)
                                   if getattr(p, "text", None) is not None and not getattr(p, "thought", None)), None)
                if target is None and signature_kind != "signature_only":
                    target = parts[-1] if parts else None
                if target is None:
                    parts.append(types.Part(text="", thought_signature=message_sig))
                else:
                    parts[parts.index(target)] = target.model_copy(update={"thought_signature": message_sig})

        if not parts:
            continue

        if current_gemini_role == "model":
            for p_idx, p in enumerate(parts):
                inline = getattr(p, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    if not getattr(p, "thought_signature", None):
                        resolved_sig = image_signature_store.get(inline.data)
                        if not resolved_sig and require_sig:
                            resolved_sig = SKIP_VALIDATOR_SENTINEL
                            print("⚠️ [生图历史] 历史回复中的图片思考签名未在缓存命中，已自动补全官方跳过校验哨兵。")
                        elif resolved_sig:
                            print("✅ [生图历史] 已成功从缓存恢复历史生图的完整思考签名。")
                        if resolved_sig:
                            try:
                                parts[p_idx] = p.model_copy(update={"thought_signature": resolved_sig})
                            except Exception:
                                try:
                                    setattr(parts[p_idx], "thought_signature", resolved_sig)
                                except Exception:
                                    pass

        raw_gemini_messages.append(types.Content(role=current_gemini_role, parts=parts))

    def _is_text_only(content: types.Content) -> bool:
        for p in content.parts or []:
            if getattr(p, "function_call", None) is not None:
                return False
            if getattr(p, "function_response", None) is not None:
                return False
            if getattr(p, "inline_data", None) is not None:
                return False
            if getattr(p, "text", None) is None:
                return False
        return True

    def _is_function_response_only(content: types.Content) -> bool:
        return bool(content.parts) and all(
            getattr(p, "function_response", None) is not None for p in content.parts)

    merged_messages = []
    for msg in raw_gemini_messages:
        if merged_messages and merged_messages[-1].role == msg.role:
            previous = merged_messages[-1]
            previous_is_results = _is_function_response_only(previous)
            current_is_results = _is_function_response_only(msg)
            if previous_is_results != current_is_results:
                merged_messages.append(msg)
                continue
            if _is_text_only(previous) and _is_text_only(msg):
                previous.parts.append(types.Part.from_text(text="\n\n"))
            previous.parts.extend(msg.parts)
        else:
            merged_messages.append(msg)

    if not merged_messages:
        merged_messages.append(types.Content(role="user", parts=[types.Part.from_text(text="继续")]))

    return merged_messages


MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REMOTE_IMAGE_REDIRECTS = 3


def _is_blocked_host(host: str) -> bool:
    import ipaddress
    import socket

    if not host:
        return True
    host = host.strip("[]")
    candidates = []
    try:
        candidates.append(ipaddress.ip_address(host))
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except Exception:
            return True
        for info in infos:
            try:
                candidates.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
    if not candidates:
        return True
    for ip in candidates:
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def fetch_remote_image(url: str, timeout: float = 10.0) -> Optional[Tuple[bytes, str]]:
    from urllib.parse import urlparse

    check_host = not app_config.PROXY_URL
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"不支持的 URL 协议：{parsed.scheme or '(空)'}")
        if check_host and _is_blocked_host(parsed.hostname or ""):
            raise ValueError("目标地址指向内网/环回/链路本地，已拒绝")

        client_args = {"timeout": timeout, "follow_redirects": False}
        if app_config.PROXY_URL:
            client_args["proxy"] = app_config.PROXY_URL
        if getattr(app_config, "SSL_CERT_FILE", None):
            client_args["verify"] = app_config.SSL_CERT_FILE

        with httpx.Client(**client_args) as client:
            current = url
            for _ in range(MAX_REMOTE_IMAGE_REDIRECTS + 1):
                resp = client.get(current)
                if resp.is_redirect:
                    current = str(resp.next_request.url) if resp.next_request else ""
                    nxt = urlparse(current)
                    if nxt.scheme not in ("http", "https"):
                        raise ValueError(f"重定向到不支持的协议：{nxt.scheme or '(空)'}")
                    if check_host and _is_blocked_host(nxt.hostname or ""):
                        raise ValueError("重定向指向内网/环回/链路本地，已拒绝")
                    continue

                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                if mime and not mime.startswith("image/"):
                    raise ValueError(f"响应不是图片（content-type={mime}）")
                data = resp.content
                if len(data) > MAX_REMOTE_IMAGE_BYTES:
                    raise ValueError(
                        f"图片超过 {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)}MB 上限（{len(data)} 字节）")
                return data, mime or "image/jpeg"
            raise ValueError("重定向次数过多")
    except Exception as e:
        print(f"⚠️ [图片处理] 获取远程图片失败，已跳过：{url[:120]}，原因：{e}")
        return None


def _wire_image_part(raw_bytes: bytes, mime_type: str, for_model_role: bool = False) -> Dict[str, Any]:
    if for_model_role:
        sig = image_signature_store.get(raw_bytes)
        if sig:
            return {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(raw_bytes).decode("utf-8"),
                },
                "thoughtSignature": base64.b64encode(sig).decode("ascii") if isinstance(sig, bytes) else str(sig),
            }
        opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
        sig = image_signature_store.get(opt_bytes)
        res = {
            "inlineData": {
                "mimeType": opt_mime,
                "data": base64.b64encode(opt_bytes).decode("utf-8"),
            }
        }
        if sig:
            res["thoughtSignature"] = base64.b64encode(sig).decode("ascii") if isinstance(sig, bytes) else str(sig)
        return res

    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
    return {
        "inlineData": {
            "mimeType": opt_mime,
            "data": base64.b64encode(opt_bytes).decode("utf-8"),
        }
    }


def openai_content_to_wire_parts(content: Any, for_model_role: bool = False) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if content is None:
        return parts

    def _add_text_with_inline_images(text: str):
        if not text:
            return
        img_parts, clean_text = _extract_markdown_images_to_parts(text, for_model_role=for_model_role)
        if clean_text:
            parts.append({"text": clean_text})
        for ip in img_parts:
            blob = getattr(ip, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                data = blob.data
                if isinstance(data, bytes):
                    data = base64.b64encode(data).decode("utf-8")
                wire_part = {"inlineData": {"mimeType": getattr(blob, "mime_type", "image/jpeg"),
                                            "data": data}}
                sig = getattr(ip, "thought_signature", None) if for_model_role else None
                if sig:
                    wire_part["thoughtSignature"] = base64.b64encode(sig).decode("ascii") if isinstance(sig, bytes) else str(sig)
                parts.append(wire_part)

    if isinstance(content, str):
        _add_text_with_inline_images(content)
        return parts

    if isinstance(content, list):
        for item in content:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if isinstance(item, str):
                _add_text_with_inline_images(item)
                continue
            if not isinstance(item, dict):
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    _add_text_with_inline_images(text_attr)
                continue

            item_type = item.get("type", "")
            if item_type == "text":
                _add_text_with_inline_images(item.get("text", ""))
            elif item_type == "image_url":
                url = item.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                if url.startswith("data:"):
                    m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
                    if m:
                        mime_type, b64_data = m.groups()
                        try:
                            parts.append(
                                _wire_image_part(
                                    base64.b64decode(b64_data), mime_type, for_model_role=for_model_role
                                )
                            )
                        except Exception as e:
                            print(f"⚠️ [图片处理] 解析内联图片失败，已跳过：{e}")
                elif url.startswith("http"):
                    fetched = fetch_remote_image(url)
                    if fetched:
                        parts.append(_wire_image_part(*fetched, for_model_role=for_model_role))
    return parts


def _rating_fields(r: Any) -> Tuple[str, str, Optional[float], Optional[float]]:
    def _enum_name(v):
        return getattr(v, "name", None) or (str(v) if v is not None else "")

    if isinstance(r, dict):
        cat = str(r.get("category") or "")
        prob = str(r.get("probability") or "")
        ps, ss = r.get("probabilityScore"), r.get("severityScore")
    else:
        cat = _enum_name(getattr(r, "category", None))
        prob = _enum_name(getattr(r, "probability", None))
        ps, ss = getattr(r, "probability_score", None), getattr(r, "severity_score", None)
    cat = cat.replace("HARM_CATEGORY_", "").replace("_", " ").title()
    return cat, prob, (ps if isinstance(ps, (int, float)) else None), (ss if isinstance(ss, (int, float)) else None)


def _create_safety_ratings_html(safety_ratings: list) -> str:
    if not safety_ratings:
        return ""
    normalized = [_rating_fields(r) for r in safety_ratings]
    highest = max(normalized, key=lambda t: t[2] if t[2] is not None else -1.0)
    highest_score = highest[2] if highest[2] is not None else -1.0

    if highest_score < 0:
        color = "#888"
    elif highest_score <= 0.33:
        color = "#0f8"
    elif highest_score <= 0.66:
        color = "yellow"
    else:
        color = "#bf555d"

    def _fmt(cat, prob, ps, ss):
        ps_s = f"{ps:.7f}" if ps is not None else "None"
        ss_s = f"{ss:.8f}" if ss is not None else "None"
        return f"{cat}: {prob} (Score: {ps_s}, Severity: {ss_s})"

    summary_line = _fmt(*highest)
    all_ratings_str = "\n".join(_fmt(*t) for t in normalized)

    css_style = "<style>.cb{border:1px solid #444;margin:10px;border-radius:4px;background:#111}.cb summary{padding:8px;cursor:pointer;background:#222}.cb pre{margin:0;padding:10px;border-top:1px solid #444;white-space:pre-wrap}</style>"
    html_output = (
        f"{css_style}"
        f"<details class='cb'>"
        f"<summary style='color:{color}'>{summary_line} ▼</summary>"
        f"<pre>\n--- Safety Ratings ---\n{all_ratings_str}\n</pre>"
        f"</details>"
    )
    return html_output


def _convert_image_to_markdown(image_data: bytes, mime_type: str) -> str:
    try:
        b64_data = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"
        return f"![Image]({data_url})"
    except Exception as e:
        print(f"⚠️ [图片处理] 将 Gemini 图片转换为 Markdown 失败：{e}")
        return "[Image could not be displayed]"


def parse_gemini_response_for_reasoning_and_content(gemini_response_candidate: Any) -> Tuple[str, str]:
    reasoning_text_parts = []
    normal_text_parts = []
    candidate_part_text = ""
    if hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None:
        candidate_part_text = str(gemini_response_candidate.text)

    gemini_candidate_content = None
    if hasattr(gemini_response_candidate, "content"):
        gemini_candidate_content = gemini_response_candidate.content

    if gemini_candidate_content and hasattr(gemini_candidate_content, "parts") and gemini_candidate_content.parts:
        for part_item in gemini_candidate_content.parts:
            if hasattr(part_item, "function_call") and part_item.function_call is not None:
                continue

            part_text = ""
            if hasattr(part_item, "text") and part_item.text is not None:
                part_text = str(part_item.text)
            elif hasattr(part_item, "inline_data") and part_item.inline_data is not None:
                inline_data = part_item.inline_data
                if hasattr(inline_data, "data") and hasattr(inline_data, "mime_type"):
                    image_bytes = inline_data.data
                    mime_type = inline_data.mime_type
                    sig = getattr(part_item, "thought_signature", None) or getattr(gemini_response_candidate, "thought_signature", None)
                    if sig and image_bytes:
                        image_signature_store.put(image_bytes, sig)
                    part_text = _convert_image_to_markdown(image_bytes, mime_type)
            elif hasattr(part_item, "file_data") and part_item.file_data is not None:
                file_data = part_item.file_data
                if hasattr(file_data, "file_uri"):
                    file_uri = file_data.file_uri
                    part_text = f"![Image]({file_uri})"

            part_is_thought = hasattr(part_item, "thought") and part_item.thought is True

            if part_is_thought:
                reasoning_text_parts.append(part_text)
            elif part_text:
                normal_text_parts.append(part_text)

    elif candidate_part_text:
        normal_text_parts.append(candidate_part_text)
    elif gemini_candidate_content and hasattr(gemini_candidate_content, "text") and gemini_candidate_content.text is not None:
        normal_text_parts.append(str(gemini_candidate_content.text))
    elif hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None and not gemini_candidate_content:
        normal_text_parts.append(str(gemini_response_candidate.text))

    return "".join(reasoning_text_parts), "".join(normal_text_parts)


def process_gemini_response_to_openai_dict(
    gemini_response_obj: Any,
    request_model_str: str,
    synthetic_tool_name: str = "",
    prefill_text: str = "",
) -> Dict[str, Any]:
    choices = []
    response_timestamp = int(time.time())
    base_id = make_bad_woman_response_id(
        default_prefix="chatcmpl",
        sandbox_enabled=bool(synthetic_tool_name),
    )

    if hasattr(gemini_response_obj, "candidates") and gemini_response_obj.candidates:
        for i, candidate in enumerate(gemini_response_obj.candidates):
            message_payload = {"role": "assistant"}

            raw_finish_reason = getattr(candidate, "finish_reason", None)
            openai_finish_reason = "stop"
            if raw_finish_reason:
                if hasattr(raw_finish_reason, "name"):
                    raw_gemini_finish_reason_str = raw_finish_reason.name.upper()
                else:
                    raw_gemini_finish_reason_str = str(raw_finish_reason).upper()

                if raw_gemini_finish_reason_str == "STOP":
                    openai_finish_reason = "stop"
                elif raw_gemini_finish_reason_str == "MAX_TOKENS":
                    openai_finish_reason = "length"
                elif raw_gemini_finish_reason_str == "SAFETY":
                    openai_finish_reason = "content_filter"
                elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]:
                    openai_finish_reason = "tool_calls"

            parts_in_order = list(getattr(getattr(candidate, "content", None), "parts", None) or [])
            reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)

            # 预扫描是否命中了合成输出工具
            synthetic_texts: List[str] = []
            synthetic_hit = False
            if synthetic_tool_name:
                for p_chk in parts_in_order:
                    fc_chk = getattr(p_chk, "function_call", None)
                    if fc_chk is not None and getattr(fc_chk, "name", "") == synthetic_tool_name:
                        synthetic_hit = True
                        extracted = extract_synthetic_content(
                            getattr(fc_chk, "args", None),
                            emit_tool_name=synthetic_tool_name,
                            prefill_text=prefill_text,
                        )
                        if extracted:
                            synthetic_texts.append(extracted)
            if synthetic_hit:
                # 单一事实来源（防拼接）：使用合成工具提取出的 content 覆盖普通文本，保留 reasoning_str
                normal_content_str = "".join(synthetic_texts)

            if app_state.get_setting("safety_score", app_config.SAFETY_SCORE) and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                normal_content_str += _create_safety_ratings_html(candidate.safety_ratings)

            tool_index = 0
            ordinary_metadata = []
            order_descriptors = []
            message_signature = None
            message_signature_kind = None
            orphan_synth_sig: Optional[bytes] = None
            for part in parts_in_order:
                fc = getattr(part, "function_call", None)
                sig = _signature_bytes(part, fc)
                if fc is not None:
                    if synthetic_tool_name and getattr(fc, "name", "") == synthetic_tool_name:
                        synth_piece = extract_synthetic_content(
                            getattr(fc, "args", None),
                            emit_tool_name=synthetic_tool_name,
                            prefill_text=prefill_text,
                        )
                        ordinary_index = len(ordinary_metadata)
                        ordinary_metadata.append(ordinary_part_metadata("text", synth_piece, sig))
                        order_descriptors.append({"type": "ordinary", "index": ordinary_index})
                        if sig:
                            message_signature = sig
                            message_signature_kind = "text"
                            orphan_synth_sig = sig
                        continue
                    # 若合成工具排在前面占用了真实签名，而首个真实工具无签名，直接继承真实签名供无扩展客户端回放
                    if sig is None and tool_index == 0 and orphan_synth_sig is not None:
                        sig = orphan_synth_sig
                        part = part.model_copy(update={"thought_signature": sig})
                    missing_state = (SignatureState.UNSIGNED_FOLLOWER if tool_index > 0
                                     else SignatureState.UNKNOWN)
                    tool_call_id = build_tool_call_id(fc, part, missing_state=missing_state)
                    tool_payload = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args or {}, ensure_ascii=False)
                        }
                    }
                    extra = thought_signature_extra(sig)
                    if extra:
                        tool_payload["extra_content"] = extra
                    message_payload.setdefault("tool_calls", []).append(tool_payload)
                    order_descriptors.append({"type": "tool_call", "index": tool_index})
                    tool_index += 1
                    openai_finish_reason = "tool_calls"
                    continue

                is_thought = getattr(part, "thought", None) is True
                if synthetic_hit and not is_thought and getattr(part, "inline_data", None) is None and getattr(part, "file_data", None) is None:
                    # 合成工具命中时，跳过普通多嘴文本 Part，但保留纯签名 Part
                    if sig:
                        message_signature = sig
                        message_signature_kind = "text"
                    continue

                text_value = getattr(part, "text", None)
                if text_value is None and getattr(part, "inline_data", None) is not None:
                    inline = part.inline_data
                    text_value = _convert_image_to_markdown(inline.data, inline.mime_type)
                    if getattr(inline, "data", None):
                        img_sig = sig or getattr(candidate, "thought_signature", None) or next(
                            (_signature_bytes(p_c) for p_c in parts_in_order if _signature_bytes(p_c)), None
                        )
                        if img_sig:
                            image_signature_store.put(inline.data, img_sig)
                            try:
                                opt_b, _ = optimize_image_bytes(inline.data, inline.mime_type)
                                if opt_b != inline.data:
                                    image_signature_store.put(opt_b, img_sig)
                            except Exception:
                                pass
                elif text_value is None and getattr(part, "file_data", None) is not None:
                    text_value = f"![Image]({part.file_data.file_uri})"
                text_value = "" if text_value is None else str(text_value)
                if is_thought:
                    ordinary_kind = "thought"
                elif text_value == "" and sig:
                    ordinary_kind = "signature_only"
                else:
                    ordinary_kind = "text"
                ordinary_index = len(ordinary_metadata)
                ordinary_metadata.append(ordinary_part_metadata(ordinary_kind, text_value, sig))
                order_descriptors.append({"type": "ordinary", "index": ordinary_index})
                if sig:
                    message_signature = sig
                    message_signature_kind = ordinary_kind

            if not message_payload.get("tool_calls") and openai_finish_reason == "tool_calls":
                openai_finish_reason = "stop"
            if normal_content_str or not message_payload.get("tool_calls"):
                message_payload["content"] = normal_content_str
            else:
                message_payload["content"] = None
            if reasoning_str:
                message_payload["reasoning_content"] = reasoning_str

            if message_signature:
                message_payload["extra_content"] = thought_signature_extra(
                    message_signature, message_signature_kind)
            if ordinary_metadata:
                google = message_payload.setdefault("extra_content", {}).setdefault("google", {})
                google["ordinary_parts"] = ordinary_metadata
            if message_payload.get("tool_calls") and order_descriptors:
                google = message_payload.setdefault("extra_content", {}).setdefault("google", {})
                google["part_order"] = order_descriptors

            choice_item = {"index": i, "message": message_payload, "finish_reason": openai_finish_reason}
            if hasattr(candidate, "logprobs") and candidate.logprobs is not None:
                choice_item["logprobs"] = candidate.logprobs
            choices.append(choice_item)

    elif hasattr(gemini_response_obj, "text") and gemini_response_obj.text is not None:
        content_str = gemini_response_obj.text or ""
        choices.append({"index": 0, "message": {"role": "assistant", "content": content_str}, "finish_reason": "stop"})
    else:
        choices.append({"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"})

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(gemini_response_obj, "usage_metadata"):
        um = gemini_response_obj.usage_metadata
        if hasattr(um, "prompt_token_count"):
            usage_data["prompt_tokens"] = um.prompt_token_count
        if hasattr(um, "candidates_token_count"):
            usage_data["completion_tokens"] = um.candidates_token_count
            if hasattr(um, "total_token_count"):
                usage_data["total_tokens"] = um.total_token_count
            else:
                usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
        elif hasattr(um, "total_token_count"):
            usage_data["total_tokens"] = um.total_token_count
            if usage_data["prompt_tokens"] > 0 and usage_data["total_tokens"] > usage_data["prompt_tokens"]:
                usage_data["completion_tokens"] = usage_data["total_tokens"] - usage_data["prompt_tokens"]
        else:
            usage_data["total_tokens"] = usage_data["prompt_tokens"]

    return {
        "id": base_id, "object": "chat.completion", "created": response_timestamp,
        "model": request_model_str, "choices": choices,
        "usage": usage_data
    }


def convert_to_openai_format(
    gemini_response: Any,
    model: str,
    synthetic_tool_name: str = "",
    prefill_text: str = "",
) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(
        gemini_response,
        model,
        synthetic_tool_name=synthetic_tool_name,
        prefill_text=prefill_text,
    )