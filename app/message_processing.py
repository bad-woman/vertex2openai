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
    SKIP_VALIDATOR_SENTINEL,
    SignatureRecord,
    SignatureState,
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


def _signature_bytes(part: Any, fc: Any) -> Optional[bytes]:
    """从 part / function_call 上取出思考签名，统一成 bytes。"""
    sig = getattr(part, "thought_signature", None)
    if sig is None:
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
    """该模型是否强校验思考签名（仅 Gemini 3.x / 4.x 家族）。"""
    if not model_name:
        return False
    try:
        return mc.get_profile(model_name)["family"] == "g3"
    except Exception:
        return False




def _extract_markdown_images_to_parts(text: str) -> Tuple[List[types.Part], str]:
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
                opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
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


_TAG_OPEN_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_\-~.:]*)\s*>")


def detect_unclosed_tag(text: str) -> Optional[str]:
    """找出文本里"开了但没闭合"的最后一个标签名，没有则返回 None。"""
    if not text:
        return None
    last: Optional[str] = None
    for m in _TAG_OPEN_RE.finditer(text):
        name = m.group(1)
        if re.search(rf"</\s*{re.escape(name)}\s*>", text[m.end():]):
            continue
        last = name
    return last


def build_cot_guard(tag: str) -> str:
    """生成"必须先写完思维链再写正文"的强化要求（思维链守卫）。"""
    return (
        f"（格式硬性要求：下方已写好的开头正停在未闭合的 <{tag}> 标签内部；"
        f"请直接从断点处继续写出 <{tag}> 内的全部思考内容，写完后用 </{tag}> 闭合，然后才开始写正文；"
        f"严禁再次重复输出 <{tag}> 开标签，严禁跳过思考直接写正文，也不允许只写一个空标签。）"
    )


def apply_prefill_compat(
    messages: List[OpenAIMessage],
    mode: str = "smart",
    allow_model_last: bool = False,
    instruction_template: str = "",
    cot_guard: bool = False,
) -> Tuple[List[OpenAIMessage], str, bool]:
    """
    预填充(prefill)兼容：Gemini 3.x / 4.x 拒绝以 assistant/model 结尾的请求（400）。
    返回 (处理后的消息列表, 需拼回输出开头的预填充文本, 是否检测到预填充)。
    """
    if not messages or mode == "off":
        return messages, "", False

    idx = len(messages) - 1
    while idx >= 0 and _is_empty_message(messages[idx]):
        idx -= 1
    if idx < 0:
        return messages, "", False

    last = messages[idx]
    if last.role != "assistant" or getattr(last, "tool_calls", None):
        return messages, "", False

    raw_prefill = _message_text(last.content)
    if not raw_prefill or not raw_prefill.strip():
        return messages, "", False

    prefill = raw_prefill.lstrip("\r\n").rstrip(" \t\r")
    if prefill.endswith("\n\n"):
        prefill = prefill.rstrip("\n") + "\n"

    if mode == "minimal":
        new_msgs = list(messages[:idx + 1])
        new_msgs.append(OpenAIMessage(role="user", content="(请继续)"))
        return new_msgs, "", True

    if allow_model_last:
        return messages, prefill, True

    unclosed_tag = detect_unclosed_tag(prefill) if cot_guard else None

    if mode == "keep_turn":
        new_msgs = list(messages[:idx + 1])
        nudge = (instruction_template or "").strip() or DEFAULT_KEEP_TURN_NUDGE
        if unclosed_tag:
            nudge = nudge + "\n\n" + build_cot_guard(unclosed_tag)
        new_msgs.append(OpenAIMessage(role="user", content=nudge))
        return new_msgs, prefill, True

    # smart 模式：将 cot_guard 放在 intro 与 prefill 之间，确保整条消息严格以 prefill 原文断点收尾
    new_msgs = list(messages[:idx])
    intro = (instruction_template or "").strip() or DEFAULT_PREFILL_INSTRUCTION
    if unclosed_tag:
        intro = intro + "\n" + build_cot_guard(unclosed_tag)
    instruction = intro + "\n\n" + prefill

    if new_msgs and new_msgs[-1].role == "user":
        prev_user = new_msgs[-1]
        if isinstance(prev_user.content, str):
            merged = prev_user.content + "\n\n" + instruction
            new_msgs[-1] = prev_user.model_copy(update={"content": merged})
        elif isinstance(prev_user.content, list):
            merged_list = list(prev_user.content) + [{"type": "text", "text": "\n\n" + instruction}]
            new_msgs[-1] = prev_user.model_copy(update={"content": merged_list})
        else:
            new_msgs.append(OpenAIMessage(role="user", content=instruction))
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
        tag_pattern = re.compile(rf"<\s*{re.escape(unclosed_tag)}\s*>", re.IGNORECASE)
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

    def __init__(self, prefill: str, window_cap: int = 600):
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
    安全聚合成完整的函数调用事件，同时 100% 兼容普通单块 FunctionCall。"""

    def __init__(self):
        self._pending: Optional[Dict[str, Any]] = None

    def has_pending(self) -> bool:
        return self._pending is not None

    def feed(self, fc: Any, thought_signature: Optional[bytes] = None) -> List[Dict[str, Any]]:
        """喂入一个 function_call 对象或字典，返回目前已聚合完成的调用列表。"""
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
        return [self._finalize_pending()]

    def _finalize_pending(self) -> Dict[str, Any]:
        item = self._pending
        self._pending = None
        return {
            "id": item.get("id"),
            "name": item.get("name") or "unknown",
            "args": item.get("args") or {},
            "thought_signature": item.get("thought_signature"),
        }


def create_gemini_prompt(messages: List[OpenAIMessage], model_name: str = "") -> List[types.Content]:
    """OpenAI 消息 → Gemini contents。"""
    print("🔄 [消息转换] 正在将 OpenAI 格式消息转换为 Gemini contents。")
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
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    ordinary_parts.append(types.Part(text=reasoning, thought=True))
                if isinstance(message.content, str) and message.content:
                    image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                    if clean_text:
                        ordinary_parts.append(types.Part.from_text(text=clean_text))
                    ordinary_parts.extend(image_parts)

                message_sig, signature_kind = signature_from_extra(message)
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
                parts.extend(ordered)
            else:
                parts.extend(tool_parts)
                parts.extend(ordinary_parts)
        else:
            message_sig, signature_kind = signature_from_extra(message)
            reasoning = getattr(message, "reasoning_content", None)
            if message.content is None and not reasoning and not message_sig:
                continue

            current_gemini_role = role
            if current_gemini_role in ("assistant", "model"):
                current_gemini_role = "model"
            if current_gemini_role not in SUPPORTED_ROLES:
                current_gemini_role = "user"

            exact_ordinary_parts = ordinary_parts_from_extra(message)
            if exact_ordinary_parts is not None:
                parts.extend(exact_ordinary_parts)

            if reasoning and exact_ordinary_parts is None:
                parts.append(types.Part(text=reasoning, thought=True))

            if exact_ordinary_parts is None and isinstance(message.content, str):
                image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
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
                        image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
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
                            mime_match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                            if mime_match:
                                mime_type, b64_data = mime_match.groups()
                                raw_bytes = base64.b64decode(b64_data)
                                opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                        elif image_url.startswith(("http://", "https://")):
                            fetched = fetch_remote_image(image_url)
                            if fetched:
                                img_bytes, mime_type = fetched
                                opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))

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


def _wire_image_part(raw_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
    return {"inlineData": {"mimeType": opt_mime,
                           "data": base64.b64encode(opt_bytes).decode("utf-8")}}


def openai_content_to_wire_parts(content: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if content is None:
        return parts

    def _add_text_with_inline_images(text: str):
        if not text:
            return
        img_parts, clean_text = _extract_markdown_images_to_parts(text)
        if clean_text:
            parts.append({"text": clean_text})
        for ip in img_parts:
            blob = getattr(ip, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                data = blob.data
                if isinstance(data, bytes):
                    data = base64.b64encode(data).decode("utf-8")
                parts.append({"inlineData": {"mimeType": getattr(blob, "mime_type", "image/jpeg"),
                                             "data": data}})

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
                            parts.append(_wire_image_part(base64.b64decode(b64_data), mime_type))
                        except Exception as e:
                            print(f"⚠️ [图片处理] 解析内联图片失败，已跳过：{e}")
                elif url.startswith("http"):
                    fetched = fetch_remote_image(url)
                    if fetched:
                        parts.append(_wire_image_part(*fetched))
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


def process_gemini_response_to_openai_dict(gemini_response_obj: Any, request_model_str: str) -> Dict[str, Any]:
    choices = []
    response_timestamp = int(time.time())
    base_id = f"chatcmpl-{response_timestamp}-{random.randint(1000,9999)}"

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
            if app_state.get_setting("safety_score", app_config.SAFETY_SCORE) and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                normal_content_str += _create_safety_ratings_html(candidate.safety_ratings)

            tool_index = 0
            ordinary_metadata = []
            order_descriptors = []
            message_signature = None
            message_signature_kind = None
            for part in parts_in_order:
                fc = getattr(part, "function_call", None)
                sig = _signature_bytes(part, fc)
                if fc is not None:
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
                text_value = getattr(part, "text", None)
                if text_value is None and getattr(part, "inline_data", None) is not None:
                    inline = part.inline_data
                    text_value = _convert_image_to_markdown(inline.data, inline.mime_type)
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


def convert_to_openai_format(gemini_response: Any, model: str) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(gemini_response, model)