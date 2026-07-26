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
from signature_store import signature_store, SKIP_VALIDATOR_SENTINEL

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

    # 在安全体积内的图片，原样发送，不损耗画质
    if len(image_data) <= max_size_bytes:
        return image_data, original_mime

    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # 抹平透明通道以防转成 JPEG 时报错
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

            # 二次压缩兜底（锁死在阈值以下）
            if len(opt_data) > max_size_bytes:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=max(40, quality - 15), optimize=True)
                opt_data = output.getvalue()

            return opt_data, "image/jpeg"
    except Exception as e:
        print(f"⚠️ [图片处理] 输入图片压缩失败，已回退为原图传输：{e}")
        return image_data, original_mime

SUPPORTED_ROLES = ["user", "model", "function"] 

# ============================================================
# 思考签名（thought signature）的出入站处理（P0-4）
#
# 官方约束（核对于 2026-07-26）：
#   - 函数调用强校验，Gemini 3 缺签名直接 400；纯文本不强校验但会掉质量。
#   - Gemini 3 的签名总在**第一个** function call part 上，必须原样回传。
#   - 并行调用必须按 FC1,FC2,FR1,FR2 顺序回传，交错会 400。
#   - 拿不到签名时可用 skip_thought_signature_validator 哨兵跳过校验（最后手段）。
#
# 这里的两个函数是出/入站的唯一入口。此前 api_helpers 与本文件各有一份复制粘贴的
# 实现，且已经出现细微不一致（一个用 response_id、一个用 base_id 拼 fallback id）。
# ============================================================

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


def build_tool_call_id(fc: Any, part: Any) -> str:
    """生成 OpenAI 侧的 tool_call_id，并把思考签名登记进旁路缓存。

    返回**短 id**（≤40 字符）。旧实现把 base64 签名拼进 id，动辄上千字符，
    很多前端会截断，回传时签名丢失 → 400。

    注意：这里只是不再**生成**旧的内嵌格式，`resolve_tool_call_signature` 仍然
    **解析**它——升级前发出的 id 可能还留在客户端的对话历史里。
    """
    real_id = getattr(fc, "id", None) or ""
    if not isinstance(real_id, str):
        real_id = str(real_id)
    if not real_id:
        real_id = "call_" + uuid.uuid4().hex[:16]   # 共 21 字符

    sig = _signature_bytes(part, fc)
    if sig:
        signature_store.put(real_id, sig)
    return real_id


def resolve_tool_call_signature(tool_call_id: str,
                                require_signature: bool = False) -> Tuple[str, Optional[bytes]]:
    """从客户端回传的 tool_call_id 还原 (真实 id, 思考签名)。

    三层降级：旁路缓存 → 旧的 `__thought__` 内嵌格式 → 官方哨兵值。
    require_signature=True（Gemini 3.x）时绝不返回 None，避免整个请求 400。
    """
    real_id = tool_call_id or ""
    sig: Optional[bytes] = None

    if LEGACY_THOUGHT_SEP in real_id:
        real_id, _, encoded = real_id.partition(LEGACY_THOUGHT_SEP)
        try:
            sig = base64.b64decode(encoded) or None
        except Exception:
            sig = None

    if sig is None:
        sig = signature_store.get(real_id)

    if sig is None and require_signature:
        sig = SKIP_VALIDATOR_SENTINEL
        print("⚠️ [工具调用] 未能恢复思考签名（可能被前端截断或服务已重启），"
              "已使用官方跳过校验哨兵。请求不会失败，但模型表现会下降。")

    return real_id, sig


def _requires_signature(model_name: str) -> bool:
    """该模型是否强校验思考签名（仅 Gemini 3.x 家族）。"""
    if not model_name:
        return False
    try:
        return mc.get_profile(model_name)["family"] == "g3"
    except Exception:
        return False


def extract_reasoning_by_tags(full_text: str, tag_name: str) -> Tuple[str, str]:
    if not tag_name or not isinstance(full_text, str):
        return "", full_text if isinstance(full_text, str) else ""
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    pattern = re.compile(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", re.DOTALL)
    reasoning_parts = pattern.findall(full_text)
    normal_text = pattern.sub("", full_text)
    reasoning_content = "".join(reasoning_parts)
    return reasoning_content.strip(), normal_text.strip()

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
        # 仅在**确实抽走了图片**时才压平空白：抠掉 data URL 会留下成片空格。
        # 旧实现无条件执行，导致所有纯文本消息的缩进/多空格（代码块、ASCII 图、
        # 酒馆预设的排版）都被悄悄改写。文本保真优先。
        remaining_text = re.sub(r"[ \t]+", " ", remaining_text).strip()

    return parts, remaining_text

def _coerce_tool_response(content: Any) -> Dict[str, Any]:
    """把 OpenAI 工具结果安全地转成 function_response 需要的对象。

    修复：旧实现用 `isinstance(str) and (...) or (...)` 的错误优先级，
    当 content 为 list 时会对其调用 .strip() 抛 AttributeError。
    """
    if content is None:
        return {"result": ""}
    if not isinstance(content, str):
        # content 可能是 OpenAI 的分段 list / dict
        try:
            return {"result": json.dumps(content, ensure_ascii=False)}
        except Exception:
            return {"result": str(content)}
    s = content.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
            # function_response 的 response 需要是对象；数组则包一层
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
    "[继续输出] 下面是你这条回复已经写好的开头，请从断点处无缝继续，"
    "不要重复开头内容，也不要添加任何前言、解释或标注："
)


# keep_turn 模式用的收尾指令。预填充留在 model 轮次里，这里只需要一句极短的推动，
# 越短越不干扰预设本身。
DEFAULT_KEEP_TURN_NUDGE = (
    "[继续] 从你上一条的断点处无缝往下写，不要重复已写内容，不要任何前言或解释。"
)


# 生图模型专用的续写指令。通用那句是"从断点处无缝往下写"，模型会照办——
# 继续写**文本**，于是吐出一段字符画而不是图片（实测）。生图必须明确要图。
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
    """把控制台配置的 system 指令 / 预填充注入到消息里。

    面向 RikkaHub 这类轻量前端：它们没有酒馆的预设系统，尤其**从不发送
    assistant 预填充**，因此在这些前端下预填充这个杠杆完全用不上，
    "预填充时压制原生思考"也永远不会触发。

    注入发生在 `apply_prefill_compat` **之前**，注入完的消息与"前端自己发了
    预填充"完全同形，下游四种兼容模式原样复用，不引入新分支。

    四条护栏（缺一个就会和现有功能打架）：
      1. 客户端已经发了预填充 → 不注入，否则两段预填充叠一起（酒馆场景）；
      2. 请求带 tools → 不注入，凭空多一个 model 轮次会打乱函数调用往返；
      3. 生图模型 → 默认不注入预填充，除非 allow_image_prefill=True。
         预填充对生图确有很强的引导力（实测：同一句"画一只猫"，预填充承诺
         "纯黑白钢笔线稿"就真的输出线稿，不加则是彩色写实照片），但角色扮演
         用的预填充落到生图请求上会让模型改吐文本，所以做成开关而非默认放行；
      4. 两个字段都留空 → 整个函数是空操作。

    返回 (新消息列表, 说明做了什么的日志行列表)。
    """
    notes: List[str] = []
    system_text = (system_text or "").strip()
    prefill_text = (prefill_text or "").strip()
    if not system_text and not prefill_text:
        return messages, notes

    new_msgs = list(messages or [])

    if system_text:
        # 追加在客户端 system 之后：越靠后越不容易被前面的内容淹没，
        # 也保证前端自己的系统提示仍然在场。
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


def apply_prefill_compat(
    messages: List[OpenAIMessage],
    mode: str = "smart",
    allow_model_last: bool = False,
    instruction_template: str = "",
) -> Tuple[List[OpenAIMessage], str, bool]:
    """
    预填充(prefill)兼容：Gemini 3.x 拒绝以 assistant/model 结尾的请求（400）。

    - mode="off"：不处理（可能 400）。
    - mode="minimal"：末尾非 user 时追加一个占位 user，仅保证不报错（不还原预填充）。
    - mode="smart"：
      * allow_model_last=True（模型允许以 model 轮次结尾，如 2.5 及更早）→ **原生透传**：
        消息保持原样发给上游，模型直接续写末尾轮次，最忠实；
      * 否则（3.x 等）→ 把末尾 assistant 预填充取出，转成末尾 user 的“续写指令”
        （模板可用 instruction_template 自定义，留空用内置默认）。
    - mode="keep_turn"（3.x 上比 smart 更贴合原意）：
      **保留** assistant 预填充作为 model 轮次，只在其后补一句极短的 user 推动语。
      3.x 拒绝的是“以 model 轮次**结尾**”，并不禁止 model 轮次出现在中间。
      smart 把预填充塞进 user 消息，模型于是把自己写的话当成“用户给的参考文本”，
      倾向另起一句；keep_turn 让预填充留在模型自己的声音里，续写是逐字接续的。
      预填充的主要用途是用预设自带的思维链顶掉原生思维链，此时预填充往往是
      `<thinking>` 这类**未闭合的开头**——它必须处在 model 轮次里，模型才会
      当作“自己已经写了一半”继续填充，而不是当成用户贴来的样例。
      2.5 系仍走原生透传，不受影响。
      以上模式都返回 prefill 文本，由上游把它拼回输出开头（配合去重）。

    返回 (处理后的消息列表, 需拼回输出开头的预填充文本, 是否检测到预填充)。
    第三项供“预填充时压制原生思考”等联动逻辑使用。
    与模型名无关：按请求形状 + 能力档案触发；新加模型 ID 自动生效。
    """
    if not messages or mode == "off":
        return messages, "", False

    # 找到最后一条“非空”消息
    idx = len(messages) - 1
    while idx >= 0 and _is_empty_message(messages[idx]):
        idx -= 1
    if idx < 0:
        return messages, "", False

    last = messages[idx]
    if last.role != "assistant" or getattr(last, "tool_calls", None):
        return messages, "", False  # 已是 user 结尾 / 末尾是工具调用 → 无需处理

    prefill = _message_text(last.content).strip()
    if not prefill:
        return messages, "", False

    if mode == "minimal":
        new_msgs = list(messages)
        new_msgs.append(OpenAIMessage(role="user", content="(请继续)"))
        return new_msgs, "", True

    # 模型支持 model 结尾 → 原生预填充透传（不改消息，模型直接续写）
    if allow_model_last:
        return messages, prefill, True

    if mode == "keep_turn":
        # 保留预填充所在的 assistant 轮次，只补一句极短 user 推动语。
        # 必须截到 idx+1：预填充后面可能还跟着空消息，带上它们会再次以非 user 结尾。
        new_msgs = list(messages[:idx + 1])
        nudge = (instruction_template or "").strip() or DEFAULT_KEEP_TURN_NUDGE
        new_msgs.append(OpenAIMessage(role="user", content=nudge))
        return new_msgs, prefill, True

    # smart：丢弃末尾预填充 assistant（及其后的空消息），转成续写指令
    new_msgs = list(messages[:idx])
    intro = (instruction_template or "").strip() or DEFAULT_PREFILL_INSTRUCTION
    instruction = intro + "\n\n" + prefill
    if new_msgs and new_msgs[-1].role == "user" and isinstance(new_msgs[-1].content, str):
        merged = new_msgs[-1].content + "\n\n" + instruction
        new_msgs[-1] = OpenAIMessage(role="user", content=merged)
    else:
        new_msgs.append(OpenAIMessage(role="user", content=instruction))
    return new_msgs, prefill, True


def strip_prefill_overlap(prefill: str, output: str, min_overlap: int = 8) -> str:
    """预填充去重：若模型无视指令、把预填充的结尾复述在输出开头，裁掉重叠部分。

    - 输出以整段预填充开头（完整复述）→ 裁掉整段；
    - 否则找 预填充结尾 与 输出开头 的最长重叠（≥ min_overlap 字符，避免误伤）。
    """
    if not prefill or not output:
        return output
    if output.startswith(prefill):
        return output[len(prefill):]
    kmax = min(len(prefill), len(output))
    for k in range(kmax, min_overlap - 1, -1):
        if prefill[-k:] == output[:k]:
            return output[k:]
    return output


class PrefillDeduper:
    """流式版预填充去重器。

    预填充文本由代理先行发给客户端；若模型复述了预填充开头，需要在
    流式输出的起始处裁掉重叠。做法：先攒下输出开头的一小段（窗口 =
    min(len(prefill)+32, 600) 字符），做一次去重判定后放行，之后的
    文本原样透传（不再增加任何延迟）。

    用法：out = deduper.feed(chunk_text)（可能返回空串表示还在攒）；
    流结束时调用 deduper.flush() 取回剩余文本。
    """

    def __init__(self, prefill: str, window_cap: int = 600):
        self.prefill = prefill or ""
        self.window = min(len(self.prefill) + 32, window_cap) if self.prefill else 0
        self.buffer = ""
        self.done = self.window == 0

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buffer += text
        # P2-4：只要已经能判定「不可能是预填充的重复」，立刻放行，不必攒满窗口。
        # 旧实现固定攒到 min(len(prefill)+32, 600) 字符才放行，首 token 延迟明显，
        # 与注释里的“零额外延迟”不符。
        if len(self.buffer) >= self.window or not self._still_ambiguous():
            return self._resolve()
        return ""

    def _still_ambiguous(self) -> bool:
        """当前缓冲是否还不足以判定去重量。

        strip_prefill_overlap 只看输出的前 len(prefill) 个字符，因此：
          - 缓冲长度已达 len(prefill) → 信息完备，重叠量已确定，立即放行；
          - 否则只有当缓冲仍是预填充的某个子串时，才可能延展成更长的重叠，需要继续等。
        """
        buf, pre = self.buffer, self.prefill
        if not buf or not pre:
            return False
        if len(buf) >= len(pre):
            return False
        return buf in pre

    def flush(self) -> str:
        if self.done:
            return ""
        return self._resolve()

    def _resolve(self) -> str:
        out, self.buffer, self.done = self.buffer, "", True
        return strip_prefill_overlap(self.prefill, out)


def create_gemini_prompt(messages: List[OpenAIMessage], model_name: str = "") -> List[types.Content]:
    """OpenAI 消息 → Gemini contents。

    model_name 用于判断是否需要对缺失的思考签名启用官方哨兵（仅 Gemini 3.x 强校验）。
    留空表示“未知模型”，此时不注入哨兵——调用方（express_sdk）总会传真实模型名。
    """
    print("🔄 [消息转换] 正在将 OpenAI 格式消息转换为 Gemini contents。")
    require_sig = _requires_signature(model_name)
    raw_gemini_messages = []
    for idx, message in enumerate(messages):
        role = message.role
        if role == "system":
            continue

        parts = []
        current_gemini_role = "" 

        if role == "tool":
            tool_call_id_str = message.tool_call_id or ""

            if not message.name:
                # 没有函数名无法构造规范的 function_response，降级为文本观测
                mock_text = f"[System Observation - Tool Result]:\n{message.content}"
                parts.append(types.Part.from_text(text=mock_text))
                current_gemini_role = "user"
            else:
                # 无论是否带 __thought__ 后缀，都构造规范的 function_response，
                # 让标准 OpenAI 客户端的工具往返也能正确工作（修复退化为纯文本的问题）。
                tool_output_data = _coerce_tool_response(message.content)

                # 函数**结果**不需要签名（官方只对 functionCall 强校验），因此不注入哨兵；
                # 但如果能取回签名就照常带上，保持与上游一致。
                real_tool_id, thought_sig_bytes = resolve_tool_call_signature(
                    tool_call_id_str, require_signature=False)

                func_resp_kwargs = {"name": message.name, "response": tool_output_data}
                if real_tool_id:
                    func_resp_kwargs["id"] = real_tool_id

                try:
                    part_kwargs = {"function_response": types.FunctionResponse(**func_resp_kwargs)}
                    if thought_sig_bytes:
                        part_kwargs["thought_signature"] = thought_sig_bytes
                    resp_part = types.Part(**part_kwargs)
                except Exception as e:
                    print(f"⚠️ [工具调用] 构造 FunctionResponse 失败，将回退为基础形式：{e}")
                    resp_part = types.Part.from_function_response(name=message.name, response=tool_output_data)

                parts.append(resp_part)
                current_gemini_role = "function"

        elif role == "assistant" and message.tool_calls:
            current_gemini_role = "model"
            for tool_call in message.tool_calls:
                function_call_data = tool_call.get("function", {})
                function_name = function_call_data.get("name", "unknown")
                arguments_str = function_call_data.get("arguments", "{}")
                tool_call_id_str = tool_call.get("id", "") or ""

                try:
                    parsed_arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else (arguments_str or {})
                except json.JSONDecodeError:
                    parsed_arguments = {}

                # 函数**调用**是强校验点：Gemini 3.x 缺签名直接 400，
                # 因此这里允许降级到官方哨兵，保证请求不会整体失败。
                real_tool_id, thought_sig_bytes = resolve_tool_call_signature(
                    tool_call_id_str, require_signature=require_sig)

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

                parts.append(fc_part)

            if message.content:
                if isinstance(message.content, str):
                    image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                    if clean_text: parts.append(types.Part.from_text(text=clean_text))
                    parts.extend(image_parts)
        else: 
            if message.content is None: continue
            
            current_gemini_role = role
            if current_gemini_role == "assistant": current_gemini_role = "model"
            if current_gemini_role not in SUPPORTED_ROLES:
                current_gemini_role = "user"

            if isinstance(message.content, str):
                image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                if clean_text: parts.append(types.Part.from_text(text=clean_text))
                parts.extend(image_parts)

            elif isinstance(message.content, list):
                for part_item in message.content:
                    # F-1：pydantic 会把标准 part 解析成 ContentPartText / ContentPartImage 实例，
                    # 归一成 dict 后只保留一条处理路径（原先 dict 与实例两套分支已出现行为分歧：
                    # 实例分支直接 from_text，跳过了 markdown 内联图片抽取）。
                    part_item = normalize_content_part(part_item)
                    if not isinstance(part_item, dict):
                        text_attr = getattr(part_item, "text", None)
                        if isinstance(text_attr, str):
                            parts.append(types.Part.from_text(text=text_attr))
                        continue

                    if part_item.get("type") == "text":
                        text_content = part_item.get("text", "\n")
                        image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
                        if clean_text: parts.append(types.Part.from_text(text=clean_text))
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
                            # 统一走加固后的 fetch_remote_image（SSRF 与体积防护都在那里，见 F-3）。
                            fetched = fetch_remote_image(image_url)
                            if fetched:
                                img_bytes, mime_type = fetched
                                opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))

        if not parts: continue
        raw_gemini_messages.append(types.Content(role=current_gemini_role, parts=parts))

    def _is_text_only(content: types.Content) -> bool:
        """该 Content 是否只含文本 part（没有 function_call / function_response / inline_data）。"""
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

    merged_messages = []
    for msg in raw_gemini_messages:
        if merged_messages and merged_messages[-1].role == msg.role:
            # 只有两侧都是纯文本时才插入 "\n\n" 分隔符。
            # 若两个连续 model 轮次都带 function call，插入文本 part 会打断
            # 官方要求的 FC 连续排列，可能触发 Malformed_Function_Call 或签名校验失败。
            if _is_text_only(merged_messages[-1]) and _is_text_only(msg):
                merged_messages[-1].parts.append(types.Part.from_text(text="\n\n"))
            merged_messages[-1].parts.extend(msg.parts)
        else:
            merged_messages.append(msg)

    if not merged_messages:
        merged_messages.append(types.Content(role="user", parts=[types.Part.from_text(text="继续")]))

    return merged_messages

# F-3：远程图片抓取的防护参数。
# 本服务常部署在能访问内网/云元数据服务的环境里，而图片 URL 完全由请求方控制，
# 不加限制等于把代理变成一个任意内网 GET 的跳板（SSRF）。
MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024   # 单张图上限，防止超大响应打爆内存
MAX_REMOTE_IMAGE_REDIRECTS = 3


def _is_blocked_host(host: str) -> bool:
    """目标是否指向内网/环回/链路本地等不该被代理访问的地址。

    云元数据服务（169.254.169.254）属于链路本地段，已被 is_link_local 覆盖。
    主机名先解析再判断，避免用 DNS 指向内网的域名绕过。
    """
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
            return True   # 解析不了就不放行
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
    """同步抓取远程图片，返回 (bytes, mime)。失败返回 None。

    调用方须保证不在事件循环线程里直接调用（Express/Cookie 两条通道都用
    asyncio.to_thread 包住整个消息转换）。

    F-3 防护：只允许 http/https、拒绝内网与链路本地地址（重定向后逐跳复查）、
    限制响应体积、校验 content-type。配置了 PROXY_URL 时出站本就经代理，
    到不了内网，此时跳过地址检查。
    """
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
                    # 逐跳复查：只校验首个 URL 的话，一个 302 就能把请求带进内网。
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
    """压缩后转成 batchGraphql 的 inlineData part（camelCase）。"""
    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
    return {"inlineData": {"mimeType": opt_mime,
                           "data": base64.b64encode(opt_bytes).decode("utf-8")}}


def openai_content_to_wire_parts(content: Any) -> List[Dict[str, Any]]:
    """OpenAI 的 message.content → batchGraphql 的 parts 列表（P1-2）。

    与 Express 通道行为对齐：
      - 解析正文里的 markdown data-URL 图片（多轮修图时上一张图不会再被当成巨大文本发出）
      - data: 与 http(s): 两种 image_url 都支持
      - 统一走 optimize_image_bytes 做输入图压缩（控制台开关对两条通道都生效）
    """
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


def _create_safety_ratings_html(safety_ratings: list) -> str:
    if not safety_ratings:
        return ""
    highest_rating = max(safety_ratings, key=lambda r: r.probability_score)
    highest_score = highest_rating.probability_score

    if highest_score <= 0.33: color = "#0f8"  
    elif highest_score <= 0.66: color = "yellow"
    else: color = "#bf555d"

    summary_category = highest_rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
    summary_probability = highest_rating.probability.name
    summary_score_str = f"{highest_rating.probability_score:.7f}" if highest_rating.probability_score is not None else "None"
    summary_severity_str = f"{highest_rating.severity_score:.8f}" if highest_rating.severity_score is not None else "None"
    summary_line = f"{summary_category}: {summary_probability} (Score: {summary_score_str}, Severity: {summary_severity_str})"

    ratings_list = []
    for rating in safety_ratings:
        category = rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
        probability = rating.probability.name
        score_str = f"{rating.probability_score:.7f}" if rating.probability_score is not None else "None"
        severity_str = f"{rating.severity_score:.8f}" if rating.severity_score is not None else "None"
        ratings_list.append(f"{category}: {probability} (Score: {score_str}, Severity: {severity_str})")
    all_ratings_str = "\n".join(ratings_list)

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

            if part_is_thought: reasoning_text_parts.append(part_text)
            elif part_text: normal_text_parts.append(part_text)
            
    elif candidate_part_text: normal_text_parts.append(candidate_part_text)
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
                if hasattr(raw_finish_reason, "name"): raw_gemini_finish_reason_str = raw_finish_reason.name.upper()
                else: raw_gemini_finish_reason_str = str(raw_finish_reason).upper()

                if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
                elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
                elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
                elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"
            
            function_call_detected = False
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call is not None: 
                        fc = part.function_call

                        # 统一走 build_tool_call_id：短 id + 签名进旁路缓存
                        tool_call_id = build_tool_call_id(fc, part)

                        if "tool_calls" not in message_payload:
                            message_payload["tool_calls"] = []
                        
                        message_payload["tool_calls"].append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args or {})
                            }
                        })
                        message_payload["content"] = None 
                        openai_finish_reason = "tool_calls" 
                        function_call_detected = True
            
            if not function_call_detected:
                reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
                if app_state.get_setting("safety_score", app_config.SAFETY_SCORE) and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    safety_html = _create_safety_ratings_html(candidate.safety_ratings)
                    if reasoning_str: reasoning_str += safety_html
                    else: normal_content_str += safety_html
                
                message_payload["content"] = normal_content_str
                if reasoning_str: message_payload["reasoning_content"] = reasoning_str
            
            choice_item = {"index": i, "message": message_payload, "finish_reason": openai_finish_reason}
            if hasattr(candidate, "logprobs") and candidate.logprobs is not None: choice_item["logprobs"] = candidate.logprobs
            choices.append(choice_item)
            
    elif hasattr(gemini_response_obj, "text") and gemini_response_obj.text is not None:
         content_str = gemini_response_obj.text or ""
         choices.append({"index": 0, "message": {"role": "assistant", "content": content_str}, "finish_reason": "stop"})
    else: 
         choices.append({"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"})

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(gemini_response_obj, "usage_metadata"):
        um = gemini_response_obj.usage_metadata
        if hasattr(um, "prompt_token_count"): usage_data["prompt_tokens"] = um.prompt_token_count
        if hasattr(um, "candidates_token_count"):
            usage_data["completion_tokens"] = um.candidates_token_count
            if hasattr(um, "total_token_count"): usage_data["total_tokens"] = um.total_token_count
            else: usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
        elif hasattr(um, "total_token_count"): 
             usage_data["total_tokens"] = um.total_token_count
             if usage_data["prompt_tokens"] > 0 and usage_data["total_tokens"] > usage_data["prompt_tokens"]:
                 usage_data["completion_tokens"] = usage_data["total_tokens"] - usage_data["prompt_tokens"]
        else: usage_data["total_tokens"] = usage_data["prompt_tokens"] 

    return {
        "id": base_id, "object": "chat.completion", "created": response_timestamp,
        "model": request_model_str, "choices": choices,
        "usage": usage_data
    }

def convert_to_openai_format(gemini_response: Any, model: str) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(gemini_response, model)