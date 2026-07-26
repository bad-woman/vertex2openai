from pydantic import BaseModel, ConfigDict
from typing import Any, Literal

# Define data models
class ImageUrl(BaseModel):
    url: str

class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str

def normalize_content_part(part: Any) -> Any:
    """把 message.content 里的单个 part 归一成普通 dict。

    F-1：`content` 的类型是 `list[ContentPartText | ContentPartImage | dict]`，
    pydantic 会优先把 `{"type": "text", ...}` 解析成 `ContentPartText` 实例，
    因此 `isinstance(p, dict)` 对标准 OpenAI 请求恒为 False——凡是靠这个判断
    筛选 part 的地方都会静默丢内容（分段 system、`--ar` 检测、markdown 图片抽取）。
    统一先过这个函数再按 dict 处理。
    """
    if isinstance(part, dict):
        return part
    if hasattr(part, "model_dump"):
        return part.model_dump()
    return part


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[ContentPartText | ContentPartImage | dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    # 默认 None = 客户端未显式传入（便于区分“省略”与“显式 1.0”，从而应用控制台默认值/按模型剥离）
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # 兼容最新版 OpenAI 客户端
    top_p: float | None = None
    top_k: int | None = None
    stream: bool | None = False
    # P1-3：OpenAI 允许 stop 是字符串或数组，旧定义只收数组，客户端发字符串会 422。
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    # P1-3：OpenAI 的 logprobs 是 bool、top_logprobs 才是整数；
    # 这里两种都收下，由 create_generation_config 统一规范成 Gemini 的形状。
    logprobs: bool | int | None = None
    top_logprobs: int | None = None
    response_logprobs: bool | None = None
    response_format: dict[str, Any] | None = None # 兼容强制 JSON 格式化输出
    n: int | None = None 
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # 流式用量开关（OpenAI: {"include_usage": true}）
    stream_options: dict[str, Any] | None = None

    model_config = ConfigDict(extra='allow')