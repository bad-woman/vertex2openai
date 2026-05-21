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

class OpenAIMessage(BaseModel):
    role: str
    content: str | list[ContentPartText | ContentPartImage | dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

class OpenAIRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # 兼容最新版 OpenAI 客户端
    top_p: float | None = 1.0
    top_k: int | None = None
    stream: bool | None = False
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: int | None = None
    response_logprobs: bool | None = None
    response_format: dict[str, Any] | None = None # 兼容强制 JSON 格式化输出
    n: int | None = None 
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    model_config = ConfigDict(extra='allow')