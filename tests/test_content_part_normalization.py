"""F-1：pydantic 会把标准 OpenAI content part 解析成 ContentPartText / ContentPartImage 实例。

凡是用 `isinstance(p, dict)` 筛 part 的地方，对真实请求都恒为 False，内容被静默丢弃。
这些用例全部用 **OpenAIMessage 构造**（而不是手写 dict），才能复现真实的 pydantic 行为——
用 dict 直接调内部函数是测不出这个 bug 的。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from models import OpenAIMessage, OpenAIRequest, ContentPartText, normalize_content_part  # noqa: E402
import model_capabilities as mc  # noqa: E402
from upstreams.cookie_proxy import _convert_messages_to_contents  # noqa: E402


def test_pydantic_really_coerces_dict_to_model():
    """守住前提：一旦 pydantic 不再强转，本文件其余用例就失去意义。"""
    msg = OpenAIMessage(role="system", content=[{"type": "text", "text": "A"}])
    assert isinstance(msg.content[0], ContentPartText)
    assert not isinstance(msg.content[0], dict)


def test_normalize_handles_model_dict_and_other():
    assert normalize_content_part({"type": "text", "text": "x"}) == {"type": "text", "text": "x"}
    normalized = normalize_content_part(ContentPartText(type="text", text="y"))
    assert isinstance(normalized, dict) and normalized["text"] == "y"
    assert normalize_content_part("plain") == "plain"


def test_multipart_system_is_not_dropped():
    msgs = [OpenAIMessage(role="system",
                          content=[{"type": "text", "text": "A"},
                                   {"type": "text", "text": "B"}]),
            OpenAIMessage(role="user", content="hi")]
    _, system_text = _convert_messages_to_contents(msgs)
    assert system_text is not None, "分段 system 被整段丢弃了"
    assert "A" in system_text and "B" in system_text


def test_multipart_system_mixed_with_string_system():
    msgs = [OpenAIMessage(role="system", content="S1"),
            OpenAIMessage(role="system", content=[{"type": "text", "text": "S2"}]),
            OpenAIMessage(role="user", content="hi")]
    _, system_text = _convert_messages_to_contents(msgs)
    assert "S1" in system_text and "S2" in system_text


def test_prompt_aspect_ratio_found_in_list_content():
    req = OpenAIRequest(model="gemini-3.1-flash-image",
                        messages=[OpenAIMessage(role="user",
                                                content=[{"type": "text", "text": "一只猫 --ar 16:9"}])])
    assert mc._prompt_aspect_ratio(req) == "16:9"


def test_prompt_aspect_ratio_still_works_for_string_content():
    req = OpenAIRequest(model="gemini-3.1-flash-image",
                        messages=[OpenAIMessage(role="user", content="一只猫 --ar 4:3")])
    assert mc._prompt_aspect_ratio(req) == "4:3"


def test_list_content_text_goes_through_markdown_image_extraction():
    """实例分支原先直接 from_text，跳过了 markdown 内联图片抽取。"""
    from message_processing import create_gemini_prompt

    tiny_png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
                "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    msgs = [OpenAIMessage(role="user",
                          content=[{"type": "text",
                                    "text": f"看图 ![x](data:image/png;base64,{tiny_png})"}])]
    contents = create_gemini_prompt(msgs)
    kinds = [p for c in contents for p in (c.parts or [])]
    assert any(getattr(p, "inline_data", None) is not None for p in kinds), \
        "markdown 内联图片没有被抽成图片 part"
