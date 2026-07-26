# -*- coding: utf-8 -*-
"""P2-4：去重器应在能判定时立即放行，而不是固定攒满窗口。"""
from message_processing import PrefillDeduper, strip_prefill_overlap


def test_releases_immediately_when_clearly_different():
    """输出开头与预填充毫无关系时，第一个 chunk 就该放行。"""
    d = PrefillDeduper("从前有座山，" * 30)      # 长预填充 → 旧实现窗口很大
    out = d.feed("完全不相干的开头")
    assert out == "完全不相干的开头"
    assert d.done is True


def test_still_buffers_while_ambiguous():
    d = PrefillDeduper("从前有座山")
    assert d.feed("从前") == ""                 # 仍可能是复述，先攒着
    assert d.done is False


def test_strips_full_repetition():
    d = PrefillDeduper("从前有座山")
    out = d.feed("从前有座山，山里有座庙")
    assert out == "，山里有座庙"


def test_strips_tail_overlap():
    """注意 min_overlap 默认为 8：短于 8 个字符的重叠会被有意忽略，避免误伤正文。"""
    pre = "夜色渐深，她轻声说道，声音很轻："      # 结尾 11 字与输出开头重合
    assert strip_prefill_overlap(pre, "她轻声说道，声音很轻：好的") == "好的"


def test_short_overlap_is_intentionally_kept():
    assert strip_prefill_overlap("……他说：", "他说：你好") == "他说：你好"


def test_char_by_char_feeding():
    d = PrefillDeduper("从前有座山")
    out = "".join(d.feed(c) for c in "从前有座山，山里有座庙")
    assert out == "，山里有座庙"


def test_flush_returns_buffered_tail():
    d = PrefillDeduper("从前有座山")
    d.feed("从前")
    assert d.flush() == "从前"


def test_empty_prefill_is_passthrough():
    d = PrefillDeduper("")
    assert d.feed("abc") == "abc"
