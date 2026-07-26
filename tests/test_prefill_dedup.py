# -*- coding: utf-8 -*-
"""预填充重叠去重：strip_prefill_overlap 与流式 PrefillDeduper。"""
from message_processing import strip_prefill_overlap, PrefillDeduper


def test_strip_full_echo():
    assert strip_prefill_overlap("从前有座山，", "从前有座山，山里有座庙。") == "山里有座庙。"


def test_strip_partial_tail_overlap():
    pf = "<thinking>\n让我梳理一下剧情走向："
    out = "让我梳理一下剧情走向：主角推开了门。"
    assert strip_prefill_overlap(pf, out) == "主角推开了门。"


def test_strip_no_overlap_untouched():
    assert strip_prefill_overlap("ABCDEFGH", "12345678") == "12345678"


def test_strip_short_overlap_ignored():
    # 少于 min_overlap(8) 字符的重叠不裁，避免误伤正常开头
    assert strip_prefill_overlap("……结尾是叹号!", "!正常正文") == "!正常正文"


def test_strip_empty_inputs():
    assert strip_prefill_overlap("", "xyz") == "xyz"
    assert strip_prefill_overlap("abc", "") == ""


def test_deduper_stream_echo_removed():
    d = PrefillDeduper("从前有座山，这是一个很长很长的预填充开头哦")
    fed = []
    for piece in ["从前有座山，这是一", "个很长很长的预填充开头哦", "然后故事继续发展。"]:
        fed.append(d.feed(piece))
    fed.append(d.flush())
    assert "".join(fed) == "然后故事继续发展。"


def test_deduper_no_prefill_passthrough():
    d = PrefillDeduper("")
    assert d.feed("abc") == "abc"
    assert d.flush() == ""


def test_deduper_short_reply_flush():
    """短回复且与预填充毫无重叠时立即放行（P2-4 之前会一直攒到窗口上限）。

    不变量：feed + flush 的总输出恒等于原文，只是放行时机提前了。
    """
    d = PrefillDeduper("这是一个非常长的预填充" * 10)
    out = d.feed("短回复")
    assert out + d.flush() == "短回复"
    assert out == "短回复", "与预填充无重叠时不应再无谓地攒着"


def test_deduper_buffers_only_while_overlap_possible():
    """仍可能是预填充复述时才继续攒。"""
    d = PrefillDeduper("这是一个非常长的预填充")
    assert d.feed("这是一个") == ""
    assert d.flush() == "这是一个"


def test_deduper_no_overlap_passthrough():
    d = PrefillDeduper("预填充文本预填充文本")
    out1 = d.feed("完全不同的开头，" * 50)  # 超过窗口，立即判定
    assert out1.startswith("完全不同的开头，")
    assert d.feed("后续") == "后续"
