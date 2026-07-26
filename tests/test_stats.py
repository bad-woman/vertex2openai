# -*- coding: utf-8 -*-
"""大盘统计：Express 通道 token 行解析计入；Cookie 通道成功数单独计入且不打 💰。"""
import logger
import upstreams.cookie_proxy as cp


def test_add_success_direct():
    s = logger.ProxyStats()
    s.add_success()
    s.add_success()
    assert s.get_json_stats()["success"] == 2
    assert s.get_json_stats()["prompt_tokens"] == 0


def test_usage_recorded_directly_not_by_log_regex():
    """P1-5：token 统计改为在产生 usage 的地方直接记账。

    旧实现用正则从中文日志文本里反解，改一句文案就静默失效。
    """
    from api_helpers import _record_usage

    class FakeUM:
        prompt_token_count = 111
        candidates_token_count = 222
        total_token_count = 333

    class FakeResp:
        usage_metadata = FakeUM()

    before = logger.stats.get_json_stats()
    usage = _record_usage(FakeResp())
    after = logger.stats.get_json_stats()
    assert usage == {"prompt_tokens": 111, "completion_tokens": 222, "total_tokens": 333}
    assert after["prompt_tokens"] - before["prompt_tokens"] == 111
    assert after["completion_tokens"] - before["completion_tokens"] == 222
    assert after["success"] - before["success"] == 1


def test_printing_token_line_does_not_double_count():
    """关键不变量：日志钩子不得再解析 💰 行，否则每次统计都会翻倍。"""
    before = logger.stats.get_json_stats()
    print("💰 [算力消耗统计] 提示词: 111 | 思考与生成: 222 | 总计: 333 Tokens")
    after = logger.stats.get_json_stats()
    assert after["prompt_tokens"] == before["prompt_tokens"]
    assert after["success"] == before["success"]


def test_no_usage_metadata_is_noop():
    from api_helpers import _record_usage

    class Empty:
        usage_metadata = None

    before = logger.stats.get_json_stats()
    assert _record_usage(Empty()) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert logger.stats.get_json_stats()["success"] == before["success"]


def test_cookie_map_usage_no_token_line(capsys):
    before = logger.stats.get_json_stats()
    usage = cp._map_usage({"promptTokenCount": 9, "candidatesTokenCount": 4})
    captured = capsys.readouterr()
    assert "💰" not in captured.out  # 不再打印统计行 → 不计入大盘 token
    assert usage == {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    after = logger.stats.get_json_stats()
    assert after["prompt_tokens"] == before["prompt_tokens"]
    assert after["success"] == before["success"]


def test_cookie_map_usage_handles_none():
    assert cp._map_usage(None) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert cp._map_usage({}) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
