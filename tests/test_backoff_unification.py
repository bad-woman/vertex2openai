"""F-5：退避时长必须来自控制台的 retry_backoff_seconds。

整改包声称"重试语义已统一"，但只统一了**次数**：Express 真流式与非流式仍在用
硬编码的 `2 ** (attempt % 4)` / `min(8, 2 ** (attempt % 4))`，控制台把退避设成
多少都没有效果。这里用源码级断言锁住——退避写在深层重试循环里，
用例无法在不打真实上游的前提下驱动它。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

APP = Path(__file__).resolve().parent.parent / "app"

import config as app_config  # noqa: E402
from api_helpers import get_retry_settings  # noqa: E402
from runtime_state import app_state  # noqa: E402


def test_no_hardcoded_exponential_backoff_left():
    """全项目不得再出现 2 ** (attempt % 4) 这类无视配置的退避。"""
    offenders = []
    for path in APP.rglob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]        # 注释里可以提旧写法，代码里不行
            if re.search(r"2\s*\*\*\s*\(?\s*(attempt|wave_index)", code):
                offenders.append(f"{path.name}:{line_no}")
    assert not offenders, f"仍有硬编码指数退避：{offenders}"


def test_retry_loops_use_configured_backoff():
    """三条通道的等待时长都应源自 get_retry_settings() 的第二个返回值。"""
    text = (APP / "api_helpers.py").read_text(encoding="utf-8")
    assert text.count("wait_time = backoff_sec") >= 2, "Express 真流式/非流式未改用配置退避"
    # 每处使用前都必须真正解包出 backoff（而不是 `_backoff` 丢掉）
    assert "max_retries, _backoff = get_retry_settings()" not in text


def test_backoff_is_clamped_and_falls_back(monkeypatch):
    monkeypatch.setattr(app_state, "get_setting", lambda k, d=None: {"retry_max": 3,
                                                                    "retry_backoff_seconds": 999}.get(k, d))
    assert get_retry_settings() == (3, 120.0)

    monkeypatch.setattr(app_state, "get_setting", lambda k, d=None: {"retry_max": 3,
                                                                    "retry_backoff_seconds": -5}.get(k, d))
    assert get_retry_settings() == (3, 0.0)

    monkeypatch.setattr(app_state, "get_setting", lambda k, d=None: {"retry_max": "x",
                                                                    "retry_backoff_seconds": "y"}.get(k, d))
    assert get_retry_settings() == (app_config.DEFAULT_SETTINGS["retry_max"],
                                    float(app_config.DEFAULT_SETTINGS["retry_backoff_seconds"]))


def test_tenacity_fully_removed():
    """假流式改成手写退避后，tenacity 应从代码与依赖里一并消失。"""
    assert "tenacity" not in (APP / "requirements.txt").read_text(encoding="utf-8")
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import tenacity", "from tenacity")):
                raise AssertionError(f"{path.name} 仍在导入 tenacity")
