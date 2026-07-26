# -*- coding: utf-8 -*-
"""pytest 基础设施：
- 把 app/ 加入 sys.path（项目模块用顶层导入，如 `import config`）
- 每个测试自动切到独立临时目录（runtime_state 的 web_state.json 按 CWD 落盘）
- 每个测试自动重置 app_state 的内存态，避免测试间串扰
"""
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from runtime_state import app_state  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """状态隔离：状态文件指到临时目录 + 清空内存态。

    P1-4 之后 AppState 改成内存优先（只在启动/reload 时读盘），
    因此这里直接重置 `_state`，并把 STATE_FILE 指到临时目录，
    避免测试互相污染、也避免误写仓库根目录。
    """
    import runtime_state

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_state, "STATE_FILE", str(tmp_path / "web_state.json"))
    with app_state._lock:
        app_state._state = {"use_web_proxy": False}
    yield
    with app_state._lock:
        app_state._state = {"use_web_proxy": False}


@pytest.fixture(autouse=True)
def _clear_signature_store():
    """清空思考签名旁路缓存，避免用例之间串扰（P0-4）。"""
    from signature_store import signature_store
    signature_store.clear()
    yield
    signature_store.clear()


class FakeReq:
    """轻量请求对象：模拟 OpenAIRequest 的属性访问（含 model_extra）。"""

    def __init__(self, **kw):
        self.model = kw.pop("model", "gemini-2.5-flash")
        self.messages = kw.pop("messages", [])
        self.model_extra = kw.pop("model_extra", {})
        self.reasoning_effort = kw.pop("reasoning_effort", None)
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture
def fake_req():
    return FakeReq
