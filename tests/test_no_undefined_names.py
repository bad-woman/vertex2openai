"""静态守卫：全项目不得存在未定义的名字。

起因是一个真实事故：给 `express_sdk.py` 加了一处 `app_config.DEFAULT_SETTINGS[...]`，
但该模块并没有 `import config as app_config`。`compileall` 只查语法、不查名字解析，
其余测试也没走到那条聊天路径，于是问题一路溜到线上，表现为每次请求都
`HTTP 500 name 'app_config' is not defined`。

pyflakes 能在不执行代码的前提下解析出这类未定义名，正好补上这个缺口。
"""
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"


def _pyflakes_available() -> bool:
    try:
        import pyflakes  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _pyflakes_available(), reason="需要 pyflakes：pip install pyflakes")
def test_no_undefined_names_in_app():
    files = [str(p) for p in sorted(APP.rglob("*.py")) if "__pycache__" not in str(p)]
    assert files, "没找到任何待检查的源文件"

    proc = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                          capture_output=True, text=True)
    # 只挑"未定义名"这一类——其余风格问题不在本用例的职责范围内。
    offenders = [line for line in proc.stdout.splitlines()
                 if "undefined name" in line and "unable to detect" not in line]
    assert not offenders, "存在未定义的名字：\n" + "\n".join(offenders)


@pytest.mark.skipif(not _pyflakes_available(), reason="需要 pyflakes：pip install pyflakes")
def test_modules_using_app_config_actually_import_it():
    """针对本次事故的直接断言，即使将来没装 pyflakes 也想保住这条不变量。"""
    for path in sorted(APP.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        if "app_config." in text:
            assert "import config as app_config" in text, \
                f"{path.name} 用了 app_config 却没有导入它"
