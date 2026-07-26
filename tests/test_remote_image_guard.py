"""F-3：远程图片抓取的 SSRF / 体积 / 类型防护。

图片 URL 完全由请求方控制，而本服务常部署在能访问内网与云元数据服务的环境里。
实机验证时确认过：加固前代理会老老实实去抓 http://127.0.0.1:8777/ 并把内容送给模型。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config as app_config  # noqa: E402
import message_processing as mp  # noqa: E402


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """默认按“无出站代理”测试——此时地址检查才生效。"""
    monkeypatch.setattr(app_config, "PROXY_URL", None, raising=False)


@pytest.mark.parametrize("host", [
    "127.0.0.1",          # 环回
    "localhost",
    "169.254.169.254",    # 云元数据服务
    "10.0.0.5",           # 私网
    "192.168.1.1",
    "172.16.0.1",
    "0.0.0.0",
])
def test_internal_addresses_are_blocked(host):
    assert mp._is_blocked_host(host) is True


def test_public_address_is_allowed():
    assert mp._is_blocked_host("8.8.8.8") is False


def test_unresolvable_host_is_blocked():
    assert mp._is_blocked_host("no-such-host.invalid") is True


def test_empty_host_is_blocked():
    assert mp._is_blocked_host("") is True


def test_fetch_rejects_loopback_url(capsys):
    assert mp.fetch_remote_image("http://127.0.0.1:8777/x.png") is None
    assert "内网" in capsys.readouterr().out


def test_fetch_rejects_non_http_scheme(capsys):
    assert mp.fetch_remote_image("file:///etc/passwd") is None
    assert "协议" in capsys.readouterr().out


def test_fetch_rejects_metadata_service():
    assert mp.fetch_remote_image("http://169.254.169.254/latest/meta-data/") is None


class _Resp:
    is_redirect = False

    def __init__(self, content=b"", headers=None, status=200):
        self.content = content
        self.headers = headers or {"content-type": "image/png"}
        self._status = status
        self.next_request = None

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class _Client:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url):
        return self._resp


def test_oversized_response_is_rejected(monkeypatch, capsys):
    big = b"x" * (mp.MAX_REMOTE_IMAGE_BYTES + 1)
    monkeypatch.setattr(mp.httpx, "Client", lambda **kw: _Client(_Resp(big)))
    monkeypatch.setattr(mp, "_is_blocked_host", lambda h: False)
    assert mp.fetch_remote_image("https://example.com/big.png") is None
    assert "上限" in capsys.readouterr().out


def test_non_image_content_type_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(mp.httpx, "Client",
                        lambda **kw: _Client(_Resp(b"<html>", {"content-type": "text/html"})))
    monkeypatch.setattr(mp, "_is_blocked_host", lambda h: False)
    assert mp.fetch_remote_image("https://example.com/page") is None
    assert "不是图片" in capsys.readouterr().out


def test_ordinary_image_still_fetched(monkeypatch):
    monkeypatch.setattr(mp.httpx, "Client",
                        lambda **kw: _Client(_Resp(b"PNGDATA", {"content-type": "image/png"})))
    monkeypatch.setattr(mp, "_is_blocked_host", lambda h: False)
    assert mp.fetch_remote_image("https://example.com/ok.png") == (b"PNGDATA", "image/png")


def test_redirect_into_internal_address_is_blocked(monkeypatch, capsys):
    """只校验首个 URL 的话，一个 302 就能把请求带进内网。"""
    class _RedirResp(_Resp):
        is_redirect = True

    class _Req:
        url = "http://169.254.169.254/latest/meta-data/"

    redir = _RedirResp()
    redir.next_request = _Req()
    monkeypatch.setattr(mp.httpx, "Client", lambda **kw: _Client(redir))
    assert mp.fetch_remote_image("https://example.com/redir.png") is None
    assert "重定向" in capsys.readouterr().out


def test_host_check_skipped_when_outbound_proxy_configured(monkeypatch):
    """配了 PROXY_URL 时出站本就经代理、到不了内网，不该误伤。"""
    monkeypatch.setattr(app_config, "PROXY_URL", "http://proxy:8080", raising=False)
    monkeypatch.setattr(mp.httpx, "Client",
                        lambda **kw: _Client(_Resp(b"IMG", {"content-type": "image/jpeg"})))
    assert mp.fetch_remote_image("http://10.0.0.5/x.jpg") == (b"IMG", "image/jpeg")
