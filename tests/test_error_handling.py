# -*- coding: utf-8 -*-
"""上游错误提取：把 SDK 的 404/403/400 如实转成状态码+简明消息。"""
from api_helpers import extract_upstream_error


class _ClientError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def test_extract_404_from_code_attr():
    e = _ClientError(404, "{'error': {'code': 404, 'message': 'Publisher model X was not found or your project does not have access to it.'}}")
    code, msg = extract_upstream_error(e)
    assert code == 404
    assert "was not found" in msg
    assert "'code'" not in msg  # 已提取干净 message


def test_extract_403_permission():
    e = _ClientError(403, "Permission denied on resource")
    code, msg = extract_upstream_error(e)
    assert code == 403


def test_extract_from_text_only():
    code, _ = extract_upstream_error(Exception("Model not found: gemini-x"))
    assert code == 404
    code2, _ = extract_upstream_error(Exception("INVALID_ARGUMENT: bad request"))
    assert code2 == 400
    code3, _ = extract_upstream_error(Exception("some random failure"))
    assert code3 == 500


def test_extract_status_code_attr():
    class _E(Exception):
        status_code = 429
    code, _ = extract_upstream_error(_E("rate limited"))
    assert code == 429
