"""输出附加安全分：渲染、崩溃防护、两条通道的一致性。

用户反馈"开了没任何区别"，查下来是三个独立问题叠加：
  1. 有思考时安全分被塞进 reasoning_content，埋在前端折叠的思考区里；
  2. 流式路径每个 chunk 都带 safetyRatings，逐块追加会重复，实际未生效；
  3. Cookie 通道下发 OFF——分类器整个关掉，上游根本不回传评分，且没有实现。
另外发现一个崩溃：部分分类只给 probability 不给 probability_score，
max(key=probability_score) 拿 None 和 float 比较 → TypeError → 整个请求 500。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import message_processing as mp  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app"


class _SdkRating:
    """模拟 google-genai 的 SafetyRating（枚举属性 + 蛇形字段）。"""
    def __init__(self, cat, prob, ps=None, ss=None):
        self.category = type("C", (), {"name": cat})()
        self.probability = type("P", (), {"name": prob})()
        self.probability_score = ps
        self.severity_score = ss


def test_missing_probability_score_does_not_crash():
    """实测 JAILBREAK 常常没有 probability_score，此前会 500。"""
    html = mp._create_safety_ratings_html([
        _SdkRating("HARM_CATEGORY_HATE_SPEECH", "NEGLIGIBLE", 0.1, 0.2),
        _SdkRating("HARM_CATEGORY_JAILBREAK", "LOW"),          # 两个分数都缺
    ])
    assert "<details" in html and "Jailbreak" in html


def test_all_scores_missing_still_renders():
    html = mp._create_safety_ratings_html([_SdkRating("HARM_CATEGORY_HARASSMENT", "NEGLIGIBLE")])
    assert "<details" in html and "None" in html


def test_highest_rating_ignores_scoreless_entries():
    """缺分数的不能被选成"最高分"，否则摘要行会是一条没有分数的记录。"""
    html = mp._create_safety_ratings_html([
        _SdkRating("HARM_CATEGORY_JAILBREAK", "LOW"),
        _SdkRating("HARM_CATEGORY_HATE_SPEECH", "HIGH", 0.9, 0.8),
    ])
    summary = html.split("<summary", 1)[1].split("</summary>", 1)[0]
    assert "Hate Speech" in summary


def test_renderer_accepts_camelcase_dicts_from_cookie_channel():
    """Cookie 通道拿到的是 batchGraphql 的 camelCase 字典，必须共用同一个渲染器。"""
    html = mp._create_safety_ratings_html([
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE",
         "probabilityScore": 0.01, "severityScore": 0.02},
        {"category": "HARM_CATEGORY_JAILBREAK", "probability": "LOW"},
    ])
    assert "<details" in html and "Hate Speech" in html and "Jailbreak" in html


def test_empty_ratings_render_nothing():
    assert mp._create_safety_ratings_html([]) == ""
    assert mp._create_safety_ratings_html(None) == ""


def test_score_thresholds_pick_a_color():
    low = mp._create_safety_ratings_html([_SdkRating("HARM_CATEGORY_HARASSMENT", "NEGLIGIBLE", 0.1)])
    high = mp._create_safety_ratings_html([_SdkRating("HARM_CATEGORY_HARASSMENT", "HIGH", 0.9)])
    assert "#0f8" in low and "#bf555d" in high


# ---- 位置与通道一致性（源码级断言：这些路径没法在离线用例里驱动） ----

def test_safety_block_goes_to_content_not_reasoning():
    """附在正文里才看得见。旧实现"有思考就进思考字段"，
    在剥离思考的配置下整块消失，开了开关也毫无变化。"""
    for f in ("message_processing.py", "api_helpers.py"):
        src = (APP / f).read_text(encoding="utf-8")
        for line in src.split("\n"):
            if "_create_safety_ratings_html" in line and "def " not in line and "import" not in line:
                assert "reasoning" not in line, f"{f}: 安全分仍在往思考字段里塞 —— {line.strip()}"


def test_streaming_only_appends_on_the_final_chunk():
    """上游每个 chunk 都带 safetyRatings，必须只在最后一块附加，
    否则同一份评分会被重复插进正文里。"""
    lines = (APP / "api_helpers.py").read_text(encoding="utf-8").split("\n")
    uses = [i for i, ln in enumerate(lines)
            if "_safety_score_enabled()" in ln and not ln.lstrip().startswith("def ")]
    assert uses, "流式路径找不到安全分开关的判断"
    for i in uses:
        window = "\n".join(lines[max(0, i - 2): i + 2])
        assert "openai_finish_reason" in window, \
            f"第 {i + 1} 行的安全分未用 finish_reason 限定在最后一块：{window!r}"


def test_cookie_channel_requests_scores_only_when_enabled():
    """OFF 会让上游不返回评分，所以开关打开时必须改用 BLOCK_NONE；
    关闭时保持 OFF，不改变既有的安全行为。"""
    src = (APP / "upstreams" / "cookie_proxy.py").read_text(encoding="utf-8")
    assert '_threshold = "BLOCK_NONE" if _want_scores else "OFF"' in src
    assert '"threshold": "OFF"' not in src, "仍有硬编码 OFF 的分类，开关对它无效"


def test_cookie_channel_actually_emits_the_block():
    src = (APP / "upstreams" / "cookie_proxy.py").read_text(encoding="utf-8")
    assert 'yield ("safety", ratings)' in src, "Cookie 通道未透出 safetyRatings"
    assert "_safety_html_if_enabled" in src
    assert "full_text + safety_html" in src, "Cookie 非流式未把安全分附到正文"
