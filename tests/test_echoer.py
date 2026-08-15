"""Echo-er 测试：回复整理 / 流畅 / 标点规范 / 双语。"""
import re

from sng.core.pool import AnalysisResult, GenerationRequest, VerificationReport
from sng.echoer import Echoer, normalize_punctuation


def _req(results, passed=True):
    return GenerationRequest("t1", results,
                             VerificationReport("t1", passed, [] if passed else ["背包超容量"]))


def test_normalize_punctuation_dedup():
    assert normalize_punctuation("你好。。你好吗？？") == "你好。你好吗？"
    assert normalize_punctuation(" 前面有空格 。") == "前面有空格。"
    assert normalize_punctuation("没有句号") == "没有句号。"


def test_reply_is_organized_into_sections():
    e = Echoer()
    req = _req([
        AnalysisResult("t1", "memory", "你位于客厅，背包里有：钥匙。", 0.99),
        AnalysisResult("t1", "domain", "大门锁着，你需要钥匙。", 0.98),
    ])
    text = e.format(req)
    assert "【状态】" in text
    assert "【分析】" in text
    assert "【结果】" in text
    # 分节有先后顺序
    assert text.index("【状态】") < text.index("【分析】") < text.index("【结果】")


def test_reply_is_fluent_and_punctuated():
    e = Echoer()
    req = _req([
        AnalysisResult("t1", "memory", "背包里有钥匙和苹果", 0.99),
        AnalysisResult("t1", "domain", "大门锁着需要钥匙", 0.98),
    ])
    text = e.format(req)
    for line in text.split("\n"):
        if not line:
            continue
        # 每行都以中文句末标点结束
        assert re.search(r"[。！？]$", line), f"标点不规范: {line!r}"
        # 每行至少包含一个句末标点，且不以标点开头
        assert not re.match(r"^[，。！？、]", line)
    # 流畅性：无空段、无碎片
    assert len(text.split("\n")) >= 3


def test_verification_failure_reported():
    e = Echoer()
    req = _req([AnalysisResult("t1", "memory", "背包超容量。", 0.9)], passed=False)
    text = e.format(req)
    assert "未通过验证" in text
    assert "背包超容量" in text


def test_bilingual_output():
    e = Echoer()
    req = _req([AnalysisResult("t1", "memory", "你位于客厅。", 0.99)])
    zh = e.answer(req, bilingual=False)
    bi = e.answer(req, bilingual=True)
    assert "【状态】" in zh
    assert "[Status]" in bi
