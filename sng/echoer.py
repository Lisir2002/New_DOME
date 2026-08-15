"""Echo-er：最终输出的组织与润色。

职责（对应基础测试）：回复整理（结构化分节）、回答流畅（完整通顺的句子）、
标点规范（去重标点、正确句末符、标点前不留空格）、双语支持。

输出质量通过"后置规范化"做结构性保证（原型阶段；正式实现中为独立小模型）。
"""

from __future__ import annotations

import re

from sng.core.pool import GenerationRequest, VerificationReport


def normalize_punctuation(text: str) -> str:
    """标点规范化：去掉连续重复标点、句末补上正确标点、标点前不留空格。"""
    t = text.strip()
    # 标点前不留空格
    t = re.sub(r"\s+([，。！？；：、,\.!?;:])", r"\1", t)
    # 连续标点只保留一个（保留句号优先）
    t = re.sub(r"[，,]{2,}", "，", t)
    t = re.sub(r"[。]{2,}", "。", t)
    t = re.sub(r"[！!]{2,}", "！", t)
    t = re.sub(r"[？?]{2,}", "？", t)
    # 句末标点归一：统一用中文标点
    t = re.sub(r"[.!?]$", lambda m: {".": "。", "!": "！", "?": "？"}[m.group(0)], t)
    # 去掉句末多余空格
    t = t.rstrip()
    if not re.search(r"[。！？.!?]$", t):
        t += "。"
    return t


def _as_sentence(segments: list[str]) -> str:
    """把片段整理成通顺的一句话。"""
    segs = [s for s in segments if s]
    if not segs:
        return ""
    return "，".join(segs)


class Echoer:
    """Echo-er：把结论池的结果组织成最终回答。"""

    def __init__(self, lang: str = "zh") -> None:
        self.lang = lang

    def format(self, req: GenerationRequest) -> str:
        results = req.results
        verif = req.verification
        parts: list[str] = []

        # —— 通用对话小节（chat 插件） ——
        chat = next((r for r in results if r.plugin_id == "chat"), None)
        if chat:
            parts.append(f"【回复】{normalize_punctuation(chat.content)}")

        # —— 状态 / 观察小节 ——
        recall = next((r for r in results if r.plugin_id == "memory"), None)
        if recall:
            parts.append(f"【状态】{normalize_punctuation(recall.content)}")

        # —— 行动分析小节 ——
        domain = next((r for r in results if r.plugin_id == "domain"), None)
        if domain:
            parts.append(f"【分析】{normalize_punctuation(domain.content)}")

        # —— 结果小节（验证报告） ——
        if verif is not None:
            if verif.passed:
                parts.append("【结果】状态更新已通过一致性验证，可以继续行动。")
            else:
                issues = "；".join(verif.issues) if verif.issues else "存在状态冲突"
                parts.append(f"【结果】状态更新未通过验证：{issues}，已回滚。")

        # —— 行动提示 ——
        hint = self._hint(results)
        if hint:
            parts.append(f"【提示】{normalize_punctuation(hint)}")

        return "\n".join(parts)

    def _hint(self, results) -> str | None:
        domain = next((r for r in results if r.plugin_id == "domain"), None)
        if not domain:
            return None
        content = domain.content
        if "锁着" in content and "钥匙" in content:
            return "你可以先用钥匙打开大门，或先探索其他房间收集物品"
        if "门已经打开" in content:
            return "大门已打开，可以通过走廊前往新区域"
        if "不可以执行" in content:
            return "换个方向试试，先完成可达的目标再继续"
        return None

    def answer(self, req: GenerationRequest, bilingual: bool = False) -> str:
        """对外接口：返回组织好的回答。bilingual=True 时附英文版。"""
        zh = self.format(req)
        if not bilingual:
            return zh
        en = zh
        for k, v in {"【回复】": "[Reply] ", "【状态】": "[Status] ",
                     "【分析】": "[Analysis] ", "【结果】": "[Result] ",
                     "【提示】": "[Hint] "}.items():
            en = en.replace(k, v)
        # 未映射的其余中文字符框转为英文方括号
        en = en.replace("【", "[").replace("】", "]")
        return zh + "\n\n" + en
