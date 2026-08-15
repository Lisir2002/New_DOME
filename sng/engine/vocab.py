"""词汇表与分词器。

原型采用确定性的固定词表：先对引擎所有模板文本做分词，收集唯一 token。
中文使用"已知词最长匹配 + 单字兜底"，英文使用空白分词（统一小写）。
"""

from __future__ import annotations

import re
from collections import Counter

# 中文已知词（用于最长匹配，顺序无关紧要，内部按长度排序）
KNOWN_ZH = [
    "你现在在", "你位于", "这里的东西", "你的背包里", "出口", "物品", "背包",
    "帮助", "查看", "环顾", "打开", "拾取", "拿走", "放下", "扔", "走到", "去",
    "客厅", "厨房", "花园", "书房", "走廊", "大厅", "大门", "书房门", "厨房门",
    "钥匙", "苹果", "书", "信", "面包", "杯子", "有", "没有", "锁着", "开着",
    "可以", "需要", "已经", "获得了", "把", "放进", "了", "一个", "这", "那里",
]

_EN_WORD_RE = re.compile(r"[a-zA-Z']+")


def tokenize_zh(text: str) -> list[str]:
    """中文分词：已知词最长匹配 + 单字兜底。"""
    tokens: list[str] = []
    i = 0
    # 已知词按长度降序，便于最长匹配
    known = sorted(KNOWN_ZH, key=len, reverse=True)
    n = len(text)
    while i < n:
        ch = text[i]
        if not _is_cjk(ch):
            # 跳过标点/空白，但保留英文词与数字
            m = _EN_WORD_RE.match(text, i)
            if m:
                tokens.append(m.group(0).lower())
                i = m.end()
                continue
            i += 1
            continue
        matched = False
        for w in known:
            if text.startswith(w, i):
                tokens.append(w)
                i += len(w)
                matched = True
                break
        if not matched:
            tokens.append(ch)
            i += 1
    return tokens


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize_en(text: str) -> list[str]:
    """英文分词：空白 + 标点剥离，统一小写。"""
    return [t.lower() for t in _EN_WORD_RE.findall(text)]


def tokenize(text: str, lang: str) -> list[str]:
    return tokenize_zh(text) if lang == "zh" else tokenize_en(text)


class Vocabulary:
    """固定词表：token -> id。未知 token 统一映射到 <unk>。"""

    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self) -> None:
        self.itos: list[str] = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def build(self, texts: list[tuple[str, str]]) -> None:
        """从 (text, lang) 语料构建词表。"""
        for text, lang in texts:
            for tok in tokenize(text, lang):
                self.add(tok)

    def encode(self, text: str, lang: str,
               max_len: int | None = None) -> list[int]:
        toks = [self.BOS] + tokenize(text, lang) + [self.EOS]
        ids = [self.stoi.get(t, self.stoi[self.UNK]) for t in toks]
        if max_len is not None:
            ids = ids[: max_len]
            ids += [self.stoi[self.PAD]] * (max_len - len(ids))
        return ids

    @property
    def size(self) -> int:
        return len(self.itos)

    def freq(self, texts: list[tuple[str, str]]) -> Counter:
        c: Counter = Counter()
        for text, lang in texts:
            c.update(tokenize(text, lang))
        return c
