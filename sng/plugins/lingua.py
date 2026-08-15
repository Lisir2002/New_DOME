"""双语插件（lingua）：中英文本转换。

能力 translate：把内容在 zh / en 之间转换。原型用确定性词表翻译
（演示双语能力；正式实现中该插件是独立小模型）。
"""

from __future__ import annotations

from sng.plugins.base import BasePlugin, PluginManifest

# 演示用最小词表
_LUT = {
    "你现在在": "You are in the ", "客厅": "living room", "厨房": "kitchen",
    "花园": "garden", "书房": "study", "走廊": "hallway",
    "出口": "Exits", "这里的东西": "Items here", "你的背包里有": "Inventory",
    "大门": "front door", "锁着": "is locked", "已经打开": "is open",
    "钥匙": "key", "苹果": "apple", "书": "book", "信": "letter",
    "面包": "bread", "杯子": "cup", "空无一物": "nothing",
}


class LinguaPlugin(BasePlugin):
    manifest = PluginManifest(
        plugin_id="lingua",
        capabilities=["translate"],
        size_mb=3.0,
        description="中英双语转换",
        judge_rules=["payload 需要双语输出时开机"],
        task_types=["lingua"],
    )

    def run(self, payload):
        text = payload.get("text", "")
        target = payload.get("target_lang", "en")
        if target == "en":
            out = text
            for k, v in _LUT.items():
                out = out.replace(k, v)
        else:
            out = text  # 原型：英文转中文不做
        return out, 0.9, [{"type": "lingua", "claim": "完成双语转换", "precondition": True}]
