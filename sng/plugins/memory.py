"""记忆插件（memory）：召回背包 / 事实。

能力 recall：把当前状态关键事实整理成结构化摘要，供 Echo-er 使用。
纯规则实现，输出断言（可验证）。
"""

from __future__ import annotations

from sng.engine.world import ITEM_ZH
from sng.plugins.base import BasePlugin, PluginManifest


class MemoryPlugin(BasePlugin):
    manifest = PluginManifest(
        plugin_id="memory",
        capabilities=["recall"],
        size_mb=2.0,
        description="记忆召回：背包 / 位置 / 事实摘要",
        judge_rules=["payload 含 recall 请求时开机"],
        task_types=["retrieval", "system"],
    )

    def run(self, payload):
        st = payload["state"]
        inv = [ITEM_ZH.get(i, i) for i in st.inventory]
        room = payload.get("room_zh") or st.location
        content = (f"当前记忆：你位于{room}，背包里有{'、'.join(inv) if inv else '无物品'}；"
                   f"大门{'已打开' if st.flags.get('door_open') else '锁着'}。")
        return content, 0.99, [
            {"type": "recall", "claim": f"背包={inv}",
             "precondition": True, "check": "inventory"},
        ]
