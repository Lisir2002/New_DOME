"""领域插件（domain）：文本游戏领域规则查询。

能力 domain_query：物品位置 / 门状态 / 动作合法性。纯规则实现，
接收引擎状态快照，输出结构化分析 + 断言（供验证器与 Echo-er 使用）。
"""

from __future__ import annotations

from sng.engine.world import DOOR_ZH, ITEM_ZH, ROOM_ZH
from sng.plugins.base import BasePlugin, PluginManifest


class DomainPlugin(BasePlugin):
    manifest = PluginManifest(
        plugin_id="domain",
        capabilities=["domain_query"],
        size_mb=4.0,
        description="领域规则查询：物品位置 / 门状态 / 动作合法性",
        judge_rules=["payload 含 domain_query 请求时开机"],
        task_types=["reasoning", "comprehension"],
    )

    def run(self, payload):
        st = payload["state"]
        # 动作可行性基于动作前状态判断；描述性内容用动作后状态
        pre = payload.get("state_before", st)
        a = payload["action"]
        room_zh = ROOM_ZH.get(st.location, st.location)
        items = [ITEM_ZH.get(i, i) for i in st.items_at.get(st.location, [])]
        door_txt = "已打开" if st.flags.get("door_open") else (
            "锁着" if st.flags.get("door_locked") else "未锁")
        verdict = self._rule_verdict(pre, a)
        action_txt = payload.get("action_zh") or f"{a.verb} {a.target or ''}".strip()
        content = (f"你位于{room_zh}，此处物品有{'、'.join(items) if items else '无'}；"
                   f"大门{door_txt}。动作“{action_txt}”"
                   f"{'可以执行' if verdict[0] else '不可以执行'}：{verdict[1]}。")
        return content, 0.98, [
            {"type": "domain", "claim": f"玩家位于 {room_zh}",
             "precondition": True, "check": "location"},
            {"type": "domain", "claim": f"大门{door_txt}",
             "precondition": True, "check": "door"},
        ]

    def _rule_verdict(self, st, a):
        if a.verb == "go":
            from sng.engine.world import ADJACENCY
            ok = a.target in ADJACENCY[st.location]
            return (ok, "该房间可达" if ok else "该房间不在出口列表中")
        if a.verb == "take":
            ok = a.target in st.items_at.get(st.location, [])
            return (ok, "物品在此处" if ok else "此处没有该物品")
        if a.verb == "open":
            if st.flags.get("door_open"):
                return (False, "门已经打开")
            if st.flags.get("door_locked"):
                if "key" in st.inventory:
                    return (True, "你有钥匙，可以打开锁着的门")
                return (False, "门锁着，需要钥匙")
            return (True, "门未锁，可以直接打开")
        if a.verb == "drop":
            return (a.target in st.inventory, "物品在背包中" if a.target in st.inventory else "背包中没有该物品")
        return (True, "可执行")
