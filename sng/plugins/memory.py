"""记忆插件（memory）：召回背包 / 事实。

能力 recall：把当前状态关键事实整理成结构化摘要，供 Echo-er 使用。
实现两层：
- 训练态：由 MemoryTracker 神经状态追踪器回忆（专项训练后启用）——
  只凭"已执行动作轨迹"回忆状态，不看当前状态，对应"记忆插件用自己的记忆"；
- 回退态：纯规则实现（读 payload["state"]），保证未训练时也能用。
输出断言（可验证）。
"""

from __future__ import annotations

import os

from sng.engine.world import ITEM_ZH, ITEMS, ROOM_ZH, ROOMS
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

    def __init__(self, tracker=None, tok=None) -> None:
        self._tracker = tracker
        self._tok = tok
        if self._tracker is None:
            self._try_load()

    def _try_load(self) -> None:
        """自动加载训练好的记忆追踪器（若权重文件存在）。"""
        try:
            from sng.plugins.memory_model import load_memory_tracker
            loaded = load_memory_tracker()
            if loaded is not None:
                self._tracker, self._tok = loaded
        except Exception:
            self._tracker, self._tok = None, None

    def run(self, payload):
        # 训练态：只用动作轨迹回忆状态（不看当前状态）
        history = payload.get("history")
        if self._tracker is not None and self._tok is not None and history is not None:
            try:
                return self._recall_with_model(history)
            except Exception:
                pass  # 模型异常 -> 回退规则实现
        # 回退态：规则实现
        st = payload["state"]
        inv = [ITEM_ZH.get(i, i) for i in st.inventory]
        room = payload.get("room_zh") or ROOM_ZH.get(st.location, st.location)
        content = (f"当前记忆：你位于{room}，背包里有{'、'.join(inv) if inv else '无物品'}；"
                   f"大门{'已打开' if st.flags.get('door_open') else '锁着'}。")
        return content, 0.99, [
            {"type": "recall", "claim": f"背包={inv}",
             "precondition": True, "check": "inventory"},
        ]

    def _recall_with_model(self, history) -> tuple[str, float, list[dict]]:
        import torch

        from sng.plugins.memory_model import encode_history
        with torch.no_grad():
            seq = encode_history(history, self._tok)
            loc_l, inv_l, do_l, dl_l = self._tracker(seq)
        loc = ROOMS[loc_l.argmax().item()]
        inv = [ITEMS[i] for i, v in enumerate((torch.sigmoid(inv_l) > 0.5)[0]) if bool(v)]
        door_open = bool(torch.sigmoid(do_l)[0].item() > 0.5)
        door_locked = bool(torch.sigmoid(dl_l)[0].item() > 0.5)
        room = ROOM_ZH[loc]
        inv_zh = [ITEM_ZH[i] for i in inv]
        content = (f"当前记忆：你位于{room}，背包里有{'、'.join(inv_zh) if inv_zh else '无物品'}；"
                   f"大门{'已打开' if door_open else '锁着'}。")
        return content, 0.97, [
            {"type": "recall", "claim": f"背包={inv}",
             "precondition": True, "check": "inventory"},
            {"type": "recall", "claim": f"位置={loc}",
             "precondition": True, "check": "location"},
        ]
