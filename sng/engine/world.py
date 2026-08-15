"""迷你文本游戏引擎（原型数据源）。

模拟一个 TextWorld 风格的简化世界：房间 / 物品 / 门 / 背包 / flag，
支持中英双语指令解析与观察生成，并为每一步输出四类标注：
- rule_id   ：命中的规则（CRM 监督）
- slot_ops  ：状态槽更新（WS 监督）
- dispatch  ：需要的插件集合（调度监督）
- obs_{zh,en}：双语观察（语言理解 / Echo-er 素材）
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

from sng.config import Config


class Action(NamedTuple):
    """结构化动作。verb 与 target 均为规范化的枚举字符串。"""
    verb: str       # go / take / drop / open / look / inv / help
    target: str = ""


@dataclass
class GameState:
    location: str = "living_room"
    inventory: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)
    items_at: dict[str, list[str]] = field(default_factory=dict)

    def clone(self) -> "GameState":
        return GameState(
            location=self.location,
            inventory=list(self.inventory),
            flags=dict(self.flags),
            items_at={r: list(v) for r, v in self.items_at.items()},
        )


# ---- 世界拓扑 ----
ROOMS = ["living_room", "kitchen", "garden", "study", "hallway"]
ROOM_ZH = {
    "living_room": "客厅", "kitchen": "厨房", "garden": "花园",
    "study": "书房", "hallway": "走廊",
}
ADJACENCY = {
    "living_room": ["kitchen", "garden", "hallway"],
    "kitchen": ["living_room", "study"],
    "garden": ["living_room"],
    "study": ["kitchen"],
    "hallway": ["living_room"],
}
ITEMS = ["key", "apple", "book", "letter", "bread", "cup"]
ITEM_ZH = {
    "key": "钥匙", "apple": "苹果", "book": "书", "letter": "信",
    "bread": "面包", "cup": "杯子",
}
DOORS = ["front_door"]  # 原型简化：只有一个可开合的门（大门）
DOOR_ZH = {"front_door": "大门", "study_door": "书房门"}

# 每个房间的物品：位置/初始布局
INITIAL_ITEMS = {
    "living_room": ["apple"],
    "kitchen": ["bread"],
    "garden": ["key"],
    "study": ["book", "letter"],
    "hallway": ["cup"],
}
INITIAL_FLAGS = {"door_open": False, "door_locked": True}

# 动作 -> 需要的插件集合（调度监督 ground truth）
DISPATCH_GT = {
    "go": {"memory"},
    "take": {"memory", "domain"},
    "drop": {"memory"},
    "open": {"domain"},
    "look": {"domain", "memory"},
    "inv": {"memory"},
    "help": {"memory"},
}

# 动作 -> 任务类型（通用词表，用于仲裁）
ACTION_TASKTYPE = {
    "go": "system", "take": "retrieval", "drop": "retrieval",
    "open": "reasoning", "look": "comprehension",
    "inv": "retrieval", "help": "comprehension",
}

# 中文指令词表：目标词 -> 归一化 token
_VERB_ZH = {"去": "go", "走到": "go", "拿": "take", "拾取": "take", "拿走": "take",
            "扔": "drop", "放下": "drop", "打开": "open", "查看": "look",
            "环顾": "look", "背包": "inv", "帮助": "help"}
_VERB_EN = {"go": "go", "take": "take", "pick": "take", "get": "take",
            "drop": "drop", "open": "open", "look": "look", "inv": "inv",
            "inventory": "inv", "help": "help"}
_TARGET_ZH = {v: k for k, v in ROOM_ZH.items()}
_TARGET_ZH.update({v: k for k, v in ITEM_ZH.items()})
_TARGET_ZH.update({v: k for k, v in DOOR_ZH.items()})
_TARGET_ZH.update({"门": "front_door", "钥匙": "key"})


class MiniGameEngine:
    """世界与规则引擎。step 产出带四类标注的回合样本。"""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> GameState:
        self.state = GameState(
            location="living_room",
            inventory=["key"],
            flags=dict(INITIAL_FLAGS),
            items_at={r: list(v) for r, v in INITIAL_ITEMS.items()},
        )
        # 初始给玩家一把钥匙，方便演示"打开锁着的门"
        return self.state

    # ---- 语言理解：指令 -> 结构化动作 ----
    def parse_command(self, text: str, lang: str = "zh") -> Action | None:
        """把自然语言指令解析为结构化动作。失败返回 None。"""
        verb, target = None, None
        if lang == "zh":
            for key, v in _VERB_ZH.items():
                if key in text:
                    verb = v
                    break
            if verb is None:
                # 裸名词：如 "钥匙" -> 拾取钥匙
                for name, t in _TARGET_ZH.items():
                    if name in text:
                        return Action("take", t)
                return None
            if verb in ("go", "take", "drop", "open"):
                for name, t in _TARGET_ZH.items():
                    if name in text:
                        target = t
                        break
                if target is None:
                    return None
        else:
            words = [w for w in text.lower().replace(".", " ").split()]
            for w in words:
                if w in _VERB_EN:
                    verb = _VERB_EN[w]
                    break
            if verb is None:
                return None
            if verb in ("go", "take", "drop", "open"):
                # 取动词后第一个已知目标
                rest = words[words.index(next(k for k in words if k in _VERB_EN)) + 1:]
                for w in rest:
                    t = self._resolve_target_en(w)
                    if t:
                        target = t
                        break
                if target is None:
                    return None
        return Action(verb, target or "")

    def _resolve_target_en(self, w: str) -> str | None:
        en_map = {"kitchen": "kitchen", "garden": "garden", "study": "study",
                  "hallway": "hallway", "living": "living_room", "room": "living_room",
                  "key": "key", "apple": "apple", "book": "book", "letter": "letter",
                  "bread": "bread", "cup": "cup", "door": "front_door",
                  "front": "front_door", "study_door": "study_door"}
        return en_map.get(w)

    # ---- 合法性：动作 -> 规则 ----
    def _classify(self, a: Action, st: GameState) -> tuple[str, bool]:
        """返回 (rule_id, is_valid)。"""
        if a.verb == "go":
            if a.target in ADJACENCY[st.location]:
                return "move_valid", True
            return "move_invalid", False
        if a.verb == "take":
            if a.target in st.items_at.get(st.location, []):
                return "take_valid", True
            return "take_invalid", False
        if a.verb == "drop":
            if a.target in st.inventory:
                return "drop_valid", True
            return "drop_invalid", False
        if a.verb == "open":
            if a.target not in DOORS:
                return "open_invalid", False
            if st.flags.get("door_open", False):
                return "open_already", False
            if st.flags.get("door_locked", False):
                if "key" in st.inventory:
                    return "open_locked_with_key", True
                return "open_locked_no_key", False
            return "open_unlocked", True
        if a.verb in ("look", "inv", "help"):
            return f"{a.verb}_valid", True
        return "invalid_action", False

    # ---- 状态迁移 + 槽更新 ----
    def transition_ops(self, rule_id: str, a: Action) -> list[tuple[str, str, object]]:
        """根据规则 id 返回槽操作（不改变状态，供 WS 确定性写入）。"""
        if rule_id == "move_valid":
            return [("location", "set", a.target)]
        if rule_id == "take_valid":
            return [("room", "remove", a.target), ("inv", "append", a.target)]
        if rule_id == "drop_valid":
            return [("inv", "remove", a.target), ("room", "add", a.target)]
        if rule_id == "open_locked_with_key":
            return [("flag.door_open", "set", True), ("flag.door_locked", "set", False)]
        if rule_id == "open_unlocked":
            return [("flag.door_open", "set", True)]
        return []

    def _apply(self, a: Action, st: GameState, rule_id: str) -> list[tuple[str, str, object]]:
        """执行合法动作，返回 (槽名, op, value) 列表。"""
        ops: list[tuple[str, str, object]] = []
        if rule_id == "move_valid":
            ops.append(("location", "set", a.target))
            st.location = a.target
        elif rule_id == "take_valid":
            st.items_at[st.location].remove(a.target)
            st.inventory.append(a.target)
            ops.append(("room", "remove", a.target))
            ops.append(("inv", "append", a.target))
        elif rule_id == "drop_valid":
            st.inventory.remove(a.target)
            st.items_at.setdefault(st.location, []).append(a.target)
            ops.append(("inv", "remove", a.target))
            ops.append(("room", "add", a.target))
        elif rule_id == "open_locked_with_key":
            st.flags["door_open"] = True
            st.flags["door_locked"] = False
            ops.append(("flag.door_open", "set", True))
            ops.append(("flag.door_locked", "set", False))
        elif rule_id == "open_unlocked":
            st.flags["door_open"] = True
            ops.append(("flag.door_open", "set", True))
        return ops

    # ---- 观察生成（双语）----
    def action_text(self, a: Action, lang: str = "zh") -> str:
        """结构化动作 -> 自然语言指令（供控制器输入 / 训练监督）。"""
        verb_zh = {"go": "去", "take": "拾取", "drop": "放下", "open": "打开",
                   "look": "查看", "inv": "背包", "help": "帮助"}
        if lang == "zh":
            t = ROOM_ZH.get(a.target) or ITEM_ZH.get(a.target) \
                or DOOR_ZH.get(a.target) or ""
            return verb_zh[a.verb] + t
        return f"{a.verb} {a.target}".strip()

    def observe(self, lang: str = "zh") -> str:
        st = self.state
        if lang == "zh":
            items = "、".join(ITEM_ZH[i] for i in st.items_at.get(st.location, [])) or "空无一物"
            inv = "、".join(ITEM_ZH[i] for i in st.inventory) or "空"
            dest = "、".join(ROOM_ZH[r] for r in ADJACENCY[st.location])
            door_txt = ""
            if st.flags.get("door_locked"):
                door_txt = "大门还锁着。"
            elif st.flags.get("door_open"):
                door_txt = "大门已经打开。"
            return (f"你现在在{ROOM_ZH[st.location]}。出口：{dest}。"
                    f"这里的东西：{items}。你的背包里有：{inv}。{door_txt}")
        rooms = ", ".join(r.replace("_", " ") for r in ADJACENCY[st.location])
        items = ", ".join(st.items_at.get(st.location, [])) or "nothing"
        inv = ", ".join(st.inventory) or "empty"
        door_txt = "The front door is locked." if st.flags.get("door_locked") else \
            ("The front door is open." if st.flags.get("door_open") else "")
        return (f"You are in the {st.location.replace('_', ' ')}. Exits: {rooms}. "
                f"Items here: {items}. Inventory: {inv}. {door_txt}")

    # ---- 一步执行（带全部标注）----
    def step(self, action: Action) -> tuple[str, str, GameState, dict]:
        st = self.state.clone()
        rule_id, is_valid = self._classify(action, st)
        if is_valid:
            ops = self._apply(action, st, rule_id)
        else:
            ops = []
        self.state = st
        info = {
            "rule_id": rule_id,
            "valid": is_valid,
            "slot_ops": ops,
            "dispatch": set(DISPATCH_GT[action.verb]) if action.verb in DISPATCH_GT else set(),
            "task_type": ACTION_TASKTYPE.get(action.verb, "comprehension"),
            "action": action,
            "reward": 1.0 if is_valid and action.verb != "help" else 0.0,
            "done": False,
        }
        return self.observe("zh"), self.observe("en"), self.state, info

    # ---- 状态特征（供 CRM 匹配与调度头的 ground truth / 输入）----
    def state_features(self, st: GameState | None = None) -> dict[str, float]:
        st = st or self.state
        return {
            "has_key": float("key" in st.inventory),
            "door_open": float(st.flags.get("door_open", False)),
            "door_locked": float(st.flags.get("door_locked", True)),
            "inv_len": float(len(st.inventory)),
        }

    def action_features(self, a: Action) -> dict[str, float]:
        return {
            f"verb:{a.verb}": 1.0,
            f"target:{a.target}": 1.0,
        }

    # ---- CRM 规则谓词（供可微索引与 DFA 编译）----
    def rule_predicates(self, a: Action, st: GameState | None = None) -> dict[str, bool]:
        """计算 (状态, 动作) 的规则相关布尔谓词。"""
        st = st or self.state
        return {
            "adjacent": a.verb == "go" and a.target in ADJACENCY[st.location],
            "item_at_loc": a.verb == "take" and a.target in st.items_at.get(st.location, []),
            "item_in_inv": a.verb == "drop" and a.target in st.inventory,
            "door_locked": bool(st.flags.get("door_locked", False)),
            "has_key": "key" in st.inventory,
            "door_open": bool(st.flags.get("door_open", False)),
            "special": a.verb in ("look", "inv", "help"),
        }

    # ---- CRM 规则表（真值规则，供可微索引学习 + DFA 编译）----
    def rule_table(self) -> list[dict]:
        """返回规则列表。每项：rule_id, 状态谓词, 动作谓词, 迁移描述。"""
        return [
            {"id": "move_valid", "state": {"adjacent": True}, "action": {"verb": "go"},
             "transition": "location=target"},
            {"id": "move_invalid", "state": {"adjacent": False}, "action": {"verb": "go"},
             "transition": "no-op"},
            {"id": "take_valid", "state": {"item_at_loc": True}, "action": {"verb": "take"},
             "transition": "inv+=item, room-=item"},
            {"id": "take_invalid", "state": {"item_at_loc": False}, "action": {"verb": "take"},
             "transition": "no-op"},
            {"id": "drop_valid", "state": {"item_in_inv": True}, "action": {"verb": "drop"},
             "transition": "inv-=item, room+=item"},
            {"id": "drop_invalid", "state": {"item_in_inv": False}, "action": {"verb": "drop"},
             "transition": "no-op"},
            {"id": "open_locked_with_key", "state": {"door_locked": True, "has_key": True},
             "action": {"verb": "open", "target": "front_door"}, "transition": "door_open=T,door_locked=F"},
            {"id": "open_locked_no_key", "state": {"door_locked": True, "has_key": False},
             "action": {"verb": "open", "target": "front_door"}, "transition": "no-op"},
            {"id": "open_unlocked", "state": {"door_locked": False}, "action": {"verb": "open"},
             "transition": "door_open=T"},
            {"id": "open_already", "state": {"door_open": True}, "action": {"verb": "open"},
             "transition": "no-op"},
            {"id": "look_valid", "state": {}, "action": {"verb": "look"}, "transition": "no-op"},
            {"id": "inv_valid", "state": {}, "action": {"verb": "inv"}, "transition": "no-op"},
            {"id": "help_valid", "state": {}, "action": {"verb": "help"}, "transition": "no-op"},
            {"id": "invalid_action", "state": {}, "action": {"verb": "<other>"}, "transition": "no-op"},
        ]

    # ---- 生成训练数据：随机 rollout ----
    def rollout(self, n_steps: int = 6, lang: str = "zh") -> list[dict]:
        """随机策略 rollout，返回带标注的样本序列。"""
        self.reset()
        samples = []
        for _ in range(n_steps):
            st_before = self.state.clone()
            valid = self.valid_actions()
            a = self.rng.choice(valid + [Action("look"), Action("inv"), Action("help")])
            obs_zh, obs_en, st_after, info = self.step(a)
            samples.append({
                "obs_zh": obs_zh, "obs_en": obs_en,
                "state_before": st_before, "state_after": st_after,
                "action": a, "rule_id": info["rule_id"], "valid": info["valid"],
                "slot_ops": info["slot_ops"], "dispatch": info["dispatch"],
                "task_type": info["task_type"], "reward": info["reward"],
            })
        return samples

    def valid_actions(self, st: GameState | None = None) -> list[Action]:
        st = st or self.state
        acts: list[Action] = [Action("go", r) for r in ADJACENCY[st.location]]
        acts += [Action("take", i) for i in st.items_at.get(st.location, [])]
        acts += [Action("drop", i) for i in st.inventory]
        acts += [Action("open", d) for d in DOORS]
        acts += [Action("look"), Action("inv"), Action("help")]
        return acts
