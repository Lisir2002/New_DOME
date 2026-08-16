"""记忆插件专项训练：记忆追踪器（MemoryTracker）。

任务定义（记忆召回 / 长叙事回忆）：
- 输入  ：已执行的动作轨迹（结构化 Action 序列，含当前动作）；
          追踪器**不看当前状态**，只凭轨迹回忆状态 —— 与"记忆插件用自己的记忆
          回忆状态"的架构定位一致，区别于控制器内置的 WS 状态引擎。
- 输出  ：当前状态槽位
    location    : 5 分类（ROOMS）
    inventory   : 6 多标签（ITEMS）
    door_open   : 二元
    door_locked : 二元

模型：小型 GRU 状态追踪器。
数据：由迷你文本游戏引擎 rollout 自动产出（动作轨迹前缀 -> 该步引擎真值状态），
     属于"身份分训"阶段插件模型的独立监督数据，免人工标注。
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn

from sng.engine.world import (
    Action, DOORS, ITEMS, ROOMS, MiniGameEngine,
)

# 记忆追踪器权重文件（与主控制器 MODEL_PATH 同一约定，*.pt 不入库）
MEMORY_MODEL_PATH = "/workspace/sng_memory.pt"

# 动作词表
_VERBS = ["go", "take", "drop", "open", "look", "inv", "help"]
_TARGETS = list(ROOMS) + list(ITEMS) + list(DOORS) + [""]
_PAD = 0  # 左填充 token


def build_action_vocab() -> tuple[dict, dict, int]:
    """返回 (verb->idx, target->idx, target 数)。动作 token = verb*N + target + 1。"""
    vt = {v: i for i, v in enumerate(_VERBS)}
    tt = {t: i for i, t in enumerate(_TARGETS)}
    return vt, tt, len(_TARGETS)


def action_token(a: Action, vt: dict, tt: dict, n_targets: int) -> int:
    """结构化动作 -> 单整数 token。"""
    return vt[a.verb] * n_targets + tt[a.target] + 1


def encode_history(actions: list[Action], tok, max_len: int = 32) -> torch.Tensor:
    """动作轨迹 -> 左填充 token 序列（一维 (T,)；批量时由调用方 stack）。"""
    seq = [tok(a) for a in actions[-max_len:]]
    padded = [_PAD] * (max_len - len(seq)) + seq
    return torch.tensor(padded, dtype=torch.long)


class MemoryTracker(nn.Module):
    """小型 GRU 状态追踪器：动作轨迹 -> 当前状态槽位。"""

    def __init__(self, vocab_size: int, n_rooms: int = len(ROOMS),
                 n_items: int = len(ITEMS), embed: int = 24, hidden: int = 40) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed, padding_idx=_PAD)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.loc_head = nn.Linear(hidden, n_rooms)
        self.inv_head = nn.Linear(hidden, n_items)
        self.door_open_head = nn.Linear(hidden, 1)
        self.door_locked_head = nn.Linear(hidden, 1)

    def forward(self, seq: torch.Tensor):
        """输入 (B, T) 或 (T,) -> (loc_logits, inv_logits, door_open_logit, door_locked_logit)。"""
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        e = self.emb(seq) * (seq != _PAD).unsqueeze(-1)  # pad 位归零
        h, _ = self.gru(e)
        h = h[:, -1]
        return (self.loc_head(h), self.inv_head(h),
                self.door_open_head(h).squeeze(-1),
                self.door_locked_head(h).squeeze(-1))


def state_target(st) -> tuple[int, list[float], float, float]:
    """引擎状态 -> 监督目标（位置 idx / 背包 mask / 门旗标）。"""
    loc = ROOMS.index(st.location)
    inv = [float(i in st.inventory) for i in ITEMS]
    return loc, inv, float(st.flags.get("door_open", False)), \
        float(st.flags.get("door_locked", True))


def make_dataset(engine: MiniGameEngine, n_rollouts: int = 400,
                 max_steps: int = 10, seed: int = 7) -> list[tuple[list, tuple]]:
    """引擎 rollout -> (动作轨迹前缀, 该步真值状态) 样本序列。"""
    rng = random.Random(seed)
    samples: list[tuple[list, tuple]] = []
    for _ in range(n_rollouts):
        engine.reset()
        history: list[Action] = []
        # 空轨迹 -> 初始状态（模型学会初始条件：客厅 / 有钥匙 / 门锁着）
        samples.append((list(history), state_target(engine.state)))
        for _ in range(max_steps):
            valid = engine.valid_actions()
            a = rng.choice(valid + [Action("look"), Action("inv"), Action("help")])
            _, _, st_after, info = engine.step(a)
            if info["valid"]:
                history.append(a)
            samples.append((list(history), state_target(st_after)))
    return samples


def train_memory_tracker(engine: MiniGameEngine, n_rollouts: int = 400,
                         max_steps: int = 10, epochs: int = 40,
                         hidden: int = 40, lr: float = 2e-3,
                         batch_size: int = 64, seed: int = 7,
                         verbose: bool = True) -> MemoryTracker:
    """训练记忆追踪器，返回训练好的模型（不保存，由调用方决定）。"""
    vt, tt, n_t = build_action_vocab()
    tok = lambda a: action_token(a, vt, tt, n_t)  # noqa: E731
    vocab_size = len(_VERBS) * n_t + 1
    data = make_dataset(engine, n_rollouts, max_steps, seed)
    model = MemoryTracker(vocab_size, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loc_loss = nn.CrossEntropyLoss()
    inv_loss = nn.BCEWithLogitsLoss()
    flag_loss = nn.BCEWithLogitsLoss()

    rng = random.Random(seed)
    max_len = 32
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        tot_loc, tot_inv, tot_do, tot_dl, n = 0.0, 0.0, 0.0, 0.0, 0
        for i in range(0, len(order), batch_size):
            batch = [data[j] for j in order[i:i + batch_size]]
            seq = torch.stack([encode_history(h, tok, max_len) for h, _ in batch])
            t_loc = torch.tensor([t[0] for _, t in batch])
            t_inv = torch.tensor([t[1] for _, t in batch])
            t_do = torch.tensor([t[2] for _, t in batch]).float()
            t_dl = torch.tensor([t[3] for _, t in batch]).float()
            loc, inv, do, dl = model(seq)
            loss = (loc_loss(loc, t_loc) + inv_loss(inv, t_inv)
                    + flag_loss(do, t_do) + flag_loss(dl, t_dl))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loc += (loc.argmax(1) == t_loc).sum().item()
            tot_inv += ((torch.sigmoid(inv) > 0.5) == (t_inv > 0.5)).all(1).sum().item()
            tot_do += ((torch.sigmoid(do) > 0.5) == (t_do > 0.5)).sum().item()
            tot_dl += ((torch.sigmoid(dl) > 0.5) == (t_dl > 0.5)).sum().item()
            n += len(batch)
        if verbose and (ep + 1) % 5 == 0:
            print(f"  记忆追踪器 epoch {ep+1:02d}  loss={loss.item():.3f} "
                  f"位置={(100*tot_loc/n):.1f}% 背包={(100*tot_inv/n):.1f}% "
                  f"门开={(100*tot_do/n):.1f}% 门锁={(100*tot_dl/n):.1f}%")
    model.eval()
    if verbose:
        loc_a, inv_a, do_a, dl_a = evaluate_memory(model, tok, engine, seed + 1, 60, 8)
        print(f"  记忆追踪器验证（held-out） 位置={loc_a:.1f}% 背包={inv_a:.1f}% "
              f"门开={do_a:.1f}% 门锁={dl_a:.1f}%")
    return model


def evaluate_memory(model: MemoryTracker, tok, engine: MiniGameEngine,
                    seed: int, n_rollouts: int = 60,
                    max_steps: int = 8) -> tuple[float, float, float, float]:
    """held-out rollout 上的槽位准确率。"""
    rng = random.Random(seed)
    loc_ok = inv_ok = do_ok = dl_ok = n = 0
    with torch.no_grad():
        for _ in range(n_rollouts):
            engine.reset()
            history: list[Action] = []
            for t in range(max_steps + 1):
                seq = encode_history(history, tok)
                loc, inv, do, dl = model(seq)
                s = engine.state
                loc_ok += (loc.argmax().item() == ROOMS.index(s.location))
                inv_ok += bool(((torch.sigmoid(inv) > 0.5)[0] ==
                                torch.tensor([float(i in s.inventory) for i in ITEMS])).all())
                do_ok += (torch.sigmoid(do)[0].item() > 0.5) == bool(s.flags.get("door_open", False))
                dl_ok += (torch.sigmoid(dl)[0].item() > 0.5) == bool(s.flags.get("door_locked", True))
                n += 1
                if t < max_steps:
                    valid = engine.valid_actions()
                    a = rng.choice(valid + [Action("look"), Action("inv"), Action("help")])
                    _, _, _, info = engine.step(a)
                    if info["valid"]:
                        history.append(a)
    return 100 * loc_ok / n, 100 * inv_ok / n, 100 * do_ok / n, 100 * dl_ok / n


def load_memory_tracker(path: str = MEMORY_MODEL_PATH) -> tuple[MemoryTracker, callable] | None:
    """加载训练好的记忆追踪器；文件不存在返回 None。"""
    if not os.path.exists(path):
        return None
    vt, tt, n_t = build_action_vocab()
    model = MemoryTracker(len(_VERBS) * n_t + 1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    tok = lambda a: action_token(a, vt, tt, n_t)  # noqa: E731
    return model, tok
