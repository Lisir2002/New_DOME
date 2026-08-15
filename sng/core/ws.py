"""WS 状态读写层（State Stream）。

持久状态矩阵 S ∈ R^{128×128}（128 槽 × 128 维）。混合寻址：
- 类型化槽：语义固定直写（schema 决定槽位，无需学习寻址）→ 可精确验证；
- 自由槽：content-based 软寻址（可微，模型自定存什么）。

对外提供：
- encode_state / decode_state：与引擎真值状态互转；
- read：注意力读取（读取结果供调度/决策）；
- write_candidate：可微写入（自由槽软写 + 类型化槽回归预测，训练用）；
- apply_ops：确定性写入（由 CRM 迁移/引擎槽操作驱动，推理保证一致性）。
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sng.config import Config
from sng.engine.world import GameState, ITEMS, ROOMS

# 状态词表：房间 + 物品 + flag
FLAG_KEYS = ["door_open", "door_locked"]
STATE_VOCAB = {name: i for i, name in enumerate(ROOMS + ITEMS + FLAG_KEYS)}
STATE_VOCAB_SIZE = len(STATE_VOCAB)


class StateStream(nn.Module):
    """WS 状态读写层。不持有状态矩阵本身，而是提供读写算子与状态编解码。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.emb_dim
        self.n_slots = cfg.n_slots
        self.n_typed = cfg.n_typed
        self.slot_free = cfg.slot_free
        # 状态词表嵌入（类型化槽的"值"编码）
        self.state_emb = nn.Embedding(STATE_VOCAB_SIZE, self.dim)
        # 自由槽软写：查询投影 + 值生成
        self.free_q = nn.Linear(self.dim, self.dim)
        self.free_w = nn.Linear(self.dim, self.dim)
        # 类型化槽可微写（训练用）：预测 n_typed 个槽的目标值
        self.typed_write = nn.Linear(self.dim, self.n_typed * self.dim)

    # ---- 状态编解码 ----
    def encode_state(self, st: GameState) -> torch.Tensor:
        """引擎真值状态 -> 状态矩阵 S（[n_slots, dim]）。"""
        S = torch.zeros(self.n_slots, self.dim, dtype=torch.float32)
        ids = [STATE_VOCAB[st.location]]
        for it in st.inventory[: self.cfg.max_inv]:
            ids.append(STATE_VOCAB[it])
        vec = self.state_emb(torch.tensor(ids, dtype=torch.long))
        S[self.cfg.slot_location] = vec[0]
        for k in range(len(st.inventory[: self.cfg.max_inv])):
            S[self.cfg.slot_inv + k] = vec[1 + k]
        for j, fk in enumerate(FLAG_KEYS):
            if st.flags.get(fk, False):
                S[self.cfg.slot_flag + j] = self.state_emb(
                    torch.tensor(STATE_VOCAB[fk], dtype=torch.long))
        return S

    def decode_state(self, S: torch.Tensor) -> GameState:
        """状态矩阵 -> 可读状态（类型化槽取最近词向量 argmax）。"""
        S = S.detach()
        emb = self.state_emb.weight
        emb_n = F.normalize(emb, dim=-1)
        sn = F.normalize(S[: self.cfg.n_typed], dim=-1)
        sim = sn @ emb_n.t()  # [n_typed, vocab]
        best = sim.argmax(dim=-1).tolist()
        rev = {i: n for n, i in STATE_VOCAB.items()}
        st = GameState(location=rev[best[0]])
        inv = []
        for k in range(1, self.cfg.max_inv + 1):
            name = rev[best[k]]
            if name in ITEMS and (self.cfg.slot_inv + k - 1) < self.cfg.slot_flag:
                inv.append(name)
        st.inventory = inv
        flags: dict[str, bool] = {}
        for j, fk in enumerate(FLAG_KEYS):
            idx = self.cfg.slot_flag + j
            flags[fk] = torch.nn.functional.cosine_similarity(
                S[idx: idx + 1], emb[STATE_VOCAB[fk]: STATE_VOCAB[fk] + 1]
            ).item() > 0.5
        st.flags = flags
        st.items_at = {}
        return st

    # ---- 读取 ----
    def read(self, S: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """content-based 读取。返回 (读出向量 r, 槽位注意力 alpha)。"""
        S = S.unsqueeze(0)                      # [1, n_slots, dim]
        q = q.reshape(1, 1, self.dim)           # [1, 1, dim]
        alpha = torch.softmax(q @ S.transpose(1, 2) / (self.dim ** 0.5), dim=-1)  # [1,1,n_slots]
        r = (alpha @ S)                         # [1, 1, dim]
        return r.reshape(self.dim), alpha.reshape(self.n_slots)

    # ---- 可微写入（训练用：世界模型）----
    def write_candidate(self, S: torch.Tensor, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回候选更新：
        - typed_targets: [n_typed, dim] 类型化槽目标值（L_state 监督用）
        - free_delta: [n_free, dim] 自由槽软写增量
        - free_attn: [n_free] 自由槽寻址注意力
        """
        typed_targets = self.typed_write(h).view(self.cfg.n_typed, self.dim)
        q_w = self.free_q(h)
        S_free = S[self.slot_free:]
        beta = torch.softmax(q_w @ S_free.t() / self.cfg.free_slot_temp, dim=-1)
        w = self.free_w(h).expand(self.n_slots - self.slot_free, -1)
        free_delta = beta.unsqueeze(-1) * w
        return {"typed_targets": typed_targets, "free_delta": free_delta, "free_attn": beta}

    def apply_learned(self, S: torch.Tensor, upd: dict[str, torch.Tensor],
                      weight: float = 1.0) -> torch.Tensor:
        """把可微候选更新应用到状态矩阵（训练/探索用）。"""
        S = S.clone()
        S[: self.cfg.n_typed] = S[: self.cfg.n_typed] * (1 - weight) + upd["typed_targets"] * weight
        S[self.slot_free:] = S[self.slot_free:] * (1 - weight) + upd["free_delta"] * weight
        return S

    # ---- 确定性写入（推理用：由槽操作驱动）----
    def apply_ops(self, S: torch.Tensor, ops: list[tuple[str, str, object]],
                  emb: Callable[[str], torch.Tensor]) -> torch.Tensor:
        """按槽操作确定性地更新类型化槽（schema 直写，可精确验证）。"""
        S = S.clone()
        for key, op, val in ops:
            if key == "location":
                S[self.cfg.slot_location] = emb(str(val))
            elif key == "inv":
                if op == "append":
                    # 找第一个空背包槽
                    free_slot = None
                    for k in range(self.cfg.max_inv):
                        if S[self.cfg.slot_inv + k].norm() < 1e-6:
                            free_slot = self.cfg.slot_inv + k
                            break
                    if free_slot is not None:
                        S[free_slot] = emb(str(val))
                elif op == "remove":
                    for k in range(self.cfg.max_inv):
                        slot = self.cfg.slot_inv + k
                        if torch.argmax(S[slot] @ self.state_emb.weight.t()) == STATE_VOCAB.get(str(val), 0):
                            S[slot] = torch.zeros_like(S[slot])
                            break
            elif key.startswith("flag."):
                flag = key.split(".")[1]
                idx = self.cfg.slot_flag + FLAG_KEYS.index(flag)
                S[idx] = emb(flag) if val else torch.zeros_like(S[idx])
            elif key == "room":
                # 房间物品仅由引擎状态追踪，WS 类型化槽不保存（保持一致性时跳过）
                continue
        return S
