"""CRM 可编译规则记忆层（Compilable Rule Memory）。

训练时可微索引学习：用"状态谓词 + 动作"对规则表做软注意力匹配，
监督来自引擎的真值规则（L_match = CE(α, rule*) + 稀疏正则）；
推理时编译为确定性 DFA（哈希迁移表），O(1) 查表，状态一致性为结构性保证。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sng.config import Config
from sng.engine.world import Action, GameState, ITEMS, ROOMS

VERBS = ["go", "take", "drop", "open", "look", "inv", "help"]
TARGETS = ["<none>"] + ROOMS + ITEMS + ["front_door"]
PREDS = ["adjacent", "item_at_loc", "item_in_inv", "door_locked", "has_key",
         "door_open", "special"]

# 规则 id 顺序（与引擎 rule_table 一致）
RULE_ORDER = [
    "move_valid", "move_invalid", "take_valid", "take_invalid",
    "drop_valid", "drop_invalid", "open_locked_with_key", "open_locked_no_key",
    "open_unlocked", "open_already", "look_valid", "inv_valid",
    "help_valid", "invalid_action",
]
RULE_INDEX = {rid: i for i, rid in enumerate(RULE_ORDER)}


def _one_hot(idx: int, n: int) -> list[float]:
    v = [0.0] * n
    v[idx] = 1.0
    return v


def rule_feature_vec(verb: str, target: str, preds: dict[str, bool]) -> torch.Tensor:
    """构建规则匹配输入特征：[谓词7 + 动词onehot7 + 目标onehot12]。"""
    feat = [1.0 if preds.get(p, False) else 0.0 for p in PREDS]
    feat += _one_hot(VERBS.index(verb), len(VERBS))
    t = target if target in TARGETS else "<none>"
    feat += _one_hot(TARGETS.index(t), len(TARGETS))
    return torch.tensor(feat, dtype=torch.float32)


class CRM(nn.Module):
    """可微规则索引 + DFA 编译。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.feat_dim = len(PREDS) + len(VERBS) + len(TARGETS)
        self.n_rules = cfg.n_rules
        self.proj = nn.Linear(self.feat_dim, cfg.emb_dim)
        self.rule_emb = nn.Parameter(torch.randn(self.n_rules, cfg.emb_dim) * 0.1)

    def forward(self, verb: str, target: str,
                preds: dict[str, bool]) -> torch.Tensor:
        """软注意力索引 -> 规则概率 α（训练用，可微）。"""
        x = rule_feature_vec(verb, target, preds)
        q = self.proj(x)
        logits = q @ self.rule_emb.t() / (self.cfg.emb_dim ** 0.5)
        return torch.softmax(logits, dim=-1)

    def match_loss(self, alpha: torch.Tensor, rule_id: str,
                   sparse_lambda: float | None = None) -> torch.Tensor:
        """L_match：CE(α, 真值规则) + 稀疏正则（逼向单一匹配）。"""
        gt = torch.tensor(RULE_INDEX.get(rule_id, RULE_INDEX["invalid_action"]),
                          dtype=torch.long)
        ce = F.cross_entropy(alpha.unsqueeze(0).log(), gt.unsqueeze(0))
        lam = sparse_lambda if sparse_lambda is not None else self.cfg.crm_sparse_lambda
        sparse = lam * (alpha * alpha.log().clamp_min(-20)).sum()
        return ce + sparse

    def predict_rule(self, verb: str, target: str,
                     preds: dict[str, bool]) -> tuple[str, torch.Tensor]:
        """推理：取 argmax（训练收敛后等价于确定性匹配）。"""
        alpha = self.forward(verb, target, preds)
        rid = RULE_ORDER[int(alpha.argmax())]
        return rid, alpha

    # ---- DFA 编译：确定性哈希迁移表 ----
    @staticmethod
    def build_dfa(engine) -> dict:
        """枚举 (动词, 目标, 谓词组合) 的有限集合，生成 (键 -> 规则) 迁移表。"""
        dfa: dict = {}
        for verb in VERBS:
            if verb == "go":
                for adj in (0, 1):
                    p = {"adjacent": bool(adj), "special": False}
                    dfa[(verb, "?", adj, 0, 0, 0, 0, 0)] = "move_valid" if adj else "move_invalid"
                    _ = p
            elif verb == "take":
                for at in (0, 1):
                    dfa[(verb, "?", 0, at, 0, 0, 0, 0)] = "take_valid" if at else "take_invalid"
            elif verb == "drop":
                for in_inv in (0, 1):
                    dfa[(verb, "?", 0, 0, in_inv, 0, 0, 0)] = "drop_valid" if in_inv else "drop_invalid"
            elif verb == "open":
                for opened in (0, 1):
                    for locked in (0, 1):
                        for has_key in (0, 1):
                            if opened:
                                rid = "open_already"
                            elif locked and has_key:
                                rid = "open_locked_with_key"
                            elif locked:
                                rid = "open_locked_no_key"
                            else:
                                rid = "open_unlocked"
                            dfa[(verb, "front_door", 0, 0, 0, locked, has_key, opened)] = rid
                # 打开非门对象 -> 无效
                for t in TARGETS:
                    if t != "front_door" and t != "<none>":
                        dfa[(verb, t, 0, 0, 0, 0, 0, 0)] = "open_invalid"
            elif verb in ("look", "inv", "help"):
                dfa[(verb, "?", 0, 0, 0, 0, 0, 0)] = f"{verb}_valid"
            else:
                dfa[(verb, "?", 0, 0, 0, 0, 0, 0)] = "invalid_action"
        return dfa

    @staticmethod
    def dfa_lookup(dfa: dict, engine, a: Action, st: GameState) -> str:
        """DFA O(1) 查表：确定性返回规则 id。

        按动词规范化键（只保留与迁移相关的谓词位，与 build_dfa 存储一致）：
        go->adjacent；take->item_at_loc；drop->item_in_inv；
        open->target/door_locked/has_key/door_open；其余无谓词。
        """
        p = engine.rule_predicates(a, st)
        if a.verb == "go":
            key = ("go", "?", 1 if p["adjacent"] else 0, 0, 0, 0, 0, 0)
        elif a.verb == "take":
            key = ("take", "?", 0, 1 if p["item_at_loc"] else 0, 0, 0, 0, 0)
        elif a.verb == "drop":
            key = ("drop", "?", 0, 0, 1 if p["item_in_inv"] else 0, 0, 0, 0)
        elif a.verb == "open":
            t = a.target if a.target in TARGETS else "<none>"
            key = ("open", t, 0, 0, 0, 1 if p["door_locked"] else 0,
                   1 if p["has_key"] else 0, 1 if p["door_open"] else 0)
        else:
            key = (a.verb, "?", 0, 0, 0, 0, 0, 0)
        return dfa.get(key, "invalid_action")
