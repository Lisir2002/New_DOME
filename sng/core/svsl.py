"""SVSL 自验证循环头（Self-Verifying State Loop）。

在计算图内做"候选更新 -> 验证打分 -> 失败则修正 -> 重试（≤K 次）"。
- verify_head：学到的判别打分器，输出 P(更新合法)；
- correct_head：失败时产出修正后的更新；
- 确定性检查（consistency_check）：对类型化槽做规则一致性校验，
  训练时与软打分结合，推理时可完全用确定性检查兜底。

更新差异以"逐槽 L2 范数"作为紧凑特征（哪个槽变了、变了多少），
既保留可微性又控制验证/修正头的输入规模。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sng.config import Config
from sng.core.ws import FLAG_KEYS


class SVSL(nn.Module):
    """自验证循环头。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.emb_dim
        n_typed = cfg.n_typed
        # 验证打分器：h + 状态特征(4) + 逐槽差异特征(n_typed)
        self.verify_head = nn.Sequential(
            nn.Linear(d + 4 + n_typed, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        # 修正头：h + 状态特征 + 逐槽差异特征 -> 修正后的类型化槽目标值
        self.correct_head = nn.Sequential(
            nn.Linear(d + 4 + n_typed, d // 2), nn.ReLU(),
            nn.Linear(d // 2, n_typed * d))

    # ---- 逐槽差异特征 ----
    @staticmethod
    def diff_feat(diff: torch.Tensor) -> torch.Tensor:
        """diff 可为 [n_typed, d] 矩阵或任意向量，归约为 [n_typed] 逐槽范数。"""
        if diff.dim() == 2 and diff.size(0) > 1:
            return diff.norm(dim=-1)
        return diff.reshape(-1)[: 16] if diff.numel() != 16 else diff

    # ---- 确定性一致性检查（规则层）----
    @staticmethod
    def consistency_check(S: torch.Tensor, cfg: Config) -> tuple[bool, list[str]]:
        """对类型化槽做一致性校验。返回 (是否一致, 问题列表)。"""
        issues: list[str] = []
        # 背包槽：无重复物品、数量不超过容量
        seen: set[int] = set()
        count = 0
        for k in range(cfg.max_inv):
            slot = S[cfg.slot_inv + k]
            if slot.norm() < 1e-6:
                continue
            count += 1
            tok = int(slot.argmax().item()) if slot.norm() > 0 else -1
            if tok in seen:
                issues.append("背包出现重复物品")
            seen.add(tok)
        if count > cfg.max_inv:
            issues.append("背包超容量")
        # flag 一致性：door_open 为真则不应 door_locked 为真
        o = FLAG_KEYS.index("door_open")
        l = FLAG_KEYS.index("door_locked")
        open_v = S[cfg.slot_flag + o].norm().item() > 0.5
        locked_v = S[cfg.slot_flag + l].norm().item() > 0.5
        if open_v and locked_v:
            issues.append("门同时开着且锁着（flag 冲突）")
        return (len(issues) == 0, issues)

    # ---- 学到的验证打分 ----
    def verify_score(self, h: torch.Tensor, state_feat: torch.Tensor,
                     diff: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, state_feat, self.diff_feat(diff)])
        return torch.sigmoid(self.verify_head(x)).squeeze(0)

    def verify_loss(self, h: torch.Tensor, state_feat: torch.Tensor,
                    diff: torch.Tensor, label: float) -> torch.Tensor:
        p = self.verify_score(h, state_feat, diff)
        target = torch.tensor(label, dtype=torch.float32)
        return nn.functional.binary_cross_entropy(p, target)

    # ---- 修正头 ----
    def correct(self, h: torch.Tensor, state_feat: torch.Tensor,
                fail_diff: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, state_feat, self.diff_feat(fail_diff)])
        return self.correct_head(x).view(self.cfg.n_typed, self.cfg.emb_dim)

    def correct_loss(self, h: torch.Tensor, state_feat: torch.Tensor,
                     fail_diff: torch.Tensor,
                     target_typed: torch.Tensor) -> torch.Tensor:
        pred = self.correct(h, state_feat, fail_diff)
        return nn.functional.mse_loss(pred, target_typed)

    # ---- 循环（推理用）：≤K 次验证-修正 ----
    def loop(self, S: torch.Tensor, S_candidate: torch.Tensor, h: torch.Tensor,
             state_feat: torch.Tensor, diff: torch.Tensor) -> tuple[torch.Tensor, int, float]:
        """验证-修正循环。返回 (最终状态, 迭代次数, 最终验证分数)。"""
        S_cur = S_candidate
        v = 0.0
        cur_diff = diff
        for it in range(1, self.cfg.svsl_max_iters + 1):
            ok, _ = self.consistency_check(S_cur, self.cfg)
            v = self.verify_score(h, state_feat, cur_diff).item()
            if ok and v > 0.5:
                return S_cur, it, v
            # 失败：用修正头产出修正后的类型化槽
            corrected = self.correct(h, state_feat, cur_diff)
            S_cur = S_cur.clone()
            S_cur[: self.cfg.n_typed] = corrected
            cur_diff = S_cur[: self.cfg.n_typed].detach() - S[: self.cfg.n_typed].detach()
        return S_cur, self.cfg.svsl_max_iters, v
