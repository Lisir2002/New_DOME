"""主控制器（新架构的心脏）。

把 编码器 + WS 状态读写 + CRM 规则记忆 + SVSL 自验证循环 + 调度头
整合为一个统一前向计算图。调度决策与状态管理是同一前向的一部分，
可端到端训练。

训练时输出统一的 8 项损失（§6）；推理时给出调度决策 + 确定性状态更新。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sng.config import Config
from sng.core.crm import CRM
from sng.core.dispatch import DispatchHead
from sng.core.svsl import SVSL
from sng.core.ws import StateStream
from sng.engine.world import Action, MiniGameEngine
from sng.engine.vocab import Vocabulary

# 复杂度 ground truth：任务类型 -> 复杂度（推理任务更复杂）
COMPLEXITY_GT = {"comprehension": 0.2, "reasoning": 0.8, "retrieval": 0.5,
                 "generation": 0.3, "lingua": 0.4, "system": 0.1}


class Controller(nn.Module):
    """主控制器。"""

    def __init__(self, cfg: Config, vocab: Vocabulary) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.embed = nn.Embedding(vocab.size, cfg.emb_dim, padding_idx=0)
        self.encoder = nn.GRU(cfg.emb_dim, cfg.enc_hidden, batch_first=True)
        # 状态读取查询投影
        self.read_q = nn.Linear(cfg.enc_hidden, cfg.emb_dim)
        self.ws = StateStream(cfg)
        self.crm = CRM(cfg)
        self.svsl = SVSL(cfg)
        self.dispatch = DispatchHead(cfg)

    # ---- 编码 ----
    def encode_obs(self, obs_ids: list[int]) -> torch.Tensor:
        x = torch.tensor(obs_ids, dtype=torch.long).unsqueeze(0)
        e = self.embed(x)
        _, h = self.encoder(e)
        return h.squeeze(0).squeeze(0)  # [enc_hidden]

    def state_feat_vec(self, state_feats: dict[str, float]) -> torch.Tensor:
        order = ["has_key", "door_open", "door_locked", "inv_len"]
        return torch.tensor([state_feats[k] for k in order], dtype=torch.float32)

    # ---- 训练：单样本统一前向，返回全部损失 ----
    def forward_train(self, sample: dict, engine: MiniGameEngine) -> dict:
        obs_ids = self.vocab.encode(sample["obs_zh"], "zh", max_len=48)
        a: Action = sample["action"]
        h = self.encode_obs(obs_ids)

        S0 = self.ws.encode_state(sample["state_before"])
        r, _ = self.ws.read(S0, self.read_q(h))

        # 调度头（监督：task_type + 插件集合 + 复杂度）
        disp = self.dispatch.forward(h, r)
        gt_tt = [sample["task_type"]]
        L_dispatch = self.dispatch.dispatch_loss(disp["task_logits"], gt_tt)
        L_plugin = self.dispatch.plugin_loss(disp["plugin_logits"], sample["dispatch"])
        L_comp = nn.functional.mse_loss(
            disp["complexity"].squeeze(0),
            torch.tensor(COMPLEXITY_GT[sample["task_type"]], dtype=torch.float32))

        # CRM 软索引（监督：真值规则）
        preds = engine.rule_predicates(a, sample["state_before"])
        alpha = self.crm.forward(a.verb, a.target, preds)
        L_crm = self.crm.match_loss(alpha, sample["rule_id"])

        # WS 可微写（世界模型，监督：引擎下一状态）
        upd = self.ws.write_candidate(S0, h)
        S1_gt = self.ws.encode_state(sample["state_after"])
        L_state = nn.functional.mse_loss(upd["typed_targets"], S1_gt[: self.cfg.n_typed])

        # SVSL 验证：正样本（真实迁移）+ 负样本（扰动迁移）
        sf = self.state_feat_vec(engine.state_features(sample["state_before"]))
        real_diff = (S1_gt[: self.cfg.n_typed] - S0[: self.cfg.n_typed]).detach()
        L_verify_pos = self.svsl.verify_loss(h, sf, real_diff, 1.0)
        corrupt = torch.randn_like(real_diff) * 0.5
        L_verify_neg = self.svsl.verify_loss(h, sf, corrupt, 0.0)

        # SVSL 修正头：当可微写有误时，修正头应输出引擎真值
        pred_diff = (upd["typed_targets"] - S0[: self.cfg.n_typed]).detach()
        fail_diff = pred_diff - real_diff
        L_correct = self.svsl.correct_loss(h, sf, fail_diff, S1_gt[: self.cfg.n_typed])

        # 稀疏正则（自由槽寻址 + 规则索引）
        L_sparse = upd["free_attn"].sum() * 1e-3 + (alpha * alpha.log().clamp_min(-20)).abs().sum() * 1e-3

        return {
            "L_dispatch": L_dispatch, "L_comp": L_comp, "L_crm": L_crm,
            "L_plugin": L_plugin, "L_state": L_state,
            "L_verify": (L_verify_pos + L_verify_neg) * 0.5,
            "L_correct": L_correct, "L_sparse": L_sparse,
        }

    # ---- 推理：确定性状态更新 + 调度决策 ----
    def infer(self, obs_ids: list[int], a: Action, S: torch.Tensor,
              engine: MiniGameEngine, state: object,
              use_dfa: bool = True) -> dict:
        h = self.encode_obs(obs_ids)
        r, _ = self.ws.read(S, self.read_q(h))
        disp = self.dispatch.predict(h, r, self.cfg)

        preds = engine.rule_predicates(a, state)
        if use_dfa:
            dfa = self.crm.build_dfa(engine)
            rule_id = self.crm.dfa_lookup(dfa, engine, a, state)
        else:
            rule_id, alpha = self.crm.predict_rule(a.verb, a.target, preds)
        ops = engine.transition_ops(rule_id, a)
        if ops:
            S_new = self.ws.apply_ops(S, ops, lambda s: self.ws.state_emb(
                torch.tensor(type(self).tok(s), dtype=torch.long)))
        else:
            S_new = S

        sf = self.state_feat_vec(engine.state_features(state))
        diff = S_new[: self.cfg.n_typed] - S[: self.cfg.n_typed]
        S_final, iters, v = self.svsl.loop(S, S_new, h, sf, diff)
        return {
            "rule_id": rule_id, "ops": ops, "S_final": S_final,
            "svsl_iters": iters, "verify_score": v, **disp,
        }

    @staticmethod
    def tok(name: str) -> int:
        from sng.core.ws import STATE_VOCAB
        return STATE_VOCAB.get(name, 0)
