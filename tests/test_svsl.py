"""SVSL 自验证循环头测试：一致性检查与验证打分。"""
import torch

from sng.config import Config
from sng.core.svsl import SVSL
from sng.engine.world import MiniGameEngine


def test_consistency_check_clean_state():
    cfg = Config()
    e = MiniGameEngine()
    S = __import__("sng.core.ws", fromlist=["StateStream"]).StateStream(cfg).encode_state(e.state)
    ok, issues = SVSL.consistency_check(S, cfg)
    assert ok is True
    assert issues == []


def test_consistency_check_detects_conflict():
    cfg = Config()
    e = MiniGameEngine()
    ws = __import__("sng.core.ws", fromlist=["StateStream"]).StateStream(cfg)
    S = ws.encode_state(e.state)
    # 人为制造 flag 冲突：门同时开着且锁着
    S[cfg.slot_flag + 0] = S[cfg.slot_flag + 0] + ws.state_emb.weight[11]  # door_open
    ok, issues = SVSL.consistency_check(S, cfg)
    assert ok is False
    assert any("冲突" in i for i in issues)


def test_verify_head_separates_valid_and_corrupt():
    cfg = Config()
    svsl = SVSL(cfg)
    h = torch.randn(cfg.emb_dim)
    sf = torch.tensor([1.0, 0.0, 1.0, 1.0])
    real_diff = torch.randn(cfg.n_typed, cfg.emb_dim) * 0.1
    corrupt = torch.randn(cfg.n_typed, cfg.emb_dim) * 2.0
    opt = torch.optim.Adam(svsl.verify_head.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = svsl.verify_loss(h, sf, real_diff, 1.0) + svsl.verify_loss(h, sf, corrupt, 0.0)
        loss.backward()
        opt.step()
    assert svsl.verify_score(h, sf, real_diff).item() > 0.5   # 真实迁移判为合法
    assert svsl.verify_score(h, sf, corrupt).item() < 0.5     # 扰动迁移判为不合法
