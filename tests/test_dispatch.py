"""调度头测试：多意图预测 + 能力映射。"""
import torch

from sng.config import Config
from sng.core.dispatch import DispatchHead


def test_tasktype_vocab_complete():
    from sng.core.dispatch import TASK_TYPES
    assert set(TASK_TYPES) >= {"comprehension", "reasoning", "retrieval",
                               "generation", "lingua", "system"}


def test_predict_returns_candidate_plugins():
    cfg = Config()
    head = DispatchHead(cfg)
    h = torch.randn(cfg.emb_dim)
    r = torch.randn(cfg.emb_dim)
    out = head.predict(h, r, cfg)
    assert isinstance(out["task_types"], list)
    assert isinstance(out["candidate_plugins"], set)
    assert 0.0 <= out["complexity"] <= 1.0


def test_dispatch_loss_learning():
    import torch.nn as nn
    cfg = Config()
    head = DispatchHead(cfg)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    h = torch.randn(cfg.emb_dim, requires_grad=False)
    r = torch.randn(cfg.emb_dim, requires_grad=False)
    for _ in range(200):
        opt.zero_grad()
        logits = head.forward(h, r)["task_logits"]
        loss = head.dispatch_loss(logits, ["reasoning"])
        loss.backward()
        opt.step()
    out = head.predict(h, r, cfg)
    assert "reasoning" in out["task_types"]
